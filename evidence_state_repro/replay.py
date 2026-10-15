"""Replay aggregate metrics from text-free frozen score projections.

This verifies aggregation and selected membership. It is not a substitute for
answer-level rescoring, retraining, or historical API-output regeneration.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from .selection import fixed_budget


def load_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    qids = [r["qid"] for r in rows]
    if len(qids) != len(set(qids)):
        raise ValueError(f"Duplicate QIDs in {Path(path).name}")
    return rows


def verify_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "provenance/file_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        target = (root / record["path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Manifest path escapes candidate root")
        if not target.is_file():
            raise ValueError(f"Missing candidate file: {record['path']}")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Candidate hash mismatch: {record['path']}")
    return len(manifest["files"])


def binary_summary(rows, method, initial="Initial", metric="em"):
    values = [r["methods"][method] for r in rows]
    before = [r["methods"][initial] for r in rows]
    correct = sum(int(v[metric]) for v in values)
    w2r = sum(not bool(b[metric]) and bool(v[metric]) for b, v in zip(before, values))
    r2w = sum(bool(b[metric]) and not bool(v[metric]) for b, v in zip(before, values))
    result = {"n": len(rows), "correct_count": correct,
              "em_percent": correct / len(rows) * 100,
              "W2R": w2r, "R2W": r2w, "Net": w2r - r2w}
    if "f1" in values[0]:
        result["token_f1_percent"] = statistics.mean(v["f1"] for v in values) * 100
    return result


def assert_close(actual, expected, name, tolerance=1e-8):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert_close(actual[key], value, name + "." + key, tolerance)
    elif isinstance(expected, (int, float)):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
            raise AssertionError(f"{name}: {actual} != frozen expectation {expected}")
    elif actual != expected:
        raise AssertionError(f"{name}: differs from frozen expectation")


def replay(root):
    root = Path(root)
    verified_files = verify_manifest(root)
    expectations = json.loads((root / "provenance/expected_metrics.json").read_text(encoding="utf-8"))
    dev = load_rows(root / "frozen_scores/dev1096_oof.jsonl")
    if len(dev) != 1096:
        raise AssertionError("Dev1096 population mismatch")
    k = math.ceil(len(dev) * 0.20)
    oof = {}
    for variant in dev[0]["scores"]:
        ranked = sorted(dev, key=lambda r: (-r["scores"][variant], r["qid"]))
        selected = ranked[:k]
        errors = sum(r["initial_wrong_label"] for r in selected)
        oof[variant] = {"selected": k, "errors": errors,
                        "error_precision_percent": errors / k * 100}
        if set(r["qid"] for r in selected) != set(r["qid"] for r in dev if r["saved_selected"][variant]):
            raise AssertionError(f"OOF saved selection mismatch: {variant}")
    locked = load_rows(root / "frozen_scores/locked1341.jsonl")
    if len(locked) != 1341:
        raise AssertionError("locked1341 population mismatch")
    locked_result = {m: binary_summary(locked, m) for m in locked[0]["methods"]}
    random_names = sorted(m for m in locked_result if m.startswith("Random-"))
    random_aggregate = {
        "em_mean_percent": statistics.mean(locked_result[m]["em_percent"] for m in random_names),
        "em_sample_sd_percent": statistics.stdev(locked_result[m]["em_percent"] for m in random_names),
        "f1_mean_percent": statistics.mean(locked_result[m]["token_f1_percent"] for m in random_names),
        "f1_sample_sd_percent": statistics.stdev(locked_result[m]["token_f1_percent"] for m in random_names),
    }
    controlled = load_rows(root / "frozen_scores/dev1096_controlled.jsonl")
    controlled_result = {m: binary_summary(controlled, m) for m in ("A0", "A4", "A6")}
    m3 = load_rows(root / "frozen_scores/m3vqa13125.jsonl")
    if len(m3) != 13125:
        raise AssertionError("M3 population mismatch")
    m3_result = {}
    for method in ("Initial", "B3-Refit", "Random-20"):
        result = binary_summary(m3, method, metric="full_match")
        result["strict_full_match_percent"] = result.pop("em_percent")
        result["answer_list_accuracy_percent"] = statistics.mean(r["methods"][method]["accuracy"] for r in m3) * 100
        m3_result[method] = result
    selection_results = {}
    for dataset, population, forced_n, budget in (("locked1341", 1341, 6, 269), ("m3vqa13125", 13125, 515, 2625)):
        scores = load_rows(root / f"frozen_scores/{dataset}_selection.jsonl")
        if len(scores) != population:
            raise AssertionError("Selection population mismatch")
        forced = [r["qid"] for r in scores if r["forced"]]
        if len(forced) != forced_n:
            raise AssertionError("Forced-case count mismatch")
        ranking_scores = {r["qid"]: r["risk_score"] for r in scores if not r["forced"]}
        selected = fixed_budget([r["qid"] for r in scores], ranking_scores, forced)
        saved = {r["qid"] for r in scores if r["selected"]}
        if len(selected) != budget or set(selected) != saved:
            raise AssertionError(f"Frozen selected membership mismatch: {dataset}")
        selection_results[dataset] = {"population": population, "selected": budget, "forced": forced_n, "membership_equal": True}
    assert_close(oof, expectations["dev_oof"], "dev_oof")
    assert_close(locked_result, expectations["locked1341"], "locked1341")
    assert_close(controlled_result, expectations["controlled"], "controlled")
    assert_close(m3_result, expectations["m3"], "m3")
    assert_close(random_aggregate, expectations["random20_aggregate"], "random20_aggregate")
    return {"status": "PASS_FROZEN_SCORE_AGGREGATION_AND_SELECTION_REPLAY",
            "boundary": "Frozen score aggregation and selection membership only; answer rescoring is a separate private QA step.",
            "verified_candidate_files": verified_files,
            "dev_oof": oof, "locked1341": locked_result,
            "random20_aggregate": random_aggregate, "controlled": controlled_result,
            "m3": m3_result, "selection": selection_results,
            "training_calls": 0, "generation_api_calls": 0,
            "original_files_written": 0}
