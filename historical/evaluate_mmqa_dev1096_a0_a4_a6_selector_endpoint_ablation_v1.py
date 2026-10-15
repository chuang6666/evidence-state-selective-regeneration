#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""One-time offline Gold evaluation for frozen Dev1096 endpoint predictions.

The script refuses to read Gold until all Stage-A files and their before-Gold
manifest have passed integrity checks and the statistical protocol has been
frozen.  It never calls an API, trains a model, or changes a selection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
EXPECTED_N = 1096
EXPECTED_TOP_K = 220
VARIANTS = ("A0", "A4", "A6")
TOP20_ERROR_PRECISION = {"A0": 0.863636, "A4": 0.904545, "A6": 0.913636}
SELECTED_SET_SHA = {
    "A0": "8308040BC6179DDE8AAF0ECC723EDFFF1E321E82A1544D2FE196D31E75BA0E3A",
    "A4": "9F93EFB5E9D9584571AB674C2432C4A422368274BEC93A707F7025EBB4410965",
    "A6": "ADCEB7F524F3D8BE1E562752CC71637A7AB6D48C10D315B10E6D83C2ACC20D66",
}
EXPECTED_CANDIDATES_SHA = "AC411E5DF2F998679F6093557FD1A622F4E265DDE63D35C3FB64F3CC9FAB423D"

OUT_DIR = ROOT / "outputs" / "mmqa_dev1096_a0_a4_a6_selector_endpoint_ablation_v1"
FINAL_FILES = {
    "A0": OUT_DIR / "a0_dev1096_endpoint_final_predictions_before_gold_v1.jsonl",
    "A4": OUT_DIR / "a4_dev1096_endpoint_final_predictions_before_gold_v1.jsonl",
    "A6": OUT_DIR / "a6_dev1096_endpoint_final_predictions_before_gold_v1.jsonl",
}
STAGE_A_CONTRACT = OUT_DIR / "selector_endpoint_final_prediction_contract_v1.json"
STAGE_A_AUDIT = OUT_DIR / "selector_endpoint_before_gold_audit_v1.json"
STAGE_A_MEMBERSHIP = OUT_DIR / "selector_endpoint_final_prediction_membership_audit_v1.json"
STAGE_A_SHAS = OUT_DIR / "SHA256SUMS_BEFORE_GOLD.json"

GOLD = ROOT / "data" / "raw" / "mmqa_official" / "MMQA_dev.jsonl.gz"
EXPECTED_GOLD_SHA = "2E348CA574B2DC368E84671709689070F943A59F2E08E6B6E374705DEB712D31"
OFFICIAL_EVALUATOR = ROOT / "audit_mmqa_v3_locked1341_frozen_boundary_v1_1.py"
EXPECTED_OFFICIAL_EVALUATOR_SHA = "16C3BB79BF93DB73381977A2A032F6C54771F7BBF7E1EE19CB20D4713A87C85B"
CLEAN_V2_OFFICIAL_PROTOCOL = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_one_time_official_eval_v1"
    / "one_time_official_eval_protocol_v1.json"
)
EXPECTED_CLEAN_V2_OFFICIAL_PROTOCOL_SHA = "1DF18E87A697AB15D54563877B890B4A9196B1E76CE63586E99DC1AFC7E90A48"
OOF_LABELS = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_answer_risk_oof_v1"
    / "dev1096_saved_answer_error_labels_and_shared_folds_v1.jsonl"
)
EXPECTED_OOF_LABELS_SHA = "712195863BA08DAE57E92462B18531C3A80E4D9CF798404A8FD5FF9F1DB4DE88"

EVAL_PROTOCOL = OUT_DIR / "selector_endpoint_gold_evaluation_protocol_v1.json"
DETAILS = OUT_DIR / "selector_endpoint_official_gold_eval_details_v1.jsonl"
SUMMARY = OUT_DIR / "selector_endpoint_official_gold_eval_summary_v1.json"
GROUPED = OUT_DIR / "selector_endpoint_grouped_evaluation_v1.json"
SIGNIFICANCE = OUT_DIR / "selector_endpoint_pairwise_significance_v1.json"
SHA_AFTER_GOLD = OUT_DIR / "SHA256SUMS_AFTER_GOLD.json"

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_EM_SEED = 260817
BOOTSTRAP_F1_SEED = 260818
PAIRWISE = (("A4", "A0"), ("A6", "A0"), ("A6", "A4"))


class EvaluationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise EvaluationError(f"FAIL-STOP: {message}")


def safe(value: Any) -> str:
    return "" if value is None else str(value).strip()


def answer_text(value: Any) -> str:
    """Preserve Stage-A frozen answer text exactly."""
    return "" if value is None else str(value)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        fail(f"missing required file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected.upper():
        fail(f"SHA mismatch: {path}; expected={expected}; actual={actual}")


def qid_set_sha(qids: Iterable[str]) -> str:
    text = "\n".join(sorted(set(qids))) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing JSON: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        fail(f"JSON root is not an object: {path}")
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                fail(f"malformed JSONL {path}:{line_no}: {exc}")
            if not isinstance(row, dict):
                fail(f"non-object JSONL row {path}:{line_no}")
            rows.append(row)
    return rows


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                fail(f"malformed official Gold JSONL {path}:{line_no}: {exc}")
            if not isinstance(row, dict):
                fail(f"non-object official Gold row {path}:{line_no}")
            rows.append(row)
    return rows


def index_qids(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = safe(row.get("qid") or row.get("question_id") or row.get("id"))
        if not qid:
            fail(f"blank QID in {label}")
        if qid in result:
            fail(f"duplicate QID in {label}: {qid}")
        result[qid] = dict(row)
    return result


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def frozen_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != data:
            fail(f"existing frozen output differs: {path}")
        return
    path.write_bytes(data)


def import_official_evaluator() -> Any:
    require_sha(OFFICIAL_EVALUATOR, EXPECTED_OFFICIAL_EVALUATOR_SHA)
    spec = importlib.util.spec_from_file_location("dev1096_frozen_official_evaluator", OFFICIAL_EVALUATOR)
    if spec is None or spec.loader is None:
        fail("cannot import frozen MMQA official evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("extract_gold_answers", "stringify_answer", "best_score"):
        if not hasattr(module, name):
            fail(f"official evaluator missing function: {name}")
    return module


def exact_mcnemar_p(left_better: int, right_better: int) -> float:
    n = left_better + right_better
    if n == 0:
        return 1.0
    k = min(left_better, right_better)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_delta(
    left: Sequence[float], right: Sequence[float], seed: int
) -> dict[str, Any]:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) != EXPECTED_N:
        fail("invalid paired bootstrap vectors")
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for i in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, len(a), size=len(a))
        deltas[i] = float((a[idx] - b[idx]).mean())
    low, high = np.quantile(deltas, [0.025, 0.975])
    observed = float((a - b).mean())
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "observed_delta": observed,
        "observed_delta_percentage_points": 100.0 * observed,
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci95": [float(low), float(high)],
        "ci95_percentage_points": [100.0 * float(low), 100.0 * float(high)],
        "ci_method": "paired percentile bootstrap",
    }


def score_summary(scores: Mapping[str, Mapping[str, float]], qids: Sequence[str]) -> dict[str, Any]:
    n = len(qids)
    em_count = sum(int(scores[qid]["em"]) for qid in qids)
    f1_sum = sum(float(scores[qid]["f1"]) for qid in qids)
    return {
        "n": n,
        "em_count": em_count,
        "em": em_count / n if n else None,
        "em_percent": 100.0 * em_count / n if n else None,
        "f1": f1_sum / n if n else None,
        "f1_percent": 100.0 * f1_sum / n if n else None,
    }


def migration(
    initial_scores: Mapping[str, Mapping[str, float]],
    final_scores: Mapping[str, Mapping[str, float]],
    qids: Sequence[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for qid in qids:
        before = int(initial_scores[qid]["em"])
        after = int(final_scores[qid]["em"])
        key = {
            (0, 1): "W2R", (1, 0): "R2W", (0, 0): "W2W", (1, 1): "R2R"
        }[(before, after)]
        counts[key] += 1
    result = {key: counts[key] for key in ("W2R", "R2W", "W2W", "R2R")}
    result["Net"] = result["W2R"] - result["R2W"]
    if sum(result[key] for key in ("W2R", "R2W", "W2W", "R2R")) != len(qids):
        fail("migration partition does not sum to the evaluation universe")
    return result


def validate_stage_a() -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    manifest = read_json(STAGE_A_SHAS)
    if manifest.get("status") != "A0_A4_A6_ENDPOINT_FINAL_PREDICTIONS_FROZEN_BEFORE_GOLD":
        fail("Stage-A SHA manifest status mismatch")
    if manifest.get("gold_access") != 0 or manifest.get("api_calls") != 0:
        fail("Stage-A boundary flags are not zero")
    expected_files = [STAGE_A_CONTRACT, STAGE_A_AUDIT, STAGE_A_MEMBERSHIP, *(FINAL_FILES[v] for v in VARIANTS)]
    listed = manifest.get("files")
    if not isinstance(listed, dict) or set(listed) != {path.name for path in expected_files}:
        fail("Stage-A SHA manifest file set mismatch")
    for path in expected_files:
        require_sha(path, safe(listed[path.name]))

    contract = read_json(STAGE_A_CONTRACT)
    audit = read_json(STAGE_A_AUDIT)
    membership = read_json(STAGE_A_MEMBERSHIP)
    if contract.get("status") != "MMQA_DEV1096_A0_A4_A6_ENDPOINT_FINAL_PREDICTION_CONTRACT_V1_FROZEN_BEFORE_GOLD":
        fail("Stage-A contract status mismatch")
    if audit.get("status") != "A0_A4_A6_ENDPOINT_FINAL_PREDICTIONS_FROZEN_BEFORE_GOLD":
        fail("Stage-A audit status mismatch")
    if membership.get("status") != "MMQA_DEV1096_A0_A4_A6_ENDPOINT_MEMBERSHIP_AUDIT_PASS":
        fail("Stage-A membership audit status mismatch")
    frozen_candidate = contract.get("frozen_candidate", {})
    if frozen_candidate.get("sha256") != EXPECTED_CANDIDATES_SHA:
        fail("Stage-A candidate SHA contract mismatch")

    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    qid_universe: set[str] | None = None
    for variant in VARIANTS:
        rows = read_jsonl(FINAL_FILES[variant])
        by_qid = index_qids(rows, f"Stage-A {variant} predictions")
        if len(by_qid) != EXPECTED_N:
            fail(f"Stage-A {variant} does not contain 1096 unique QIDs")
        if qid_universe is None:
            qid_universe = set(by_qid)
        elif set(by_qid) != qid_universe:
            fail("Stage-A variant QID universes differ")
        selected = {qid for qid, row in by_qid.items() if row.get("selected") is True}
        if len(selected) != EXPECTED_TOP_K or qid_set_sha(selected) != SELECTED_SET_SHA[variant]:
            fail(f"Stage-A {variant} Top220 membership changed")
        for qid, row in by_qid.items():
            if safe(row.get("variant")) != variant or row.get("gold_used") is not False:
                fail(f"Stage-A row contract mismatch: {variant}:{qid}")
            initial = answer_text(row.get("initial_answer"))
            final = answer_text(row.get("final_answer"))
            action = safe(row.get("action"))
            candidate = answer_text(row.get("candidate_answer"))
            if action == "KEEP_INITIAL_NOT_SELECTED":
                if row.get("selected") is not False or final != initial:
                    fail(f"non-selected merge rule violation: {variant}:{qid}")
            elif action == "USE_FROZEN_REGEN":
                if row.get("selected") is not True or safe(row.get("candidate_status")) != "success" or final != candidate:
                    fail(f"successful merge rule violation: {variant}:{qid}")
            elif action == "KEEP_INITIAL_TERMINAL":
                if row.get("selected") is not True or safe(row.get("candidate_status")) != "keep_initial_terminal" or final != initial:
                    fail(f"terminal merge rule violation: {variant}:{qid}")
            else:
                fail(f"unknown Stage-A action: {variant}:{qid}:{action}")
        indexed[variant] = by_qid
    assert qid_universe is not None
    for qid in sorted(qid_universe):
        initials = {answer_text(indexed[v][qid].get("initial_answer")) for v in VARIANTS}
        if len(initials) != 1:
            fail(f"Initial answer differs across variants: {qid}")
    return indexed, manifest


def evaluation_protocol(stage_a_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "MMQA_DEV1096_A0_A4_A6_ENDPOINT_GOLD_EVALUATION_PROTOCOL_V1_FROZEN",
        "stage_a": {
            "status_required": "A0_A4_A6_ENDPOINT_FINAL_PREDICTIONS_FROZEN_BEFORE_GOLD",
            "sha_manifest_path": str(STAGE_A_SHAS),
            "sha_manifest_sha256": sha256_file(STAGE_A_SHAS),
            "files": dict(stage_a_manifest["files"]),
        },
        "authoritative_gold": {
            "path": str(GOLD), "sha256": EXPECTED_GOLD_SHA,
            "same_source_as_clean_v2_locked_official_evaluation": True,
        },
        "official_scoring": {
            "source": str(OFFICIAL_EVALUATOR),
            "source_sha256": EXPECTED_OFFICIAL_EVALUATOR_SHA,
            "functions": ["extract_gold_answers", "stringify_answer", "best_score"],
            "same_as_clean_v2_locked_official_evaluation": True,
        },
        "historic_oof_label_boundary": {
            "path": str(OOF_LABELS), "sha256": EXPECTED_OOF_LABELS_SHA,
            "frozen_top20_error_precision_reused_not_retrained": TOP20_ERROR_PRECISION,
            "official_initial_em_vs_saved_oof_label_parity_reported_after_gold": True,
            "parity_diagnostic_must_not_change_selection_or_candidates": True,
        },
        "reported_metrics": [
            "Initial EM", "Initial F1", "Final EM", "Final F1",
            "W2R", "R2W", "W2W", "R2R", "Net",
            "selected Initial/Final/delta EM/F1", "Repair Success Rate", "Harm Rate",
        ],
        "paired_statistics": {
            "comparisons": [f"{left}-{right}" for left, right in PAIRWISE],
            "mcnemar": "two-sided exact binomial on paired exact-correctness discordances",
            "bootstrap": "paired QID resampling; percentile 95% CI",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "em_seed": BOOTSTRAP_EM_SEED,
            "f1_seed": BOOTSTRAP_F1_SEED,
        },
        "prohibitions": {
            "api_calls": 0, "regeneration": 0, "risk_training": 0,
            "variant_reselection": 0, "top220_modification": 0,
            "locked1341_gold_access": 0, "m3_gold_access": 0,
            "post_result_statistical_method_selection": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--authorize-dev-gold", action="store_true")
    args = parser.parse_args()
    if not args.run_all or not args.authorize_dev_gold:
        fail("both --run-all and --authorize-dev-gold are required")

    predictions, stage_a_manifest = validate_stage_a()
    require_sha(OOF_LABELS, EXPECTED_OOF_LABELS_SHA)
    require_sha(CLEAN_V2_OFFICIAL_PROTOCOL, EXPECTED_CLEAN_V2_OFFICIAL_PROTOCOL_SHA)
    clean_v2_protocol = read_json(CLEAN_V2_OFFICIAL_PROTOCOL)
    if safe(clean_v2_protocol.get("gold", {}).get("sha256")) != EXPECTED_GOLD_SHA:
        fail("Clean V2 official protocol Gold SHA differs")
    if safe(clean_v2_protocol.get("official_scoring_contract", {}).get("source_sha256")) != EXPECTED_OFFICIAL_EVALUATOR_SHA:
        fail("Clean V2 official scorer SHA differs")

    # Freeze the full evaluation/statistical protocol before touching Gold.
    frozen_write(EVAL_PROTOCOL, json_bytes(evaluation_protocol(stage_a_manifest)))

    require_sha(GOLD, EXPECTED_GOLD_SHA)
    official = import_official_evaluator()
    gold_all = index_qids(read_gzip_jsonl(GOLD), "official MMQA Dev Gold")
    qids = sorted(predictions["A0"])
    if not set(qids).issubset(gold_all):
        fail("official Gold does not cover the exact frozen Dev1096 universe")
    gold = {qid: official.extract_gold_answers(gold_all[qid]) for qid in qids}
    if any(not isinstance(gold[qid], list) or not gold[qid] for qid in qids):
        fail("empty/malformed Gold answer list in Dev1096")

    initial_answers = {qid: answer_text(predictions["A0"][qid].get("initial_answer")) for qid in qids}
    final_answers = {
        variant: {qid: answer_text(predictions[variant][qid].get("final_answer")) for qid in qids}
        for variant in VARIANTS
    }
    initial_scores: dict[str, dict[str, float]] = {}
    final_scores: dict[str, dict[str, dict[str, float]]] = {v: {} for v in VARIANTS}
    for qid in qids:
        em, f1 = official.best_score(initial_answers[qid], gold[qid])
        initial_scores[qid] = {"em": int(em), "f1": float(f1)}
        for variant in VARIANTS:
            em, f1 = official.best_score(final_answers[variant][qid], gold[qid])
            final_scores[variant][qid] = {"em": int(em), "f1": float(f1)}

    label_rows = index_qids(read_jsonl(OOF_LABELS), "saved OOF labels")
    if set(label_rows) != set(qids):
        fail("saved OOF label QID universe differs from endpoint universe")
    label_mismatch_qids = [
        qid for qid in qids
        if int(label_rows[qid].get("initial_wrong_label")) != 1 - int(initial_scores[qid]["em"])
    ]
    frozen_precision_checks: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [qid for qid in qids if predictions[variant][qid].get("selected") is True]
        old_label_precision = sum(int(label_rows[qid]["initial_wrong_label"]) for qid in selected) / EXPECTED_TOP_K
        if abs(old_label_precision - TOP20_ERROR_PRECISION[variant]) > 1e-6:
            fail(f"frozen Top20 Error Precision mismatch for {variant}")
        frozen_precision_checks[variant] = {
            "reported_frozen_value": TOP20_ERROR_PRECISION[variant],
            "recomputed_from_saved_oof_labels_for_integrity_only": old_label_precision,
            "selection_or_model_recomputed": False,
        }

    initial_summary = score_summary(initial_scores, qids)
    grouped_out: dict[str, Any] = {}
    variants_summary: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [qid for qid in qids if predictions[variant][qid].get("selected") is True]
        unselected = [qid for qid in qids if qid not in set(selected)]
        full_final = score_summary(final_scores[variant], qids)
        mig = migration(initial_scores, final_scores[variant], qids)
        selected_initial = score_summary(initial_scores, selected)
        selected_final = score_summary(final_scores[variant], selected)
        selected_mig = migration(initial_scores, final_scores[variant], selected)
        initial_wrong_selected = sum(1 - int(initial_scores[qid]["em"]) for qid in selected)
        initial_correct_selected = EXPECTED_TOP_K - initial_wrong_selected
        repair_rate = selected_mig["W2R"] / initial_wrong_selected if initial_wrong_selected else None
        harm_rate = selected_mig["R2W"] / initial_correct_selected if initial_correct_selected else None
        terminal = [qid for qid in selected if safe(predictions[variant][qid].get("action")) == "KEEP_INITIAL_TERMINAL"]
        if any(final_answers[variant][qid] != initial_answers[qid] for qid in terminal):
            fail(f"terminal fallback parity failure after Gold: {variant}")
        if any(final_answers[variant][qid] != initial_answers[qid] for qid in unselected):
            fail(f"low-risk KEEP parity failure after Gold: {variant}")
        variants_summary[variant] = {
            "top20_error_precision_frozen": TOP20_ERROR_PRECISION[variant],
            "final": full_final,
            "delta_em_percentage_points": full_final["em_percent"] - initial_summary["em_percent"],
            "delta_f1_percentage_points": full_final["f1_percent"] - initial_summary["f1_percent"],
            "migration": mig,
            "selected220": {
                "initial": selected_initial, "final": selected_final,
                "delta_em_percentage_points": selected_final["em_percent"] - selected_initial["em_percent"],
                "delta_f1_percentage_points": selected_final["f1_percent"] - selected_initial["f1_percent"],
                "initial_wrong_selected": initial_wrong_selected,
                "initial_correct_selected": initial_correct_selected,
                "migration": selected_mig,
                "repair_success_rate": repair_rate,
                "harm_rate": harm_rate,
                "terminal_keep_initial": len(terminal),
            },
        }
        grouped_out[variant] = {
            "selected220": variants_summary[variant]["selected220"],
            "not_selected876": {
                "initial": score_summary(initial_scores, unselected),
                "final": score_summary(final_scores[variant], unselected),
                "prediction_equal_initial": sum(
                    final_answers[variant][qid] == initial_answers[qid] for qid in unselected
                ),
            },
            "protocol_terminal": {
                "n": len(terminal),
                "fallback_equal_initial": sum(
                    final_answers[variant][qid] == initial_answers[qid] for qid in terminal
                ),
            },
        }

    pairwise_out: dict[str, Any] = {}
    for left, right in PAIRWISE:
        left_em = [int(final_scores[left][qid]["em"]) for qid in qids]
        right_em = [int(final_scores[right][qid]["em"]) for qid in qids]
        left_f1 = [float(final_scores[left][qid]["f1"]) for qid in qids]
        right_f1 = [float(final_scores[right][qid]["f1"]) for qid in qids]
        left_better = sum(a == 1 and b == 0 for a, b in zip(left_em, right_em))
        right_better = sum(a == 0 and b == 1 for a, b in zip(left_em, right_em))
        pairwise_out[f"{left}_vs_{right}"] = {
            "direction": f"{left} minus {right}",
            "em": {
                "left_better": left_better,
                "right_better": right_better,
                "same": EXPECTED_N - left_better - right_better,
                "mcnemar_exact_two_sided_p": exact_mcnemar_p(left_better, right_better),
                "paired_bootstrap": paired_bootstrap_delta(left_em, right_em, BOOTSTRAP_EM_SEED),
            },
            "f1": {
                "paired_bootstrap": paired_bootstrap_delta(left_f1, right_f1, BOOTSTRAP_F1_SEED),
            },
        }

    detail_rows: list[dict[str, Any]] = []
    for qid in qids:
        row: dict[str, Any] = {
            "qid": qid,
            "gold_answers": gold[qid],
            "initial_answer": initial_answers[qid],
            "initial_em": int(initial_scores[qid]["em"]),
            "initial_f1": float(initial_scores[qid]["f1"]),
            "saved_oof_initial_wrong_label": int(label_rows[qid]["initial_wrong_label"]),
            "saved_oof_label_matches_official_initial_em": qid not in set(label_mismatch_qids),
        }
        for variant in VARIANTS:
            pred = predictions[variant][qid]
            row[variant] = {
                "selected": bool(pred.get("selected")),
                "action": safe(pred.get("action")),
                "final_answer": final_answers[variant][qid],
                "final_em": int(final_scores[variant][qid]["em"]),
                "final_f1": float(final_scores[variant][qid]["f1"]),
            }
        detail_rows.append(row)

    summary_out = {
        "status": "MMQA_DEV1096_A0_A4_A6_SELECTOR_ENDPOINT_ABLATION_V1_COMPLETE",
        "n": EXPECTED_N,
        "initial": initial_summary,
        "variants": variants_summary,
        "frozen_top20_error_precision_integrity": frozen_precision_checks,
        "oof_label_vs_current_official_initial_em": {
            "matches": EXPECTED_N - len(label_mismatch_qids),
            "mismatches": len(label_mismatch_qids),
            "mismatch_qids": label_mismatch_qids,
            "note": "Diagnostic only; no selection, candidate, fallback, or prediction was changed.",
        },
        "gold_source": {"path": str(GOLD), "sha256": sha256_file(GOLD)},
        "official_scorer": {"path": str(OFFICIAL_EVALUATOR), "sha256": sha256_file(OFFICIAL_EVALUATOR)},
        "api_calls": 0, "regeneration": 0, "risk_training": 0,
        "variant_reselection": 0, "top220_modification": 0,
        "locked1341_gold_access": 0, "m3_gold_access": 0,
    }
    significance_out = {
        "status": "MMQA_DEV1096_A0_A4_A6_ENDPOINT_PAIRED_SIGNIFICANCE_V1_COMPLETE",
        "protocol_frozen_before_gold": True,
        "comparisons": pairwise_out,
    }
    grouped_payload = {
        "status": "MMQA_DEV1096_A0_A4_A6_ENDPOINT_GROUPED_EVALUATION_V1_COMPLETE",
        "variants": grouped_out,
    }

    frozen_write(DETAILS, jsonl_bytes(detail_rows))
    frozen_write(GROUPED, json_bytes(grouped_payload))
    frozen_write(SIGNIFICANCE, json_bytes(significance_out))
    frozen_write(SUMMARY, json_bytes(summary_out))
    generated = [EVAL_PROTOCOL, DETAILS, GROUPED, SIGNIFICANCE, SUMMARY]
    sha_out = {
        "status": "MMQA_DEV1096_A0_A4_A6_SELECTOR_ENDPOINT_ABLATION_V1_COMPLETE",
        "files": {path.name: sha256_file(path) for path in generated},
        "gold_read_after_protocol_freeze": True,
        "api_calls": 0,
    }
    frozen_write(SHA_AFTER_GOLD, json_bytes(sha_out))

    print("=" * 100)
    print("MMQA DEV1096 A0/A4/A6 SELECTOR ENDPOINT ABLATION — OFFICIAL GOLD")
    print(f"Initial EM={initial_summary['em_percent']:.4f} F1={initial_summary['f1_percent']:.4f}")
    for variant in VARIANTS:
        item = variants_summary[variant]
        print(
            f"{variant}: Top20P={item['top20_error_precision_frozen']:.6f} "
            f"EM={item['final']['em_percent']:.4f} F1={item['final']['f1_percent']:.4f} "
            f"W2R={item['migration']['W2R']} R2W={item['migration']['R2W']} Net={item['migration']['Net']}"
        )
    print(f"OOF-label/current-official parity mismatches={len(label_mismatch_qids)}")
    print("API calls=0 | regeneration=0 | risk training=0 | selection change=0")
    print(f"Output: {OUT_DIR}")
    print("STATUS = MMQA_DEV1096_A0_A4_A6_SELECTOR_ENDPOINT_ABLATION_V1_COMPLETE")


if __name__ == "__main__":
    main()
