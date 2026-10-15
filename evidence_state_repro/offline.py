"""Portable frozen-answer rescoring and complete main-table aggregation.

No provider, network, model loading or training imports. Reference answers must
be supplied separately by the user under the benchmark providers' terms.
"""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from . import mmqa_endpoint_historical as mmqa
from . import m3_frozen as m3
from .replay import load_rows, binary_summary, verify_manifest


def safe_output(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Output must be outside the repository (avoid accidental data publication)")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty; existing results are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    return output


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, values):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in values), encoding="utf-8")


def indexed(path):
    return {row["qid"]: row for row in load_rows(path)}


def prepare_references(root, mmqa_data, m3_data, output):
    """Local projection from provider downloads; never downloads/redistributes Gold."""
    root = Path(root).resolve()
    verify_manifest(root)
    def provider_rows(path):
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig") as handle:
            result = {}
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row.get("qid", row.get("question_id", row.get("data_id", ""))))
                if not qid or qid in result:
                    raise ValueError("Provider file has empty or duplicate QIDs")
                result[qid] = row
            return result
    mmqa_rows, m3_rows = provider_rows(mmqa_data), provider_rows(m3_data)
    datasets = {}
    for name in ("locked1341", "dev1096_controlled", "m3vqa13125"):
        qids = sorted(indexed(root / "frozen_predictions" / (name + ".jsonl")))
        source = m3_rows if name == "m3vqa13125" else mmqa_rows
        if not set(qids) <= set(source):
            raise ValueError(name + ": provider file is missing required QIDs")
        if name == "m3vqa13125":
            datasets[name] = [{"qid": qid, "answer_evals": m3.validate_answer_evals(source[qid].get("answer_evals"), qid)} for qid in qids]
        else:
            datasets[name] = [{"qid": qid, "answers": mmqa.extract_gold_answers(source[qid])} for qid in qids]
    output = safe_output(root, output)
    for name, data in datasets.items():
        write_rows(output / (name + ".jsonl"), data)
    receipt = {"status": "LOCAL_REFERENCE_FILES_PREPARED_NOT_FOR_UPLOAD", "populations": {n: len(v) for n, v in datasets.items()},
               "provider_file_sha256": {"mmqa": hashlib.sha256(Path(mmqa_data).read_bytes()).hexdigest(),
                                         "m3": hashlib.sha256(Path(m3_data).read_bytes()).hexdigest()},
               "rule": "MMQA original reference members; M3 answer_evals copied unchanged, QID=data_id for raw M3 questions."}
    dump(output / "reference_preparation.json", receipt)
    return receipt


def rescore(root, reference_dir, output):
    root = Path(root).resolve()
    verify_manifest(root)
    reference_dir = Path(reference_dir).resolve()
    datasets = ("locked1341", "dev1096_controlled", "m3vqa13125")
    prepared, summary = {}, {}
    for name in datasets:
        predictions = load_rows(root / "frozen_predictions" / (name + ".jsonl"))
        references = indexed(reference_dir / (name + ".jsonl"))
        saved = indexed(root / "frozen_scores" / (name + ".jsonl"))
        if set(references) != set(saved) or {p["qid"] for p in predictions} != set(saved):
            raise ValueError(name + ": exact QID population mismatch")
        scored, mismatches = [], []
        for item in sorted(predictions, key=lambda r: r["qid"]):
            qid, methods = item["qid"], {}
            if set(item) != {"qid", "methods"} or set(item["methods"]) != set(saved[qid]["methods"]):
                raise ValueError(name + ": prediction/arm schema mismatch")
            for method, answer in item["methods"].items():
                if name == "m3vqa13125":
                    refs = m3.validate_answer_evals(references[qid]["answer_evals"], qid)
                    accuracy = m3.official_accuracy(m3.convert_to_list(answer), refs)
                    score = {"accuracy": accuracy, "full_match": bool(abs(accuracy - 1.0) <= 1e-12)}
                else:
                    refs = references[qid].get("answers")
                    if not isinstance(refs, list) or not refs or not all(isinstance(x, str) for x in refs):
                        raise ValueError(name + ": answers must be a nonempty list of reference-member strings")
                    em, f1 = mmqa.best_score(answer, refs)
                    score = {"em": em, "f1": f1}
                for metric, value in score.items():
                    if not math.isclose(float(value), float(saved[qid]["methods"][method][metric]), abs_tol=1e-12, rel_tol=0):
                        mismatches.append({"qid": qid, "method": method, "metric": metric})
                methods[method] = score
            scored.append({"qid": qid, "methods": methods})
        if mismatches:
            raise ValueError(name + ": rescoring mismatch; original scores retained; first mismatches=" + json.dumps(mismatches[:5]))
        prepared[name] = scored
        summary[name] = {"qids": len(scored), "methods": len(scored[0]["methods"]),
                         "score_mismatches": len(mismatches),
                         "reference_file_sha256": hashlib.sha256((reference_dir / (name + ".jsonl")).read_bytes()).hexdigest()}
    output = safe_output(root, output)
    for name, data in prepared.items():
        write_rows(output / (name + ".jsonl"), data)
    result = {"status": "PASS_ANSWER_LEVEL_RESCORING", "datasets": summary,
              "score_files_sha256": {name + ".jsonl": hashlib.sha256((output / (name + ".jsonl")).read_bytes()).hexdigest() for name in datasets},
              "boundary": "Fixed historical outputs rescored with the frozen endpoint implementations; no new inference or training.",
              "gold_redistributed_by_this_command": False}
    dump(output / "rescoring_validation.json", result)
    return result


def ranking(rows, variant):
    ordered = sorted(rows, key=lambda r: (-r["scores"][variant], r["qid"]))
    errors_total = sum(r["initial_wrong_label"] for r in ordered)
    negatives = len(ordered) - errors_total
    k = math.ceil(0.20 * len(ordered))
    errors = sum(r["initial_wrong_label"] for r in ordered[:k])
    ap = wins = 0.0
    seen_pos = seen_neg = offset = 0
    while offset < len(ordered):
        stop = offset + 1
        while stop < len(ordered) and ordered[stop]["scores"][variant] == ordered[offset]["scores"][variant]:
            stop += 1
        p = sum(r["initial_wrong_label"] for r in ordered[offset:stop])
        n = stop - offset - p
        seen_pos += p
        seen_neg += n
        ap += p / errors_total * seen_pos / stop
        wins += p * (negatives - seen_neg + 0.5 * n)
        offset = stop
    chosen = {r["qid"] for r in ordered[:k]}
    if chosen != {r["qid"] for r in rows if r["saved_selected"][variant]}:
        raise AssertionError("Saved Top-k membership differs: " + variant)
    return {"variant": variant, "selected": k, "errors": errors,
            "error_precision_percent": errors / k * 100,
            "error_recall_percent": errors / errors_total * 100,
            "AP_percent": ap * 100, "ROC_AUC_percent": wins / (errors_total * negatives) * 100}


def csv_table(path, data):
    columns = list(dict.fromkeys(key for row in data for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)


def main_tables(root, output, scores_dir=None):
    root = Path(root).resolve()
    verify_manifest(root)
    score_root = Path(scores_dir).resolve() if scores_dir else root / "frozen_scores"
    if scores_dir:
        receipt = json.loads((score_root / "rescoring_validation.json").read_text(encoding="utf-8"))
        if receipt.get("status") != "PASS_ANSWER_LEVEL_RESCORING":
            raise ValueError("Score input must have passed answer-level rescoring")
        for filename, expected_hash in receipt["score_files_sha256"].items():
            if hashlib.sha256((score_root / filename).read_bytes()).hexdigest() != expected_hash:
                raise ValueError("Rescored score file changed: " + filename)
    formal = load_rows(root / "frozen_scores/dev1096_oof.jsonl")
    explanatory = load_rows(root / "frozen_scores/dev1096_explanatory_oof.jsonl")
    expected = json.loads((root / "provenance/ranking_metric_expectations.json").read_text(encoding="utf-8"))
    contracts = json.loads((root / "contracts/exact_variant_key_sets.json").read_text(encoding="utf-8"))
    tables = {}
    for title, data, family in (("Table2_formal_oof", formal, "formal"), ("Table6_explanatory_oof", explanatory, "explanatory")):
        values = []
        for variant in data[0]["scores"]:
            metric = ranking(data, variant)
            saved = expected[family][variant]
            if metric["errors"] != saved["errors"] or abs(metric["AP_percent"] / 100 - saved["average_precision"]) > 1e-12 or abs(metric["ROC_AUC_percent"] / 100 - saved["roc_auc"]) > 1e-12:
                raise AssertionError("Frozen ranking expectation mismatch: " + variant)
            key = variant if variant in contracts else next(key for key in contracts if key.startswith(variant + "_"))
            metric["raw_allowed_keys"] = len(contracts[key])
            values.append(metric)
        tables[title] = values
    locked = load_rows(score_root / "locked1341.jsonl")
    tables["Table3_mmqa_endpoints"] = [{"method": m, **binary_summary(locked, m)} for m in locked[0]["methods"]]
    random = [r for r in tables["Table3_mmqa_endpoints"] if r["method"].startswith("Random-")]
    tables["Table3_random_mean_sample_SD"] = [{"metric": field,
        "mean": statistics.mean(r[field] for r in random), "sample_SD": statistics.stdev(r[field] for r in random)}
        for field in ("em_percent", "token_f1_percent", "W2R", "R2W", "Net")]
    membership = indexed(root / "frozen_scores/locked1341_all_selector_membership.jsonl")
    tables["Table4_selected_error_and_repair_yield"] = []
    for method in membership[next(iter(membership))]["selected"]:
        selected = [r for r in locked if membership[r["qid"]]["selected"][method]]
        wrong = sum(not r["methods"]["Initial"]["em"] for r in selected)
        w2r = sum(not r["methods"]["Initial"]["em"] and r["methods"][method]["em"] for r in selected)
        tables["Table4_selected_error_and_repair_yield"].append({"method": method, "selected": len(selected),
            "selected_wrong": wrong, "error_rate_percent": wrong / len(selected) * 100,
            "W2R": w2r, "repair_yield_percent": w2r / wrong * 100})
    m3rows = load_rows(score_root / "m3vqa13125.jsonl")
    tables["Table5_m3_answer_list_endpoints"] = []
    initial_accuracy = statistics.mean(r["methods"]["Initial"]["accuracy"] for r in m3rows) * 100
    for method in ("Initial", "Random-20", "B3-Refit"):
        item = binary_summary(m3rows, method, metric="full_match")
        item["strict_full_match_percent"] = item.pop("em_percent")
        accuracy = statistics.mean(r["methods"][method]["accuracy"] for r in m3rows) * 100
        tables["Table5_m3_answer_list_endpoints"].append({"method": method, **item,
            "answer_list_accuracy_percent": accuracy, "delta_vs_initial_pp": accuracy - initial_accuracy})
    controlled = load_rows(score_root / "dev1096_controlled.jsonl")
    member = indexed(root / "frozen_scores/dev1096_controlled_membership.jsonl")
    tables["Table7_controlled_endpoints"] = [{"method": method, **binary_summary(controlled, method),
        "selected": sum(member[r["qid"]]["selected"][method] for r in controlled),
        "selected_wrong": sum(member[r["qid"]]["selected"][method] and not r["methods"]["Initial"]["em"] for r in controlled)}
        for method in ("A0", "A4", "A6")]
    output = safe_output(root, output)
    for name, table in tables.items():
        csv_table(output / (name + ".csv"), table)
    report = {"status": "PASS_MAIN_NUMERIC_TABLE_AGGREGATION", "tables": tables,
              "score_source": "answer_level_rescored" if scores_dir else "frozen_saved_scores",
              "not_covered": ["Table1 method specification", "Table8 complete evidence text", "all SI tables", "confidence intervals (separate statistics command)"]}
    dump(output / "main_tables.json", report)
    return report
