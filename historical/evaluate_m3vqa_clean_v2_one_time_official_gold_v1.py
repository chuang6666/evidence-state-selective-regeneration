from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


EXPECTED_FINAL_SHA = "9BB95BE3FE68A10744159F3058C4DE4968ADD86E3B7D08C21B68A13D1B0CC4B9"
EXPECTED_SELECTION_SHA = "0529295C7A57EA51A69560E46A1623E796D6F34DFCA68697C9C25A888A763FE9"
EXPECTED_TOTAL = 13_125
EXPECTED_SELECTED = 2_625
EXPECTED_LOW = 10_500
EXPECTED_SUCCESS = 2_430
EXPECTED_TERMINAL = 195
EXPECTED_EMPTY = 180
EXPECTED_FORCED = 515
EXPECTED_RANKED = 2_110
EXPECTED_OVERLAP = 2_532
EXPECTED_CHANGED_EACH_SIDE = 93
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20_260_823
EPS = 1e-12
FINAL_STATUS = "M3VQA_CLEAN_V2_ONE_TIME_OFFICIAL_GOLD_EVAL_V1_COMPLETE"


class ProtocolError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the frozen pre-Gold boundary and stop without loading Gold.",
    )
    return parser.parse_args()


def safe(value: Any) -> str:
    return "" if value is None else str(value).strip()


def qid_of(row: Mapping[str, Any]) -> str:
    qid = safe(row.get("qid") or row.get("question_id") or row.get("data_id"))
    if not qid:
        raise ProtocolError("row has no qid")
    return qid


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolError(f"{path.name}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def index_unique(label: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        qid = qid_of(row)
        if qid in result:
            raise ProtocolError(f"{label}: duplicate qid={qid}")
        result[qid] = row
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def paths_for(root: Path) -> Dict[str, Path]:
    merge = root / "outputs/m3vqa_clean_v2_final_prediction_merge_before_gold_v1"
    select = root / "outputs/m3vqa_clean_b3_zero_tune_risk_top2625_selection_before_gold_v1"
    old_eval = root / "outputs/m3vqa_frozen13125_one_time_official_eval_v1"
    return {
        "final": merge / "m3vqa_clean_v2_final_predictions_13125_frozen_before_gold_v1.jsonl",
        "merge_summary": merge / "m3vqa_clean_v2_final_prediction_merge_summary_v1.json",
        "merge_audit": merge / "m3vqa_clean_v2_final_merge_audit_v1.json",
        "leakage_audit": merge / "m3vqa_clean_v2_before_gold_leakage_audit_v1.json",
        "selection": select / "m3vqa_clean_b3_top2625_selection_v1.jsonl",
        "low": select / "m3vqa_clean_b3_low_risk_keep10500_v1.jsonl",
        "overlap": select / "new_clean_vs_old_top2625_intersection_v1.jsonl",
        "new_only": select / "new_clean_vs_old_top2625_new_only_v1.jsonl",
        "old_only": select / "new_clean_vs_old_top2625_old_only_v1.jsonl",
        "initial": root / "outputs/m3vqa_initial_multimodal_qa_glm5v_batch_v2_frozen/m3vqa_initial_multimodal_qa_glm5v_batch_v2_frozen_results.jsonl",
        "gold": root / "outputs/m3vqa_dataset_adapter_v1_1/m3vqa_external_zero_shot_eval_only_v1_1.jsonl",
        "old_final": root / "outputs/m3vqa_frozen_top6_plain_regen_qwen37plus_v1_2/m3vqa_frozen13125_final_predictions_before_eval_v1.jsonl",
        "old_details": old_eval / "m3vqa_frozen13125_official_eval_per_qid_v1.jsonl",
        "old_summary": old_eval / "m3vqa_frozen13125_official_eval_summary_v1.json",
        "old_significance": root / "outputs/m3vqa_frozen13125_budget_significance_v1/m3vqa_frozen13125_budget_significance_summary_v1.json",
        "official_evaluator": root / "evaluate_m3vqa_frozen13125_one_time_official_v1.py",
        "output": root / "outputs/m3vqa_clean_v2_one_time_official_gold_eval_v1",
    }


def validate_pre_gold(paths: Mapping[str, Path]) -> Dict[str, Any]:
    # This function deliberately opens only frozen non-Gold assets.
    observed_final_sha = sha256_file(paths["final"])
    observed_selection_sha = sha256_file(paths["selection"])
    if observed_final_sha != EXPECTED_FINAL_SHA:
        raise ProtocolError("final predictions SHA mismatch; Gold was not opened")
    if observed_selection_sha != EXPECTED_SELECTION_SHA:
        raise ProtocolError("Top-2625 SHA mismatch; Gold was not opened")

    final_rows = load_jsonl(paths["final"])
    selection_rows = load_jsonl(paths["selection"])
    low_rows = load_jsonl(paths["low"])
    final = index_unique("clean_final", final_rows)
    selected = index_unique("selection", selection_rows)
    low = index_unique("low", low_rows)
    if len(final_rows) != EXPECTED_TOTAL or len(final) != EXPECTED_TOTAL:
        raise ProtocolError("final population is not 13,125 unique QIDs")
    if len(selection_rows) != EXPECTED_SELECTED or len(selected) != EXPECTED_SELECTED:
        raise ProtocolError("selection is not 2,625 unique QIDs")
    if len(low_rows) != EXPECTED_LOW or len(low) != EXPECTED_LOW:
        raise ProtocolError("low-risk KEEP is not 10,500 unique QIDs")
    if set(selected) & set(low) or set(selected) | set(low) != set(final):
        raise ProtocolError("selected/low partition is not a disjoint 13,125 cover")
    for qid, row in final.items():
        if bool(row.get("selected")) != (qid in selected):
            raise ProtocolError(f"{qid}: final selected flag disagrees with selection")

    selected_status = Counter(safe(final[qid].get("generation_status")) for qid in selected)
    selected_success = selected_status["success"]
    selected_terminal = selected_status["keep_initial_terminal"]
    unresolved = EXPECTED_SELECTED - selected_success - selected_terminal
    empty = sum(not safe(row.get("final_answer")) for row in final_rows)
    counts = {
        "total": len(final),
        "selected": len(selected),
        "low_risk_KEEP": len(low),
        "selected_success": selected_success,
        "protocol_terminal": selected_terminal,
        "unresolved": unresolved,
        "empty_final_prediction": empty,
    }
    expected = {
        "total": EXPECTED_TOTAL,
        "selected": EXPECTED_SELECTED,
        "low_risk_KEEP": EXPECTED_LOW,
        "selected_success": EXPECTED_SUCCESS,
        "protocol_terminal": EXPECTED_TERMINAL,
        "unresolved": 0,
        "empty_final_prediction": EXPECTED_EMPTY,
    }
    if counts != expected:
        raise ProtocolError(f"pre-Gold frozen counts mismatch: observed={counts}, expected={expected}")

    forced = sum(safe(row.get("partition")) == "FORCED_UNAVAILABLE" for row in selection_rows)
    ranked = sum(safe(row.get("partition")) == "RISK_RANKED" for row in selection_rows)
    ranks = sorted(int(row["selection_rank"]) for row in selection_rows)
    if forced != EXPECTED_FORCED or ranked != EXPECTED_RANKED:
        raise ProtocolError(f"selection partition mismatch: forced={forced}, ranked={ranked}")
    if ranks != list(range(1, EXPECTED_SELECTED + 1)):
        raise ProtocolError("selection ranks are not exactly 1..2625")

    merge_summary = load_json(paths["merge_summary"])
    merge_audit = load_json(paths["merge_audit"])
    leakage_audit = load_json(paths["leakage_audit"])
    if safe(merge_summary.get("status")) != "M3VQA_CLEAN_V2_FINAL_PREDICTIONS_FROZEN_BEFORE_GOLD_V1_COMPLETE":
        raise ProtocolError("before-Gold merge status is not the frozen COMPLETE status")
    if merge_audit.get("all_checks_pass") is not True:
        raise ProtocolError("before-Gold merge audit did not pass")
    if safe(leakage_audit.get("status")) != "M3VQA_CLEAN_V2_BEFORE_GOLD_LEAKAGE_AUDIT_PASS":
        raise ProtocolError("before-Gold leakage audit did not pass")
    if int(leakage_audit.get("gold_or_reference_values_read", -1)) != 0:
        raise ProtocolError("before-Gold leakage audit reports Gold/reference access")

    return {
        "status": "M3VQA_CLEAN_V2_ONE_TIME_OFFICIAL_GOLD_EVAL_PRE_GOLD_GATE_PASS",
        "gold_opened": False,
        "all_checks_pass": True,
        "frozen_sha256": {
            "final_predictions_expected": EXPECTED_FINAL_SHA,
            "final_predictions_observed": observed_final_sha,
            "top2625_expected": EXPECTED_SELECTION_SHA,
            "top2625_observed": observed_selection_sha,
        },
        "population": counts,
        "selection_partition": {"forced": forced, "risk_ranked": ranked},
        "before_gold_merge_audit_pass": True,
        "before_gold_leakage_audit_pass": True,
        "hard_check_failures": [],
    }


def load_official_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_m3_official_eval_v1", path)
    if spec is None or spec.loader is None:
        raise ProtocolError("cannot load frozen official evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_two_sided_binomial_p(successes: int, failures: int) -> float:
    n = successes + failures
    if n == 0:
        return 1.0
    tail = min(successes, failures)
    numerator = sum(math.comb(n, k) for k in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2**n))


def paired_bootstrap_ci(deltas: np.ndarray) -> Dict[str, Any]:
    if deltas.ndim != 1 or len(deltas) != EXPECTED_TOTAL:
        raise ProtocolError("paired bootstrap received the wrong population")
    values, counts = np.unique(deltas, return_counts=True)
    probabilities = counts.astype(np.float64) / len(deltas)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.multinomial(len(deltas), probabilities, size=BOOTSTRAP_REPS)
    means = (draws @ values.astype(np.float64)) / len(deltas)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "method": "paired empirical percentile bootstrap",
        "repetitions": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "delta_accuracy_pp_95ci_lower": float(lower * 100.0),
        "delta_accuracy_pp_95ci_upper": float(upper * 100.0),
        "bootstrap_probability_delta_le_zero": float(np.mean(means <= 0.0)),
        "unique_delta_values": int(len(values)),
    }


def transition(initial_full: bool, final_full: bool) -> str:
    if not initial_full and final_full:
        return "W2R"
    if initial_full and not final_full:
        return "R2W"
    return "R2R" if initial_full else "W2W"


def full_match_key(score_key: str) -> str:
    mapping = {
        "initial_official_accuracy": "initial_full_match",
        "clean_v2_official_accuracy": "clean_v2_full_match",
        "old_v1_official_accuracy": "old_v1_full_match",
    }
    try:
        return mapping[score_key]
    except KeyError as exc:
        raise ProtocolError(f"no frozen full-match field for score key={score_key}") from exc


def summarize(rows: Sequence[Mapping[str, Any]], before: str, after: str) -> Dict[str, Any]:
    n = len(rows)
    before_scores = [float(row[before]) for row in rows]
    after_scores = [float(row[after]) for row in rows]
    before_full_key = full_match_key(before)
    after_full_key = full_match_key(after)
    before_full = sum(bool(row[before_full_key]) for row in rows)
    after_full = sum(bool(row[after_full_key]) for row in rows)
    w2r = sum(not bool(row[before_full_key]) and bool(row[after_full_key]) for row in rows)
    r2w = sum(bool(row[before_full_key]) and not bool(row[after_full_key]) for row in rows)
    improved = sum(a - b > EPS for b, a in zip(before_scores, after_scores))
    regressed = sum(a - b < -EPS for b, a in zip(before_scores, after_scores))
    return {
        "N": n,
        "Initial_or_before_Accuracy": math.fsum(before_scores) / n if n else 0.0,
        "Final_or_after_Accuracy": math.fsum(after_scores) / n if n else 0.0,
        "Delta_Accuracy_pp": (math.fsum(after_scores) - math.fsum(before_scores)) / n * 100.0 if n else 0.0,
        "Initial_or_before_correct_count": before_full,
        "Final_or_after_correct_count": after_full,
        "W2R": w2r,
        "R2W": r2w,
        "Net": w2r - r2w,
        "Score_improved": improved,
        "Score_regressed": regressed,
        "Score_same": n - improved - regressed,
        "Official_score_sum_change": math.fsum(after_scores) - math.fsum(before_scores),
    }


def pair_significance(rows: Sequence[Mapping[str, Any]], before: str, after: str) -> Dict[str, Any]:
    if len(rows) != EXPECTED_TOTAL:
        raise ProtocolError("significance population must contain all 13,125 QIDs")
    before_full_key = full_match_key(before)
    after_full_key = full_match_key(after)
    w2r = sum(not bool(row[before_full_key]) and bool(row[after_full_key]) for row in rows)
    r2w = sum(bool(row[before_full_key]) and not bool(row[after_full_key]) for row in rows)
    deltas = np.asarray([float(row[after]) - float(row[before]) for row in rows], dtype=np.float64)
    bootstrap = paired_bootstrap_ci(deltas)
    mcnemar_p = exact_two_sided_binomial_p(w2r, r2w)
    return {
        "full_match_McNemar_exact": {
            "W2R": w2r,
            "R2W": r2w,
            "Net": w2r - r2w,
            "exact_two_sided_p": mcnemar_p,
        },
        "paired_official_accuracy_bootstrap": bootstrap,
        "significance_at_alpha_0_05": {
            "McNemar": mcnemar_p < 0.05,
            "bootstrap_CI_excludes_zero": (
                bootstrap["delta_accuracy_pp_95ci_lower"] > 0.0
                or bootstrap["delta_accuracy_pp_95ci_upper"] < 0.0
            ),
        },
    }


def evaluate(root: Path, paths: Mapping[str, Path], gate: Mapping[str, Any]) -> None:
    # GOLD ACCESS BOUNDARY: no Gold/reference artifact is opened above this line.
    official = load_official_evaluator(paths["official_evaluator"])
    initial_rows = load_jsonl(paths["initial"])
    clean_rows = load_jsonl(paths["final"])
    selected_rows = load_jsonl(paths["selection"])
    low_rows = load_jsonl(paths["low"])
    old_final_rows = load_jsonl(paths["old_final"])
    old_detail_rows = load_jsonl(paths["old_details"])
    old_summary = load_json(paths["old_summary"])
    old_significance = load_json(paths["old_significance"])
    overlap_rows = load_jsonl(paths["overlap"])
    new_only_rows = load_jsonl(paths["new_only"])
    old_only_rows = load_jsonl(paths["old_only"])
    gold_rows = load_jsonl(paths["gold"])  # first and only official Gold load

    initial = index_unique("initial", initial_rows)
    clean = index_unique("clean", clean_rows)
    selected = index_unique("selected", selected_rows)
    low = index_unique("low", low_rows)
    old_final = index_unique("old_final", old_final_rows)
    old_details = index_unique("old_details", old_detail_rows)
    gold = index_unique("gold", gold_rows)
    overlap = set(index_unique("overlap", overlap_rows))
    new_only = set(index_unique("new_only", new_only_rows))
    old_only = set(index_unique("old_only", old_only_rows))
    population = set(clean)
    for label, indexed in (("initial", initial), ("old_final", old_final), ("old_details", old_details), ("gold", gold)):
        if len(indexed) != EXPECTED_TOTAL or set(indexed) != population:
            raise ProtocolError(f"{label}: population mismatch after authorized Gold load")
    if len(overlap) != EXPECTED_OVERLAP or len(new_only) != EXPECTED_CHANGED_EACH_SIDE or len(old_only) != EXPECTED_CHANGED_EACH_SIDE:
        raise ProtocolError("new/old selection overlap partition size mismatch")
    if overlap & new_only or overlap & old_only or new_only & old_only:
        raise ProtocolError("new/old selection comparison sets overlap illegally")
    old_selected = overlap | old_only
    if set(selected) != overlap | new_only or len(old_selected) != EXPECTED_SELECTED:
        raise ProtocolError("selection overlap files disagree with frozen new selection")

    forced = {qid for qid, row in selected.items() if safe(row.get("partition")) == "FORCED_UNAVAILABLE"}
    ranked = set(selected) - forced
    terminal = {qid for qid, row in clean.items() if safe(row.get("generation_status")) == "keep_initial_terminal"}
    empty = {qid for qid, row in clean.items() if not safe(row.get("final_answer"))}
    if (len(forced), len(ranked), len(terminal), len(empty)) != (EXPECTED_FORCED, EXPECTED_RANKED, EXPECTED_TERMINAL, EXPECTED_EMPTY):
        raise ProtocolError("post-Gold group identities disagree with frozen boundary")

    details: List[Dict[str, Any]] = []
    terminal_fallback_matches = 0
    low_raw_matches = 0
    old_only_clean_initial_matches = 0
    old_frozen_score_max_diff = 0.0
    for qid in sorted(population):
        gold_row = gold[qid]
        if gold_row.get("eval_only") is not True or gold_row.get("must_not_be_loaded_during_inference") is not True:
            raise ProtocolError(f"{qid}: M3 eval-only Gold guardrail invalid")
        answer_evals = official.validate_answer_evals(gold_row.get("answer_evals"), qid)
        initial_answer = safe(initial[qid].get("draft_answer"))
        clean_answer = safe(clean[qid].get("final_answer"))
        old_answer = safe(old_final[qid].get("final_answer"))
        initial_score = official.official_accuracy(official.convert_to_list(initial_answer), answer_evals)
        clean_score = official.official_accuracy(official.convert_to_list(clean_answer), answer_evals)
        old_score = official.official_accuracy(official.convert_to_list(old_answer), answer_evals)
        initial_full = abs(initial_score - 1.0) <= EPS
        clean_full = abs(clean_score - 1.0) <= EPS
        old_full = abs(old_score - 1.0) <= EPS
        old_frozen_score_max_diff = max(
            old_frozen_score_max_diff,
            abs(old_score - float(old_details[qid]["final_official_accuracy"])),
            abs(initial_score - float(old_details[qid]["initial_official_accuracy"])),
        )
        if qid in terminal and clean_answer == initial_answer:
            terminal_fallback_matches += 1
        if qid in low and clean_answer == initial_answer:
            low_raw_matches += 1
        if qid in old_only and clean_answer == initial_answer:
            old_only_clean_initial_matches += 1
        details.append(
            {
                "qid": qid,
                "initial_answer": initial_answer,
                "clean_v2_final_answer": clean_answer,
                "old_v1_final_answer": old_answer,
                "initial_official_accuracy": initial_score,
                "clean_v2_official_accuracy": clean_score,
                "old_v1_official_accuracy": old_score,
                "initial_full_match": initial_full,
                "clean_v2_full_match": clean_full,
                "old_v1_full_match": old_full,
                "initial_to_clean_migration": transition(initial_full, clean_full),
                "old_v1_to_clean_migration": transition(old_full, clean_full),
                "initial_to_clean_score_delta": clean_score - initial_score,
                "old_v1_to_clean_score_delta": clean_score - old_score,
                "selected": qid in selected,
                "low_risk_keep": qid in low,
                "forced": qid in forced,
                "risk_ranked": qid in ranked,
                "protocol_terminal": qid in terminal,
                "empty_clean_final_prediction": qid in empty,
                "selection_relation": (
                    "OVERLAP" if qid in overlap else "NEW_ONLY" if qid in new_only else "OLD_ONLY" if qid in old_only else "NEITHER"
                ),
                "generation_status": safe(clean[qid].get("generation_status")),
                "source": safe(clean[qid].get("source")),
            }
        )
    if old_frozen_score_max_diff > EPS:
        raise ProtocolError(f"official evaluator parity with old frozen per-QID scores failed: {old_frozen_score_max_diff}")

    overall = summarize(details, "initial_official_accuracy", "clean_v2_official_accuracy")
    if overall["Final_or_after_correct_count"] - overall["Initial_or_before_correct_count"] != overall["Net"]:
        raise ProtocolError("Final correct - Initial correct != W2R - R2W")
    if low_raw_matches != EXPECTED_LOW:
        raise ProtocolError("low-risk 10,500 Final predictions are not strictly identical to Initial")
    low_summary = summarize([row for row in details if row["low_risk_keep"]], "initial_official_accuracy", "clean_v2_official_accuracy")
    if low_summary["W2R"] != 0 or low_summary["R2W"] != 0 or abs(low_summary["Delta_Accuracy_pp"]) > EPS:
        raise ProtocolError("low-risk KEEP group changed under official evaluation")
    if terminal_fallback_matches != EXPECTED_TERMINAL:
        raise ProtocolError("protocol-terminal fallback parity failed")
    if old_only_clean_initial_matches != EXPECTED_CHANGED_EACH_SIDE:
        raise ProtocolError("old-only93 Clean V2 predictions are not all Initial")

    selected_summary = summarize([row for row in details if row["selected"]], "initial_official_accuracy", "clean_v2_official_accuracy")
    forced_summary = summarize([row for row in details if row["forced"]], "initial_official_accuracy", "clean_v2_official_accuracy")
    ranked_summary = summarize([row for row in details if row["risk_ranked"]], "initial_official_accuracy", "clean_v2_official_accuracy")
    terminal_summary = summarize([row for row in details if row["protocol_terminal"]], "initial_official_accuracy", "clean_v2_official_accuracy")
    empty_summary = summarize([row for row in details if row["empty_clean_final_prediction"]], "initial_official_accuracy", "clean_v2_official_accuracy")
    grouped = {
        "Selected2625": selected_summary,
        "Low_risk10500": {**low_summary, "strict_Final_equals_Initial": f"{low_raw_matches}/{EXPECTED_LOW}"},
        "Forced515": forced_summary,
        "Risk_ranked2110": ranked_summary,
        "Protocol_terminal195": {**terminal_summary, "fallback_parity": f"{terminal_fallback_matches}/{EXPECTED_TERMINAL}"},
        "Empty180": empty_summary,
    }

    clean_significance = pair_significance(details, "initial_official_accuracy", "clean_v2_official_accuracy")
    old_comparison_summary = summarize(details, "old_v1_official_accuracy", "clean_v2_official_accuracy")
    old_significance_result = pair_significance(details, "old_v1_official_accuracy", "clean_v2_official_accuracy")
    clean_better = sum(row["clean_v2_official_accuracy"] - row["old_v1_official_accuracy"] > EPS for row in details)
    old_better = sum(row["clean_v2_official_accuracy"] - row["old_v1_official_accuracy"] < -EPS for row in details)
    same_score = EXPECTED_TOTAL - clean_better - old_better
    raw_same = sum(row["clean_v2_final_answer"] == row["old_v1_final_answer"] for row in details)

    old_overall = old_summary.get("overall", {})
    expected_old = {
        "Initial_official_accuracy": 0.28529968446741555,
        "Final_official_accuracy": 0.31254204112779194,
        "Delta_official_accuracy_pp": 2.7242356660376355,
        "Full_match_W2R": 159,
        "Full_match_R2W": 1,
        "Full_match_Net": 158,
    }
    for key, expected in expected_old.items():
        observed = old_overall.get(key)
        if isinstance(expected, float):
            if abs(float(observed) - expected) > EPS:
                raise ProtocolError(f"old V1 frozen summary mismatch for {key}")
        elif int(observed) != expected:
            raise ProtocolError(f"old V1 frozen summary mismatch for {key}")
    archived_sig = old_significance.get("significance", {})
    if abs(float(archived_sig["full_match_mcnemar"]["exact_two_sided_p"]) - exact_two_sided_binomial_p(159, 1)) > 1e-60:
        raise ProtocolError("old significance implementation parity failed")

    def subset(qids: set[str]) -> List[Mapping[str, Any]]:
        return [row for row in details if row["qid"] in qids]

    overlap_subset = subset(overlap)
    new_only_subset = subset(new_only)
    old_only_subset = subset(old_only)
    neither = population - overlap - new_only - old_only
    decomposition_groups = {
        "overlap2532": summarize(overlap_subset, "old_v1_official_accuracy", "clean_v2_official_accuracy"),
        "new_only93_initial_to_clean": summarize(new_only_subset, "initial_official_accuracy", "clean_v2_official_accuracy"),
        "old_only93_old_to_clean_initial": summarize(old_only_subset, "old_v1_official_accuracy", "clean_v2_official_accuracy"),
        "neither10407": summarize(subset(neither), "old_v1_official_accuracy", "clean_v2_official_accuracy"),
    }
    overlap_raw_same = sum(row["clean_v2_final_answer"] == row["old_v1_final_answer"] for row in overlap_subset)
    overlap_normalized_same = sum(
        official.convert_to_list(row["clean_v2_final_answer"]) == official.convert_to_list(row["old_v1_final_answer"])
        for row in overlap_subset
    )
    overlap_score_same = sum(abs(row["clean_v2_official_accuracy"] - row["old_v1_official_accuracy"]) <= EPS for row in overlap_subset)
    decomposition_score_change = math.fsum(
        group["Official_score_sum_change"] for group in decomposition_groups.values()
    )
    if abs(decomposition_score_change - old_comparison_summary["Official_score_sum_change"]) > EPS:
        raise ProtocolError("selection-change Accuracy decomposition does not reconcile")

    old_comparison = {
        "Clean_V2_vs_Old_V1": old_comparison_summary,
        "Clean_V2_better": clean_better,
        "Old_V1_better": old_better,
        "same_official_score": same_score,
        "same_raw_prediction": raw_same,
        "paired_significance": old_significance_result,
        "difference_significant": (
            old_significance_result["full_match_McNemar_exact"]["exact_two_sided_p"] < 0.05
            and old_significance_result["paired_official_accuracy_bootstrap"]["delta_accuracy_pp_95ci_lower"] *
            old_significance_result["paired_official_accuracy_bootstrap"]["delta_accuracy_pp_95ci_upper"] > 0
        ),
        "frozen_Old_V1_reference": expected_old,
        "old_per_qid_recompute_max_abs_difference": old_frozen_score_max_diff,
        "overlap2532_prediction_parity": {
            "N": len(overlap_subset),
            "raw_exact_same": overlap_raw_same,
            "parsed_prediction_list_same": overlap_normalized_same,
            "official_score_same": overlap_score_same,
        },
        "selection_change_decomposition": decomposition_groups,
        "decomposition_official_score_change_sum": decomposition_score_change,
    }

    output = paths["output"]
    details_path = output / "m3vqa_clean_v2_one_time_official_gold_eval_details_v1.jsonl"
    grouped_path = output / "m3vqa_clean_v2_grouped_evaluation_v1.json"
    significance_path = output / "m3vqa_clean_v2_significance_v1.json"
    old_comparison_path = output / "m3vqa_clean_v2_old_v1_paired_comparison_v1.json"
    summary_path = output / "m3vqa_clean_v2_one_time_official_gold_eval_summary_v1.json"
    atomic_jsonl(details_path, details)
    atomic_json(grouped_path, grouped)
    atomic_json(significance_path, {
        "Initial_vs_Clean_V2": clean_significance,
        "Old_V1_vs_Clean_V2": old_significance_result,
        "protocol": {"bootstrap_repetitions": BOOTSTRAP_REPS, "bootstrap_seed": BOOTSTRAP_SEED, "McNemar": "exact two-sided binomial"},
    })
    atomic_json(old_comparison_path, old_comparison)

    summary = {
        "status": FINAL_STATUS,
        "pre_gold_gate": gate,
        "official_evaluator": {
            "source": str(paths["official_evaluator"]),
            "source_sha256": sha256_file(paths["official_evaluator"]),
            "normalization": "exact frozen M3-VQA official evaluator normalization",
            "accuracy": "exact frozen M3-VQA official Accuracy implementation",
        },
        "main_result": overall,
        "grouped_evaluation_file": grouped_path.name,
        "significance": clean_significance,
        "old_v1_paired_comparison_file": old_comparison_path.name,
        "identity_checks": {
            "Final_correct_minus_Initial_correct": overall["Final_or_after_correct_count"] - overall["Initial_or_before_correct_count"],
            "W2R_minus_R2W": overall["Net"],
            "low_risk_W2R_R2W_zero": low_summary["W2R"] == 0 and low_summary["R2W"] == 0,
            "low_risk_final_equals_initial": f"{low_raw_matches}/{EXPECTED_LOW}",
            "terminal_fallback_parity": f"{terminal_fallback_matches}/{EXPECTED_TERMINAL}",
            "old_only93_clean_equals_initial": f"{old_only_clean_initial_matches}/{EXPECTED_CHANGED_EACH_SIDE}",
        },
        "empty180": {"N": len(empty), "correct_count": empty_summary["Final_or_after_correct_count"], "Accuracy": empty_summary["Final_or_after_Accuracy"]},
        "protocol": {
            "Gold_read_only_after_preflight_PASS": True,
            "Gold_access_count": 1,
            "API_calls": 0,
            "predictions_modified": False,
            "regeneration": False,
            "risk_training": False,
            "risk_rescoring": False,
            "selection_modified": False,
            "calibration": False,
            "empty_predictions_repaired": False,
            "post_gold_tuning": False,
        },
        "input_sha256": {key: sha256_file(paths[key]) for key in (
            "final", "selection", "low", "initial", "gold", "old_final", "old_details", "old_summary", "old_significance", "overlap", "new_only", "old_only"
        )},
        "hard_check_failures": [],
    }
    atomic_json(summary_path, summary)

    output_files = [summary_path, details_path, significance_path, old_comparison_path, grouped_path, output / "pre_gold_gate_v1.json"]
    manifest = {
        "status": FINAL_STATUS,
        "algorithm": "SHA-256",
        "files": {path.name: sha256_file(path) for path in output_files},
        "script": {Path(__file__).name: sha256_file(Path(__file__))},
        "frozen_primary_inputs": {
            paths["final"].name: sha256_file(paths["final"]),
            paths["selection"].name: sha256_file(paths["selection"]),
        },
    }
    atomic_json(output / "SHA256SUMS.json", manifest)

    print("=" * 100)
    print("M3VQA CLEAN V2 — ONE-TIME OFFICIAL GOLD EVALUATION V1")
    print(f"Status: {FINAL_STATUS}")
    print(f"Initial Accuracy: {overall['Initial_or_before_Accuracy'] * 100:.6f}%")
    print(f"Clean V2 Accuracy: {overall['Final_or_after_Accuracy'] * 100:.6f}%")
    print(f"Delta Accuracy: {overall['Delta_Accuracy_pp']:+.6f} pp")
    print(f"Correct: {overall['Initial_or_before_correct_count']} -> {overall['Final_or_after_correct_count']}")
    print(f"W2R/R2W/Net: {overall['W2R']}/{overall['R2W']}/{overall['Net']:+d}")
    print(f"McNemar exact p: {clean_significance['full_match_McNemar_exact']['exact_two_sided_p']:.8g}")
    ci = clean_significance["paired_official_accuracy_bootstrap"]
    print(f"Delta Accuracy bootstrap 95% CI: [{ci['delta_accuracy_pp_95ci_lower']:.6f}, {ci['delta_accuracy_pp_95ci_upper']:.6f}] pp")
    print(f"Clean V2 - Old V1 Accuracy: {old_comparison_summary['Delta_Accuracy_pp']:+.6f} pp")
    print(f"Output: {output}")
    print("No retraining, rescoring, reselection, regeneration, or post-Gold modification performed.")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = paths_for(root)
    gate = validate_pre_gold(paths)
    atomic_json(paths["output"] / "pre_gold_gate_v1.json", gate)
    print("PRE-GOLD GATE: PASS")
    print(json.dumps(gate["frozen_sha256"], ensure_ascii=False, sort_keys=True))
    print(json.dumps(gate["population"], ensure_ascii=False, sort_keys=True))
    if args.preflight_only:
        print("STOPPED BEFORE GOLD AS REQUESTED BY --preflight-only")
        return
    evaluate(root, paths, gate)


if __name__ == "__main__":
    main()
