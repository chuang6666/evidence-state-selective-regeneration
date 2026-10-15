from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import random
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(r"E:\RAG_Projects\MultimodalEvidenceChain")
OUT_DIR = ROOT / "outputs" / "mmqa_clean_risk_v2_one_time_official_eval_v1"

FINAL_PRED = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_final_prediction_merge_before_gold_v1"
    / "clean_v2_locked1341_final_predictions_frozen_before_gold_v1.jsonl"
)
TOP269 = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_final_full_dev_locked_selection_v1"
    / "locked1341_clean_b3_top269_selection_v1.jsonl"
)
LOW1072 = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_final_full_dev_locked_selection_v1"
    / "locked1341_clean_b3_low1072_v1.jsonl"
)
INITIAL = (
    ROOT / "outputs" / "mmqa_v3_heldout1341_pre_natural_v1"
    / "mmqa_v3_heldout1341_initial_base_sanitized_v1.jsonl"
)
TERMINAL_AUDIT = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_final_prediction_merge_before_gold_v1"
    / "clean_v2_terminal_fallback_audit_v1.jsonl"
)
GOLD = ROOT / "data" / "raw" / "mmqa_official" / "MMQA_dev.jsonl.gz"

OFFICIAL_EVALUATOR = ROOT / "audit_mmqa_v3_locked1341_frozen_boundary_v1_1.py"
OLD_V1_PRED = (
    ROOT / "outputs" / "mmqa_v3_heldout1341_final_plain_regen_qwen37plus_v1"
    / "mmqa_v3_locked1341_final_predictions_frozen_before_eval_v1.jsonl"
)
RANDOM_DIR = (
    ROOT / "outputs" / "mmqa_v3_random20_matched_qwen37plus_v1"
    / "frozen_seed_predictions_before_eval_v1"
)
VC_DIR = (
    ROOT / "outputs" / "mmqa_vc20_matched_regeneration_union_qwen37plus_v1"
    / "frozen_variant_predictions_before_eval_v1"
)

EXPECTED_FINAL_SHA = "CA0D6B81120748C844119850D1E59DDA79E3C27D9DD1776367196B5A4E51A7A5"
EXPECTED_TOP269_SHA = "DCE4FFF6218F321DB71950ED9D21128EC24D512BE162E73B05C707DAFE4A74BE"
EXPECTED_LOW1072_SHA = "B2AE91631CBBDC4A8500699A077C4E1905431CAFE89557E580628EEBA72B123C"
EXPECTED_INITIAL_SHA = "6F068067A5F5DC489644D68EC5B072E31B624264AE506D7D89CCE36498437E81"
EXPECTED_EVALUATOR_SHA = "16C3BB79BF93DB73381977A2A032F6C54771F7BBF7E1EE19CB20D4713A87C85B"

RANDOM_SEEDS = (20260822, 20260823, 20260824, 20260825, 20260826)
VC_TIE_SEEDS = RANDOM_SEEDS
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260828
STATUS = "MMQA_CLEAN_RISK_V2_ONE_TIME_OFFICIAL_EVAL_V1_COMPLETE"


class EvalError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise EvalError(f"Missing required file: {path}")
    actual = sha256(path)
    if actual != expected.upper():
        raise EvalError(f"SHA mismatch for {path}: expected={expected}, actual={actual}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: List[Dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise EvalError(f"Non-object JSON row: {path}:{line_no}")
            rows.append(row)
    return rows


def index_qids(rows: Iterable[Mapping[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid", ""))
        if not qid:
            raise EvalError(f"Empty QID in {label}")
        if qid in result:
            raise EvalError(f"Duplicate QID in {label}: {qid}")
        result[qid] = dict(row)
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def import_official_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location("mmqa_frozen_official_evaluator", OFFICIAL_EVALUATOR)
    if spec is None or spec.loader is None:
        raise EvalError("Could not load frozen official evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("extract_gold_answers", "stringify_answer", "best_score"):
        if not hasattr(module, name):
            raise EvalError(f"Frozen evaluator is missing required function: {name}")
    return module


def answer_from(row: Mapping[str, Any], key: str = "final_answer") -> str:
    value = row.get(key, row.get("answer", ""))
    return OFFICIAL.stringify_answer(value)


def score_answers(
    answers: Mapping[str, str], gold: Mapping[str, Sequence[str]], qids: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    if set(answers) != set(qids):
        missing = sorted(set(qids) - set(answers))[:5]
        extra = sorted(set(answers) - set(qids))[:5]
        raise EvalError(f"Prediction universe mismatch: missing={missing}, extra={extra}")
    result: Dict[str, Dict[str, float]] = {}
    for qid in qids:
        em, f1 = OFFICIAL.best_score(answers[qid], gold[qid])
        result[qid] = {"em": int(em), "f1": float(f1)}
    return result


def summarize(scores: Mapping[str, Mapping[str, float]], qids: Sequence[str]) -> Dict[str, Any]:
    em_count = sum(int(scores[q]["em"]) for q in qids)
    return {
        "n": len(qids),
        "em_count": em_count,
        "em": em_count / len(qids),
        "em_percent": 100.0 * em_count / len(qids),
        "f1": sum(float(scores[q]["f1"]) for q in qids) / len(qids),
        "f1_percent": 100.0 * sum(float(scores[q]["f1"]) for q in qids) / len(qids),
    }


def migration(
    base: Mapping[str, Mapping[str, float]],
    target: Mapping[str, Mapping[str, float]],
    qids: Sequence[str],
) -> Dict[str, int]:
    counts = Counter()
    for qid in qids:
        b = int(base[qid]["em"])
        t = int(target[qid]["em"])
        if b == 0 and t == 1:
            counts["wrong_to_right"] += 1
        elif b == 1 and t == 0:
            counts["right_to_wrong"] += 1
        elif b == 1 and t == 1:
            counts["right_to_right"] += 1
        else:
            counts["wrong_to_wrong"] += 1
    return {
        "wrong_to_right": counts["wrong_to_right"],
        "right_to_wrong": counts["right_to_wrong"],
        "net": counts["wrong_to_right"] - counts["right_to_wrong"],
        "right_to_right": counts["right_to_right"],
        "wrong_to_wrong": counts["wrong_to_wrong"],
    }


def method_summary(
    scores: Mapping[str, Mapping[str, float]],
    initial_scores: Mapping[str, Mapping[str, float]],
    qids: Sequence[str],
) -> Dict[str, Any]:
    result = summarize(scores, qids)
    initial_result = summarize(initial_scores, qids)
    result["delta_em"] = result["em"] - initial_result["em"]
    result["delta_em_pp"] = result["em_percent"] - initial_result["em_percent"]
    result["delta_f1"] = result["f1"] - initial_result["f1"]
    result["delta_f1_pp"] = result["f1_percent"] - initial_result["f1_percent"]
    result["migration_from_initial"] = migration(initial_scores, scores, qids)
    return result


def exact_mcnemar_p(a_better: int, b_better: int) -> float:
    n = a_better + b_better
    if n == 0:
        return 1.0
    k = min(a_better, b_better)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_delta(
    values_a: Sequence[float], values_b: Sequence[float], seed: int
) -> Dict[str, Any]:
    if len(values_a) != len(values_b) or not values_a:
        raise EvalError("Invalid paired bootstrap vectors")
    deltas = [float(a) - float(b) for a, b in zip(values_a, values_b)]
    observed = statistics.mean(deltas)
    rng = random.Random(seed)
    n = len(deltas)
    boot: List[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        boot.append(sum(deltas[rng.randrange(n)] for _j in range(n)) / n)
    boot.sort()

    def quantile(p: float) -> float:
        pos = p * (len(boot) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return boot[lo]
        w = pos - lo
        return boot[lo] * (1.0 - w) + boot[hi] * w

    return {
        "delta": observed,
        "delta_pp": 100.0 * observed,
        "ci95_low": quantile(0.025),
        "ci95_high": quantile(0.975),
        "ci95_low_pp": 100.0 * quantile(0.025),
        "ci95_high_pp": 100.0 * quantile(0.975),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
    }


def pairwise(
    a_name: str,
    a: Mapping[str, Mapping[str, float]],
    b_name: str,
    b: Mapping[str, Mapping[str, float]],
    qids: Sequence[str],
) -> Dict[str, Any]:
    a_em = [float(a[q]["em"]) for q in qids]
    b_em = [float(b[q]["em"]) for q in qids]
    a_f1 = [float(a[q]["f1"]) for q in qids]
    b_f1 = [float(b[q]["f1"]) for q in qids]
    a_better = sum(int(x == 1 and y == 0) for x, y in zip(a_em, b_em))
    b_better = sum(int(x == 0 and y == 1) for x, y in zip(a_em, b_em))
    return {
        "comparison": f"{a_name}_minus_{b_name}",
        "n": len(qids),
        "em": {
            "delta": statistics.mean(a_em) - statistics.mean(b_em),
            "delta_pp": 100.0 * (statistics.mean(a_em) - statistics.mean(b_em)),
            "clean_better_discordant": a_better,
            "baseline_better_discordant": b_better,
            "mcnemar_exact_p": exact_mcnemar_p(a_better, b_better),
            "paired_bootstrap": paired_bootstrap_delta(a_em, b_em, BOOTSTRAP_SEED),
        },
        "f1": {
            "delta": statistics.mean(a_f1) - statistics.mean(b_f1),
            "delta_pp": 100.0 * (statistics.mean(a_f1) - statistics.mean(b_f1)),
            "paired_bootstrap": paired_bootstrap_delta(a_f1, b_f1, BOOTSTRAP_SEED + 1),
        },
    }


def aggregate_seed_metrics(items: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"n_seeds": len(items), "seeds": list(items)}
    for metric in ("em", "em_percent", "f1", "f1_percent", "delta_em_pp", "delta_f1_pp"):
        values = [float(v[metric]) for v in items.values()]
        result[metric] = {
            "mean": statistics.mean(values),
            "population_sd": statistics.pstdev(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    for metric in ("wrong_to_right", "right_to_wrong", "net"):
        values = [int(v["migration_from_initial"][metric]) for v in items.values()]
        result[metric] = {
            "mean": statistics.mean(values),
            "population_sd": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def averaged_seed_scores(
    seed_scores: Mapping[str, Mapping[str, Mapping[str, float]]], qids: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    return {
        q: {
            "em": statistics.mean(float(s[q]["em"]) for s in seed_scores.values()),
            "f1": statistics.mean(float(s[q]["f1"]) for s in seed_scores.values()),
        }
        for q in qids
    }


def pairwise_fractional(
    clean: Mapping[str, Mapping[str, float]],
    baseline_mean: Mapping[str, Mapping[str, float]],
    label: str,
    qids: Sequence[str],
) -> Dict[str, Any]:
    clean_em = [float(clean[q]["em"]) for q in qids]
    base_em = [float(baseline_mean[q]["em"]) for q in qids]
    clean_f1 = [float(clean[q]["f1"]) for q in qids]
    base_f1 = [float(baseline_mean[q]["f1"]) for q in qids]
    return {
        "comparison": f"clean_v2_minus_{label}",
        "note": "Seed-mean comparator is fractional per QID; McNemar is not applicable.",
        "em": {"paired_bootstrap": paired_bootstrap_delta(clean_em, base_em, BOOTSTRAP_SEED)},
        "f1": {"paired_bootstrap": paired_bootstrap_delta(clean_f1, base_f1, BOOTSTRAP_SEED + 1)},
    }


def holm_adjust(raw: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(raw.items(), key=lambda x: x[1])
    m = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        value = min(1.0, (m - i) * p)
        running = max(running, value)
        adjusted[name] = running
    return {name: adjusted[name] for name in raw}


def main() -> int:
    global OFFICIAL
    print("=" * 152)
    print("MMQA CLEAN RISK V2 — ONE-TIME OFFICIAL GOLD EVALUATION V1")
    print("=" * 152)
    print("Prediction mutation: NO | Regeneration: NO | Training: NO | Reselection: NO | API: 0")
    print("Gold access: AUTHORIZED ONLY AFTER frozen-input SHA/protocol checks")

    if OUT_DIR.exists():
        raise EvalError(f"One-time output directory already exists: {OUT_DIR}")

    random_paths = {
        str(seed): RANDOM_DIR / f"mmqa_v3_locked1341_random20_seed_{seed}_predictions_frozen_before_eval_v1.jsonl"
        for seed in RANDOM_SEEDS
    }
    vc_paths = {
        "vc20_qid_ascending": VC_DIR / "mmqa_vc20_qid_ascending_locked1341_predictions_frozen_before_eval_v1.jsonl",
        **{
            f"vc20_random_tie_{seed}": VC_DIR / f"mmqa_vc20_random_tie_seed_{seed}_locked1341_predictions_frozen_before_eval_v1.jsonl"
            for seed in VC_TIE_SEEDS
        },
    }

    # Pre-Gold gate: immutable primary inputs and evaluator contract.
    require_hash(FINAL_PRED, EXPECTED_FINAL_SHA)
    require_hash(TOP269, EXPECTED_TOP269_SHA)
    require_hash(LOW1072, EXPECTED_LOW1072_SHA)
    require_hash(INITIAL, EXPECTED_INITIAL_SHA)
    require_hash(OFFICIAL_EVALUATOR, EXPECTED_EVALUATOR_SHA)
    for path in [OLD_V1_PRED, TERMINAL_AUDIT, *random_paths.values(), *vc_paths.values(), GOLD]:
        if not path.is_file():
            raise EvalError(f"Missing required input: {path}")

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    protocol = {
        "status": "MMQA_CLEAN_RISK_V2_ONE_TIME_OFFICIAL_EVAL_PROTOCOL_FROZEN",
        "primary_denominator": 1341,
        "selected_budget": 269,
        "low_risk_keep": 1072,
        "immutable_primary_inputs": {
            "final_predictions": {"path": str(FINAL_PRED), "sha256": sha256(FINAL_PRED)},
            "top269": {"path": str(TOP269), "sha256": sha256(TOP269)},
            "low1072": {"path": str(LOW1072), "sha256": sha256(LOW1072)},
            "initial": {"path": str(INITIAL), "sha256": sha256(INITIAL)},
        },
        "official_scoring_contract": {
            "source": str(OFFICIAL_EVALUATOR),
            "source_sha256": sha256(OFFICIAL_EVALUATOR),
            "functions_imported_without_modification": ["extract_gold_answers", "stringify_answer", "best_score"],
        },
        "gold": {"path": str(GOLD), "sha256": sha256(GOLD), "access_authorized": True},
        "comparators": {
            "old_v1": {"path": str(OLD_V1_PRED), "sha256": sha256(OLD_V1_PRED)},
            "random20": {k: {"path": str(v), "sha256": sha256(v)} for k, v in random_paths.items()},
            "vc20": {k: {"path": str(v), "sha256": sha256(v)} for k, v in vc_paths.items()},
        },
        "significance": {
            "mcnemar": "two-sided exact binomial on EM discordances",
            "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
            "em_seed": BOOTSTRAP_SEED,
            "f1_seed": BOOTSTRAP_SEED + 1,
            "confidence_interval": "percentile 95%",
        },
        "prediction_mutation": False,
        "regeneration": False,
        "training": False,
        "selection_change": False,
        "external_api_calls": 0,
    }
    protocol_path = OUT_DIR / "one_time_official_eval_protocol_v1.json"
    write_json(protocol_path, protocol)
    print("Protocol freeze/preflight: PASS")
    print(f"Final SHA: {EXPECTED_FINAL_SHA}")
    print(f"Top269 SHA: {EXPECTED_TOP269_SHA}")

    OFFICIAL = import_official_evaluator()

    clean_rows = index_qids(read_jsonl(FINAL_PRED), "Clean V2 final predictions")
    top_rows = index_qids(read_jsonl(TOP269), "Top269")
    low_rows = index_qids(read_jsonl(LOW1072), "Low1072")
    initial_rows = index_qids(read_jsonl(INITIAL), "Initial")
    terminal_rows = index_qids(read_jsonl(TERMINAL_AUDIT), "terminal audit")
    qids = sorted(clean_rows)
    if len(qids) != 1341 or len(top_rows) != 269 or len(low_rows) != 1072:
        raise EvalError("Frozen cardinality mismatch")
    if set(top_rows) & set(low_rows) or set(top_rows) | set(low_rows) != set(qids):
        raise EvalError("Top269/Low1072 partition mismatch")
    if set(initial_rows) != set(qids):
        raise EvalError("Initial/Clean QID universe mismatch")
    if len(terminal_rows) != 5 or not set(terminal_rows).issubset(top_rows):
        raise EvalError("Terminal5 boundary mismatch")
    for qid in qids:
        flag = bool(clean_rows[qid].get("selected_for_intervention"))
        if flag != (qid in top_rows):
            raise EvalError(f"Selection flag mismatch: {qid}")

    initial_answers = {q: answer_from(initial_rows[q], "draft_answer") for q in qids}
    clean_answers = {q: answer_from(clean_rows[q]) for q in qids}
    low_exact = [q for q in low_rows if clean_answers[q] != initial_answers[q]]
    if low_exact:
        raise EvalError(f"Low-risk Final != Initial: {low_exact[:5]}")
    terminal_fallback_errors = [
        q for q in terminal_rows
        if clean_answers[q] != initial_answers[q]
        or not bool(clean_rows[q].get("terminal_fallback_applied"))
        or clean_rows[q].get("generation_status") != "keep_initial_terminal"
    ]
    if terminal_fallback_errors:
        raise EvalError(f"Terminal fallback mismatch: {terminal_fallback_errors}")
    empty_qids = [q for q in qids if clean_answers[q] == ""]
    if len(empty_qids) != 3:
        raise EvalError(f"Expected 3 empty final predictions; observed {len(empty_qids)}")

    # This is the first and only point where official Gold/reference content is read.
    gold_all = index_qids(read_jsonl(GOLD), "official MMQA Gold")
    if not set(qids).issubset(gold_all):
        raise EvalError("Official Gold does not cover locked-1341")
    gold = {q: OFFICIAL.extract_gold_answers(gold_all[q]) for q in qids}

    initial_scores = score_answers(initial_answers, gold, qids)
    clean_scores = score_answers(clean_answers, gold, qids)

    old_rows = index_qids(read_jsonl(OLD_V1_PRED), "old V1")
    old_answers = {q: answer_from(old_rows[q]) for q in qids}
    old_scores = score_answers(old_answers, gold, qids)

    random_answers: Dict[str, Dict[str, str]] = {}
    random_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for seed, path in random_paths.items():
        rows = index_qids(read_jsonl(path), f"Random20 seed {seed}")
        random_answers[seed] = {q: answer_from(rows[q]) for q in qids}
        random_scores[seed] = score_answers(random_answers[seed], gold, qids)

    vc_answers: Dict[str, Dict[str, str]] = {}
    vc_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, path in vc_paths.items():
        rows = index_qids(read_jsonl(path), name)
        vc_answers[name] = {q: answer_from(rows[q]) for q in qids}
        vc_scores[name] = score_answers(vc_answers[name], gold, qids)

    # Scoring-equivalence gate against previously frozen official results.
    initial_check = summarize(initial_scores, qids)
    old_check = summarize(old_scores, qids)
    vc_check = summarize(vc_scores["vc20_qid_ascending"], qids)
    expected_checks = [
        ("Initial EM count", initial_check["em_count"], 641),
        ("Old V1 EM count", old_check["em_count"], 694),
        ("VC20 primary EM count", vc_check["em_count"], 743),
    ]
    for label, actual, expected in expected_checks:
        if actual != expected:
            raise EvalError(f"Official scorer equivalence failed: {label}: {actual} != {expected}")
    if abs(initial_check["f1"] - 0.5499891914604916) > 1e-12:
        raise EvalError("Initial F1 scorer-equivalence gate failed")
    if abs(old_check["f1"] - 0.5873644117115852) > 1e-12:
        raise EvalError("Old V1 F1 scorer-equivalence gate failed")
    if abs(vc_check["f1"] - 0.6363706572936254) > 1e-12:
        raise EvalError("VC20 F1 scorer-equivalence gate failed")

    selected_qids = sorted(top_rows)
    low_qids = sorted(low_rows)
    terminal_qids = sorted(terminal_rows)
    group_summary = {
        "selected269": {
            "initial": summarize(initial_scores, selected_qids),
            "clean_v2": method_summary(clean_scores, initial_scores, selected_qids),
        },
        "low_risk1072": {
            "initial": summarize(initial_scores, low_qids),
            "clean_v2": method_summary(clean_scores, initial_scores, low_qids),
            "final_equals_initial_count": len(low_qids),
            "final_equals_initial_verified": True,
        },
        "protocol_terminal5": {
            "qids": terminal_qids,
            "initial": summarize(initial_scores, terminal_qids),
            "clean_v2": method_summary(clean_scores, initial_scores, terminal_qids),
            "fallback_exact_count": len(terminal_qids),
        },
        "empty_final3": {
            "qids": sorted(empty_qids),
            "initial": summarize(initial_scores, sorted(empty_qids)),
            "clean_v2": method_summary(clean_scores, initial_scores, sorted(empty_qids)),
        },
    }

    random_metrics = {k: method_summary(v, initial_scores, qids) for k, v in random_scores.items()}
    vc_metrics = {k: method_summary(v, initial_scores, qids) for k, v in vc_scores.items()}
    baseline_comparison = {
        "initial": summarize(initial_scores, qids),
        "old_v1": method_summary(old_scores, initial_scores, qids),
        "random20_seeds": random_metrics,
        "random20_aggregate": aggregate_seed_metrics(random_metrics),
        "vc20_variants": vc_metrics,
        "vc20_random_tie_aggregate": aggregate_seed_metrics(
            {k: v for k, v in vc_metrics.items() if k.startswith("vc20_random_tie_")}
        ),
        "clean_v2": method_summary(clean_scores, initial_scores, qids),
    }

    primary_pairs = {
        "initial": pairwise("clean_v2", clean_scores, "initial", initial_scores, qids),
        "old_v1": pairwise("clean_v2", clean_scores, "old_v1", old_scores, qids),
        "vc20_qid_ascending": pairwise(
            "clean_v2", clean_scores, "vc20_qid_ascending", vc_scores["vc20_qid_ascending"], qids
        ),
    }
    seedwise_em = {}
    for name, scores in {**{f"random20_{k}": v for k, v in random_scores.items()}, **vc_scores}.items():
        a_better = sum(int(clean_scores[q]["em"] == 1 and scores[q]["em"] == 0) for q in qids)
        b_better = sum(int(clean_scores[q]["em"] == 0 and scores[q]["em"] == 1) for q in qids)
        seedwise_em[name] = {
            "clean_better_discordant": a_better,
            "baseline_better_discordant": b_better,
            "mcnemar_exact_p": exact_mcnemar_p(a_better, b_better),
        }
    random_mean_scores = averaged_seed_scores(random_scores, qids)
    vc_tie_mean_scores = averaged_seed_scores(
        {k: v for k, v in vc_scores.items() if k.startswith("vc20_random_tie_")}, qids
    )
    raw_p = {
        "initial": primary_pairs["initial"]["em"]["mcnemar_exact_p"],
        "old_v1": primary_pairs["old_v1"]["em"]["mcnemar_exact_p"],
        **{k: v["mcnemar_exact_p"] for k, v in seedwise_em.items()},
    }
    significance = {
        "primary_pairwise": primary_pairs,
        "random20_seed_mean_fractional_comparison": pairwise_fractional(
            clean_scores, random_mean_scores, "random20_seed_mean", qids
        ),
        "vc20_random_tie_mean_fractional_comparison": pairwise_fractional(
            clean_scores, vc_tie_mean_scores, "vc20_random_tie_mean", qids
        ),
        "seedwise_em_mcnemar": seedwise_em,
        "holm_adjusted_mcnemar_across_reported_discrete_comparisons": holm_adjust(raw_p),
        "note": "Primary inference is Clean V2 vs Initial. Comparator tests are secondary and reported transparently.",
    }

    per_qid: List[Dict[str, Any]] = []
    for qid in qids:
        initial_em = int(initial_scores[qid]["em"])
        clean_em = int(clean_scores[qid]["em"])
        if initial_em == 0 and clean_em == 1:
            move = "W2R"
        elif initial_em == 1 and clean_em == 0:
            move = "R2W"
        elif initial_em == 1:
            move = "R2R"
        else:
            move = "W2W"
        row = clean_rows[qid]
        per_qid.append({
            "qid": qid,
            "group": "SELECTED269" if qid in top_rows else "LOW_RISK1072",
            "selected_for_intervention": qid in top_rows,
            "high_risk_rank": top_rows.get(qid, {}).get("high_risk_rank"),
            "generation_status": row.get("generation_status"),
            "failure_class": row.get("failure_class"),
            "terminal_fallback_applied": bool(row.get("terminal_fallback_applied")),
            "empty_final_prediction": clean_answers[qid] == "",
            "gold_answers": gold[qid],
            "initial_answer": initial_answers[qid],
            "clean_v2_answer": clean_answers[qid],
            "initial_em": initial_em,
            "initial_f1": float(initial_scores[qid]["f1"]),
            "clean_v2_em": clean_em,
            "clean_v2_f1": float(clean_scores[qid]["f1"]),
            "migration": move,
            "old_v1": {"answer": old_answers[qid], **old_scores[qid]},
            "random20": {
                k: {"answer": random_answers[k][qid], **random_scores[k][qid]} for k in random_scores
            },
            "vc20": {k: {"answer": vc_answers[k][qid], **vc_scores[k][qid]} for k in vc_scores},
        })

    terminal_empty_rows = [r for r in per_qid if r["terminal_fallback_applied"] or r["empty_final_prediction"]]
    details_path = OUT_DIR / "mmqa_clean_risk_v2_one_time_official_eval_details_v1.jsonl"
    baseline_path = OUT_DIR / "mmqa_clean_risk_v2_baseline_comparison_v1.json"
    significance_path = OUT_DIR / "mmqa_clean_risk_v2_significance_v1.json"
    terminal_path = OUT_DIR / "mmqa_clean_risk_v2_terminal_empty_official_audit_v1.jsonl"
    write_jsonl(details_path, per_qid)
    write_json(baseline_path, baseline_comparison)
    write_json(significance_path, significance)
    write_jsonl(terminal_path, terminal_empty_rows)

    summary = {
        "status": STATUS,
        "n": len(qids),
        "official_scoring_equivalence_gate": "PASS",
        "input_validation": {
            "final_prediction_sha256": sha256(FINAL_PRED),
            "final_prediction_sha_match": True,
            "top269_sha256": sha256(TOP269),
            "top269_sha_match": True,
            "prediction_mutation": False,
            "regeneration": False,
            "training": False,
            "selection_change": False,
            "external_api_calls": 0,
        },
        "initial": summarize(initial_scores, qids),
        "clean_v2": method_summary(clean_scores, initial_scores, qids),
        "group_results": group_summary,
        "terminal_and_empty": {
            "protocol_terminal_count": len(terminal_qids),
            "protocol_terminal_qids": terminal_qids,
            "terminal_fallback_correct_count": len(terminal_qids),
            "empty_final_prediction_count": len(empty_qids),
            "empty_final_prediction_qids": sorted(empty_qids),
        },
        "comparators": baseline_comparison,
        "significance_primary": primary_pairs["initial"],
        "gold": {"path": str(GOLD), "sha256": sha256(GOLD), "read_once_for_this_frozen_evaluation": True},
        "stop_condition": "FROZEN_AFTER_ONE_TIME_OFFICIAL_EVALUATION; NO_POST_HOC_CHANGE",
    }
    summary_path = OUT_DIR / "mmqa_clean_risk_v2_one_time_official_eval_summary_v1.json"
    write_json(summary_path, summary)

    files_to_hash = sorted(p for p in OUT_DIR.iterdir() if p.is_file())
    manifest = {
        "status": STATUS,
        "files": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files_to_hash
        ],
        "source_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "immutable_source_inputs_revalidated_after_evaluation": {
            "final_predictions": sha256(FINAL_PRED),
            "top269": sha256(TOP269),
        },
    }
    manifest_path = OUT_DIR / "SHA256SUMS.json"
    write_json(manifest_path, manifest)

    clean = summary["clean_v2"]
    init = summary["initial"]
    print("Evaluation freeze: PASS")
    print(f"Initial: EM={init['em_percent']:.4f}% F1={init['f1_percent']:.4f}%")
    print(
        f"Clean V2: EM={clean['em_percent']:.4f}% F1={clean['f1_percent']:.4f}% "
        f"DeltaEM={clean['delta_em_pp']:+.4f}pp DeltaF1={clean['delta_f1_pp']:+.4f}pp"
    )
    m = clean["migration_from_initial"]
    print(f"W2R={m['wrong_to_right']} R2W={m['right_to_wrong']} Net={m['net']:+d}")
    print(f"Selected={len(selected_qids)} Low-risk KEEP={len(low_qids)} Terminal={len(terminal_qids)} Empty={len(empty_qids)}")
    print(f"Summary: {summary_path}")
    print(f"Summary SHA256: {sha256(summary_path)}")
    print(f"Manifest: {manifest_path}")
    print(f"STATUS = {STATUS}")
    return 0


OFFICIAL: Any = None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvalError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
