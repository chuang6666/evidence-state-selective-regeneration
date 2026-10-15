#!/usr/bin/env python3
"""One-time, read-only official evaluation of frozen M3-VQA 13,125 predictions.

The evaluator opens M3 Gold/eval-only data only after verifying the passed
before-Gold audit and every frozen input hash.  It never calls an API, never
regenerates an answer, and never modifies inference artifacts.

Primary metric: the official M3-VQA answer-list Accuracy implemented in
https://github.com/CASIA-IVA-Lab/M3VQA/blob/main/run_eval.py
Additional paired diagnostics compare Initial and Final on the exact same
13,125 qids.  "Full match" means official per-question Accuracy == 1.0.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


EXPECTED_TOTAL = 13_125
EXPECTED_HIGH = 2_625
EXPECTED_LOW = 10_500
EXPECTED_INITIAL_SUCCESS = 12_726
EXPECTED_INITIAL_UNAVAILABLE = 399
EXPECTED_REGEN_SUCCESS = 2_434
EXPECTED_TERMINAL_KEEP = 191
EPS = 1e-12

OFFICIAL_EVAL_URL = "https://github.com/CASIA-IVA-Lab/M3VQA/blob/main/run_eval.py"
BEFORE_GOLD_PASS = "M3VQA_FROZEN13125_BEFORE_GOLD_AUDIT_PASS"
PASS_STATUS = "M3VQA_FROZEN13125_ONE_TIME_OFFICIAL_EVAL_PASS"
FAIL_STATUS = "M3VQA_FROZEN13125_ONE_TIME_OFFICIAL_EVAL_FAIL"


DIGIT_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "entailment": "yes",
    "true": "yes",
    "contradiction": "no",
    "false": "no",
}

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't",
    "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
    "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's",
    "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}


class ProtocolError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="MultimodalEvidenceChain project root",
    )
    return parser.parse_args()


def safe(value: Any) -> str:
    return "" if value is None else str(value).strip()


def qid_of(row: Mapping[str, Any]) -> str:
    qid = safe(row.get("qid") or row.get("question_id") or row.get("data_id"))
    if not qid:
        raise ProtocolError("row has no qid/data_id")
    return qid


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
            try:
                value = json.loads(line)
            except Exception as exc:
                raise ProtocolError(
                    f"{path.name}:{line_number}: malformed JSON: {exc}"
                ) from exc
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def normalize_answer(text: Any) -> str:
    """Exact normalization order from the official M3-VQA run_eval.py."""
    value = safe(text).lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation + "‘’´`_"))
    value = re.sub(r"\b(the answer is|a|an|the)\b", " ", value)
    words: List[str] = []
    for word in value.split():
        word = DIGIT_MAP.get(word, word)
        word = CONTRACTIONS.get(word, word)
        words.append(word)
    return " ".join(" ".join(words).split())


def convert_to_list(value: Any) -> List[Any]:
    """Safe equivalent of the official answer-string list conversion."""
    if isinstance(value, list):
        return value
    text = safe(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    if "[" in text and "]" in text:
        content = text[text.find("[") + 1 : text.find("]")]
    elif "[" in text:
        content = text[text.find("[") + 1 :]
    elif "]" in text:
        content = text[: text.find("]")]
    else:
        content = text
    return [item.strip() for item in content.split(",") if item.strip()]


def validate_answer_evals(value: Any, qid: str) -> List[List[Any]]:
    if not isinstance(value, list) or not value:
        raise ProtocolError(f"{qid}: answer_evals is not a nonempty list")
    result: List[List[Any]] = []
    for index, eval_set in enumerate(value):
        if not isinstance(eval_set, list) or not eval_set:
            raise ProtocolError(f"{qid}: answer_evals[{index}] is not a nonempty list")
        result.append(eval_set)
    return result


def official_accuracy(predicted_answers: Sequence[Any], answer_evals: Sequence[Sequence[Any]]) -> float:
    """Replicate official M3-VQA evaluate_predictions()."""
    if not predicted_answers:
        return 0.0
    normalized_pred = [normalize_answer(answer) for answer in predicted_answers]
    normalized_evals = [
        [normalize_answer(answer) for answer in eval_set]
        for eval_set in answer_evals
    ]
    flattened = {answer for eval_set in normalized_evals for answer in eval_set}
    if not (set(normalized_pred) & flattened):
        return 0.0
    keys: List[List[int]] = []
    for answer in normalized_pred:
        indices = [
            index
            for index, eval_set in enumerate(normalized_evals)
            if answer in eval_set
        ]
        if indices:
            keys.append(indices)
    matched = min(len(keys), len({index for indices in keys for index in indices}))
    denominator = len(normalized_pred) + len(normalized_evals) - matched
    return matched / denominator if denominator else 0.0


def integer_field(row: Mapping[str, Any], key: str, qid: str) -> int:
    try:
        return int(row[key])
    except Exception as exc:
        raise ProtocolError(f"{qid}: invalid {key}={row.get(key)!r}") from exc


def bucket(value: int) -> str:
    return str(value) if value in (1, 2, 3) else ">3"


def score_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    initial_total = math.fsum(float(row["initial_official_accuracy"]) for row in rows)
    final_total = math.fsum(float(row["final_official_accuracy"]) for row in rows)
    initial_full = sum(bool(row["initial_full_match"]) for row in rows)
    final_full = sum(bool(row["final_full_match"]) for row in rows)
    improved = sum(float(row["score_delta"]) > EPS for row in rows)
    regressed = sum(float(row["score_delta"]) < -EPS for row in rows)
    unchanged = n - improved - regressed
    migrations = Counter(safe(row["full_match_migration"]) for row in rows)
    return {
        "N": n,
        "Initial_official_accuracy": initial_total / n if n else 0.0,
        "Final_official_accuracy": final_total / n if n else 0.0,
        "Delta_official_accuracy_pp": (final_total - initial_total) / n * 100.0 if n else 0.0,
        "Initial_official_score_sum": initial_total,
        "Final_official_score_sum": final_total,
        "Net_official_score_gain": final_total - initial_total,
        "Score_improved": improved,
        "Score_regressed": regressed,
        "Score_unchanged": unchanged,
        "Initial_full_match_count": initial_full,
        "Initial_full_match_rate": initial_full / n if n else 0.0,
        "Final_full_match_count": final_full,
        "Final_full_match_rate": final_full / n if n else 0.0,
        "Full_match_W2R": migrations["W2R"],
        "Full_match_R2W": migrations["R2W"],
        "Full_match_Net": migrations["W2R"] - migrations["R2W"],
        "Full_match_migrations": dict(sorted(migrations.items())),
    }


def grouped_metrics(
    rows: Sequence[Mapping[str, Any]], key: str
) -> Dict[str, Dict[str, Any]]:
    groups: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[safe(row.get(key)) or "UNKNOWN"].append(row)
    return {
        name: score_summary(group)
        for name, group in sorted(groups.items(), key=lambda item: item[0])
    }


def paths_for(root: Path) -> Dict[str, Path]:
    jobs_dir = root / "outputs/m3vqa_frozen_top6_plain_regen_jobs_v1"
    run_dir = root / "outputs/m3vqa_frozen_top6_plain_regen_qwen37plus_v1_2"
    initial_dir = root / "outputs/m3vqa_initial_multimodal_qa_glm5v_batch_v2_frozen"
    return {
        "high_jobs": jobs_dir / "m3vqa_frozen_high_risk_plain_regen_jobs_2625_v1.jsonl",
        "low_keep": jobs_dir / "m3vqa_frozen_low_risk_keep_initial_10500_v1.jsonl",
        "protocol": jobs_dir / "m3vqa_frozen_plain_regen_protocol_v1.json",
        "initial_success": initial_dir / "m3vqa_initial_multimodal_qa_glm5v_batch_v2_frozen_successes.jsonl",
        "generation_summary": run_dir / "m3vqa_frozen_plain_regen_generation_summary_v1.json",
        "high_frozen": run_dir / "m3vqa_high_risk2625_effective_predictions_frozen_v1.jsonl",
        "final_13125": run_dir / "m3vqa_frozen13125_final_predictions_before_eval_v1.jsonl",
        "initial_results": initial_dir / "m3vqa_initial_multimodal_qa_glm5v_batch_v2_frozen_results.jsonl",
        "before_gold_audit": (
            root / "outputs/m3vqa_frozen13125_before_gold_audit_v1/"
            "m3vqa_frozen13125_before_gold_audit_summary_v1.json"
        ),
        "gold_eval_only": (
            root / "outputs/m3vqa_dataset_adapter_v1_1/"
            "m3vqa_external_zero_shot_eval_only_v1_1.jsonl"
        ),
    }


def validate_frozen_boundary(paths: Mapping[str, Path]) -> Dict[str, Any]:
    audit = load_json(paths["before_gold_audit"])
    if safe(audit.get("status")) != BEFORE_GOLD_PASS:
        raise ProtocolError("before-Gold boundary audit status is not PASS")
    if audit.get("hard_check_failures"):
        raise ProtocolError("before-Gold boundary audit contains hard failures")
    expected_hashes = audit.get("input_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise ProtocolError("before-Gold audit has no input_sha256 manifest")
    for label in (
        "high_jobs", "low_keep", "protocol", "initial_success",
        "generation_summary", "high_frozen", "final_13125",
    ):
        expected = safe(expected_hashes.get(label))
        if not expected:
            raise ProtocolError(f"before-Gold audit has no hash for {label}")
        observed = sha256_file(paths[label])
        if observed != expected:
            raise ProtocolError(f"frozen input hash changed after before-Gold audit: {label}")
    population = audit.get("population")
    if not isinstance(population, Mapping):
        raise ProtocolError("before-Gold audit population manifest missing")
    if int(population.get("unique_qids", -1)) != EXPECTED_TOTAL:
        raise ProtocolError("before-Gold audited qid count changed")
    return audit


def evaluate(root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    paths = paths_for(root)
    audit = validate_frozen_boundary(paths)

    # Gold is opened only after all frozen-boundary hashes pass.
    final_rows = load_jsonl(paths["final_13125"])
    initial_rows = load_jsonl(paths["initial_results"])
    high_rows = load_jsonl(paths["high_jobs"])
    gold_rows = load_jsonl(paths["gold_eval_only"])

    final = index_unique("final_13125", final_rows)
    initial = index_unique("initial_results", initial_rows)
    high = index_unique("high_jobs", high_rows)
    gold = index_unique("gold_eval_only", gold_rows)

    population = set(final)
    if len(final_rows) != EXPECTED_TOTAL or len(final) != EXPECTED_TOTAL:
        raise ProtocolError("frozen Final is not exactly 13,125 unique qids")
    if len(initial_rows) != EXPECTED_TOTAL or len(initial) != EXPECTED_TOTAL:
        raise ProtocolError("frozen Initial results are not exactly 13,125 unique qids")
    if len(gold_rows) != EXPECTED_TOTAL or len(gold) != EXPECTED_TOTAL:
        raise ProtocolError("M3 eval-only file is not exactly 13,125 unique qids")
    if set(initial) != population or set(gold) != population:
        raise ProtocolError("Initial/Final/Gold qid sets are not identical")
    if len(high) != EXPECTED_HIGH or not set(high).issubset(population):
        raise ProtocolError("high-risk job set is not the frozen 2,625-qid subset")

    initial_status_counts = Counter(safe(row.get("status")) for row in initial_rows)
    if initial_status_counts != Counter(
        {"success": EXPECTED_INITIAL_SUCCESS, "provider_safety_abstain": EXPECTED_INITIAL_UNAVAILABLE}
    ):
        raise ProtocolError(f"unexpected Initial status counts={dict(initial_status_counts)}")
    final_source_counts = Counter(safe(row.get("source")) for row in final_rows)
    if final_source_counts != Counter(
        {
            "QWEN_REGEN": EXPECTED_REGEN_SUCCESS,
            "KEEP_INITIAL_AFTER_REGEN_FAILURE": EXPECTED_TERMINAL_KEEP,
            "LOW_RISK_INITIAL": EXPECTED_LOW,
        }
    ):
        raise ProtocolError(f"unexpected Final source counts={dict(final_source_counts)}")

    route_by_qid: Dict[str, str] = {}
    for qid, row in high.items():
        metadata = row.get("audit_metadata_not_for_solver_prompt")
        if not isinstance(metadata, Mapping):
            raise ProtocolError(f"{qid}: high-job audit metadata missing")
        route_by_qid[qid] = safe(metadata.get("localized_transition")) or "CLEAN_OR_UNCERTAIN"

    per_qid: List[Dict[str, Any]] = []
    for qid in sorted(population):
        initial_row = initial[qid]
        final_row = final[qid]
        gold_row = gold[qid]
        if gold_row.get("eval_only") is not True:
            raise ProtocolError(f"{qid}: eval_only guardrail is not true")
        if gold_row.get("must_not_be_loaded_during_inference") is not True:
            raise ProtocolError(f"{qid}: inference guardrail is not true")
        if bool(final_row.get("gold_used_in_inference")):
            raise ProtocolError(f"{qid}: final row says Gold was used in inference")

        answer_evals = validate_answer_evals(gold_row.get("answer_evals"), qid)
        initial_answer = safe(initial_row.get("draft_answer"))
        final_answer = safe(final_row.get("final_answer"))
        initial_predicted = convert_to_list(initial_answer)
        final_predicted = convert_to_list(final_answer)
        initial_score = official_accuracy(initial_predicted, answer_evals)
        final_score = official_accuracy(final_predicted, answer_evals)
        delta = final_score - initial_score
        initial_full = abs(initial_score - 1.0) <= EPS
        final_full = abs(final_score - 1.0) <= EPS
        if not initial_full and final_full:
            migration = "W2R"
        elif initial_full and not final_full:
            migration = "R2W"
        elif initial_full and final_full:
            migration = "R2R"
        else:
            migration = "W2W"

        question_hop = integer_field(gold_row, "question_hop", qid)
        entity_num = integer_field(gold_row, "entity_num", qid)
        high_risk = qid in high
        source = safe(final_row.get("source"))
        terminal_category = "NOT_TERMINAL"
        if source == "KEEP_INITIAL_AFTER_REGEN_FAILURE":
            terminal_category = (
                "TERMINAL_KEEP_WITH_INITIAL"
                if bool(final_row.get("initial_available"))
                else "TERMINAL_KEEP_WITHOUT_INITIAL"
            )
        per_qid.append(
            {
                "qid": qid,
                "risk_group": "HIGH_RISK_PLAIN_REGEN" if high_risk else "LOW_RISK_KEEP_INITIAL",
                "localized_route": route_by_qid.get(qid, "LOW_RISK_NOT_LOCALIZED"),
                "source": source,
                "generation_status": safe(final_row.get("generation_status")),
                "terminal_category": terminal_category,
                "initial_available": bool(final_row.get("initial_available")),
                "initial_answer_empty": not bool(initial_answer),
                "final_answer_empty": not bool(final_answer),
                "initial_predicted_item_count": len(initial_predicted),
                "final_predicted_item_count": len(final_predicted),
                "gold_answer_group_count": len(answer_evals),
                "initial_official_accuracy": initial_score,
                "final_official_accuracy": final_score,
                "score_delta": delta,
                "score_migration": (
                    "IMPROVED" if delta > EPS else "REGRESSED" if delta < -EPS else "UNCHANGED"
                ),
                "initial_full_match": initial_full,
                "final_full_match": final_full,
                "full_match_migration": migration,
                "question_hop": question_hop,
                "question_hop_bucket": bucket(question_hop),
                "entity_num": entity_num,
                "entity_num_bucket": bucket(entity_num),
                "question_type": safe(gold_row.get("question_type")) or "UNKNOWN",
                "parsed_complexity": safe(gold_row.get("parsed_complexity")) or "UNKNOWN",
            }
        )

    overall = score_summary(per_qid)
    high_metrics = score_summary([row for row in per_qid if row["risk_group"] == "HIGH_RISK_PLAIN_REGEN"])
    low_metrics = score_summary([row for row in per_qid if row["risk_group"] == "LOW_RISK_KEEP_INITIAL"])
    if low_metrics["Score_improved"] or low_metrics["Score_regressed"]:
        raise ProtocolError("low-risk KEEP subset changed between Initial and Final")
    if overall["Full_match_Net"] != overall["Final_full_match_count"] - overall["Initial_full_match_count"]:
        raise ProtocolError("full-match Net identity failed")

    result: Dict[str, Any] = {
        "status": PASS_STATUS,
        "project_root": str(root),
        "protocol": {
            "external_api_calls": 0,
            "answer_regeneration": False,
            "existing_experiment_files_modified": False,
            "m3_gold_eval_only_loaded": True,
            "gold_loaded_after_before_gold_hash_verification": True,
            "old_t1_t5_stage_a_stage_b_loaded": False,
            "primary_metric": "official M3-VQA answer-list Accuracy",
            "official_evaluator_source": OFFICIAL_EVAL_URL,
            "full_match_definition": "official per-qid Accuracy == 1.0",
            "paired_population": EXPECTED_TOTAL,
        },
        "overall": overall,
        "high_risk": high_metrics,
        "low_risk": low_metrics,
        "by_question_hop": grouped_metrics(per_qid, "question_hop_bucket"),
        "by_entity_num": grouped_metrics(per_qid, "entity_num_bucket"),
        "by_question_type": grouped_metrics(per_qid, "question_type"),
        "by_parsed_complexity": grouped_metrics(per_qid, "parsed_complexity"),
        "high_risk_by_localized_route": grouped_metrics(
            [row for row in per_qid if row["risk_group"] == "HIGH_RISK_PLAIN_REGEN"],
            "localized_route",
        ),
        "by_source": grouped_metrics(per_qid, "source"),
        "by_terminal_category": grouped_metrics(per_qid, "terminal_category"),
        "counts": {
            "initial_status": dict(sorted(initial_status_counts.items())),
            "final_source": dict(sorted(final_source_counts.items())),
            "final_empty": sum(bool(row["final_answer_empty"]) for row in per_qid),
        },
        "input_sha256": {
            "before_gold_audit": sha256_file(paths["before_gold_audit"]),
            "initial_results": sha256_file(paths["initial_results"]),
            "final_13125": sha256_file(paths["final_13125"]),
            "gold_eval_only": sha256_file(paths["gold_eval_only"]),
        },
        "before_gold_audit_status": audit["status"],
        "hard_check_failures": [],
        "soft_warnings": [],
    }

    report_lines = [
        "M3-VQA FROZEN-13125 ONE-TIME OFFICIAL EVALUATION V1",
        "=" * 110,
        f"Status: {PASS_STATUS}",
        f"Project root: {root}",
        "External API calls: 0",
        "Answer regeneration: NO",
        "Existing experiment files modified: NO",
        "M3 Gold/eval-only loaded: YES — evaluation only, after frozen hashes passed",
        "Old T1/T5 Stage-A/Stage-B loaded: NO",
        "Primary metric: official M3-VQA answer-list Accuracy",
        "",
        f"Evaluation qids: {overall['N']}",
        f"Initial official Accuracy: {overall['Initial_official_accuracy'] * 100:.4f}%",
        f"Final official Accuracy:   {overall['Final_official_accuracy'] * 100:.4f}%",
        f"Delta: {overall['Delta_official_accuracy_pp']:+.4f}pp",
        f"Score improved / regressed / unchanged: {overall['Score_improved']} / {overall['Score_regressed']} / {overall['Score_unchanged']}",
        f"Initial / Final full match: {overall['Initial_full_match_count']} / {overall['Final_full_match_count']}",
        f"Full-match W2R / R2W / Net: {overall['Full_match_W2R']} / {overall['Full_match_R2W']} / {overall['Full_match_Net']:+d}",
        "",
        f"High-risk Initial / Final Accuracy: {high_metrics['Initial_official_accuracy'] * 100:.4f}% / {high_metrics['Final_official_accuracy'] * 100:.4f}%",
        f"Low-risk Initial / Final Accuracy:  {low_metrics['Initial_official_accuracy'] * 100:.4f}% / {low_metrics['Final_official_accuracy'] * 100:.4f}%",
        f"Final empty answers: {result['counts']['final_empty']}",
        "Hard check failures: 0",
        "Soft warnings: 0",
    ]
    return result, per_qid, "\n".join(report_lines) + "\n"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    out_dir = root / "outputs/m3vqa_frozen13125_one_time_official_eval_v1"
    summary_file = out_dir / "m3vqa_frozen13125_official_eval_summary_v1.json"
    report_file = out_dir / "m3vqa_frozen13125_official_eval_report_v1.txt"
    per_qid_file = out_dir / "m3vqa_frozen13125_official_eval_per_qid_v1.jsonl"

    print("=" * 126)
    print("M3-VQA — FROZEN-13125 ONE-TIME OFFICIAL EVALUATION V1")
    print("=" * 126)
    print(f"Project root: {root}")
    print("External API calls: 0")
    print("Answer regeneration: NO")
    print("Existing experiment files modified: NO")
    print("Gold opens only after before-Gold hashes pass: YES")
    print("Old T1/T5 Stage-A/Stage-B loaded: NO")
    print("Primary metric: official M3-VQA answer-list Accuracy")
    print("-" * 126)

    try:
        result, per_qid, report = evaluate(root)
        write_json(summary_file, result)
        write_text(report_file, report)
        write_jsonl(per_qid_file, per_qid)
    except Exception as exc:
        print(f"PROTOCOL ERROR: {type(exc).__name__}: {exc}")
        print(f"STATUS = {FAIL_STATUS}")
        return 2

    overall = result["overall"]
    high = result["high_risk"]
    low = result["low_risk"]
    print(f"Evaluation qids: {overall['N']}")
    print(f"Initial official Accuracy: {overall['Initial_official_accuracy'] * 100:.4f}%")
    print(f"Final official Accuracy:   {overall['Final_official_accuracy'] * 100:.4f}%")
    print(f"Delta: {overall['Delta_official_accuracy_pp']:+.4f}pp")
    print(
        "Score improved / regressed / unchanged: "
        f"{overall['Score_improved']} / {overall['Score_regressed']} / {overall['Score_unchanged']}"
    )
    print(
        "Initial / Final full match: "
        f"{overall['Initial_full_match_count']} / {overall['Final_full_match_count']}"
    )
    print(
        "Full-match W2R / R2W / Net: "
        f"{overall['Full_match_W2R']} / {overall['Full_match_R2W']} / {overall['Full_match_Net']:+d}"
    )
    print("-" * 126)
    print(
        "High-risk Initial / Final Accuracy: "
        f"{high['Initial_official_accuracy'] * 100:.4f}% / {high['Final_official_accuracy'] * 100:.4f}%"
    )
    print(
        "Low-risk Initial / Final Accuracy:  "
        f"{low['Initial_official_accuracy'] * 100:.4f}% / {low['Final_official_accuracy'] * 100:.4f}%"
    )
    print(f"Final empty answers: {result['counts']['final_empty']}")
    print("Hard check failures: 0")
    print("Soft warnings: 0")
    print(f"Summary: {summary_file}")
    print(f"Report:  {report_file}")
    print(f"Per-qid: {per_qid_file}")
    print(f"STATUS = {PASS_STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
