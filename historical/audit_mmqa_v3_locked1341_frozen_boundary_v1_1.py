#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
MMQA V3 locked-1341 frozen-boundary audit V1.1.

Purpose
-------
Independently and read-only audit the immutable prediction boundary created
after the 269 high-risk Qwen3.7-plus jobs completed.  The script never calls an
API, never regenerates an answer, and never edits an existing experiment file.
It writes only to a dedicated audit output directory.

Default project layout
----------------------
Place this script in:
    E:\RAG_Projects\MultimodalEvidenceChain

Then run it without arguments.  All defaults are resolved under that folder.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_VERSION = "1.1"

EXPECTED = {
    "n": 1341,
    "initial_em_count": 641,
    "final_em_count": 694,
    "initial_em": 641 / 1341,
    "final_em": 694 / 1341,
    "final_f1": 0.5873644117115852,
    "w2r": 65,
    "r2w": 12,
    "net": 53,
    "high_risk": 269,
    "low_risk": 1072,
    "qwen_success": 266,
    "high_risk_terminal_keep": 3,
    "approved_empty_terminal_keep": 3,
    "unresolved_transport": 0,
    "route_counts": {
        "CLEAN_OR_UNCERTAIN": 112,
        "T1": 80,
        "T2": 29,
        "T3": 19,
        "T4": 1,
        "T5": 28,
    },
    # These are overlapping Expected-Topology memberships, not mutually
    # exclusive routes.  They are a soft diagnostic cross-check only.
    "transition_membership_counts": {
        "T1": 355,
        "T2": 440,
        "T3": 264,
        "T4": 264,
        "T5": 1336,
    },
}


DEFAULTS = {
    "final_predictions": (
        "outputs/mmqa_v3_heldout1341_final_plain_regen_qwen37plus_v1/"
        "mmqa_v3_locked1341_final_predictions_frozen_before_eval_v1.jsonl"
    ),
    "generation_summary": (
        "outputs/mmqa_v3_heldout1341_final_plain_regen_qwen37plus_v1/"
        "mmqa_v3_locked1341_final_generation_summary_v1.json"
    ),
    "initial_base": (
        "outputs/mmqa_v3_heldout1341_pre_natural_v1/"
        "mmqa_v3_heldout1341_initial_base_sanitized_v1.jsonl"
    ),
    "high_risk_plan": (
        "outputs/mmqa_v3_heldout1341_frozen_verifier_localization_risk_v1/"
        "heldout1341_frozen_top20_high_risk_269_v1.jsonl"
    ),
    "low_risk_plan": (
        "outputs/mmqa_v3_heldout1341_frozen_verifier_localization_risk_v1/"
        "heldout1341_frozen_low_risk_1072_v1.jsonl"
    ),
    "localization": (
        "outputs/mmqa_v3_heldout1341_frozen_verifier_localization_risk_v1/"
        "heldout1341_localization_merged_all_qids_v1.jsonl"
    ),
    "gold": "data/raw/mmqa_official/MMQA_dev.jsonl.gz",
    "output_dir": "outputs/mmqa_v3_locked1341_frozen_boundary_audit_v1_1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit V1.1 of the MMQA V3 locked-1341 frozen boundary."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root. Defaults to the directory containing this script.",
    )
    for key in (
        "final_predictions",
        "generation_summary",
        "initial_base",
        "high_risk_plan",
        "low_risk_plan",
        "localization",
        "gold",
    ):
        parser.add_argument(
            "--" + key.replace("_", "-"),
            default=DEFAULTS[key],
            help=f"Override {key} path (absolute or relative to --root).",
        )
    parser.add_argument(
        "--output-dir",
        default=DEFAULTS["output_dir"],
        help="Dedicated audit output directory (absolute or relative to --root).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal metric/normalization tests and exit.",
    )
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open_text(path) as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}, "
                    f"got {type(row).__name__}."
                )
            rows.append(row)
    return rows


def canonical_qid(value: Any) -> str:
    return "" if value is None else str(value).strip()


def index_rows(
    rows: Sequence[Mapping[str, Any]], name: str
) -> Tuple[Dict[str, Mapping[str, Any]], List[str], int]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    duplicates: List[str] = []
    empty = 0
    for row in rows:
        qid = canonical_qid(row.get("qid", row.get("question_id")))
        if not qid:
            empty += 1
            continue
        if qid in indexed:
            duplicates.append(qid)
        else:
            indexed[qid] = row
    if duplicates:
        duplicates = sorted(set(duplicates))
    return indexed, duplicates, empty


def stringify_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(stringify_answer(item) for item in value).strip()
    if isinstance(value, dict):
        for key in ("answer", "text", "value"):
            if key in value:
                return stringify_answer(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def extract_gold_answers(row: Mapping[str, Any]) -> List[str]:
    raw_answers = row.get("answers", row.get("answer", []))
    if not isinstance(raw_answers, list):
        raw_answers = [raw_answers]
    answers: List[str] = []
    for item in raw_answers:
        if isinstance(item, dict):
            value = item.get("answer", item.get("text", item.get("value", "")))
        else:
            value = item
        if isinstance(value, list):
            answers.extend(stringify_answer(x) for x in value)
        else:
            answers.append(stringify_answer(value))
    # Keep an explicit empty answer if the dataset truly provides one.
    return answers or [""]


def normalize_answer(text: Any) -> str:
    value = stringify_answer(text).lower()
    # Preserve the exact order used by the original locked evaluator:
    # articles first, then ASCII punctuation, then whitespace normalization.
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(ch for ch in value if ch not in string.punctuation)
    return " ".join(value.split())


def exact_match(prediction: Any, gold: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_score(prediction: Any, gold_answers: Sequence[str]) -> Tuple[int, float]:
    em = int(max(exact_match(prediction, gold) for gold in gold_answers))
    f1 = max(token_f1(prediction, gold) for gold in gold_answers)
    return em, f1


def add_check(
    checks: List[Dict[str, Any]],
    name: str,
    observed: Any,
    expected: Any,
    passed: bool,
    *,
    hard: bool = True,
    note: str = "",
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "severity": "HARD" if hard else "SOFT",
            "observed": observed,
            "expected": expected,
            "note": note,
        }
    )


def close_enough(observed: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance)


def truthy_gold_paths(value: Any, prefix: str = "") -> List[str]:
    hits: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if "gold" in str(key).lower():
                false_like = child in (None, False, 0, "", [], {})
                if isinstance(child, str) and child.strip().lower() in {
                    "no",
                    "false",
                    "none",
                    "not_used",
                    "not used",
                    "closed",
                }:
                    false_like = True
                if not false_like:
                    hits.append(path)
            hits.extend(truthy_gold_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(truthy_gold_paths(child, f"{prefix}[{index}]"))
    return hits


def canonical_action(row: Mapping[str, Any]) -> str:
    source = str(row.get("source", "")).strip().upper()
    status = str(row.get("generation_status", "")).strip().lower()
    action = str(row.get("frozen_action", "")).strip().upper()

    if source == "LOW_RISK_INITIAL" or (status == "not_called" and action == "KEEP_INITIAL"):
        return "LOW_RISK_INITIAL"
    if source == "QWEN_REGEN" or status == "success" or action in {"USE_REGEN", "REGEN"}:
        return "QWEN_REGEN"
    if source == "KEEP_INITIAL_AFTER_REGEN_FAILURE" or status == "keep_initial_terminal":
        return "KEEP_INITIAL_AFTER_REGEN_FAILURE"
    return source or status.upper() or action or "UNKNOWN"


def canonical_route(row: Mapping[str, Any]) -> Optional[str]:
    candidates = [
        row.get("route"),
        row.get("localized_transition"),
        row.get("selected_transition"),
        row.get("repair_route"),
        row.get("primary_transition"),
    ]
    for value in candidates:
        if value is None:
            continue
        route = str(value).strip().upper()
        if route in {"T1", "T2", "T3", "T4", "T5"}:
            return route
        if route in {"NONE", "CLEAN", "CLEAN_OR_UNCERTAIN"}:
            return "CLEAN_OR_UNCERTAIN"

    decision = str(row.get("decision", "")).strip().upper()
    if decision in {"CLEAN_OR_UNCERTAIN", "STATE_UNAVAILABLE"}:
        return "CLEAN_OR_UNCERTAIN"
    return None


def expected_transitions(row: Mapping[str, Any]) -> List[str]:
    for key in (
        "predicted_required_transitions_external_frozen",
        "required_transitions",
        "predicted_required_transitions",
        "expected_transitions",
    ):
        value = row.get(key)
        if isinstance(value, list):
            found = []
            for item in value:
                transition = str(item).strip().upper()
                if transition in {"T1", "T2", "T3", "T4", "T5"}:
                    found.append(transition)
            return sorted(set(found))
    return []


def flatten_json(value: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_json(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            flat.update(flatten_json(child, f"{prefix}[{index}]"))
    else:
        flat[prefix] = value
    return flat


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def safe_write_json(path: Path, value: Any) -> None:
    safe_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run_self_test() -> int:
    tests = [
        (normalize_answer("The, Mask!"), "mask"),
        (normalize_answer("  AN   answer  "), "answer"),
        (exact_match("The Mask", "mask"), 1.0),
        (token_f1("red blue", "blue green"), 0.5),
        (best_score("Mask", ["mask", "other"]), (1, 1.0)),
    ]
    failures = []
    for index, (observed, expected) in enumerate(tests, start=1):
        if isinstance(observed, float):
            passed = close_enough(observed, float(expected))
        else:
            passed = observed == expected
        if not passed:
            failures.append({"test": index, "observed": observed, "expected": expected})
    if failures:
        print(json.dumps({"status": "SELF_TEST_FAIL", "failures": failures}, indent=2))
        return 1
    print("STATUS = MMQA_V3_FROZEN_BOUNDARY_AUDIT_SELF_TEST_PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    root = args.root.resolve()
    paths = {
        key: resolve(root, getattr(args, key))
        for key in (
            "final_predictions",
            "generation_summary",
            "initial_base",
            "high_risk_plan",
            "low_risk_plan",
            "localization",
            "gold",
        )
    }
    output_dir = resolve(root, args.output_dir)

    print("=" * 116)
    print("MMQA V3 — LOCKED-1341 FROZEN BOUNDARY AUDIT V1.1")
    print("=" * 116)
    print(f"Project root: {root}")
    print("External API calls: 0")
    print("Answer regeneration: NO")
    print("Existing experiment files modified: NO")
    print("MMQA Gold use: OFFLINE REPRODUCIBILITY AUDIT ONLY")
    print("M3-VQA accessed: NO")
    print("-" * 116)

    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        print("REQUIRED INPUTS MISSING")
        for item in missing:
            print(f"  - {item}")
        print("STATUS = MMQA_V3_FROZEN_BOUNDARY_AUDIT_INPUT_MISSING")
        return 2

    input_paths = list(paths.values())
    if output_dir in input_paths or any(output_dir == path.parent for path in input_paths):
        print("Refusing unsafe output directory: it overlaps an input location.")
        print("STATUS = MMQA_V3_FROZEN_BOUNDARY_AUDIT_UNSAFE_OUTPUT")
        return 2

    hashes = {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }

    final_rows = load_jsonl(paths["final_predictions"])
    initial_rows = load_jsonl(paths["initial_base"])
    high_rows = load_jsonl(paths["high_risk_plan"])
    low_rows = load_jsonl(paths["low_risk_plan"])
    localization_rows = load_jsonl(paths["localization"])
    gold_rows = load_jsonl(paths["gold"])
    generation_summary = load_json(paths["generation_summary"])

    final_by_qid, final_duplicates, final_empty_qids = index_rows(final_rows, "final")
    initial_by_qid, initial_duplicates, initial_empty_qids = index_rows(initial_rows, "initial")
    high_by_qid, high_duplicates, high_empty_qids = index_rows(high_rows, "high-risk")
    low_by_qid, low_duplicates, low_empty_qids = index_rows(low_rows, "low-risk")
    loc_by_qid, loc_duplicates, loc_empty_qids = index_rows(localization_rows, "localization")
    gold_by_qid, gold_duplicates, gold_empty_qids = index_rows(gold_rows, "gold")

    checks: List[Dict[str, Any]] = []

    # Structural integrity.
    add_check(checks, "final_unique_qids", len(final_by_qid), EXPECTED["n"], len(final_by_qid) == EXPECTED["n"])
    add_check(checks, "final_duplicate_qids", len(final_duplicates), 0, not final_duplicates)
    add_check(checks, "final_empty_qids", final_empty_qids, 0, final_empty_qids == 0)
    add_check(checks, "initial_unique_qids", len(initial_by_qid), EXPECTED["n"], len(initial_by_qid) == EXPECTED["n"])
    add_check(checks, "initial_duplicate_qids", len(initial_duplicates), 0, not initial_duplicates)
    add_check(checks, "initial_empty_qids", initial_empty_qids, 0, initial_empty_qids == 0)
    add_check(checks, "high_risk_unique_qids", len(high_by_qid), EXPECTED["high_risk"], len(high_by_qid) == EXPECTED["high_risk"])
    add_check(checks, "high_risk_duplicate_qids", len(high_duplicates), 0, not high_duplicates)
    add_check(checks, "high_risk_empty_qids", high_empty_qids, 0, high_empty_qids == 0)
    add_check(checks, "low_risk_unique_qids", len(low_by_qid), EXPECTED["low_risk"], len(low_by_qid) == EXPECTED["low_risk"])
    add_check(checks, "low_risk_duplicate_qids", len(low_duplicates), 0, not low_duplicates)
    add_check(checks, "low_risk_empty_qids", low_empty_qids, 0, low_empty_qids == 0)
    add_check(checks, "localization_unique_qids", len(loc_by_qid), EXPECTED["n"], len(loc_by_qid) == EXPECTED["n"])
    add_check(checks, "localization_duplicate_qids", len(loc_duplicates), 0, not loc_duplicates)
    add_check(checks, "localization_empty_qids", loc_empty_qids, 0, loc_empty_qids == 0)
    add_check(checks, "gold_duplicate_qids", len(gold_duplicates), 0, not gold_duplicates)
    add_check(checks, "gold_empty_qids", gold_empty_qids, 0, gold_empty_qids == 0)

    final_qids = set(final_by_qid)
    initial_qids = set(initial_by_qid)
    high_qids = set(high_by_qid)
    low_qids = set(low_by_qid)
    loc_qids = set(loc_by_qid)
    gold_qids = set(gold_by_qid)

    add_check(checks, "final_equals_initial_qid_set", len(final_qids ^ initial_qids), 0, final_qids == initial_qids)
    add_check(checks, "high_low_disjoint", len(high_qids & low_qids), 0, not (high_qids & low_qids))
    add_check(checks, "high_low_union_equals_final", len((high_qids | low_qids) ^ final_qids), 0, (high_qids | low_qids) == final_qids)
    add_check(checks, "localization_covers_final", len(final_qids - loc_qids), 0, final_qids <= loc_qids)
    add_check(checks, "gold_covers_final", len(final_qids - gold_qids), 0, final_qids <= gold_qids)

    # Frozen action and generation terminal states.
    action_by_qid = {qid: canonical_action(row) for qid, row in final_by_qid.items()}
    action_counts = Counter(action_by_qid.values())
    expected_actions = {
        "LOW_RISK_INITIAL": EXPECTED["low_risk"],
        "QWEN_REGEN": EXPECTED["qwen_success"],
        "KEEP_INITIAL_AFTER_REGEN_FAILURE": EXPECTED["high_risk_terminal_keep"],
    }
    add_check(checks, "canonical_action_counts", dict(action_counts), expected_actions, dict(action_counts) == expected_actions)

    inferred_high = {qid for qid, action in action_by_qid.items() if action != "LOW_RISK_INITIAL"}
    inferred_low = {qid for qid, action in action_by_qid.items() if action == "LOW_RISK_INITIAL"}
    add_check(checks, "prediction_high_set_matches_frozen_plan", len(inferred_high ^ high_qids), 0, inferred_high == high_qids)
    add_check(checks, "prediction_low_set_matches_frozen_plan", len(inferred_low ^ low_qids), 0, inferred_low == low_qids)

    generation_status_counts = Counter(
        str(row.get("generation_status", "")).strip().lower() for row in final_rows
    )
    failure_class_counts = Counter(
        str(row.get("failure_class", "")).strip().lower()
        for row in final_rows
        if str(row.get("failure_class", "")).strip()
    )
    add_check(
        checks,
        "generation_success_count",
        generation_status_counts.get("success", 0),
        EXPECTED["qwen_success"],
        generation_status_counts.get("success", 0) == EXPECTED["qwen_success"],
    )
    add_check(
        checks,
        "terminal_keep_count",
        generation_status_counts.get("keep_initial_terminal", 0),
        EXPECTED["high_risk_terminal_keep"],
        generation_status_counts.get("keep_initial_terminal", 0) == EXPECTED["high_risk_terminal_keep"],
    )
    unresolved_statuses = {
        "unresolved_transport",
        "transport_error",
        "pending",
        "retryable",
        "unresolved",
    }
    unresolved_count = sum(
        count for status, count in generation_status_counts.items() if status in unresolved_statuses
    )
    add_check(checks, "unresolved_transport_count", unresolved_count, EXPECTED["unresolved_transport"], unresolved_count == 0)

    empty_final_answers = [
        qid
        for qid, row in final_by_qid.items()
        if not stringify_answer(row.get("final_answer", row.get("answer", ""))).strip()
    ]
    approved_empty_terminal_qids: List[str] = []
    unexpected_empty_final_qids: List[str] = []
    for qid in empty_final_answers:
        final_row = final_by_qid[qid]
        initial_row = initial_by_qid[qid]
        initial_answer = stringify_answer(
            initial_row.get(
                "draft_answer",
                initial_row.get("initial_answer", initial_row.get("answer", "")),
            )
        ).strip()
        approved = (
            canonical_action(final_row) == "KEEP_INITIAL_AFTER_REGEN_FAILURE"
            and str(final_row.get("source", "")).strip().upper()
            == "KEEP_INITIAL_AFTER_REGEN_FAILURE"
            and str(final_row.get("generation_status", "")).strip().lower()
            == "keep_initial_terminal"
            and str(final_row.get("failure_class", "")).strip().lower()
            == "provider_safety"
            and final_row.get("initial_available") is False
            and final_row.get("gold_used_in_inference") is False
            and not initial_answer
            and initial_row.get("initial_available") is False
            and str(initial_row.get("initial_unavailable_reason", "")).strip().upper()
            == "CONTENT_FILTERED"
        )
        if approved:
            approved_empty_terminal_qids.append(qid)
        else:
            unexpected_empty_final_qids.append(qid)

    terminal_keep_qids = {
        qid
        for qid, row in final_by_qid.items()
        if canonical_action(row) == "KEEP_INITIAL_AFTER_REGEN_FAILURE"
    }
    approved_empty_terminal_set = set(approved_empty_terminal_qids)
    add_check(
        checks,
        "approved_empty_terminal_keep_count",
        len(approved_empty_terminal_qids),
        EXPECTED["approved_empty_terminal_keep"],
        len(approved_empty_terminal_qids) == EXPECTED["approved_empty_terminal_keep"],
        note=(
            "Gold-independent exception: Initial was unavailable because CONTENT_FILTERED, "
            "and regeneration reached provider_safety terminal KEEP."
        ),
    )
    add_check(
        checks,
        "terminal_keep_set_equals_approved_empty_set",
        len(terminal_keep_qids ^ approved_empty_terminal_set),
        0,
        terminal_keep_qids == approved_empty_terminal_set,
        note="All three terminal KEEP rows must be exactly the approved unavailable-Initial cases.",
    )
    add_check(
        checks,
        "unexpected_empty_final_answers",
        len(unexpected_empty_final_qids),
        0,
        not unexpected_empty_final_qids,
        note="Empty answers are otherwise forbidden.",
    )

    gold_leak_examples: List[Dict[str, Any]] = []
    for label, rows in (("final", final_rows), ("high_risk_plan", high_rows), ("low_risk_plan", low_rows)):
        for row in rows:
            hits = truthy_gold_paths(row)
            if hits:
                gold_leak_examples.append(
                    {
                        "file_group": label,
                        "qid": canonical_qid(row.get("qid", row.get("question_id"))),
                        "paths": hits,
                    }
                )
                if len(gold_leak_examples) >= 20:
                    break
        if len(gold_leak_examples) >= 20:
            break
    add_check(
        checks,
        "no_truthy_gold_fields_in_inference_artifacts",
        len(gold_leak_examples),
        0,
        not gold_leak_examples,
        note="False-valued audit flags such as gold_used_in_inference=False are allowed.",
    )

    # High-risk route audit.
    route_by_high_qid: Dict[str, str] = {}
    unresolved_route_qids: List[str] = []
    for qid, row in high_by_qid.items():
        route = canonical_route(row)
        if route is None and qid in loc_by_qid:
            route = canonical_route(loc_by_qid[qid])
        if route is None:
            unresolved_route_qids.append(qid)
        else:
            route_by_high_qid[qid] = route
    route_counts = Counter(route_by_high_qid.values())
    add_check(
        checks,
        "high_risk_route_resolution",
        len(unresolved_route_qids),
        0,
        not unresolved_route_qids,
    )
    add_check(
        checks,
        "high_risk_route_counts",
        dict(sorted(route_counts.items())),
        EXPECTED["route_counts"],
        dict(route_counts) == EXPECTED["route_counts"],
    )

    # Independent offline evaluation.
    per_qid: List[Dict[str, Any]] = []
    for qid in sorted(final_qids):
        final_row = final_by_qid[qid]
        initial_row = initial_by_qid[qid]
        gold_row = gold_by_qid[qid]
        initial_answer = stringify_answer(
            initial_row.get("draft_answer", initial_row.get("initial_answer", initial_row.get("answer", "")))
        )
        final_answer = stringify_answer(final_row.get("final_answer", final_row.get("answer", "")))
        gold_answers = extract_gold_answers(gold_row)
        initial_em, initial_f1 = best_score(initial_answer, gold_answers)
        final_em, final_f1 = best_score(final_answer, gold_answers)
        if initial_em == 0 and final_em == 1:
            migration = "W2R"
        elif initial_em == 1 and final_em == 0:
            migration = "R2W"
        elif initial_em == 1 and final_em == 1:
            migration = "R2R"
        else:
            migration = "W2W"
        per_qid.append(
            {
                "qid": qid,
                "action": action_by_qid[qid],
                "high_risk": qid in high_qids,
                "route": route_by_high_qid.get(qid),
                "initial_em": initial_em,
                "final_em": final_em,
                "initial_f1": initial_f1,
                "final_f1": final_f1,
                "migration": migration,
                "expected_transitions": expected_transitions(loc_by_qid.get(qid, {})),
            }
        )

    n = len(per_qid)
    initial_em_count = sum(row["initial_em"] for row in per_qid)
    final_em_count = sum(row["final_em"] for row in per_qid)
    initial_em = initial_em_count / n if n else 0.0
    final_em = final_em_count / n if n else 0.0
    initial_f1 = sum(row["initial_f1"] for row in per_qid) / n if n else 0.0
    final_f1 = sum(row["final_f1"] for row in per_qid) / n if n else 0.0
    migrations = Counter(row["migration"] for row in per_qid)
    w2r = migrations["W2R"]
    r2w = migrations["R2W"]
    net = w2r - r2w

    add_check(checks, "evaluation_n", n, EXPECTED["n"], n == EXPECTED["n"])
    add_check(checks, "initial_em_count", initial_em_count, EXPECTED["initial_em_count"], initial_em_count == EXPECTED["initial_em_count"])
    add_check(checks, "final_em_count", final_em_count, EXPECTED["final_em_count"], final_em_count == EXPECTED["final_em_count"])
    add_check(checks, "initial_em", initial_em, EXPECTED["initial_em"], close_enough(initial_em, EXPECTED["initial_em"]))
    add_check(checks, "final_em", final_em, EXPECTED["final_em"], close_enough(final_em, EXPECTED["final_em"]))
    add_check(checks, "final_f1", final_f1, EXPECTED["final_f1"], close_enough(final_f1, EXPECTED["final_f1"], 1e-10))
    add_check(checks, "w2r", w2r, EXPECTED["w2r"], w2r == EXPECTED["w2r"])
    add_check(checks, "r2w", r2w, EXPECTED["r2w"], r2w == EXPECTED["r2w"])
    add_check(checks, "net", net, EXPECTED["net"], net == EXPECTED["net"])
    add_check(checks, "net_identity", final_em_count - initial_em_count, net, final_em_count - initial_em_count == net)

    delta_pp = (final_em - initial_em) * 100.0
    add_check(
        checks,
        "correct_delta_percentage_points",
        delta_pp,
        EXPECTED["net"] / EXPECTED["n"] * 100.0,
        close_enough(delta_pp, EXPECTED["net"] / EXPECTED["n"] * 100.0, 1e-12),
        note="The official rounded wording is +3.95 percentage points, not +5.95pp.",
    )

    def subset_metrics(qids: set[str]) -> Dict[str, Any]:
        subset = [row for row in per_qid if row["qid"] in qids]
        subset_n = len(subset)
        subset_initial = sum(row["initial_em"] for row in subset)
        subset_final = sum(row["final_em"] for row in subset)
        subset_migrations = Counter(row["migration"] for row in subset)
        return {
            "N": subset_n,
            "Initial_EM_count": subset_initial,
            "Initial_EM": subset_initial / subset_n if subset_n else 0.0,
            "Final_EM_count": subset_final,
            "Final_EM": subset_final / subset_n if subset_n else 0.0,
            "W2R": subset_migrations["W2R"],
            "R2W": subset_migrations["R2W"],
            "Net": subset_migrations["W2R"] - subset_migrations["R2W"],
            "migrations": dict(subset_migrations),
        }

    high_metrics = subset_metrics(high_qids)
    low_metrics = subset_metrics(low_qids)
    add_check(checks, "high_risk_initial_correct", high_metrics["Initial_EM_count"], 66, high_metrics["Initial_EM_count"] == 66)
    add_check(checks, "high_risk_final_correct", high_metrics["Final_EM_count"], 119, high_metrics["Final_EM_count"] == 119)
    add_check(checks, "high_risk_w2r", high_metrics["W2R"], EXPECTED["w2r"], high_metrics["W2R"] == EXPECTED["w2r"])
    add_check(checks, "high_risk_r2w", high_metrics["R2W"], EXPECTED["r2w"], high_metrics["R2W"] == EXPECTED["r2w"])
    add_check(checks, "low_risk_initial_correct", low_metrics["Initial_EM_count"], 575, low_metrics["Initial_EM_count"] == 575)
    add_check(checks, "low_risk_final_correct", low_metrics["Final_EM_count"], 575, low_metrics["Final_EM_count"] == 575)
    add_check(checks, "low_risk_has_no_migration", low_metrics["W2R"] + low_metrics["R2W"], 0, low_metrics["W2R"] + low_metrics["R2W"] == 0)

    # Overlapping Expected-Topology transition diagnosis.
    transition_membership: Dict[str, set[str]] = {f"T{i}": set() for i in range(1, 6)}
    membership_sizes = Counter()
    for row in per_qid:
        memberships = row["expected_transitions"]
        membership_sizes[len(memberships)] += 1
        for transition in memberships:
            transition_membership[transition].add(row["qid"])

    transition_membership_counts = {
        transition: len(qids) for transition, qids in transition_membership.items()
    }
    total_memberships = sum(transition_membership_counts.values())
    multi_membership_qids = sum(count for size, count in membership_sizes.items() if size > 1)
    add_check(
        checks,
        "transition_membership_counts",
        transition_membership_counts,
        EXPECTED["transition_membership_counts"],
        transition_membership_counts == EXPECTED["transition_membership_counts"],
        hard=False,
        note="Soft check: these sets overlap and therefore do not sum to N.",
    )
    add_check(
        checks,
        "transition_memberships_are_explicitly_nonexclusive",
        {"total_memberships": total_memberships, "multi_membership_qids": multi_membership_qids},
        "total_memberships > N and at least one multi-membership qid",
        total_memberships > n and multi_membership_qids > 0,
        hard=False,
    )

    by_transition: Dict[str, Dict[str, Any]] = {
        transition: subset_metrics(qids) for transition, qids in transition_membership.items()
    }

    hard_failures = [check for check in checks if check["severity"] == "HARD" and not check["passed"]]
    soft_failures = [check for check in checks if check["severity"] == "SOFT" and not check["passed"]]
    if hard_failures:
        status = "FAIL"
    elif soft_failures:
        status = "PASS_WITH_SOFT_WARNINGS"
    else:
        status = "PASS"

    summary_flat = flatten_json(generation_summary)
    audit = {
        "schema_version": "MMQA_V3_FROZEN_BOUNDARY_AUDIT_V1_1",
        "script_version": SCRIPT_VERSION,
        "status": status,
        "safety": {
            "external_api_calls": 0,
            "answer_regeneration": False,
            "existing_experiment_files_modified": False,
            "mmqa_gold_usage": "offline_reproducibility_audit_only",
            "m3vqa_accessed": False,
        },
        "input_manifest": hashes,
        "checks": checks,
        "metrics": {
            "overall": {
                "N": n,
                "Initial_EM_count": initial_em_count,
                "Initial_EM": initial_em,
                "Initial_F1": initial_f1,
                "Final_EM_count": final_em_count,
                "Final_EM": final_em,
                "Final_F1": final_f1,
                "Delta_EM_percentage_points": delta_pp,
                "W2R": w2r,
                "R2W": r2w,
                "Net": net,
                "migrations": dict(migrations),
            },
            "high_risk_269": high_metrics,
            "low_risk_1072": low_metrics,
            "by_expected_transition_overlapping": by_transition,
        },
        "counts": {
            "canonical_actions": dict(action_counts),
            "generation_status": dict(generation_status_counts),
            "failure_class_nonempty": dict(failure_class_counts),
            "empty_final_answers_total": len(empty_final_answers),
            "approved_empty_terminal_keep": len(approved_empty_terminal_qids),
            "unexpected_empty_final_answers": len(unexpected_empty_final_qids),
            "high_risk_routes": dict(sorted(route_counts.items())),
            "transition_membership_counts": transition_membership_counts,
            "transition_membership_size_distribution": dict(sorted(membership_sizes.items())),
            "total_transition_memberships": total_memberships,
            "multi_membership_qids": multi_membership_qids,
        },
        "generation_summary_flat": summary_flat,
        "examples": {
            "duplicate_final_qids": final_duplicates[:20],
            "empty_final_answer_qids": empty_final_answers[:20],
            "approved_empty_terminal_keep_qids": sorted(approved_empty_terminal_qids),
            "unexpected_empty_final_answer_qids": sorted(unexpected_empty_final_qids)[:20],
            "unresolved_route_qids": unresolved_route_qids[:20],
            "gold_leak_examples": gold_leak_examples,
        },
        "scientific_boundary": {
            "development_1096_baseline_em": 0.379562,
            "development_baseline_is_not_direct_locked1341_comparator": True,
            "locked1341_direct_comparison": "47.80% Initial EM -> 51.75% Final EM",
            "official_delta_wording": "+3.95 percentage points",
            "mmqa_gold_state": "already opened after predictions were frozen; audit only, no retuning",
            "m3vqa_gold_state": "must remain closed",
            "empty_answer_exception_policy": (
                "Exactly three Gold-independent terminal KEEP cases are allowed to remain empty: "
                "their frozen Initial was unavailable due to CONTENT_FILTERED and Qwen regeneration "
                "also ended in provider_safety. No answer exists to copy."
            ),
        },
    }

    report_lines = [
        "MMQA V3 locked-1341 frozen-boundary audit V1.1",
        "=" * 72,
        f"STATUS: {status}",
        "",
        "Safety boundary",
        "- External API calls: 0",
        "- Answer regeneration: NO",
        "- Existing experiment files modified: NO",
        "- M3-VQA accessed: NO",
        "- MMQA Gold: offline reproducibility audit only",
        "",
        "Frozen composition",
        f"- High-risk: {len(high_qids)}",
        f"- Low-risk KEEP_INITIAL: {len(low_qids)}",
        f"- Actions: {dict(action_counts)}",
        f"- Generation status: {dict(generation_status_counts)}",
        f"- Approved empty terminal KEEP: {len(approved_empty_terminal_qids)}",
        f"- Unexpected empty final answers: {len(unexpected_empty_final_qids)}",
        f"- High-risk routes: {dict(sorted(route_counts.items()))}",
        "",
        "Independent metrics",
        f"- N: {n}",
        f"- Initial EM: {initial_em:.12f} ({initial_em_count}/{n}) = {initial_em*100:.4f}%",
        f"- Final EM: {final_em:.12f} ({final_em_count}/{n}) = {final_em*100:.4f}%",
        f"- Final F1: {final_f1:.12f} = {final_f1*100:.4f}%",
        f"- W2R: {w2r}",
        f"- R2W: {r2w}",
        f"- Net: {net:+d}",
        f"- Delta EM: {delta_pp:.6f} percentage points (report as +3.95pp)",
        "",
        "Risk subsets",
        f"- High-risk: {high_metrics}",
        f"- Low-risk: {low_metrics}",
        "",
        "Transition diagnosis",
        f"- Membership counts: {transition_membership_counts}",
        f"- Total memberships: {total_memberships}",
        f"- QIDs with >1 membership: {multi_membership_qids}",
        "- Interpretation: Expected-Topology T1-T5 groups overlap; they are not mutually exclusive routes.",
        "",
        "Scientific reporting boundary",
        "- Development-1096 baseline EM 37.9562% is not a direct comparator for locked-1341 Final EM.",
        "- Direct locked-1341 comparison is 47.80% Initial -> 51.75% Final.",
        "- MMQA predictions and the 269 budget remain immutable.",
        "- M3-VQA Gold remains closed.",
        "",
        "Failed checks",
    ]
    failed_checks = [check for check in checks if not check["passed"]]
    if failed_checks:
        for check in failed_checks:
            report_lines.append(
                f"- [{check['severity']}] {check['name']}: observed={check['observed']!r}; "
                f"expected={check['expected']!r}; note={check['note']}"
            )
    else:
        report_lines.append("- None")
    report_lines.extend(
        [
            "",
            f"FINAL STATUS = MMQA_V3_FROZEN_BOUNDARY_AUDIT_{status}",
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "mmqa_v3_locked1341_frozen_boundary_audit_summary_v1_1.json"
    report_path = output_dir / "mmqa_v3_locked1341_frozen_boundary_audit_report_v1_1.txt"
    per_qid_path = output_dir / "mmqa_v3_locked1341_frozen_boundary_audit_per_qid_v1_1.jsonl"
    safe_write_json(summary_path, audit)
    safe_write_text(report_path, "\n".join(report_lines) + "\n")
    safe_write_text(
        per_qid_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_qid),
    )

    print("INPUT HASHES FROZEN IN AUDIT MANIFEST: YES")
    print(f"Final prediction qids: {len(final_qids)}")
    print(f"High-risk / Low-risk: {len(high_qids)} / {len(low_qids)}")
    print(f"Actions: {dict(action_counts)}")
    print(f"Routes: {dict(sorted(route_counts.items()))}")
    print(
        "Empty finals: "
        f"approved terminal KEEP={len(approved_empty_terminal_qids)} | "
        f"unexpected={len(unexpected_empty_final_qids)}"
    )
    print("-" * 116)
    print(f"Initial EM: {initial_em*100:.4f}% ({initial_em_count}/{n})")
    print(f"Final EM:   {final_em*100:.4f}% ({final_em_count}/{n})")
    print(f"Final F1:   {final_f1*100:.4f}%")
    print(f"W2R={w2r} | R2W={r2w} | Net={net:+d} | Delta={delta_pp:.4f}pp")
    print("-" * 116)
    print(f"Hard check failures: {len(hard_failures)}")
    print(f"Soft check warnings: {len(soft_failures)}")
    print(f"Audit summary: {summary_path}")
    print(f"Audit report:  {report_path}")
    print(f"Per-qid audit: {per_qid_path}")
    print(f"STATUS = MMQA_V3_FROZEN_BOUNDARY_AUDIT_{status}")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
