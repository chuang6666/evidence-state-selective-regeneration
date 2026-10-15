#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Freeze Clean-B3 on full Dev1096 and select locked-1341 Top-269 without Gold.

Modes:
  --freeze-protocol   Non-training preflight and immutable selection policy.
  --run-all           Full-Dev fit, locked non-Gold features/risk/selection only.

The program never opens raw/official Gold, never evaluates EM/F1, never calls
an API, and never performs answer regeneration.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from run_mmqa_clean_risk_v2_dev1096_feature_generation_v1 import (
    EXPECTED_REASONS,
    EXPECTED_DECISIONS,
    build_explicit_candidate,
    select_variant,
)


ROOT = Path(__file__).resolve().parent
DEV_N = 1096
LOCKED_N = 1341
BUDGET = 269
FORCED_EXPECTED = 6
NATURAL_UNAVAILABLE_EXPECTED = 5
NONFORCED_BUDGET = BUDGET - FORCED_EXPECTED
VARIANT = "CLEAN_B3_GENERAL_T1_T5"
MODEL_SEED = 260817
EXPECTED_CONTRACT_SHA = "9BACED6295AE3F31E023DD9E91395ABC567861AB4DA040A5E0D337B344ED14B1"
EXPECTED_DEV_MATRIX_SHA = "FA579EE4028FFB2F07EDA664B3547D688733E0022677048783223E763330DCF7"
EXPECTED_OOF_SELECTION_SHA = "B2937560A5E90CB12F7FF95C56B710243FD50A1FAEC3F0CA800FC7EAAAA0BB72"

CONTRACT_FILE = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_feature_contract_v1"
    / "clean_risk_v2_feature_contract_v1.json"
)
DEV_FEATURE_DIR = ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_feature_generation_v1"
DEV_MATRIX_FILE = DEV_FEATURE_DIR / "dev1096_clean_risk_feature_matrix_v1.jsonl"
DEV_SCHEMA_PROVENANCE_FILE = DEV_FEATURE_DIR / "clean_risk_feature_schema_and_provenance_v1.json"
DEV_FEATURE_AUDIT_FILE = DEV_FEATURE_DIR / "feature_schema_parity_audit_v1.json"
OOF_DIR = ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_answer_risk_oof_v1"
DEV_LABEL_SPLIT_FILE = OOF_DIR / "dev1096_saved_answer_error_labels_and_shared_folds_v1.jsonl"
OOF_PROTOCOL_FILE = OOF_DIR / "clean_v2_oof_evaluation_protocol_v1.json"
OOF_SELECTION_FILE = OOF_DIR / "clean_v2_oof_variant_selection_v1.json"
OOF_SUMMARY_FILE = OOF_DIR / "clean_v2_oof_summary_v1.json"
OOF_RUNNER = ROOT / "run_mmqa_clean_risk_v2_dev1096_answer_risk_oof_v1.py"
FEATURE_RUNNER = ROOT / "run_mmqa_clean_risk_v2_dev1096_feature_generation_v1.py"

LOCKED_PRE_DIR = ROOT / "outputs" / "mmqa_v3_heldout1341_pre_natural_v1"
LOCKED_BASE_FILE = LOCKED_PRE_DIR / "mmqa_v3_heldout1341_initial_base_sanitized_v1.jsonl"
LOCKED_TOPOLOGY_FILE = LOCKED_PRE_DIR / "mmqa_v3_heldout1341_topology_predictions_v1.jsonl"
LOCKED_INITIAL_POLICY_FILE = LOCKED_PRE_DIR / "mmqa_v3_initial_unavailable_policy_frozen_v1.json"
LOCKED_PRE_SUMMARY_FILE = LOCKED_PRE_DIR / "mmqa_v3_heldout1341_pre_natural_summary_v1.json"

LOCKED_NATURAL_DIR = ROOT / "outputs" / "mmqa_v3_heldout1341_natural_state_exact_sync_v1"
LOCKED_NATURAL_SUCCESS_FILE = LOCKED_NATURAL_DIR / "natural_state_exact_sync_predictions_v1.jsonl"
LOCKED_NATURAL_UNAVAILABLE_FILE = LOCKED_NATURAL_DIR / "natural_state_terminal_unavailable_v1.jsonl"
LOCKED_NATURAL_SUMMARY_FILE = LOCKED_NATURAL_DIR / "natural_state_exact_sync_summary_v1.json"

LOCKED_VERIFIER_DIR = ROOT / "outputs" / "mmqa_v3_heldout1341_frozen_verifier_inputs_v1"
LOCKED_STATE_MANIFEST_FILE = LOCKED_VERIFIER_DIR / "mmqa_v3_heldout1341_verifier_qid_manifest_v1.jsonl"
LOCKED_VERIFIER_SUMMARY_FILE = LOCKED_VERIFIER_DIR / "mmqa_v3_heldout1341_verifier_inputs_summary_v1.json"

LOCKED_LOC_DIR = ROOT / "outputs" / "mmqa_v3_heldout1341_frozen_verifier_localization_risk_v1"
LOCKED_LOCALIZATION_FILE = LOCKED_LOC_DIR / "heldout1341_localization_merged_all_qids_v1.jsonl"
OLD_TOP269_FILE = LOCKED_LOC_DIR / "heldout1341_frozen_top20_high_risk_269_v1.jsonl"
LOCKED_ROUTING_SUMMARY_FILE = LOCKED_LOC_DIR / "heldout1341_frozen_routing_summary_v1.json"

HISTORY_MAIN_DIR = ROOT / "outputs" / "mmqa_v3_heldout1341_final_plain_regen_qwen37plus_v1"
HISTORY_MAIN_OUTCOMES = HISTORY_MAIN_DIR / "high_risk269_effective_predictions_frozen_v1.jsonl"
HISTORY_MAIN_PREFLIGHT = HISTORY_MAIN_DIR / "preflight_v1.json"
HISTORY_MAIN_SUMMARY = HISTORY_MAIN_DIR / "mmqa_v3_locked1341_final_generation_summary_v1.json"

HISTORY_RANDOM_DIR = ROOT / "outputs" / "mmqa_v3_random20_matched_qwen37plus_v1"
HISTORY_RANDOM_OUTCOMES = HISTORY_RANDOM_DIR / "new_union707_effective_predictions_frozen_v1.jsonl"
HISTORY_RANDOM_PREFLIGHT = HISTORY_RANDOM_DIR / "runner_preflight_v1.json"
HISTORY_RANDOM_SUMMARY = HISTORY_RANDOM_DIR / "mmqa_v3_random20_generation_summary_v1.json"

HISTORY_B0_DIR = ROOT / "outputs" / "mmqa_v3_b0_only15_exact_protocol_qwen37plus_v1"
HISTORY_B0_OUTCOMES = HISTORY_B0_DIR / "b0_only15_effective_predictions_frozen_before_eval_v1.jsonl"
HISTORY_B0_PREFLIGHT = HISTORY_B0_DIR / "runner_preflight_v1.json"
HISTORY_B0_SUMMARY = HISTORY_B0_DIR / "mmqa_v3_b0_only15_generation_summary_v1.json"

HISTORY_VC_DIR = ROOT / "outputs" / "mmqa_vc20_matched_regeneration_union_qwen37plus_v1"
HISTORY_VC_OUTCOMES = HISTORY_VC_DIR / "new_jobs102_effective_predictions_frozen_v1.jsonl"
HISTORY_VC_PREFLIGHT = HISTORY_VC_DIR / "runner_preflight_v1.json"
HISTORY_VC_SUMMARY = HISTORY_VC_DIR / "vc20_matched_regeneration_generation_summary_v1.json"

OUT_DIR = ROOT / "outputs" / "mmqa_clean_risk_v2_final_full_dev_locked_selection_v1"
PROTOCOL_FILE = OUT_DIR / "final_full_dev_locked_selection_protocol_v1.json"
PREFLIGHT_FILE = OUT_DIR / "final_full_dev_locked_selection_preflight_v1.json"
PROTOCOL_HASH_FILE = OUT_DIR / "PROTOCOL_SHA256SUMS.json"

PIPELINE_FILE = OUT_DIR / "clean_b3_final_full_dev_pipeline_v1.joblib"
VECTORIZER_FILE = OUT_DIR / "clean_b3_final_full_dev_vectorizer_v1.joblib"
MODEL_FILE = OUT_DIR / "clean_b3_final_full_dev_logistic_regression_v1.joblib"
BUNDLE_FILE = OUT_DIR / "clean_b3_final_full_dev_bundle_v1.joblib"
MODEL_SCHEMA_FILE = OUT_DIR / "clean_b3_final_model_schema_v1.json"
MODEL_COEFFICIENT_FILE = OUT_DIR / "clean_b3_final_model_coefficients_v1.json"

LOCKED_FEATURE_FILE = OUT_DIR / "locked1341_clean_b3_features_v1.jsonl"
LOCKED_UNAVAILABLE_FILE = OUT_DIR / "locked1341_clean_b3_unavailable_cases_v1.jsonl"
LOCKED_FEATURE_AUDIT_FILE = OUT_DIR / "locked1341_clean_b3_schema_parity_audit_v1.json"
LOCKED_FEATURE_PROVENANCE_FILE = OUT_DIR / "locked1341_clean_b3_feature_provenance_v1.json"
LOCKED_RISK_FILE = OUT_DIR / "locked1341_clean_b3_risk_scores_v1.jsonl"
NEW_TOP269_FILE = OUT_DIR / "locked1341_clean_b3_top269_selection_v1.jsonl"
NEW_LOW1072_FILE = OUT_DIR / "locked1341_clean_b3_low1072_v1.jsonl"

OVERLAP_ALL_FILE = OUT_DIR / "clean_b3_vs_old_v1_top269_membership_v1.jsonl"
INTERSECTION_FILE = OUT_DIR / "clean_b3_vs_old_v1_intersection_qids_v1.jsonl"
NEW_ONLY_FILE = OUT_DIR / "clean_b3_vs_old_v1_new_only_qids_v1.jsonl"
OLD_ONLY_FILE = OUT_DIR / "clean_b3_vs_old_v1_old_only_qids_v1.jsonl"
OVERLAP_SUMMARY_FILE = OUT_DIR / "clean_b3_vs_old_v1_overlap_summary_v1.json"

HISTORY_REGISTRY_FILE = OUT_DIR / "historical_exact_plain_regeneration_registry_v1.jsonl"
NEW_ONLY_REUSE_FILE = OUT_DIR / "new_only_historical_regeneration_reuse_audit_v1.jsonl"
REUSE_SUMMARY_FILE = OUT_DIR / "new_only_historical_regeneration_reuse_summary_v1.json"
SUMMARY_FILE = OUT_DIR / "final_full_dev_locked_selection_summary_v1.json"
HASH_FILE = OUT_DIR / "SHA256SUMS.json"

HISTORY_SOURCES = [
    {
        "name": "old_v1_main_top269",
        "outcomes": HISTORY_MAIN_OUTCOMES,
        "preflight": HISTORY_MAIN_PREFLIGHT,
        "summary": HISTORY_MAIN_SUMMARY,
        "priority": 1,
    },
    {
        "name": "random20_new_union707",
        "outcomes": HISTORY_RANDOM_OUTCOMES,
        "preflight": HISTORY_RANDOM_PREFLIGHT,
        "summary": HISTORY_RANDOM_SUMMARY,
        "priority": 2,
    },
    {
        "name": "b0_only15_exact_protocol",
        "outcomes": HISTORY_B0_OUTCOMES,
        "preflight": HISTORY_B0_PREFLIGHT,
        "summary": HISTORY_B0_SUMMARY,
        "priority": 3,
    },
    {
        "name": "vc20_new_jobs102",
        "outcomes": HISTORY_VC_OUTCOMES,
        "preflight": HISTORY_VC_PREFLIGHT,
        "summary": HISTORY_VC_SUMMARY,
        "priority": 4,
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception as exc:
                raise RuntimeError(f"Malformed JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Non-object JSONL row {path}:{line_no}")
            rows.append(row)
    return rows


def unique_by_qid(rows: Iterable[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid", "") or "").strip()
        if not qid:
            raise RuntimeError(f"Blank qid in {source}")
        if qid in out:
            raise RuntimeError(f"Duplicate qid in {source}: {qid}")
        out[qid] = row
    return out


def json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def frozen_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != data:
            raise RuntimeError(f"Frozen output differs; refusing overwrite: {path}")
        return
    path.write_bytes(data)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def exact_evidence_ids(row: dict[str, Any]) -> list[str]:
    evidence = row.get("retrieved_evidence")
    if not isinstance(evidence, list):
        return []
    return [
        text(item.get("evidence_id")) if isinstance(item, dict) else ""
        for item in evidence
    ]


def make_model() -> Pipeline:
    return Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", LogisticRegression(
            solver="liblinear", C=1.0, class_weight="balanced",
            max_iter=2000, random_state=MODEL_SEED,
        )),
    ])


def validate_history_contract(source: dict[str, Any]) -> dict[str, Any]:
    preflight = load_json(source["preflight"])
    summary = load_json(source["summary"])
    name = source["name"]
    checks: dict[str, bool] = {}
    if name == "old_v1_main_top269":
        checks = {
            "model": preflight.get("solver") == "qwen3.7-plus",
            "exact_top6": preflight.get("exact_top6") is True,
            "thinking": preflight.get("thinking") is True,
            "thinking_budget": preflight.get("thinking_budget") == 2048,
            "temperature": preflight.get("temperature") == 0.1,
            "initial_hidden": preflight.get("initial_answer_in_prompt") is False,
            "diagnosis_hidden": preflight.get("transition_diagnosis_in_prompt") is False,
            "gold_not_opened": preflight.get("gold_opened") is False,
            "complete": summary.get("high_risk_completed") == 269,
        }
    elif name == "random20_new_union707":
        checks = {
            "model": preflight.get("model") == "qwen3.7-plus",
            "exact_top6": preflight.get("exact_original_order_top6") is True,
            "thinking": preflight.get("thinking") is True,
            "thinking_budget": preflight.get("thinking_budget") == 2048,
            "temperature": preflight.get("temperature") == 0.1,
            "initial_hidden": preflight.get("initial_answer_in_solver_prompt") is False,
            "diagnosis_hidden": preflight.get("transition_diagnosis_in_solver_prompt") is False,
            "gold_not_opened": preflight.get("mmqa_gold_eval_rows_loaded") is False,
            "complete": summary.get("new_union_completed") == 707,
        }
    elif name == "b0_only15_exact_protocol":
        checks = {
            "model": preflight.get("model") == "qwen3.7-plus",
            "exact_top6": preflight.get("exact_original_order_top6") is True,
            "thinking": preflight.get("thinking") is True,
            "thinking_budget": preflight.get("thinking_budget") == 2048,
            "temperature": preflight.get("temperature") == 0.1,
            "initial_hidden": preflight.get("initial_answer_in_solver_prompt") is False,
            "diagnosis_hidden": preflight.get("transition_diagnosis_in_solver_prompt") is False,
            "gold_not_opened": preflight.get("mmqa_gold_eval_rows_loaded") is False,
            "complete": summary.get("completed") == 15,
        }
    elif name == "vc20_new_jobs102":
        checks = {
            "model": preflight.get("model") == "qwen3.7-plus",
            "thinking_budget": preflight.get("thinking_budget") == 2048,
            "temperature": preflight.get("temperature") == 0.1,
            "gold_not_opened": preflight.get("mmqa_gold_eval_rows_loaded") is False,
            "complete": summary.get("new_completed") == 102,
        }
        # The exact Top-6/plain-regeneration contract is frozen by its paired
        # preflight source and summarized in the runner input hashes.
        checks["matched_plain_regen_contract"] = (
            preflight.get("status") == "MMQA_VC20_MATCHED_REGEN_RUNNER_PREFLIGHT_V1_PASS"
        )
    else:
        raise RuntimeError(name)
    return {
        "name": name,
        "checks": checks,
        "pass": all(checks.values()),
        "preflight": file_record(source["preflight"]),
        "summary": file_record(source["summary"]),
        "outcomes": file_record(source["outcomes"]),
    }


def protocol_preflight() -> dict[str, Any]:
    contract_sha = sha256_file(CONTRACT_FILE)
    dev_matrix_sha = sha256_file(DEV_MATRIX_FILE)
    selection_sha = sha256_file(OOF_SELECTION_FILE)
    if contract_sha != EXPECTED_CONTRACT_SHA:
        raise RuntimeError(f"Feature contract SHA drift: {contract_sha}")
    if dev_matrix_sha != EXPECTED_DEV_MATRIX_SHA:
        raise RuntimeError(f"Dev feature matrix SHA drift: {dev_matrix_sha}")
    if selection_sha != EXPECTED_OOF_SELECTION_SHA:
        raise RuntimeError(f"OOF selection SHA drift: {selection_sha}")

    contract = load_json(CONTRACT_FILE)
    oof_selection = load_json(OOF_SELECTION_FILE)
    oof_summary = load_json(OOF_SUMMARY_FILE)
    if oof_selection.get("selected_variant") != VARIANT:
        raise RuntimeError("Frozen OOF-selected variant is not Clean-B3")
    if oof_summary.get("final_full_dev_model_fit") is not False:
        raise RuntimeError("Unexpected prior final full-Dev fit flag")
    b3_schema = list(contract["variants"][VARIANT]["features"])
    if len(b3_schema) != 65 or len(set(b3_schema)) != 65:
        raise RuntimeError("Clean-B3 schema is not the frozen 65-key contract")

    dev_matrix = unique_by_qid(load_jsonl(DEV_MATRIX_FILE), "Dev feature matrix")
    labels = unique_by_qid(load_jsonl(DEV_LABEL_SPLIT_FILE), "saved Dev labels")
    if len(dev_matrix) != DEV_N or set(dev_matrix) != set(labels):
        raise RuntimeError("Dev matrix/label QID mismatch")
    label_values = [labels[qid].get("initial_wrong_label") for qid in sorted(labels)]
    if Counter(label_values) != Counter({1: 680, 0: 416}):
        raise RuntimeError("Saved Dev label balance drift")

    base = unique_by_qid(load_jsonl(LOCKED_BASE_FILE), "locked base")
    topology = unique_by_qid(load_jsonl(LOCKED_TOPOLOGY_FILE), "locked topology")
    localization = unique_by_qid(load_jsonl(LOCKED_LOCALIZATION_FILE), "locked localization")
    natural_success = unique_by_qid(load_jsonl(LOCKED_NATURAL_SUCCESS_FILE), "locked natural success")
    natural_unavailable = unique_by_qid(load_jsonl(LOCKED_NATURAL_UNAVAILABLE_FILE), "locked natural unavailable")
    state_manifest = unique_by_qid(load_jsonl(LOCKED_STATE_MANIFEST_FILE), "locked verifier manifest")
    old_top269 = unique_by_qid(load_jsonl(OLD_TOP269_FILE), "old V1 Top-269")

    qids = set(base)
    if len(qids) != LOCKED_N or set(topology) != qids or set(localization) != qids:
        raise RuntimeError("Locked 1341 base/topology/localization coverage mismatch")
    if len(natural_success) != 1336 or len(natural_unavailable) != NATURAL_UNAVAILABLE_EXPECTED:
        raise RuntimeError("Locked Natural-State success/unavailable count drift")
    if set(natural_success) & set(natural_unavailable) or set(natural_success) | set(natural_unavailable) != qids:
        raise RuntimeError("Natural-State resolution partition mismatch")
    if set(state_manifest) != set(natural_success):
        raise RuntimeError("Verifier manifest must cover exactly Natural-State success QIDs")
    if len(old_top269) != BUDGET:
        raise RuntimeError("Old V1 Top-269 coverage drift")

    exact_top6_failures = []
    initial_unavailable = []
    for qid in sorted(qids):
        evidence_ids = exact_evidence_ids(base[qid])
        if len(evidence_ids) != 6 or len(set(evidence_ids)) != 6 or any(not x for x in evidence_ids):
            exact_top6_failures.append(qid)
        if base[qid].get("initial_available") is False:
            initial_unavailable.append(qid)
    if exact_top6_failures or len(initial_unavailable) != 6:
        raise RuntimeError(
            f"Locked input policy drift: top6_fail={exact_top6_failures[:10]} initial_unavailable={len(initial_unavailable)}"
        )

    natural_unavailable_qids = set(natural_unavailable)
    forced_qids = set(initial_unavailable) | natural_unavailable_qids
    if len(forced_qids) != FORCED_EXPECTED or not natural_unavailable_qids <= set(initial_unavailable):
        raise RuntimeError("Frozen unavailable policy QID relationship drift")
    old_forced_rows = [
        row for row in load_jsonl(OLD_TOP269_FILE)
        if row.get("forced_into_high_risk_budget") is True
    ]
    old_forced_order = [text(row.get("qid")) for row in old_forced_rows]
    if len(old_forced_order) != FORCED_EXPECTED or set(old_forced_order) != forced_qids:
        raise RuntimeError("Old frozen forced-budget order/coverage drift")

    scorable_decisions = Counter(
        text(localization[qid].get("decision")) for qid in natural_success
    )
    unavailable_decisions = Counter(
        text(localization[qid].get("decision")) for qid in natural_unavailable
    )
    if not set(scorable_decisions) <= EXPECTED_DECISIONS:
        raise RuntimeError(f"Locked scorable category outside Clean contract: {scorable_decisions}")
    if set(unavailable_decisions) != {"STATE_UNAVAILABLE"}:
        raise RuntimeError("Unavailable localization category drift")

    history_contracts = [validate_history_contract(source) for source in HISTORY_SOURCES]
    if not all(item["pass"] for item in history_contracts):
        raise RuntimeError(f"Historical exact-regeneration contract failed: {history_contracts}")

    summaries = {
        "locked_pre": load_json(LOCKED_PRE_SUMMARY_FILE),
        "locked_natural": load_json(LOCKED_NATURAL_SUMMARY_FILE),
        "locked_verifier": load_json(LOCKED_VERIFIER_SUMMARY_FILE),
        "locked_routing": load_json(LOCKED_ROUTING_SUMMARY_FILE),
        "locked_initial_policy": load_json(LOCKED_INITIAL_POLICY_FILE),
    }
    guardrail_failures = []
    if summaries["locked_pre"].get("official_mmqa_gold_opened") is not False:
        guardrail_failures.append("locked_pre_gold")
    if summaries["locked_natural"].get("guardrails", {}).get("official_gold_opened") is not False:
        guardrail_failures.append("locked_natural_gold")
    if summaries["locked_verifier"].get("guardrails", {}).get("official_mmqa_gold_opened") is not False:
        guardrail_failures.append("locked_verifier_gold")
    if summaries["locked_routing"].get("guardrails", {}).get("official_mmqa_gold_opened") is not False:
        guardrail_failures.append("locked_routing_gold")
    policy = summaries["locked_initial_policy"].get("future_risk_selection_policy", {})
    initial_forced_text = text(policy.get("initial_unavailable_qids"))
    natural_forced_text = text(
        summaries["locked_natural"].get("guardrails", {}).get(
            "terminal_state_unavailable_future_policy"
        )
    )
    if (
        policy.get("total_high_risk_budget") != BUDGET
        or "FORCE_INTO_HIGH_RISK_BUDGET_FIRST" not in initial_forced_text
        or "force into fixed 269 high-risk budget" not in natural_forced_text
    ):
        guardrail_failures.append("budget_policy")
    if guardrail_failures:
        raise RuntimeError(f"Locked non-Gold/budget guardrail failure: {guardrail_failures}")

    return {
        "contract_sha256": contract_sha,
        "dev_matrix_sha256": dev_matrix_sha,
        "oof_selection_sha256": selection_sha,
        "selected_variant": VARIANT,
        "b3_raw_schema": b3_schema,
        "dev_qids": len(dev_matrix),
        "dev_wrong": 680,
        "dev_correct": 416,
        "locked_qids": len(qids),
        "locked_exact_top6_failures": len(exact_top6_failures),
        "locked_natural_success_scorable": len(natural_success),
        "locked_natural_unavailable_unscored": len(natural_unavailable),
        "locked_initial_unavailable": len(initial_unavailable),
        "forced_budget_qids": len(forced_qids),
        "forced_order": old_forced_order,
        "old_top269": len(old_top269),
        "scorable_decision_counts": dict(sorted(scorable_decisions.items())),
        "unavailable_decision_counts": dict(sorted(unavailable_decisions.items())),
        "history_contracts": history_contracts,
        "input_files": {
            path.name: file_record(path) for path in [
                CONTRACT_FILE, DEV_MATRIX_FILE, DEV_LABEL_SPLIT_FILE,
                OOF_PROTOCOL_FILE, OOF_SELECTION_FILE, OOF_SUMMARY_FILE,
                LOCKED_BASE_FILE, LOCKED_TOPOLOGY_FILE,
                LOCKED_NATURAL_SUCCESS_FILE, LOCKED_NATURAL_UNAVAILABLE_FILE,
                LOCKED_STATE_MANIFEST_FILE, LOCKED_LOCALIZATION_FILE,
                OLD_TOP269_FILE, LOCKED_INITIAL_POLICY_FILE,
            ]
        },
    }


def freeze_protocol() -> None:
    validation = protocol_preflight()
    protocol = {
        "status": "MMQA_CLEAN_RISK_V2_FINAL_FULL_DEV_LOCKED_SELECTION_PROTOCOL_V1_FROZEN",
        "selected_variant": VARIANT,
        "selection_source": {
            "path": str(OOF_SELECTION_FILE),
            "sha256": validation["oof_selection_sha256"],
            "rule": "pre-frozen OOF primary metric and tie-break; no reselection",
        },
        "full_dev_training": {
            "n": DEV_N,
            "feature_matrix_sha256": validation["dev_matrix_sha256"],
            "saved_labels_only": True,
            "raw_gold_opened": False,
            "model": "DictVectorizer(sparse=True) + LogisticRegression",
            "logistic_regression": {
                "solver": "liblinear", "penalty": "l2", "C": 1.0,
                "class_weight": "balanced", "max_iter": 2000,
                "random_state": MODEL_SEED, "tol": 0.0001,
            },
            "hyperparameter_search": False,
        },
        "locked_feature_generation": {
            "total_qids": LOCKED_N,
            "scorable_qids": 1336,
            "natural_state_unavailable_unscored": NATURAL_UNAVAILABLE_EXPECTED,
            "feature_contract_sha256": validation["contract_sha256"],
            "raw_schema": validation["b3_raw_schema"],
            "raw_schema_count": 65,
            "generic_recursive_flatten": False,
            "per_transition_verifier_numeric_features": False,
            "state_unavailable_category_entering_model": False,
        },
        "locked_selection": {
            "budget": BUDGET,
            "forced_qids": FORCED_EXPECTED,
            "forced_order": validation["forced_order"],
            "nonforced_slots": NONFORCED_BUDGET,
            "nonforced_ranking": "clean_b3_risk_desc_then_qid_ascending",
            "budget_increase": 0,
            "risk_threshold_retuning": False,
        },
        "historical_regeneration_reuse_audit": {
            "scope": "new-only versus old V1 Top-269",
            "reuse_allowed_only_if": [
                "frozen historical outcome exists",
                "same Qwen3.7-Plus plain-regeneration operator",
                "same exact original-order Top-6 source",
                "same frozen route/localization source",
                "initial answer hidden",
                "transition diagnosis hidden",
                "no Gold used in inference",
            ],
            "history_contracts": validation["history_contracts"],
            "regeneration_executed_in_this_stage": False,
        },
        "prohibitions": {
            "api_calls": 0,
            "locked_gold_opened": False,
            "em_f1_computed": False,
            "regeneration": False,
            "model_reselection": False,
            "hyperparameter_search": False,
            "feature_change": False,
            "v1_modified": False,
        },
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
    }
    preflight = {
        "status": "MMQA_CLEAN_RISK_V2_FINAL_FULL_DEV_LOCKED_SELECTION_PREFLIGHT_PASS",
        "training_started": False,
        "locked_risk_scoring_started": False,
        "validation": validation,
        "raw_gold_opened": False,
        "locked_evaluation_run": False,
        "regeneration_run": False,
        "external_api_calls": 0,
        "v1_modified": False,
        "gate": "PASS: full-Dev Clean-B3 fit and locked non-Gold selection may run",
    }
    frozen_write(PROTOCOL_FILE, json_bytes(protocol))
    frozen_write(PREFLIGHT_FILE, json_bytes(preflight))
    manifest = {
        "status": "MMQA_CLEAN_RISK_V2_FINAL_FULL_DEV_LOCKED_SELECTION_PROTOCOL_SHA_FROZEN",
        "generated_files": {
            path.name: file_record(path) for path in [PROTOCOL_FILE, PREFLIGHT_FILE]
        },
        "read_only_sources": validation["input_files"],
        "runner": file_record(Path(__file__).resolve()),
        "training_started": False,
        "locked_gold_opened": False,
        "regeneration_run": False,
        "external_api_calls": 0,
        "v1_modified": False,
    }
    frozen_write(PROTOCOL_HASH_FILE, json_bytes(manifest))

    print("=" * 130)
    print("MMQA CLEAN RISK V2 — FINAL FULL-DEV + LOCKED NON-GOLD SELECTION PROTOCOL FREEZE V1")
    print("=" * 130)
    print("Preflight: PASS")
    print("Variant:", VARIANT, "| schema keys=65")
    print("Dev:", validation["dev_qids"], "saved labels only")
    print("Locked:", validation["locked_qids"], "scorable=", validation["locked_natural_success_scorable"], "unavailable=", validation["locked_natural_unavailable_unscored"])
    print("Budget:", BUDGET, "forced=", validation["forced_budget_qids"], "nonforced=", NONFORCED_BUDGET)
    print("Historical exact-protocol sources:", {x["name"]: x["pass"] for x in validation["history_contracts"]})
    print("Training/risk scoring started: NO / NO")
    print("API/Gold/eval/regeneration/V1-modified: 0 / 0 / 0 / 0 / 0")
    print("Protocol:", PROTOCOL_FILE)
    print("STATUS =", protocol["status"])


def verify_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    if not all(path.is_file() for path in [PROTOCOL_FILE, PREFLIGHT_FILE, PROTOCOL_HASH_FILE]):
        raise RuntimeError("Run --freeze-protocol first")
    protocol = load_json(PROTOCOL_FILE)
    preflight = load_json(PREFLIGHT_FILE)
    manifest = load_json(PROTOCOL_HASH_FILE)
    if protocol.get("status") != "MMQA_CLEAN_RISK_V2_FINAL_FULL_DEV_LOCKED_SELECTION_PROTOCOL_V1_FROZEN":
        raise RuntimeError("Protocol is not frozen")
    if preflight.get("status") != "MMQA_CLEAN_RISK_V2_FINAL_FULL_DEV_LOCKED_SELECTION_PREFLIGHT_PASS":
        raise RuntimeError("Preflight is not PASS")
    for name, record in manifest["generated_files"].items():
        if sha256_file(OUT_DIR / name) != record["sha256"]:
            raise RuntimeError(f"Frozen protocol changed: {name}")
    for path, expected in [
        (CONTRACT_FILE, protocol["locked_feature_generation"]["feature_contract_sha256"]),
        (DEV_MATRIX_FILE, protocol["full_dev_training"]["feature_matrix_sha256"]),
        (OOF_SELECTION_FILE, protocol["selection_source"]["sha256"]),
    ]:
        if sha256_file(path) != expected:
            raise RuntimeError(f"Frozen input changed after protocol freeze: {path}")
    return protocol, preflight


def dump_new_joblib(obj: Any, path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite frozen binary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)


def coefficient_summary(model: Pipeline) -> dict[str, Any]:
    vec: DictVectorizer = model.named_steps["vec"]
    clf: LogisticRegression = model.named_steps["clf"]
    names = list(vec.get_feature_names_out())
    values = clf.coef_[0]
    pairs = [(name, float(value)) for name, value in zip(names, values)]
    return {
        "feature_dimension": len(names),
        "intercept": float(clf.intercept_[0]),
        "n_iter": [int(x) for x in clf.n_iter_],
        "coefficient_nonzero_count": int(np.count_nonzero(values)),
        "coefficient_l1_norm": float(np.abs(values).sum()),
        "coefficient_l2_norm": float(np.linalg.norm(values)),
        "coefficient_max_abs": float(np.abs(values).max()),
        "top_positive": [
            {"feature": name, "coefficient": value}
            for name, value in sorted(pairs, key=lambda x: (-x[1], x[0]))[:50]
        ],
        "top_negative": [
            {"feature": name, "coefficient": value}
            for name, value in sorted(pairs, key=lambda x: (x[1], x[0]))[:50]
        ],
    }


def build_history_registry() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_contracts = {item["name"]: item for item in [validate_history_contract(x) for x in HISTORY_SOURCES]}
    for source in HISTORY_SOURCES:
        if not source_contracts[source["name"]]["pass"]:
            raise RuntimeError(f"Historical source contract not equivalent: {source['name']}")
        rows = unique_by_qid(load_jsonl(source["outcomes"]), source["name"])
        for qid, row in rows.items():
            if row.get("gold_used_in_inference") is not False:
                raise RuntimeError(f"Historical outcome not non-Gold: {source['name']} {qid}")
            status = text(row.get("generation_status") or row.get("status"))
            by_qid[qid].append({
                "source": source["name"],
                "source_priority": source["priority"],
                "generation_status": status,
                "failure_class": text(row.get("failure_class")),
                "model_final_answer": text(row.get("model_final_answer")),
                "effective_final_answer": text(row.get("effective_final_answer")),
                "initial_reused_downstream": bool(row.get("initial_reused_downstream", False)),
                "gold_used_in_inference": False,
            })

    registry_rows: list[dict[str, Any]] = []
    registry: dict[str, dict[str, Any]] = {}
    duplicate_qids = 0
    conflicting_duplicate_qids = 0
    for qid in sorted(by_qid):
        outcomes = sorted(by_qid[qid], key=lambda x: x["source_priority"])
        if len(outcomes) > 1:
            duplicate_qids += 1
        signatures = {
            (
                x["generation_status"], x["failure_class"],
                x["model_final_answer"], x["effective_final_answer"],
                x["initial_reused_downstream"],
            )
            for x in outcomes
        }
        conflict = len(signatures) > 1
        if conflict:
            conflicting_duplicate_qids += 1
        chosen = outcomes[0]
        row = {
            "qid": qid,
            "historical_sources": [x["source"] for x in outcomes],
            "duplicate_source_count": len(outcomes),
            "duplicate_outcome_conflict": conflict,
            "chosen_source": chosen["source"] if not conflict else None,
            "generation_status": chosen["generation_status"] if not conflict else None,
            "failure_class": chosen["failure_class"] if not conflict else None,
            "model_final_answer": chosen["model_final_answer"] if not conflict else None,
            "effective_final_answer": chosen["effective_final_answer"] if not conflict else None,
            "initial_reused_downstream": chosen["initial_reused_downstream"] if not conflict else None,
            "same_exact_plain_regeneration_protocol": True,
            "legal_reuse": not conflict,
            "gold_used_in_inference": False,
        }
        registry_rows.append(row)
        registry[qid] = row
    summary = {
        "source_counts": {
            source["name"]: len(unique_by_qid(load_jsonl(source["outcomes"]), source["name"]))
            for source in HISTORY_SOURCES
        },
        "unique_registry_qids": len(registry),
        "duplicate_qids": duplicate_qids,
        "conflicting_duplicate_qids": conflicting_duplicate_qids,
        "source_contracts": source_contracts,
    }
    return registry_rows, registry, summary


def run_all() -> None:
    protocol, preflight = verify_protocol()
    contract = load_json(CONTRACT_FILE)
    b3_schema = list(contract["variants"][VARIANT]["features"])
    b3_allowed = set(b3_schema)

    dev_matrix = unique_by_qid(load_jsonl(DEV_MATRIX_FILE), "Dev feature matrix")
    dev_labels = unique_by_qid(load_jsonl(DEV_LABEL_SPLIT_FILE), "Dev saved labels")
    dev_qids = sorted(dev_matrix)
    train_x = [dev_matrix[qid]["variants"][VARIANT] for qid in dev_qids]
    train_y = np.asarray([dev_labels[qid]["initial_wrong_label"] for qid in dev_qids], dtype=int)
    model = make_model()
    model.fit(train_x, train_y)
    vec: DictVectorizer = model.named_steps["vec"]
    clf: LogisticRegression = model.named_steps["clf"]
    model_feature_names = list(vec.get_feature_names_out())
    model_schema = {
        "status": "MMQA_CLEAN_RISK_V2_CLEAN_B3_FINAL_FULL_DEV_MODEL_SCHEMA_FROZEN",
        "variant": VARIANT,
        "training_qids": len(dev_qids),
        "training_wrong": int(train_y.sum()),
        "training_correct": int(len(train_y) - train_y.sum()),
        "raw_contract_key_count": len(b3_schema),
        "raw_contract_keys": b3_schema,
        "vectorized_feature_dimension": len(model_feature_names),
        "vectorized_feature_names": model_feature_names,
        "feature_contract_sha256": sha256_file(CONTRACT_FILE),
        "dev_feature_matrix_sha256": sha256_file(DEV_MATRIX_FILE),
        "saved_label_snapshot_sha256": sha256_file(DEV_LABEL_SPLIT_FILE),
        "oof_selection_sha256": sha256_file(OOF_SELECTION_FILE),
        "training_protocol": protocol["full_dev_training"],
        "raw_gold_opened": False,
        "hyperparameter_search": False,
    }
    model_coefficients = {
        "status": "MMQA_CLEAN_RISK_V2_CLEAN_B3_FINAL_FULL_DEV_COEFFICIENTS_FROZEN",
        "variant": VARIANT,
        **coefficient_summary(model),
    }
    frozen_write(MODEL_SCHEMA_FILE, json_bytes(model_schema))
    frozen_write(MODEL_COEFFICIENT_FILE, json_bytes(model_coefficients))

    bundle_metadata = {
        "status": "MMQA_CLEAN_RISK_V2_CLEAN_B3_FINAL_FULL_DEV_BUNDLE_FROZEN",
        "variant": VARIANT,
        "raw_schema": b3_schema,
        "model_schema_sha256": sha256_file(MODEL_SCHEMA_FILE),
        "source_sha256": {
            "feature_contract": sha256_file(CONTRACT_FILE),
            "dev_feature_matrix": sha256_file(DEV_MATRIX_FILE),
            "saved_dev_labels": sha256_file(DEV_LABEL_SPLIT_FILE),
            "oof_selection": sha256_file(OOF_SELECTION_FILE),
            "runner": sha256_file(Path(__file__).resolve()),
        },
        "dependencies": protocol["dependencies"],
        "raw_gold_opened": False,
    }
    dump_new_joblib(model, PIPELINE_FILE)
    dump_new_joblib(vec, VECTORIZER_FILE)
    dump_new_joblib(clf, MODEL_FILE)
    dump_new_joblib({"pipeline": model, "metadata": bundle_metadata}, BUNDLE_FILE)

    # Locked feature generation: 1336 valid Natural-State/localization cases.
    # The five terminal state-unavailable QIDs are never assigned fake model features.
    base = unique_by_qid(load_jsonl(LOCKED_BASE_FILE), "locked base")
    topology = unique_by_qid(load_jsonl(LOCKED_TOPOLOGY_FILE), "locked topology")
    localization = unique_by_qid(load_jsonl(LOCKED_LOCALIZATION_FILE), "locked localization")
    natural_unavailable = unique_by_qid(load_jsonl(LOCKED_NATURAL_UNAVAILABLE_FILE), "locked unavailable")
    state_manifest = unique_by_qid(load_jsonl(LOCKED_STATE_MANIFEST_FILE), "locked state manifest")
    locked_qids = sorted(base)
    unavailable_qids = set(natural_unavailable)

    forbidden_patterns = list(contract["strict_exclusions"]["exact_or_prefix"])
    feature_rows: list[dict[str, Any]] = []
    feature_by_qid: dict[str, dict[str, Any]] = {}
    unavailable_rows: list[dict[str, Any]] = []
    failures: dict[str, list[Any]] = defaultdict(list)
    categorical: dict[str, set[str]] = defaultdict(set)
    emitted_counts = Counter()
    for qid in locked_qids:
        if qid in unavailable_qids:
            unavailable_rows.append({
                "qid": qid,
                "feature_status": "UNAVAILABLE_NO_FAKE_FEATURE_ROW",
                "reason": "NATURAL_STATE_UNAVAILABLE",
                "forced_into_fixed_budget": True,
                "risk_score_available": False,
            })
            continue

        topo_row = topology[qid]
        loc_row = localization[qid]
        manifest = state_manifest.get(qid)
        if manifest is None:
            failures["manifest_missing"].append(qid)
            continue
        adapted_loc = dict(loc_row)
        adapted_loc["present_required_states"] = loc_row.get("present_states")
        candidate, validation = build_explicit_candidate(base[qid], topo_row, adapted_loc)
        features = select_variant(
            candidate, b3_schema, ("T1", "T2", "T3", "T4", "T5"),
            validation["localized"],
        )
        unexpected = sorted(set(features) - b3_allowed)
        required = b3_allowed - {
            "LOC.localized_valid_prob", "LOC.localized_threshold",
            "LOC.localized_invalid_severity", "LOC.localized_transition",
        }
        missing_required = sorted(required - set(features))
        forbidden = sorted(
            key for key in features
            if any(fnmatch.fnmatchcase(key.lower(), pattern.lower()) for pattern in forbidden_patterns)
        )
        nulls = sorted(key for key, value in features.items() if value is None)
        if unexpected:
            failures["outside_whitelist"].append({"qid": qid, "keys": unexpected})
        if missing_required:
            failures["missing_required_key"].append({"qid": qid, "keys": missing_required})
        if forbidden:
            failures["forbidden_key"].append({"qid": qid, "keys": forbidden})
        if nulls:
            failures["null_value"].append({"qid": qid, "keys": nulls})

        schema_required = topo_row.get("predicted_required_transitions_schema_constrained")
        manifest_required = manifest.get("predicted_required_transitions_external_frozen")
        if schema_required != manifest_required:
            failures["topology_manifest_required_mismatch"].append(qid)
        if topo_row.get("probabilities") != manifest.get("topology_probabilities"):
            failures["topology_manifest_probability_mismatch"].append(qid)
        if loc_row.get("missing_transitions") != manifest.get("required_but_missing"):
            failures["localization_manifest_missing_mismatch"].append(qid)
        present_transitions = [
            text(item.get("transition")) for item in loc_row.get("present_states", [])
            if isinstance(item, dict)
        ]
        if present_transitions != manifest.get("required_and_present"):
            failures["localization_manifest_present_mismatch"].append(qid)

        for key in ["DIAG.decision", "LOC.decision", "LOC.localized_reason", "LOC.localized_transition"]:
            if key in features:
                categorical[key].add(str(features[key]))
        emitted_counts.update(features.keys())
        feature_rows.append({"qid": qid, "features": features})
        feature_by_qid[qid] = features

    if any(failures.values()):
        examples = {key: values[:10] for key, values in failures.items() if values}
        raise RuntimeError(f"Locked feature parity failure: {examples}")
    if len(feature_rows) != 1336 or len(unavailable_rows) != NATURAL_UNAVAILABLE_EXPECTED:
        raise RuntimeError("Locked scorable/unavailable feature partition drift")

    dev_provenance = load_json(DEV_SCHEMA_PROVENANCE_FILE)
    feature_provenance = dev_provenance["feature_provenance"]
    if set(feature_provenance) != set(contract["variants"]["CLEAN_B3_GENERAL_T1_T5"]["features"]):
        # Dev provenance covers all 65 unique B3 keys, exactly the selected schema.
        raise RuntimeError("Dev provenance does not match Clean-B3 schema")
    if not all(item.get("deployment_available") is True for item in feature_provenance.values()):
        raise RuntimeError("A Clean-B3 feature is not deployment-available")

    locked_x = [feature_by_qid[qid] for qid in sorted(feature_by_qid)]
    locked_scorable_qids = sorted(feature_by_qid)
    transformed = vec.transform(locked_x)
    if transformed.shape != (1336, len(model_feature_names)):
        raise RuntimeError(f"Locked transformed shape mismatch: {transformed.shape}")
    locked_scores = model.predict_proba(locked_x)[:, 1]
    if not np.isfinite(locked_scores).all():
        raise RuntimeError("Locked risk scores contain nonfinite values")

    string_feature_tokens = set()
    for features in locked_x:
        for key, value in features.items():
            if isinstance(value, str):
                string_feature_tokens.add(f"{key}={value}")
    unknown_string_tokens = sorted(string_feature_tokens - set(model_feature_names))
    if unknown_string_tokens:
        raise RuntimeError(f"Locked unseen categorical vectorized features: {unknown_string_tokens}")

    dev_category_sets = load_json(DEV_FEATURE_AUDIT_FILE)["categorical_value_checks"]
    category_parity = {
        key: {
            "dev_observed": dev_category_sets[key]["observed"],
            "locked_observed": sorted(categorical[key]),
            "locked_subset_of_dev": categorical[key] <= set(dev_category_sets[key]["observed"]),
        }
        for key in dev_category_sets
    }
    if not all(item["locked_subset_of_dev"] for item in category_parity.values()):
        raise RuntimeError(f"Dev/locked categorical parity failure: {category_parity}")

    frozen_write(LOCKED_FEATURE_FILE, jsonl_bytes(feature_rows))
    frozen_write(LOCKED_UNAVAILABLE_FILE, jsonl_bytes(unavailable_rows))
    locked_provenance = {
        "status": "MMQA_CLEAN_RISK_V2_LOCKED1341_CLEAN_B3_PROVENANCE_FROZEN",
        "feature_contract_sha256": sha256_file(CONTRACT_FILE),
        "variant": VARIANT,
        "raw_schema": b3_schema,
        "feature_provenance": feature_provenance,
        "locked_source_mapping": {
            "GEN": str(LOCKED_BASE_FILE),
            "TOPO": str(LOCKED_TOPOLOGY_FILE),
            "DIAG_LOC": str(LOCKED_LOCALIZATION_FILE),
            "present_state_count_mapping": "locked present_states -> contract LOC.present_states.__len__",
        },
        "state_unavailable_feature_policy": "no fake feature row; force into existing fixed budget",
        "deployment_unavailable_features": [],
        "generic_recursive_flatten_used": False,
        "per_transition_verifier_numeric_features_added": False,
    }
    feature_audit = {
        "status": "MMQA_CLEAN_RISK_V2_LOCKED1341_CLEAN_B3_SCHEMA_PARITY_AUDIT_PASS",
        "locked_total_qids": LOCKED_N,
        "scorable_feature_rows": len(feature_rows),
        "unavailable_no_fake_feature_rows": len(unavailable_rows),
        "raw_schema_count": len(b3_schema),
        "raw_schema_equals_dev_contract": True,
        "model_vectorized_dimension": len(model_feature_names),
        "locked_transformed_shape": list(transformed.shape),
        "unknown_categorical_vectorized_features": unknown_string_tokens,
        "category_parity": category_parity,
        "exact_top6_evidence_count_values": sorted({len(base[qid]["retrieved_evidence"]) for qid in base}),
        "feature_failure_counts": {key: len(values) for key, values in failures.items()},
        "forbidden_feature_violations": 0,
        "null_value_violations": 0,
        "state_unavailable_category_entering_model": False,
        "all_features_deployment_available": True,
        "generic_recursive_flatten_used": False,
        "per_transition_verifier_numeric_features_added": False,
        "locked_gold_opened": False,
    }
    frozen_write(LOCKED_FEATURE_PROVENANCE_FILE, json_bytes(locked_provenance))
    frozen_write(LOCKED_FEATURE_AUDIT_FILE, json_bytes(feature_audit))

    # Score all scorable QIDs, then apply the frozen forced-first budget policy.
    score_by_qid = {
        qid: float(score) for qid, score in zip(locked_scorable_qids, locked_scores)
    }
    old_top_rows = load_jsonl(OLD_TOP269_FILE)
    old_top = {text(row.get("qid")): row for row in old_top_rows}
    forced_order = list(protocol["locked_selection"]["forced_order"])
    forced_set = set(forced_order)
    nonforced_candidates = [qid for qid in locked_scorable_qids if qid not in forced_set]
    nonforced_ranked = sorted(nonforced_candidates, key=lambda qid: (-score_by_qid[qid], qid))
    selected_nonforced = nonforced_ranked[:NONFORCED_BUDGET]
    selected_order = forced_order + selected_nonforced
    if len(selected_order) != BUDGET or len(set(selected_order)) != BUDGET:
        raise RuntimeError("New Top-269 budget/uniqueness failure")
    selected_set = set(selected_order)

    risk_rows = []
    for qid in locked_qids:
        loc = localization[qid]
        initial_available = base[qid].get("initial_available") is True
        if qid in unavailable_qids:
            score = None
            scoring_status = "NATURAL_STATE_UNAVAILABLE_NO_MODEL_SCORE"
        else:
            score = score_by_qid[qid]
            scoring_status = "CLEAN_B3_SCORE_AVAILABLE"
        forced_reason = (
            "NATURAL_STATE_UNAVAILABLE" if qid in unavailable_qids
            else "INITIAL_UNAVAILABLE" if not initial_available else ""
        )
        risk_rows.append({
            "qid": qid,
            "clean_b3_risk_score": score,
            "risk_scoring_status": scoring_status,
            "forced_into_fixed_budget": qid in forced_set,
            "forced_risk_reason": forced_reason,
            "selected_top269": qid in selected_set,
            "decision": text(loc.get("decision")),
            "localized_transition": text(loc.get("localized_transition")),
            "initial_available": initial_available,
            "gold_loaded": False,
        })
    frozen_write(LOCKED_RISK_FILE, jsonl_bytes(risk_rows))

    high_rows = []
    for rank, qid in enumerate(selected_order, 1):
        loc = localization[qid]
        initial_available = base[qid].get("initial_available") is True
        forced_reason = (
            "NATURAL_STATE_UNAVAILABLE" if qid in unavailable_qids
            else "INITIAL_UNAVAILABLE" if not initial_available else ""
        )
        high_rows.append({
            "qid": qid,
            "high_risk_rank": rank,
            "clean_b3_risk_score": score_by_qid.get(qid),
            "forced_into_high_risk_budget": qid in forced_set,
            "forced_risk_reason": forced_reason,
            "decision": text(loc.get("decision")),
            "localized_transition": text(loc.get("localized_transition")),
            "initial_available": initial_available,
            "frozen_action_if_future_authorized": "PLAIN_REGEN",
            "regeneration_performed": False,
            "gold_loaded": False,
        })
    low_rows = [
        {
            "qid": qid,
            "clean_b3_risk_score": score_by_qid[qid],
            "low_risk_rank": rank + 1,
            "frozen_action": "KEEP_INITIAL",
            "gold_loaded": False,
        }
        for rank, qid in enumerate(nonforced_ranked[NONFORCED_BUDGET:])
    ]
    if len(low_rows) != 1072:
        raise RuntimeError(f"Low-risk count mismatch: {len(low_rows)}")
    frozen_write(NEW_TOP269_FILE, jsonl_bytes(high_rows))
    frozen_write(NEW_LOW1072_FILE, jsonl_bytes(low_rows))

    old_set = set(old_top)
    intersection = selected_set & old_set
    new_only = selected_set - old_set
    old_only = old_set - selected_set
    if len(new_only) != len(old_only):
        raise RuntimeError("Equal-budget symmetric difference mismatch")
    membership_rows = []
    for qid in sorted(selected_set | old_set):
        membership_rows.append({
            "qid": qid,
            "in_clean_b3_top269": qid in selected_set,
            "in_old_v1_top269": qid in old_set,
            "membership": (
                "INTERSECTION" if qid in intersection
                else "NEW_ONLY" if qid in new_only else "OLD_ONLY"
            ),
            "clean_b3_rank": selected_order.index(qid) + 1 if qid in selected_set else None,
            "old_v1_rank": old_top[qid].get("high_risk_rank") if qid in old_set else None,
            "clean_b3_risk_score": score_by_qid.get(qid),
            "old_v1_risk_score": old_top[qid].get("risk_B2_GENERAL_T1_T4") if qid in old_set else None,
        })
    intersection_rows = [row for row in membership_rows if row["membership"] == "INTERSECTION"]
    new_only_rows = [row for row in membership_rows if row["membership"] == "NEW_ONLY"]
    old_only_rows = [row for row in membership_rows if row["membership"] == "OLD_ONLY"]
    frozen_write(OVERLAP_ALL_FILE, jsonl_bytes(membership_rows))
    frozen_write(INTERSECTION_FILE, jsonl_bytes(intersection_rows))
    frozen_write(NEW_ONLY_FILE, jsonl_bytes(new_only_rows))
    frozen_write(OLD_ONLY_FILE, jsonl_bytes(old_only_rows))

    registry_rows, registry, registry_summary = build_history_registry()
    frozen_write(HISTORY_REGISTRY_FILE, jsonl_bytes(registry_rows))
    reuse_rows = []
    reusable_count = 0
    for row in sorted(new_only_rows, key=lambda x: x["clean_b3_rank"]):
        qid = row["qid"]
        historical = registry.get(qid)
        legal = bool(historical and historical.get("legal_reuse") is True)
        if legal:
            reusable_count += 1
        reuse_rows.append({
            "qid": qid,
            "clean_b3_rank": row["clean_b3_rank"],
            "historical_equivalent_outcome_found": historical is not None,
            "legal_reuse": legal,
            "historical_sources": historical.get("historical_sources", []) if historical else [],
            "chosen_source": historical.get("chosen_source") if historical else None,
            "generation_status": historical.get("generation_status") if historical else None,
            "failure_class": historical.get("failure_class") if historical else None,
            "model_final_answer": historical.get("model_final_answer") if legal else None,
            "effective_final_answer": historical.get("effective_final_answer") if legal else None,
            "equivalence_basis": (
                "same frozen Qwen3.7-Plus Plain-Regeneration operator, same exact original-order Top-6 and unchanged frozen localization route"
                if legal else "no qualifying frozen historical outcome"
            ),
            "new_regeneration_performed": False,
            "gold_used": False,
        })
    frozen_write(NEW_ONLY_REUSE_FILE, jsonl_bytes(reuse_rows))

    overlap_summary = {
        "status": "MMQA_CLEAN_RISK_V2_CLEAN_B3_VS_OLD_V1_TOP269_OVERLAP_COMPLETE",
        "new_top269": len(selected_set),
        "old_top269": len(old_set),
        "intersection": len(intersection),
        "new_only": len(new_only),
        "old_only": len(old_only),
        "overlap_fraction_of_budget": len(intersection) / BUDGET,
        "jaccard": len(intersection) / len(selected_set | old_set),
        "forced_qids_preserved": len(forced_set & selected_set & old_set),
    }
    reuse_summary = {
        "status": "MMQA_CLEAN_RISK_V2_NEW_ONLY_HISTORICAL_REGENERATION_REUSE_AUDIT_COMPLETE",
        "new_only_qids": len(new_only),
        "historical_registry_unique_qids": registry_summary["unique_registry_qids"],
        "historical_registry_source_counts": registry_summary["source_counts"],
        "historical_registry_duplicate_qids": registry_summary["duplicate_qids"],
        "historical_registry_conflicting_duplicate_qids": registry_summary["conflicting_duplicate_qids"],
        "new_only_legal_reuse_count": reusable_count,
        "new_only_requires_future_new_regeneration_count": len(new_only) - reusable_count,
        "regeneration_performed": False,
        "gold_opened": False,
    }
    frozen_write(OVERLAP_SUMMARY_FILE, json_bytes(overlap_summary))
    frozen_write(REUSE_SUMMARY_FILE, json_bytes(reuse_summary))

    summary = {
        "status": "MMQA_CLEAN_RISK_V2_FINAL_FULL_DEV_LOCKED_NON_GOLD_SELECTION_V1_COMPLETE",
        "final_model": {
            "variant": VARIANT,
            "training_qids": DEV_N,
            "raw_schema_keys": len(b3_schema),
            "vectorized_dimension": len(model_feature_names),
        },
        "locked_features": {
            "total_qids": LOCKED_N,
            "scorable": len(feature_rows),
            "natural_state_unavailable_no_fake_features": len(unavailable_rows),
            "schema_parity_status": feature_audit["status"],
        },
        "locked_selection": {
            "budget": len(selected_set),
            "forced": len(forced_set),
            "nonforced": len(selected_nonforced),
            "low_risk_keep": len(low_rows),
            "nonforced_risk_cutoff": score_by_qid[selected_nonforced[-1]],
            "budget_increase": 0,
        },
        "old_v1_overlap": overlap_summary,
        "historical_reuse": reuse_summary,
        "external_api_calls": 0,
        "locked_gold_opened": False,
        "em_f1_computed": False,
        "regeneration_performed": False,
        "feature_contract_modified": False,
        "model_reselected_or_retuned": False,
        "v1_modified": False,
        "next_step": "STOP_BEFORE_ANY_NEW_QWEN_REGENERATION_OR_LOCKED_GOLD_EVALUATION",
    }
    frozen_write(SUMMARY_FILE, json_bytes(summary))

    generated = [
        PIPELINE_FILE, VECTORIZER_FILE, MODEL_FILE, BUNDLE_FILE,
        MODEL_SCHEMA_FILE, MODEL_COEFFICIENT_FILE,
        LOCKED_FEATURE_FILE, LOCKED_UNAVAILABLE_FILE,
        LOCKED_FEATURE_AUDIT_FILE, LOCKED_FEATURE_PROVENANCE_FILE,
        LOCKED_RISK_FILE, NEW_TOP269_FILE, NEW_LOW1072_FILE,
        OVERLAP_ALL_FILE, INTERSECTION_FILE, NEW_ONLY_FILE, OLD_ONLY_FILE,
        OVERLAP_SUMMARY_FILE, HISTORY_REGISTRY_FILE,
        NEW_ONLY_REUSE_FILE, REUSE_SUMMARY_FILE, SUMMARY_FILE,
    ]
    hashes = {
        "status": "MMQA_CLEAN_RISK_V2_FINAL_FULL_DEV_LOCKED_SELECTION_SHA_FROZEN",
        "generated_files": {path.name: file_record(path) for path in generated},
        "frozen_protocol_files": {
            path.name: file_record(path)
            for path in [PROTOCOL_FILE, PREFLIGHT_FILE, PROTOCOL_HASH_FILE]
        },
        "read_only_sources": {
            path.name: file_record(path)
            for path in [
                CONTRACT_FILE, DEV_MATRIX_FILE, DEV_LABEL_SPLIT_FILE,
                OOF_SELECTION_FILE, LOCKED_BASE_FILE, LOCKED_TOPOLOGY_FILE,
                LOCKED_NATURAL_UNAVAILABLE_FILE, LOCKED_STATE_MANIFEST_FILE,
                LOCKED_LOCALIZATION_FILE, OLD_TOP269_FILE,
                *[source["outcomes"] for source in HISTORY_SOURCES],
                Path(__file__).resolve(), FEATURE_RUNNER, OOF_RUNNER,
            ]
        },
        "external_api_calls": 0,
        "locked_gold_opened": False,
        "em_f1_computed": False,
        "regeneration_performed": False,
        "v1_modified": False,
    }
    frozen_write(HASH_FILE, json_bytes(hashes))

    print("=" * 150)
    print("MMQA CLEAN RISK V2 — FINAL FULL-DEV FREEZE + LOCKED NON-GOLD RISK SELECTION V1")
    print("=" * 150)
    print("Final model:", VARIANT, "Dev=1096", "raw keys=", len(b3_schema), "vectorized dim=", len(model_feature_names))
    print("Locked features: scorable=", len(feature_rows), "unavailable_no_fake_features=", len(unavailable_rows))
    print("Schema/category/provenance/deployment parity: PASS")
    print("Selection: forced=", len(forced_set), "nonforced=", len(selected_nonforced), "total=", len(selected_set), "low=", len(low_rows))
    print("Old V1 overlap: intersection=", len(intersection), "new_only=", len(new_only), "old_only=", len(old_only))
    print("New-only historical legal reuse:", reusable_count, "/", len(new_only))
    print("New-only requiring future regeneration:", len(new_only) - reusable_count)
    print("API/Gold/EM-F1/regeneration/feature-change/retune/V1-modified: 0 / 0 / 0 / 0 / 0 / 0 / 0")
    print("Bundle:", BUNDLE_FILE)
    print("Locked Top-269:", NEW_TOP269_FILE)
    print("Overlap:", OVERLAP_SUMMARY_FILE)
    print("Reuse audit:", REUSE_SUMMARY_FILE)
    print("SHA:", HASH_FILE)
    print("STATUS =", summary["status"])
    print("NEXT STEP =", summary["next_step"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze-protocol", action="store_true")
    modes.add_argument("--run-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze_protocol:
        freeze_protocol()
    elif args.run_all:
        run_all()


if __name__ == "__main__":
    main()
