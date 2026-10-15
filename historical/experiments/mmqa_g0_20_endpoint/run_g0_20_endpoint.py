#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MMQA locked1341 formal G0-20 endpoint experiment.

The experiment is isolated from all existing frozen arms.  Preflight never
loads Gold and never calls an API.  Execution reuses the already-audited MMQA
Plain-Regeneration payload builder and provider/failure policy.  Finalization
is allowed only after all 269 selected outcomes are resolved.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
N = 1341
BUDGET = math.ceil(0.20 * N)
FORCED_N = 6
NONFORCED_N = BUDGET - FORCED_N
DEV_N = 1096
VARIANT = "G0_GENERAL_ONLY"
MODEL_SEED = 260817
BOOTSTRAP_REPS = 10_000
EM_BOOTSTRAP_SEED = 20260828
F1_BOOTSTRAP_SEED = 20260829
RANDOM_SEEDS = (20260822, 20260823, 20260824, 20260825, 20260826)

CONTRACT = ROOT / "outputs/mmqa_clean_risk_v2_feature_contract_v1/clean_risk_v2_feature_contract_v1.json"
DEV_MATRIX = ROOT / "outputs/mmqa_clean_risk_v2_dev1096_feature_generation_v1/dev1096_clean_risk_feature_matrix_v1.jsonl"
DEV_LABELS = ROOT / "outputs/mmqa_clean_risk_v2_dev1096_answer_risk_oof_v1/dev1096_saved_answer_error_labels_and_shared_folds_v1.jsonl"
OOF_METRICS = ROOT / "outputs/mmqa_clean_risk_v2_dev1096_answer_risk_oof_v1/clean_v2_oof_metrics_v1.json"
LOCKED_BASE = ROOT / "outputs/mmqa_v3_heldout1341_pre_natural_v1/mmqa_v3_heldout1341_initial_base_sanitized_v1.jsonl"
LOCKED_HIGH_OLD = ROOT / "outputs/mmqa_v3_heldout1341_frozen_verifier_localization_risk_v1/heldout1341_frozen_top20_high_risk_269_v1.jsonl"
LOCKED_LOW_OLD = ROOT / "outputs/mmqa_v3_heldout1341_frozen_verifier_localization_risk_v1/heldout1341_frozen_low_risk_1072_v1.jsonl"
CLEAN_PROTOCOL = ROOT / "outputs/mmqa_clean_risk_v2_final_full_dev_locked_selection_v1/final_full_dev_locked_selection_protocol_v1.json"
CLEAN_PREFLIGHT = ROOT / "outputs/mmqa_clean_risk_v2_final_full_dev_locked_selection_v1/final_full_dev_locked_selection_preflight_v1.json"
CLEAN_SELECTION = ROOT / "outputs/mmqa_clean_risk_v2_final_full_dev_locked_selection_v1/locked1341_clean_b3_top269_selection_v1.jsonl"
CLEAN_FEATURES = ROOT / "outputs/mmqa_clean_risk_v2_final_full_dev_locked_selection_v1/locked1341_clean_b3_features_v1.jsonl"
HISTORY_REGISTRY = ROOT / "outputs/mmqa_clean_risk_v2_final_full_dev_locked_selection_v1/historical_exact_plain_regeneration_registry_v1.jsonl"
VC_SELECTION = ROOT / "outputs/mmqa_vc20_selection_freeze_v1_1/vc20_qid_tiebreak_selected269_frozen_v1.jsonl"
A_CACHE = ROOT / "outputs/mmqa_dev1096_a0_a4_a6_selector_endpoint_regen_protocol_v1/selector_endpoint_regen_cache_v1.jsonl"
CLEAN_FINAL = ROOT / "outputs/mmqa_clean_risk_v2_final_prediction_merge_before_gold_v1/clean_v2_locked1341_final_predictions_frozen_before_gold_v1.jsonl"
GOLD = ROOT / "data/raw/mmqa_official/MMQA_dev.jsonl.gz"
OFFICIAL_EVALUATOR = ROOT / "audit_mmqa_v3_locked1341_frozen_boundary_v1_1.py"

ORIGINAL_RUNNER = ROOT / "run_mmqa_v3_heldout1341_final_plain_regen_qwen37plus_v1.py"
PROMPT_CONTRACT = ROOT / "run_mmqa_v2_fresh_pilot40_b_qwen37plus.py"
V2_PROTOCOL = ROOT / "outputs/mmqa_v2_protocol_freeze_20260819/mmqa_v2_frozen_protocol.json"
BASE_RANDOM_RUNNER = ROOT / "run_mmqa_v3_random20_matched_qwen37plus_v1.py"
PROVIDER = ROOT / "run_mmqa_transition_evidence_recompose_abc_pilot_qwen37plus_v1.py"
HARD_TIMEOUT_RUNNER = ROOT / "run_mmqa_v3_b0_only15_exact_protocol_qwen37plus_v1_3.py"
JOB_BUILDER = ROOT / "build_and_freeze_mmqa_v3_b0_only15_exact_protocol_jobs_v1_1.py"

SCORES_FILE = OUT / "g0_locked1341_scores.jsonl"
SELECTED_FILE = OUT / "g0_20_selected_qids.jsonl"
REUSE_FILE = OUT / "g0_20_reuse_audit.jsonl"
MANIFEST_FILE = OUT / "g0_20_regeneration_manifest.jsonl"
CACHE_FILE = OUT / "g0_20_new_execution_cache.jsonl"
FINAL_FILE = OUT / "g0_20_final_predictions.jsonl"
METRICS_FILE = OUT / "g0_20_metrics.json"
STATS_FILE = OUT / "g0_20_statistics.json"
REPORT_FILE = OUT / "g0_20_audit_report.md"
MODEL_FILE = OUT / "g0_full_dev_model.joblib"
PREFLIGHT_FILE = OUT / "g0_20_preflight.json"
EXECUTION_FILE = OUT / "g0_20_execution_summary.json"

EXPECTED_CONTRACT_SHA = "9BACED6295AE3F31E023DD9E91395ABC567861AB4DA040A5E0D337B344ED14B1"
EXPECTED_DEV_MATRIX_SHA = "FA579EE4028FFB2F07EDA664B3547D688733E0022677048783223E763330DCF7"
EXPECTED_INITIAL_SHA = "6F068067A5F5DC489644D68EC5B072E31B624264AE506D7D89CCE36498437E81"
EXPECTED_CLEAN_FINAL_SHA = "CA0D6B81120748C844119850D1E59DDA79E3C27D9DD1776367196B5A4E51A7A5"
EXPECTED_EVALUATOR_SHA = "16C3BB79BF93DB73381977A2A032F6C54771F7BBF7E1EE19CB20D4713A87C85B"


class ProtocolError(RuntimeError):
    pass


def safe(value: Any) -> str:
    return "" if value is None else str(value).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ProtocolError(f"Expected object: {path}")
    return obj


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ProtocolError(f"Non-object row: {path}:{line_no}")
            rows.append(row)
    return rows


def index_rows(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = safe(row.get("qid"))
        if not qid or qid in out:
            raise ProtocolError(f"Empty/duplicate QID in {label}: {qid!r}")
        out[qid] = dict(row)
    return out


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_sha(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected.upper():
        raise ProtocolError(f"Frozen SHA mismatch: {path}")


def make_model() -> Pipeline:
    return Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", LogisticRegression(
            penalty="l2", solver="liblinear", C=1.0,
            class_weight="balanced", max_iter=2000,
            random_state=MODEL_SEED, fit_intercept=True,
        )),
    ])


def deterministic_random_union(qids: Sequence[str]) -> set[str]:
    selected: set[str] = set()
    for seed in RANDOM_SEEDS:
        ranked = sorted(
            qids,
            key=lambda qid: (hashlib.sha256(f"{seed}|{qid}".encode("utf-8")).hexdigest(), qid),
        )
        selected.update(ranked[:BUDGET])
    return selected


def preflight() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    feature_module = import_file(
        "mmqa_g0_formal_feature_builder",
        ROOT / "run_mmqa_clean_risk_v2_dev1096_feature_generation_v1.py",
    )
    builder = import_file("mmqa_g0_exact_job_builder", JOB_BUILDER)
    original_runner = import_file("mmqa_g0_original_runner", ORIGINAL_RUNNER)
    prompt_module = import_file("mmqa_g0_prompt_contract", PROMPT_CONTRACT)

    require_sha(CONTRACT, EXPECTED_CONTRACT_SHA)
    require_sha(DEV_MATRIX, EXPECTED_DEV_MATRIX_SHA)
    require_sha(LOCKED_BASE, EXPECTED_INITIAL_SHA)
    contract = load_json(CONTRACT)
    schema = list(contract["variants"][VARIANT]["features"])
    if len(schema) != 12 or any(not key.startswith("GEN.") for key in schema):
        raise ProtocolError("Formal G0 contract is not exactly 12 GEN keys")

    dev = index_rows(load_jsonl(DEV_MATRIX), "Dev feature matrix")
    labels = index_rows(load_jsonl(DEV_LABELS), "Dev labels")
    if len(dev) != DEV_N or set(dev) != set(labels):
        raise ProtocolError("Dev1096 universe mismatch")
    dev_qids = sorted(dev)
    x_train = [dev[q]["variants"][VARIANT] for q in dev_qids]
    y_train = np.asarray([int(labels[q]["initial_wrong_label"]) for q in dev_qids], dtype=int)
    if any(set(row) != set(schema) for row in x_train):
        raise ProtocolError("Dev G0 rows do not exactly match the formal 12-key schema")
    model = make_model()
    model.fit(x_train, y_train)
    if list(model.named_steps["vec"].get_feature_names_out()) != sorted(schema):
        raise ProtocolError("G0 vectorized model schema drift")

    base = index_rows(load_jsonl(LOCKED_BASE), "locked1341 base")
    if len(base) != N:
        raise ProtocolError("locked1341 size drift")
    locked_qids = sorted(base)
    locked_features = {qid: feature_module.general_features(base[qid]) for qid in locked_qids}
    if any(set(row) != set(schema) for row in locked_features.values()):
        raise ProtocolError("Locked G0 feature rows do not exactly match the formal schema")

    # For all scorable Clean-B3 rows, the G0 subset must be byte-value identical.
    b3_rows = index_rows(load_jsonl(CLEAN_FEATURES), "Clean-B3 locked features")
    parity_failures = [
        qid for qid, row in b3_rows.items()
        if {key: row["features"][key] for key in schema} != locked_features[qid]
    ]
    if parity_failures:
        raise ProtocolError(f"G0/B3 general-feature parity failed: {parity_failures[:5]}")

    scores = model.predict_proba([locked_features[q] for q in locked_qids])[:, 1]
    score_by_qid = {qid: float(value) for qid, value in zip(locked_qids, scores)}

    clean_protocol = load_json(CLEAN_PROTOCOL)
    clean_preflight = load_json(CLEAN_PREFLIGHT)
    forced_order = list(clean_protocol["locked_selection"]["forced_order"])
    forced_set = set(forced_order)
    evidence_for_general_fail_safe = {
        "current_clean_protocol_forced_qids": clean_protocol["locked_selection"]["forced_qids"] == FORCED_N,
        "current_clean_protocol_nonforced_slots": clean_protocol["locked_selection"]["nonforced_slots"] == NONFORCED_N,
        "current_clean_preflight_universal_unavailable_definition": (
            clean_preflight["validation"]["forced_budget_qids"] == FORCED_N
            and clean_preflight["validation"]["forced_order"] == forced_order
            and clean_preflight["validation"]["locked_initial_unavailable"] == FORCED_N
            and clean_preflight["validation"]["locked_natural_unavailable_unscored"] == 5
        ),
        "historical_general_only_risk_ablation_used_same_forced_set": True,
    }
    old_b0_source = (ROOT / "score_and_freeze_mmqa_v3_b0_locked1341_top20_v1.py").read_text(encoding="utf-8")
    evidence_for_general_fail_safe["historical_general_only_risk_ablation_used_same_forced_set"] = (
        "selected_order = forced_order + nonforced_order" in old_b0_source
        and "budget - len(forced_order)" in old_b0_source
    )
    if not all(evidence_for_general_fail_safe.values()):
        raise ProtocolError(f"Forced-first applicability not established: {evidence_for_general_fail_safe}")

    ranked = sorted(
        (qid for qid in locked_qids if qid not in forced_set),
        key=lambda qid: (-score_by_qid[qid], qid),
    )
    selected_order = forced_order + ranked[:NONFORCED_N]
    if len(selected_order) != BUDGET or len(set(selected_order)) != BUDGET:
        raise ProtocolError("G0-20 selection budget/uniqueness failure")
    selected_set = set(selected_order)

    history = index_rows(load_jsonl(HISTORY_REGISTRY), "historical exact-protocol registry")
    conflicts = [qid for qid in selected_order if qid in history and history[qid].get("duplicate_outcome_conflict")]
    if conflicts:
        raise ProtocolError(f"Conflicting reusable outcomes: {conflicts}")
    reusable = {
        qid for qid in selected_order
        if qid in history and history[qid].get("legal_reuse") is True
        and history[qid].get("same_exact_plain_regeneration_protocol") is True
    }
    new_qids = [qid for qid in selected_order if qid not in reusable]

    historical_plan = {
        **index_rows(load_jsonl(LOCKED_LOW_OLD), "historical low"),
        **index_rows(load_jsonl(LOCKED_HIGH_OLD), "historical high"),
    }
    if set(historical_plan) != set(base):
        raise ProtocolError("Historical route plan does not cover locked1341")
    v2_protocol = load_json(V2_PROTOCOL)
    selection_rank = {qid: rank for rank, qid in enumerate(selected_order, 1)}
    jobs: list[dict[str, Any]] = []
    for position, qid in enumerate(selected_order, 1):
        job = builder.make_job(
            qid, position, selection_rank[qid], base, historical_plan,
            original_runner, prompt_module, v2_protocol,
        )
        job["g0_20_selection_rank"] = selection_rank[qid]
        job["outcome_source"] = history[qid]["chosen_source"] if qid in reusable else "NEW_REQUIRED"
        job["reuse_legal"] = qid in reusable
        jobs.append(job)

    clean_set = set(index_rows(load_jsonl(CLEAN_SELECTION), "Clean-B3 selection"))
    vc_set = set(index_rows(load_jsonl(VC_SELECTION), "VC-20 selection"))
    random_union = deterministic_random_union(locked_qids)
    # The resumable A-series cache may contain more than one attempt per QID;
    # only its QID universe is relevant to the cross-experiment overlap audit.
    a_union = {safe(row.get("qid")) for row in load_jsonl(A_CACHE) if safe(row.get("qid"))}
    if a_union & set(locked_qids):
        raise ProtocolError("A0/A4/A6 was expected to be a disjoint Dev1096 experiment")

    score_rows = [
        {
            "qid": qid,
            "g0_risk_score": score_by_qid[qid],
            "forced_case": qid in forced_set,
            "selected_g0_20": qid in selected_set,
            "gold_used_for_scoring_or_selection": False,
        }
        for qid in locked_qids
    ]
    selected_rows = [
        {
            "qid": qid,
            "selection_rank": rank,
            "selection_basis": "FORCED_FAIL_SAFE" if qid in forced_set else "G0_RISK_DESC_QID_ASC",
            "g0_risk_score": score_by_qid[qid],
            "forced_case": qid in forced_set,
            "reusable_outcome": qid in reusable,
        }
        for rank, qid in enumerate(selected_order, 1)
    ]
    reuse_rows = []
    for rank, qid in enumerate(selected_order, 1):
        source = history.get(qid, {})
        reuse_rows.append({
            "qid": qid,
            "selection_rank": rank,
            "reuse_legal": qid in reusable,
            "outcome_source": source.get("chosen_source", "NEW_REQUIRED"),
            "generation_status": source.get("generation_status", "NOT_EXECUTED"),
            "same_solver_relevant_payload": qid in reusable,
            "duplicate_outcome_conflict": source.get("duplicate_outcome_conflict", False),
            "overlap_clean_b3": qid in clean_set,
            "overlap_vc20": qid in vc_set,
            "overlap_random20_five_seed_union": qid in random_union,
            "overlap_a0_a4_a6_union": qid in a_union,
        })

    write_jsonl(SCORES_FILE, score_rows)
    write_jsonl(SELECTED_FILE, selected_rows)
    write_jsonl(REUSE_FILE, reuse_rows)
    write_jsonl(MANIFEST_FILE, jobs)
    joblib.dump({
        "pipeline": model,
        "variant": VARIANT,
        "raw_schema": schema,
        "training_qids": DEV_N,
        "positive_class": "Initial Answer error",
        "locked_gold_opened": False,
    }, MODEL_FILE)

    result = {
        "status": "MMQA_LOCKED1341_G0_20_PREFLIGHT_PASS",
        "phase": "A-F_COMPLETE_NO_API_NO_GOLD",
        "protocol": {
            "variant": VARIANT,
            "raw_keys": schema,
            "raw_key_count": len(schema),
            "model": "DictVectorizer(sparse=True) + LogisticRegression",
            "penalty": "l2",
            "solver": "liblinear",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": MODEL_SEED,
            "fit_intercept": True,
            "feature_scaling": False,
            "probability_calibration": False,
            "score_use": "ranking_only",
            "full_dev_training_qids": DEV_N,
            "locked_gold_opened": False,
        },
        "oof_reference": load_json(OOF_METRICS)["variants"][VARIANT],
        "selection": {
            "population": N,
            "budget_formula": "ceil(0.20 * 1341)",
            "budget": BUDGET,
            "forced": FORCED_N,
            "risk_ranked": NONFORCED_N,
            "tie_break": "qid_ascending",
            "risk_cutoff": score_by_qid[selected_order[-1]],
            "forced_policy_evidence": evidence_for_general_fail_safe,
        },
        "reuse": {
            "selected": BUDGET,
            "reusable_outcomes": len(reusable),
            "genuinely_new_logical_executions": len(new_qids),
            "new_qids": new_qids,
            "overlap_clean_b3": len(selected_set & clean_set),
            "overlap_vc20": len(selected_set & vc_set),
            "overlap_random20_five_seed_union": len(selected_set & random_union),
            "overlap_a0_a4_a6_union": len(selected_set & a_union),
            "a0_a4_a6_note": "Dev1096 controlled endpoint; QID overlap with locked1341 is zero.",
        },
        "regeneration": {
            "model": "qwen3.7-plus",
            "temperature": 0.1,
            "thinking": True,
            "thinking_budget": 2048,
            "input": "question + exact original-order Top-6; T1 eligible raw images only (max 3)",
            "excluded": ["Frozen Initial Answer", "Expected Topology", "Observed Natural State", "Diagnosis / Localization", "G0 risk score", "Gold"],
            "client_max_tokens_argument": 4096,
            "provider_side_max_tokens_enforcement_confirmed": False,
        },
        "source_hashes": {
            "feature_contract": sha256_file(CONTRACT),
            "dev_matrix": sha256_file(DEV_MATRIX),
            "dev_labels": sha256_file(DEV_LABELS),
            "locked_base": sha256_file(LOCKED_BASE),
            "clean_protocol": sha256_file(CLEAN_PROTOCOL),
            "historical_registry": sha256_file(HISTORY_REGISTRY),
            "original_runner": sha256_file(ORIGINAL_RUNNER),
            "prompt_contract": sha256_file(PROMPT_CONTRACT),
        },
        "outputs": {
            "scores": str(SCORES_FILE),
            "selected": str(SELECTED_FILE),
            "reuse_audit": str(REUSE_FILE),
            "regeneration_manifest": str(MANIFEST_FILE),
            "model": str(MODEL_FILE),
        },
        "external_api_calls": 0,
        "frozen_protocol_deviation": False,
    }
    write_json(PREFLIGHT_FILE, result)
    write_report(result, None, None)
    return result


def load_new_jobs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not PREFLIGHT_FILE.is_file() or load_json(PREFLIGHT_FILE).get("status") != "MMQA_LOCKED1341_G0_20_PREFLIGHT_PASS":
        raise ProtocolError("Passed no-Gold preflight is required before execution")
    jobs = load_jsonl(MANIFEST_FILE)
    new = [row for row in jobs if row.get("reuse_legal") is False]
    if len(new) != 11:
        raise ProtocolError(f"Frozen new-execution count changed: {len(new)}")
    initial = index_rows(load_jsonl(LOCKED_BASE), "locked Initial")
    return new, initial


def execute(mode: str, request_interval: float) -> dict[str, Any]:
    new_jobs, initial = load_new_jobs()
    # The hard-timeout worker is used by multiprocessing.spawn on Windows and
    # therefore must be imported under its real, importable module name.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    runner = importlib.import_module("run_mmqa_v3_b0_only15_exact_protocol_qwen37plus_v1_3")
    base = import_file("mmqa_g0_matched_runner", BASE_RANDOM_RUNNER)
    provider = import_file("mmqa_g0_provider", PROVIDER)
    parser = import_file("mmqa_g0_parser", PROMPT_CONTRACT)
    runner.configure_direct_certifi_tls()
    jobs = index_rows(new_jobs, "new G0 jobs")
    completed, unresolved = base.latest_cache(CACHE_FILE, jobs)

    pilot_qids = [safe(row["qid"]) for row in new_jobs[:10]]
    if mode == "pilot":
        selected = [jobs[qid] for qid in pilot_qids if qid not in completed]
    else:
        pilot_completed = [qid for qid in pilot_qids if qid in completed]
        pilot_success = sum(completed[qid].get("status") == "success" for qid in pilot_completed)
        if len(pilot_completed) != len(pilot_qids) or pilot_success < 8:
            raise ProtocolError("Fixed Pilot10 health gate has not passed")
        selected = [row for row in new_jobs if safe(row["qid"]) not in completed]

    api_key, key_source = provider.load_api_key()
    last_request = 0.0
    consecutive_transport = 0
    consecutive_provider_terminal = 0
    for position, job in enumerate(selected, 1):
        if last_request:
            wait = request_interval - (time.monotonic() - last_request)
            if wait > 0:
                time.sleep(wait)
        qid = safe(job["qid"])
        record = runner.call_one_with_hard_timeout(
            base, BASE_RANDOM_RUNNER, PROMPT_CONTRACT, PROVIDER,
            api_key, job, safe(initial[qid].get("draft_answer")),
        )
        last_request = time.monotonic()
        append_jsonl(CACHE_FILE, record)
        status = safe(record.get("status"))
        if status == "success":
            consecutive_transport = consecutive_provider_terminal = 0
        elif status == "keep_initial_terminal":
            consecutive_transport = 0
            consecutive_provider_terminal = (
                consecutive_provider_terminal + 1
                if safe(record.get("failure_class")) == "provider_terminal" else 0
            )
        else:
            consecutive_transport += 1
            consecutive_provider_terminal = 0
        print(f"[{position}/{len(selected)}] {qid} -> {status}", flush=True)
        if consecutive_transport >= 2 or consecutive_provider_terminal >= 2:
            break

    completed, unresolved = base.latest_cache(CACHE_FILE, jobs)
    result = {
        "status": "MMQA_LOCKED1341_G0_20_EXECUTION_PROGRESS",
        "mode": mode,
        "api_key_source": key_source,
        "new_jobs": len(new_jobs),
        "completed": len(completed),
        "success": sum(row.get("status") == "success" for row in completed.values()),
        "terminal": sum(row.get("status") == "keep_initial_terminal" for row in completed.values()),
        "unresolved": len(unresolved),
        "remaining": len(new_jobs) - len(completed),
        "frozen_protocol_deviation": False,
    }
    write_json(EXECUTION_FILE, result)
    return result


def import_official_evaluator():
    require_sha(OFFICIAL_EVALUATOR, EXPECTED_EVALUATOR_SHA)
    return import_file("mmqa_g0_official_evaluator", OFFICIAL_EVALUATOR)


def answer_from(official, row: Mapping[str, Any], key: str) -> str:
    return official.stringify_answer(row.get(key, row.get("answer", "")))


def score_answers(official, answers: Mapping[str, str], gold: Mapping[str, Sequence[str]], qids: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        qid: dict(zip(("em", "f1"), official.best_score(answers[qid], gold[qid])))
        for qid in qids
    }


def summarize(scores: Mapping[str, Mapping[str, float]], qids: Sequence[str]) -> dict[str, Any]:
    em = [int(scores[q]["em"]) for q in qids]
    f1 = [float(scores[q]["f1"]) for q in qids]
    return {
        "n": len(qids), "em_count": sum(em),
        "em": statistics.mean(em), "em_percent": 100 * statistics.mean(em),
        "f1": statistics.mean(f1), "f1_percent": 100 * statistics.mean(f1),
    }


def migration(base: Mapping[str, Mapping[str, float]], target: Mapping[str, Mapping[str, float]], qids: Sequence[str]) -> dict[str, int]:
    counts = Counter()
    for qid in qids:
        b, t = int(base[qid]["em"]), int(target[qid]["em"])
        counts[(b, t)] += 1
    return {
        "wrong_to_right": counts[(0, 1)],
        "right_to_wrong": counts[(1, 0)],
        "net": counts[(0, 1)] - counts[(1, 0)],
        "right_to_right": counts[(1, 1)],
        "wrong_to_wrong": counts[(0, 0)],
    }


def exact_mcnemar_p(a_better: int, b_better: int) -> float:
    n = a_better + b_better
    if n == 0:
        return 1.0
    k = min(a_better, b_better)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], seed: int) -> dict[str, Any]:
    delta = [float(x) - float(y) for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(delta)
    values = sorted(sum(delta[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_REPS))

    def percentile(p: float) -> float:
        pos = p * (len(values) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return values[lo]
        return values[lo] * (hi - pos) + values[hi] * (pos - lo)

    return {
        "delta": statistics.mean(delta), "delta_pp": 100 * statistics.mean(delta),
        "ci95_low": percentile(0.025), "ci95_high": percentile(0.975),
        "ci95_low_pp": 100 * percentile(0.025), "ci95_high_pp": 100 * percentile(0.975),
        "repetitions": BOOTSTRAP_REPS, "seed": seed,
    }


def pairwise(name: str, a: Mapping[str, Mapping[str, float]], b: Mapping[str, Mapping[str, float]], qids: Sequence[str]) -> dict[str, Any]:
    a_em = [int(a[q]["em"]) for q in qids]
    b_em = [int(b[q]["em"]) for q in qids]
    a_f1 = [float(a[q]["f1"]) for q in qids]
    b_f1 = [float(b[q]["f1"]) for q in qids]
    a_better = sum(x == 1 and y == 0 for x, y in zip(a_em, b_em))
    b_better = sum(x == 0 and y == 1 for x, y in zip(a_em, b_em))
    return {
        "comparison": name,
        "em": {
            "a_better_discordant": a_better, "b_better_discordant": b_better,
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(a_better, b_better),
            "paired_bootstrap": paired_bootstrap(a_em, b_em, EM_BOOTSTRAP_SEED),
        },
        "token_f1": {"paired_bootstrap": paired_bootstrap(a_f1, b_f1, F1_BOOTSTRAP_SEED)},
    }


def finalize() -> tuple[dict[str, Any], dict[str, Any]]:
    pre = load_json(PREFLIGHT_FILE)
    selected = index_rows(load_jsonl(SELECTED_FILE), "selected")
    jobs = index_rows(load_jsonl(MANIFEST_FILE), "manifest")
    history = index_rows(load_jsonl(HISTORY_REGISTRY), "history")
    initial = index_rows(load_jsonl(LOCKED_BASE), "initial")
    base_runner = import_file("mmqa_g0_finalize_base_runner", BASE_RANDOM_RUNNER)
    new_jobs = {qid: row for qid, row in jobs.items() if row.get("reuse_legal") is False}
    completed, unresolved = base_runner.latest_cache(CACHE_FILE, new_jobs)
    if unresolved or set(completed) != set(new_jobs):
        raise ProtocolError(f"Cannot finalize: completed={len(completed)}/{len(new_jobs)} unresolved={len(unresolved)}")

    outcome: dict[str, dict[str, Any]] = {}
    for qid in selected:
        if qid in completed:
            outcome[qid] = completed[qid]
        elif qid in history and history[qid].get("legal_reuse") is True:
            outcome[qid] = history[qid]
        else:
            raise ProtocolError(f"Selected QID has no resolved outcome: {qid}")

    final_rows = []
    for qid in sorted(initial):
        if qid in selected:
            row = outcome[qid]
            final_answer = safe(row.get("effective_final_answer"))
            status = safe(row.get("generation_status", row.get("status")))
            failure_class = safe(row.get("failure_class"))
            source = safe(jobs[qid].get("outcome_source"))
        else:
            final_answer = safe(initial[qid].get("draft_answer"))
            status, failure_class, source = "not_selected_keep_initial", "", "FROZEN_INITIAL"
        final_rows.append({
            "qid": qid, "final_answer": final_answer,
            "selected_g0_20": qid in selected,
            "generation_status": status, "failure_class": failure_class,
            "outcome_source": source, "gold_used_in_inference": False,
        })
    write_jsonl(FINAL_FILE, final_rows)

    # Gold is opened only after the complete before-Gold prediction file exists.
    require_sha(CLEAN_FINAL, EXPECTED_CLEAN_FINAL_SHA)
    official = import_official_evaluator()
    gold_all = index_rows(load_jsonl(GOLD), "official MMQA Gold")
    qids = sorted(initial)
    gold = {qid: official.extract_gold_answers(gold_all[qid]) for qid in qids}
    g0_answers = {row["qid"]: answer_from(official, row, "final_answer") for row in final_rows}
    initial_answers = {qid: answer_from(official, initial[qid], "draft_answer") for qid in qids}
    clean_rows = index_rows(load_jsonl(CLEAN_FINAL), "Clean-B3 final")
    clean_answers = {qid: answer_from(official, clean_rows[qid], "final_answer") for qid in qids}
    g0_scores = score_answers(official, g0_answers, gold, qids)
    initial_scores = score_answers(official, initial_answers, gold, qids)
    clean_scores = score_answers(official, clean_answers, gold, qids)
    g0_summary, initial_summary, clean_summary = (
        summarize(g0_scores, qids), summarize(initial_scores, qids), summarize(clean_scores, qids)
    )
    if initial_summary["em_count"] != 641 or abs(initial_summary["f1_percent"] - 54.99891914604916) > 1e-9:
        raise ProtocolError("Frozen Initial reference drift")
    if clean_summary["em_count"] != 688 or abs(clean_summary["f1_percent"] - 58.5258363228514) > 1e-9:
        raise ProtocolError("Frozen Clean-B3 reference drift")

    g0_migration = migration(initial_scores, g0_scores, qids)
    metrics = {
        "status": "MMQA_LOCKED1341_G0_20_ENDPOINT_COMPLETE",
        "g0_20": {
            **g0_summary,
            "migration_from_initial": g0_migration,
            "delta_em_vs_initial_pp": g0_summary["em_percent"] - initial_summary["em_percent"],
            "delta_token_f1_vs_initial_pp": g0_summary["f1_percent"] - initial_summary["f1_percent"],
            "delta_em_vs_clean_b3_pp": g0_summary["em_percent"] - clean_summary["em_percent"],
            "delta_token_f1_vs_clean_b3_pp": g0_summary["f1_percent"] - clean_summary["f1_percent"],
        },
        "initial": initial_summary,
        "clean_b3": clean_summary,
        "selected": BUDGET,
        "forced": FORCED_N,
        "risk_ranked": NONFORCED_N,
        "reused": pre["reuse"]["reusable_outcomes"],
        "new_executions": len(new_jobs),
        "selected_outcome_status": dict(sorted(Counter(
            safe(row.get("generation_status", row.get("status"))) for row in outcome.values()
        ).items())),
        "frozen_protocol_deviation": False,
    }
    statistics_result = {
        "status": "MMQA_LOCKED1341_G0_20_STATISTICS_COMPLETE",
        "protocol": {
            "em": "two-sided exact McNemar + 10,000 paired percentile bootstrap",
            "token_f1": "10,000 paired percentile bootstrap",
            "em_bootstrap_seed": EM_BOOTSTRAP_SEED,
            "f1_bootstrap_seed": F1_BOOTSTRAP_SEED,
        },
        "g0_20_vs_clean_b3": pairwise("g0_20_minus_clean_b3", g0_scores, clean_scores, qids),
        "g0_20_vs_initial": pairwise("g0_20_minus_initial", g0_scores, initial_scores, qids),
    }
    write_json(METRICS_FILE, metrics)
    write_json(STATS_FILE, statistics_result)
    execution = load_json(EXECUTION_FILE) if EXECUTION_FILE.is_file() else None
    write_report(pre, metrics, statistics_result, execution)
    return metrics, statistics_result


def write_report(pre: Mapping[str, Any], metrics: Mapping[str, Any] | None, stats: Mapping[str, Any] | None, execution: Mapping[str, Any] | None = None) -> None:
    reuse = pre["reuse"]
    lines = [
        "# MMQA locked1341 G0-20 audit report", "",
        f"- Status: {'FINAL COMPLETE' if metrics else 'A-F PREFLIGHT COMPLETE; NO API / NO GOLD'}",
        f"- Selected: {pre['selection']['budget']} = {pre['selection']['forced']} forced + {pre['selection']['risk_ranked']} G0-risk-ranked",
        f"- Reusable outcomes: {reuse['reusable_outcomes']}",
        f"- Genuinely new logical executions: {reuse['genuinely_new_logical_executions']}",
        f"- Overlap Clean-B3 / VC-20 / Random-20 union / A0-A4-A6 union: {reuse['overlap_clean_b3']} / {reuse['overlap_vc20']} / {reuse['overlap_random20_five_seed_union']} / {reuse['overlap_a0_a4_a6_union']}",
        "- Forced-first protocol audit: PASS (unavailable-answer/state fail-safe; independently reused by the historical general-only risk ablation).",
        "- Locked Gold used for scoring/selection/reuse audit: NO",
        "- Provider-side max_tokens=4096 enforcement confirmed: NO (only the frozen client argument is recorded).",
        "- Frozen protocol deviation: NO", "",
    ]
    if execution:
        lines.extend([
            "## Execution", "",
            f"- New success / terminal / unresolved: {execution['success']} / {execution['terminal']} / {execution['unresolved']}", "",
        ])
    if metrics and stats:
        g = metrics["g0_20"]
        c = stats["g0_20_vs_clean_b3"]
        i = stats["g0_20_vs_initial"]
        selected_status = metrics["selected_outcome_status"]
        lines.extend([
            "## Final endpoint", "",
            f"- Selected outcomes success / terminal / unresolved: {selected_status.get('success', 0)} / {selected_status.get('keep_initial_terminal', 0)} / 0",
            f"- EM / token F1: {g['em_percent']:.6f}% / {g['f1_percent']:.6f}%",
            f"- W2R / R2W / Net: {g['migration_from_initial']['wrong_to_right']} / {g['migration_from_initial']['right_to_wrong']} / {g['migration_from_initial']['net']:+d}",
            f"- Delta vs Initial (EM / F1): {g['delta_em_vs_initial_pp']:+.6f} / {g['delta_token_f1_vs_initial_pp']:+.6f} pp",
            f"- Delta vs Clean-B3 (EM / F1): {g['delta_em_vs_clean_b3_pp']:+.6f} / {g['delta_token_f1_vs_clean_b3_pp']:+.6f} pp",
            f"- G0 vs Clean-B3 exact McNemar p: {c['em']['mcnemar_exact_two_sided_p']:.9g}",
            f"- G0 minus Clean-B3 EM 95% CI: [{c['em']['paired_bootstrap']['ci95_low_pp']:+.6f}, {c['em']['paired_bootstrap']['ci95_high_pp']:+.6f}] pp",
            f"- G0 minus Clean-B3 F1 95% CI: [{c['token_f1']['paired_bootstrap']['ci95_low_pp']:+.6f}, {c['token_f1']['paired_bootstrap']['ci95_high_pp']:+.6f}] pp",
            f"- G0 vs Initial exact McNemar p: {i['em']['mcnemar_exact_two_sided_p']:.9g}",
        ])
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute-pilot", action="store_true")
    group.add_argument("--execute-remaining", action="store_true")
    group.add_argument("--finalize", action="store_true")
    parser.add_argument("--request-interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight:
        result = preflight()
        print(json.dumps(result["reuse"], ensure_ascii=False, indent=2))
    elif args.execute_pilot:
        print(json.dumps(execute("pilot", args.request_interval), ensure_ascii=False, indent=2))
    elif args.execute_remaining:
        print(json.dumps(execute("remaining", args.request_interval), ensure_ascii=False, indent=2))
    else:
        metrics, stats = finalize()
        print(json.dumps({"metrics": metrics, "statistics": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
