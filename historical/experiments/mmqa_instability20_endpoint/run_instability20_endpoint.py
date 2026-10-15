#!/usr/bin/env python3
"""Selection, matched regeneration, and endpoint evaluation for Instability-20."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
G0_SCRIPT = ROOT / "experiments/mmqa_g0_20_endpoint/run_g0_20_endpoint.py"
ENTROPY = OUT / "entropy_scores.jsonl"
REPLICATE_SUMMARY = OUT / "replicate_summary.json"
LOCKED_BASE = ROOT / "outputs/mmqa_v3_heldout1341_pre_natural_v1/mmqa_v3_heldout1341_initial_base_sanitized_v1.jsonl"
CLEAN_PROTOCOL = ROOT / "outputs/mmqa_clean_risk_v2_final_full_dev_locked_selection_v1/final_full_dev_locked_selection_protocol_v1.json"
LOCKED_HIGH_OLD = ROOT / "outputs/mmqa_v3_heldout1341_frozen_verifier_localization_risk_v1/heldout1341_frozen_top20_high_risk_269_v1.jsonl"
LOCKED_LOW_OLD = ROOT / "outputs/mmqa_v3_heldout1341_frozen_verifier_localization_risk_v1/heldout1341_frozen_low_risk_1072_v1.jsonl"
HISTORY = ROOT / "outputs/mmqa_clean_risk_v2_final_full_dev_locked_selection_v1/historical_exact_plain_regeneration_registry_v1.jsonl"
V2_PROTOCOL = ROOT / "outputs/mmqa_v2_protocol_freeze_20260819/mmqa_v2_frozen_protocol.json"
ORIGINAL_RUNNER = ROOT / "run_mmqa_v3_heldout1341_final_plain_regen_qwen37plus_v1.py"
PROMPT_CONTRACT = ROOT / "run_mmqa_v2_fresh_pilot40_b_qwen37plus.py"
JOB_BUILDER = ROOT / "build_and_freeze_mmqa_v3_b0_only15_exact_protocol_jobs_v1_1.py"
BASE_RANDOM_RUNNER = ROOT / "run_mmqa_v3_random20_matched_qwen37plus_v1.py"
PROVIDER = ROOT / "run_mmqa_transition_evidence_recompose_abc_pilot_qwen37plus_v1.py"
HARD_TIMEOUT_RUNNER = ROOT / "run_mmqa_v3_b0_only15_exact_protocol_qwen37plus_v1_3.py"
GOLD = ROOT / "data/raw/mmqa_official/MMQA_dev.jsonl.gz"

CLEAN_FINAL = ROOT / "outputs/mmqa_clean_risk_v2_final_prediction_merge_before_gold_v1/clean_v2_locked1341_final_predictions_frozen_before_gold_v1.jsonl"
G0_FINAL = ROOT / "experiments/mmqa_g0_20_endpoint/g0_20_final_predictions.jsonl"
VC_FINAL = ROOT / "outputs/mmqa_vc20_matched_regeneration_union_qwen37plus_v1/frozen_variant_predictions_before_eval_v1/mmqa_vc20_qid_ascending_locked1341_predictions_frozen_before_eval_v1.jsonl"
RANDOM_DIR = ROOT / "outputs/mmqa_v3_random20_matched_qwen37plus_v1/frozen_seed_predictions_before_eval_v1"
RANDOM_SEEDS = (20260822, 20260823, 20260824, 20260825, 20260826)

SELECTED = OUT / "selected_qids.jsonl"
REUSE = OUT / "reuse_audit.jsonl"
REGEN_MANIFEST = OUT / "regeneration_manifest.jsonl"
REGEN_CACHE = OUT / "regeneration_new_execution_cache.jsonl"
REGEN_SUMMARY = OUT / "regeneration_execution_summary.json"
FINAL = OUT / "final_predictions.jsonl"
METRICS = OUT / "metrics.json"
STATS = OUT / "statistical_comparisons.json"
PREFLIGHT = OUT / "endpoint_preflight.json"
REPORT = OUT / "final_audit_report.md"

N = 1341
BUDGET = 269
FORCED_N = 6
RANKED_N = 263


class ProtocolError(RuntimeError):
    pass


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


g0 = import_file("instability20_g0_helpers", G0_SCRIPT)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preflight() -> dict[str, Any]:
    rep = g0.load_json(REPLICATE_SUMMARY)
    if rep.get("status") != "REPLICATES_COMPLETE" or rep.get("successful_replicates") != 6675:
        raise ProtocolError("Replicate generation is not complete")
    entropy_rows = g0.index_rows(g0.load_jsonl(ENTROPY), "entropy")
    if len(entropy_rows) != 1335 or any(row.get("entropy_natural_log") is None for row in entropy_rows.values()):
        raise ProtocolError("Expected 1,335 finite entropy scores")
    base = g0.index_rows(g0.load_jsonl(LOCKED_BASE), "locked base")
    clean_protocol = g0.load_json(CLEAN_PROTOCOL)
    forced_order = list(clean_protocol["locked_selection"]["forced_order"])
    if len(forced_order) != FORCED_N or any(qid in entropy_rows for qid in forced_order):
        raise ProtocolError("Forced/scored partition drift")
    ranked = sorted(entropy_rows, key=lambda qid: (-float(entropy_rows[qid]["entropy_natural_log"]), qid))
    selected_order = forced_order + ranked[:RANKED_N]
    if len(selected_order) != BUDGET or len(set(selected_order)) != BUDGET:
        raise ProtocolError("Selection budget drift")

    history = g0.index_rows(g0.load_jsonl(HISTORY), "history")
    conflicts = [qid for qid in selected_order if qid in history and history[qid].get("duplicate_outcome_conflict")]
    if conflicts:
        raise ProtocolError(f"Reusable outcome conflicts: {conflicts}")
    reusable = {
        qid for qid in selected_order
        if qid in history and history[qid].get("legal_reuse") is True
        and history[qid].get("same_exact_plain_regeneration_protocol") is True
    }
    historical_plan = {
        **g0.index_rows(g0.load_jsonl(LOCKED_LOW_OLD), "historical low"),
        **g0.index_rows(g0.load_jsonl(LOCKED_HIGH_OLD), "historical high"),
    }
    if set(historical_plan) != set(base):
        raise ProtocolError("Historical route plan coverage drift")
    builder = import_file("instability20_job_builder", JOB_BUILDER)
    original = import_file("instability20_original_regen", ORIGINAL_RUNNER)
    prompt = import_file("instability20_prompt_contract", PROMPT_CONTRACT)
    protocol = g0.load_json(V2_PROTOCOL)
    ranks = {qid: index for index, qid in enumerate(selected_order, 1)}
    jobs = []
    for position, qid in enumerate(selected_order, 1):
        job = builder.make_job(qid, position, ranks[qid], base, historical_plan, original, prompt, protocol)
        job["instability20_selection_rank"] = ranks[qid]
        job["reuse_legal"] = qid in reusable
        job["outcome_source"] = history[qid]["chosen_source"] if qid in reusable else "NEW_REQUIRED"
        jobs.append(job)

    selected_rows = [{
        "qid": qid, "selection_rank": rank,
        "selection_basis": "FORCED_FAIL_SAFE" if qid in forced_order else "ENTROPY_DESC_QID_ASC",
        "forced_case": qid in forced_order,
        "entropy_natural_log": None if qid in forced_order else entropy_rows[qid]["entropy_natural_log"],
        "reusable_outcome": qid in reusable,
        "gold_used": False,
    } for rank, qid in enumerate(selected_order, 1)]
    reuse_rows = [{
        "qid": qid, "selection_rank": rank, "reuse_legal": qid in reusable,
        "outcome_source": history.get(qid, {}).get("chosen_source", "NEW_REQUIRED"),
        "generation_status": history.get(qid, {}).get("generation_status", "NOT_EXECUTED"),
        "duplicate_outcome_conflict": False,
    } for rank, qid in enumerate(selected_order, 1)]
    g0.write_jsonl(SELECTED, selected_rows)
    g0.write_jsonl(REUSE, reuse_rows)
    g0.write_jsonl(REGEN_MANIFEST, jobs)
    result = {
        "status": "INSTABILITY20_ENDPOINT_PREFLIGHT_PASS",
        "selection": {"total": BUDGET, "forced": FORCED_N, "instability_ranked": RANKED_N, "tie_break": "qid_ascending"},
        "reuse": {"reused": len(reusable), "new_executions": BUDGET - len(reusable)},
        "regeneration_contract": {
            "model": "qwen3.7-plus", "temperature": 0.1, "thinking": True,
            "thinking_budget": 2048,
            "input": "question + exact original-order Top-6; T1 eligible raw images only (max 3)",
            "failure_fallback": "frozen existing policy",
        },
        "gold_used": False, "api_calls": 0, "frozen_protocol_deviation": False,
    }
    write_json(PREFLIGHT, result)
    return result


def execute(request_interval: float) -> dict[str, Any]:
    pre = g0.load_json(PREFLIGHT)
    jobs_all = g0.index_rows(g0.load_jsonl(REGEN_MANIFEST), "regeneration manifest")
    new_jobs = [row for row in jobs_all.values() if row.get("reuse_legal") is False]
    initial = g0.index_rows(g0.load_jsonl(LOCKED_BASE), "initial")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    runner = importlib.import_module("run_mmqa_v3_b0_only15_exact_protocol_qwen37plus_v1_3")
    base_runner = import_file("instability20_matched_runner", BASE_RANDOM_RUNNER)
    provider = import_file("instability20_qwen_provider", PROVIDER)
    runner.configure_direct_certifi_tls()
    jobs = g0.index_rows(new_jobs, "new regeneration jobs")
    completed, unresolved = base_runner.latest_cache(REGEN_CACHE, jobs)
    api_key, key_source = provider.load_api_key()
    last_request = 0.0
    for position, job in enumerate(new_jobs, 1):
        qid = g0.safe(job["qid"])
        if qid in completed:
            continue
        if last_request:
            wait = request_interval - (time.monotonic() - last_request)
            if wait > 0:
                time.sleep(wait)
        record = runner.call_one_with_hard_timeout(
            base_runner, BASE_RANDOM_RUNNER, PROMPT_CONTRACT, PROVIDER,
            api_key, job, g0.safe(initial[qid].get("draft_answer")),
        )
        last_request = time.monotonic()
        g0.append_jsonl(REGEN_CACHE, record)
        print(f"[{position}/{len(new_jobs)}] {qid} -> {record.get('status')}", flush=True)
        completed, unresolved = base_runner.latest_cache(REGEN_CACHE, jobs)
        write_json(REGEN_SUMMARY, {
            "status": "RUNNING", "api_key_source": key_source, "new_jobs": len(new_jobs),
            "completed": len(completed), "success": sum(r.get("status") == "success" for r in completed.values()),
            "terminal": sum(r.get("status") == "keep_initial_terminal" for r in completed.values()),
            "unresolved": len(unresolved), "remaining": len(new_jobs) - len(completed),
            "frozen_protocol_deviation": False,
        })
    completed, unresolved = base_runner.latest_cache(REGEN_CACHE, jobs)
    result = {
        "status": "COMPLETE" if len(completed) == len(new_jobs) and not unresolved else "INCOMPLETE",
        "api_key_source": key_source, "new_jobs": len(new_jobs), "completed": len(completed),
        "success": sum(r.get("status") == "success" for r in completed.values()),
        "terminal": sum(r.get("status") == "keep_initial_terminal" for r in completed.values()),
        "unresolved": len(unresolved), "remaining": len(new_jobs) - len(completed),
        "frozen_protocol_deviation": False,
    }
    write_json(REGEN_SUMMARY, result)
    return result


def answers_and_scores(official, path: Path, gold: Mapping[str, list[str]], qids: list[str]):
    rows = g0.index_rows(g0.load_jsonl(path), str(path))
    answers = {qid: g0.answer_from(official, rows[qid], "final_answer") for qid in qids}
    return g0.score_answers(official, answers, gold, qids)


def finalize() -> tuple[dict[str, Any], dict[str, Any]]:
    pre = g0.load_json(PREFLIGHT)
    selected = g0.index_rows(g0.load_jsonl(SELECTED), "selected")
    jobs = g0.index_rows(g0.load_jsonl(REGEN_MANIFEST), "jobs")
    history = g0.index_rows(g0.load_jsonl(HISTORY), "history")
    initial = g0.index_rows(g0.load_jsonl(LOCKED_BASE), "initial")
    base_runner = import_file("instability20_finalize_runner", BASE_RANDOM_RUNNER)
    new_jobs = {qid: row for qid, row in jobs.items() if row.get("reuse_legal") is False}
    completed, unresolved = base_runner.latest_cache(REGEN_CACHE, new_jobs)
    if unresolved or set(completed) != set(new_jobs):
        raise ProtocolError(f"Regeneration incomplete: {len(completed)}/{len(new_jobs)}, unresolved={len(unresolved)}")
    outcomes = {}
    for qid in selected:
        if qid in completed:
            outcomes[qid] = completed[qid]
        elif qid in history and history[qid].get("legal_reuse") is True:
            outcomes[qid] = history[qid]
        else:
            raise ProtocolError(f"No resolved selected outcome: {qid}")
    final_rows = []
    for qid in sorted(initial):
        if qid in selected:
            row = outcomes[qid]
            answer = g0.safe(row.get("effective_final_answer"))
            status = g0.safe(row.get("generation_status", row.get("status")))
            source = g0.safe(jobs[qid].get("outcome_source"))
            failure = g0.safe(row.get("failure_class"))
        else:
            answer = g0.safe(initial[qid].get("draft_answer"))
            status, source, failure = "not_selected_keep_initial", "FROZEN_INITIAL", ""
        final_rows.append({"qid": qid, "final_answer": answer, "selected_instability20": qid in selected,
                           "generation_status": status, "outcome_source": source, "failure_class": failure,
                           "gold_used_in_inference": False})
    g0.write_jsonl(FINAL, final_rows)

    official = g0.import_official_evaluator()
    gold_rows = g0.index_rows(g0.load_jsonl(GOLD), "official Gold")
    qids = sorted(initial)
    gold = {qid: official.extract_gold_answers(gold_rows[qid]) for qid in qids}
    inst_scores = answers_and_scores(official, FINAL, gold, qids)
    initial_answers = {qid: g0.answer_from(official, initial[qid], "draft_answer") for qid in qids}
    initial_scores = g0.score_answers(official, initial_answers, gold, qids)
    clean_scores = answers_and_scores(official, CLEAN_FINAL, gold, qids)
    g0_scores = answers_and_scores(official, G0_FINAL, gold, qids)
    vc_scores = answers_and_scores(official, VC_FINAL, gold, qids)
    random_scores = []
    for seed in RANDOM_SEEDS:
        path = RANDOM_DIR / f"mmqa_v3_locked1341_random20_seed_{seed}_predictions_frozen_before_eval_v1.jsonl"
        random_scores.append(answers_and_scores(official, path, gold, qids))

    summaries = {
        "instability20": g0.summarize(inst_scores, qids), "initial": g0.summarize(initial_scores, qids),
        "clean_b3": g0.summarize(clean_scores, qids), "g0_20": g0.summarize(g0_scores, qids),
        "vc_20": g0.summarize(vc_scores, qids),
        "random20_seeds": [g0.summarize(scores, qids) for scores in random_scores],
    }
    random_mean_em = statistics.mean(row["em_percent"] for row in summaries["random20_seeds"])
    random_mean_f1 = statistics.mean(row["f1_percent"] for row in summaries["random20_seeds"])
    inst = summaries["instability20"]
    deltas = {
        "initial": {"em_pp": inst["em_percent"] - summaries["initial"]["em_percent"], "f1_pp": inst["f1_percent"] - summaries["initial"]["f1_percent"]},
        "random20_five_seed_mean": {"em_pp": inst["em_percent"] - random_mean_em, "f1_pp": inst["f1_percent"] - random_mean_f1},
        "g0_20": {"em_pp": inst["em_percent"] - summaries["g0_20"]["em_percent"], "f1_pp": inst["f1_percent"] - summaries["g0_20"]["f1_percent"]},
        "clean_b3": {"em_pp": inst["em_percent"] - summaries["clean_b3"]["em_percent"], "f1_pp": inst["f1_percent"] - summaries["clean_b3"]["f1_percent"]},
        "vc_20": {"em_pp": inst["em_percent"] - summaries["vc_20"]["em_percent"], "f1_pp": inst["f1_percent"] - summaries["vc_20"]["f1_percent"]},
    }
    migration = g0.migration(initial_scores, inst_scores, qids)
    selected_status = Counter(g0.safe(row.get("generation_status", row.get("status"))) for row in outcomes.values())
    metrics = {"status": "COMPLETE", "summaries": summaries, "random20_mean": {"em_percent": random_mean_em, "f1_percent": random_mean_f1},
               "deltas": deltas, "migration_from_initial": migration,
               "selection": pre["selection"], "reuse": pre["reuse"],
               "selected_outcome_status": dict(selected_status), "frozen_protocol_deviation": False}
    stats = {"status": "COMPLETE", "instability20_vs_clean_b3": g0.pairwise("instability20_minus_clean_b3", inst_scores, clean_scores, qids),
             "instability20_vs_initial": g0.pairwise("instability20_minus_initial", inst_scores, initial_scores, qids)}
    write_json(METRICS, metrics)
    write_json(STATS, stats)
    c = stats["instability20_vs_clean_b3"]
    lines = ["# Instability-20 final audit report", "", "- Status: FINAL COMPLETE", f"- Selected: 269 = 6 forced + 263 instability-ranked",
             f"- Reused / new: {pre['reuse']['reused']} / {pre['reuse']['new_executions']}",
             f"- EM / token F1: {inst['em_percent']:.6f}% / {inst['f1_percent']:.6f}%",
             f"- W2R / R2W / Net: {migration['wrong_to_right']} / {migration['right_to_wrong']} / {migration['net']:+d}",
             f"- McNemar p vs Clean-B3: {c['em']['mcnemar_exact_two_sided_p']:.9g}",
             f"- EM 95% CI vs Clean-B3: [{c['em']['paired_bootstrap']['ci95_low_pp']:+.6f}, {c['em']['paired_bootstrap']['ci95_high_pp']:+.6f}] pp",
             f"- F1 95% CI vs Clean-B3: [{c['token_f1']['paired_bootstrap']['ci95_low_pp']:+.6f}, {c['token_f1']['paired_bootstrap']['ci95_high_pp']:+.6f}] pp",
             "- Frozen protocol deviation: NO"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--finalize", action="store_true")
    parser.add_argument("--request-interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
    elif args.execute:
        print(json.dumps(execute(args.request_interval), ensure_ascii=False, indent=2))
    else:
        metrics, stats = finalize()
        print(json.dumps({"metrics": metrics, "statistics": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
