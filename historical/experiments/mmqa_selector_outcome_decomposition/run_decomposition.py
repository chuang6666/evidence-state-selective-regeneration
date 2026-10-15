from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
G0_SCRIPT = ROOT / "experiments/mmqa_g0_20_endpoint/run_g0_20_endpoint.py"
SEEDS = (20260822, 20260823, 20260824, 20260825, 20260826)


def import_g0():
    spec = importlib.util.spec_from_file_location("g0_endpoint", G0_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def decomposition(method: str, subset: str, qids: set[str], initial_scores, final_scores, final_summary=None):
    ordered = sorted(qids)
    wrong = sum(initial_scores[qid]["em"] == 0 for qid in ordered)
    correct = len(ordered) - wrong
    w2r = sum(initial_scores[qid]["em"] == 0 and final_scores[qid]["em"] == 1 for qid in ordered)
    r2w = sum(initial_scores[qid]["em"] == 1 and final_scores[qid]["em"] == 0 for qid in ordered)
    repair_yield = w2r / wrong if wrong else 0.0
    harm_rate = r2w / correct if correct else 0.0
    row = {
        "method": method,
        "subset": subset,
        "selected": len(ordered),
        "selected_initial_wrong": wrong,
        "selected_initial_correct": correct,
        "selected_error_rate_fraction": f"{wrong}/{len(ordered)}",
        "selected_error_rate_percent": pct(wrong, len(ordered)),
        "w2r": w2r,
        "r2w": r2w,
        "net": w2r - r2w,
        "repair_yield_fraction": f"{w2r}/{wrong}",
        "repair_yield_percent": 100.0 * repair_yield,
        "failure_to_repair_fraction": f"{wrong - w2r}/{wrong}",
        "failure_to_repair_percent": 100.0 * (1.0 - repair_yield) if wrong else 0.0,
        "harm_rate_fraction": f"{r2w}/{correct}",
        "harm_rate_percent": 100.0 * harm_rate,
        "net_utility_per_selected_fraction": f"{w2r - r2w}/{len(ordered)}",
        "net_utility_per_selected_percent": pct(w2r - r2w, len(ordered)),
        "w2r_per_selected_fraction": f"{w2r}/{len(ordered)}",
        "w2r_per_selected_percent": pct(w2r, len(ordered)),
        "r2w_per_selected_fraction": f"{r2w}/{len(ordered)}",
        "r2w_per_selected_percent": pct(r2w, len(ordered)),
        "final_em_percent": "" if final_summary is None else final_summary["em_percent"],
        "final_token_f1_percent": "" if final_summary is None else final_summary["f1_percent"],
    }
    if wrong + correct != len(ordered):
        raise RuntimeError(f"Initial correctness partition failed: {method}/{subset}")
    return row


def mean_sd_rows(rows: list[dict]):
    numeric = [
        "selected", "selected_initial_wrong", "selected_initial_correct",
        "selected_error_rate_percent", "w2r", "r2w", "net",
        "repair_yield_percent", "failure_to_repair_percent", "harm_rate_percent",
        "net_utility_per_selected_percent", "w2r_per_selected_percent",
        "r2w_per_selected_percent", "final_em_percent", "final_token_f1_percent",
    ]
    templates = []
    for label, fn in (("mean", statistics.mean), ("sample_sd", statistics.stdev)):
        row = {key: "" for key in rows[0]}
        row["method"] = f"Random-20 {label}"
        row["subset"] = "total"
        for key in numeric:
            row[key] = fn(float(item[key]) for item in rows)
        templates.append(row)
    return templates


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g0 = import_g0()
    official = g0.import_official_evaluator()
    initial_rows = g0.index_rows(g0.load_jsonl(g0.LOCKED_BASE), "Frozen Initial")
    gold_all = g0.index_rows(g0.load_jsonl(g0.GOLD), "official Gold")
    all_qids = sorted(initial_rows)
    gold = {qid: gold_all[qid]["answers"] for qid in all_qids}
    initial_answers = {qid: g0.answer_from(official, initial_rows[qid], "draft_answer") for qid in all_qids}
    initial_scores = g0.score_answers(official, initial_answers, gold, all_qids)

    configs = {
        "G0-20": {
            "selection": ROOT / "experiments/mmqa_g0_20_endpoint/g0_20_selected_qids.jsonl",
            "final": ROOT / "experiments/mmqa_g0_20_endpoint/g0_20_final_predictions.jsonl",
            "forced_key": "forced_case",
        },
        "Clean-B3": {
            "selection": ROOT / "outputs/mmqa_clean_risk_v2_final_full_dev_locked_selection_v1/locked1341_clean_b3_top269_selection_v1.jsonl",
            "final": ROOT / "outputs/mmqa_clean_risk_v2_final_prediction_merge_before_gold_v1/clean_v2_locked1341_final_predictions_frozen_before_gold_v1.jsonl",
            "forced_key": "forced_into_high_risk_budget",
        },
        "Instability-20": {
            "selection": ROOT / "experiments/mmqa_instability20_endpoint/selected_qids.jsonl",
            "final": ROOT / "experiments/mmqa_instability20_endpoint/final_predictions.jsonl",
            "forced_key": "forced_case",
        },
        "VC-20": {
            "selection": ROOT / "outputs/mmqa_vc20_selection_freeze_v1_1/vc20_qid_tiebreak_selected269_frozen_v1.jsonl",
            "final": ROOT / "outputs/mmqa_vc20_matched_regeneration_union_qwen37plus_v1/frozen_variant_predictions_before_eval_v1/mmqa_vc20_qid_ascending_locked1341_predictions_frozen_before_eval_v1.jsonl",
            "forced_key": None,
        },
    }

    selections = {}
    forced_sets = {}
    final_scores = {}
    final_summaries = {}
    main_rows = []
    for method, cfg in configs.items():
        selection_rows = read_jsonl(cfg["selection"])
        selections[method] = {str(row["qid"]) for row in selection_rows}
        if len(selections[method]) != 269:
            raise RuntimeError(f"Frozen selection count changed for {method}")
        key = cfg["forced_key"]
        forced_sets[method] = {str(row["qid"]) for row in selection_rows if key and bool(row.get(key))}
        final_rows = g0.index_rows(g0.load_jsonl(cfg["final"]), f"{method} final")
        answers = {qid: g0.answer_from(official, final_rows[qid], "final_answer") for qid in all_qids}
        final_scores[method] = g0.score_answers(official, answers, gold, all_qids)
        final_summaries[method] = g0.summarize(final_scores[method], all_qids)
        main_rows.append(decomposition(method, "total", selections[method], initial_scores, final_scores[method], final_summaries[method]))

    random_rows = []
    for seed in SEEDS:
        path = ROOT / f"outputs/mmqa_v3_random20_matched_qwen37plus_v1/frozen_seed_predictions_before_eval_v1/mmqa_v3_locked1341_random20_seed_{seed}_predictions_frozen_before_eval_v1.jsonl"
        rows = g0.index_rows(g0.load_jsonl(path), f"Random-20 seed {seed}")
        selected = {qid for qid, row in rows.items() if str(row.get("frozen_action", "")) == "PLAIN_REGEN"}
        if len(selected) != 269:
            raise RuntimeError(f"Random-20 seed {seed} selected {len(selected)}, expected 269")
        answers = {qid: g0.answer_from(official, rows[qid], "final_answer") for qid in all_qids}
        scores = g0.score_answers(official, answers, gold, all_qids)
        summary = g0.summarize(scores, all_qids)
        row = decomposition(f"Random-20 seed={seed}", "total", selected, initial_scores, scores, summary)
        random_rows.append(row)
    table_rows = random_rows + mean_sd_rows(random_rows) + main_rows
    write_csv(OUT / "selector_decomposition_table.csv", table_rows)

    forced_rows = []
    for method in ("G0-20", "Clean-B3", "Instability-20"):
        forced = forced_sets[method]
        ranked = selections[method] - forced
        if len(forced) != 6 or len(ranked) != 263:
            raise RuntimeError(f"Forced/ranked partition changed for {method}: {len(forced)}/{len(ranked)}")
        for label, subset in (("forced", forced), ("ranked", ranked), ("total", selections[method])):
            forced_rows.append(decomposition(method, label, subset, initial_scores, final_scores[method], final_summaries[method] if label == "total" else None))
    write_csv(OUT / "forced_vs_ranked_decomposition.csv", forced_rows)

    pairs = (
        ("Clean-B3", "G0-20"),
        ("Clean-B3", "Instability-20"),
        ("Clean-B3", "VC-20"),
        ("Instability-20", "VC-20"),
    )
    overlap_rows = []
    for left, right in pairs:
        intersection = selections[left] & selections[right]
        union = selections[left] | selections[right]
        overlap_rows.append({
            "left_method": left,
            "right_method": right,
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard_similarity": len(intersection) / len(union),
            "left_only_count": len(selections[left] - selections[right]),
            "right_only_count": len(selections[right] - selections[left]),
        })
    write_csv(OUT / "selector_overlap_table.csv", overlap_rows)

    def unique_table(comparator: str, output_name: str):
        common = selections["Clean-B3"] & selections[comparator]
        clean_only = selections["Clean-B3"] - selections[comparator]
        comparator_only = selections[comparator] - selections["Clean-B3"]
        rows = [
            decomposition("Clean-B3", "overlap", common, initial_scores, final_scores["Clean-B3"]),
            decomposition(comparator, "overlap", common, initial_scores, final_scores[comparator]),
            decomposition("Clean-B3", "clean_only", clean_only, initial_scores, final_scores["Clean-B3"]),
            decomposition(comparator, "comparator_only", comparator_only, initial_scores, final_scores[comparator]),
        ]
        write_csv(OUT / output_name, rows)
        return rows

    clean_instability = unique_table("Instability-20", "clean_vs_instability_unique_decomposition.csv")
    clean_vc = unique_table("VC-20", "clean_vs_vc_unique_decomposition.csv")

    def find(method, rows=main_rows):
        return next(row for row in rows if row["method"] == method)

    clean, gzero, instability, vc = (find(name) for name in ("Clean-B3", "G0-20", "Instability-20", "VC-20"))
    random_mean, random_sd = mean_sd_rows(random_rows)
    report = f"""# MMQA selector–outcome decomposition diagnostic

## Audit boundary

- Purely offline post-hoc diagnostic; no provider API call, regeneration, training, reselection, or endpoint modification.
- Gold was used only for offline correctness and frozen-outcome decomposition.
- Frozen selected manifests and final predictions were read without modification.

## Main findings

- Clean-B3: selected error rate {clean['selected_error_rate_percent']:.2f}%, repair yield {clean['repair_yield_percent']:.2f}%, harm rate {clean['harm_rate_percent']:.2f}%, Net {clean['net']:+d}.
- G0-20: selected error rate {gzero['selected_error_rate_percent']:.2f}%, repair yield {gzero['repair_yield_percent']:.2f}%, harm rate {gzero['harm_rate_percent']:.2f}%, Net {gzero['net']:+d}.
- Instability-20: selected error rate {instability['selected_error_rate_percent']:.2f}%, repair yield {instability['repair_yield_percent']:.2f}%, harm rate {instability['harm_rate_percent']:.2f}%, Net {instability['net']:+d}.
- VC-20: selected error rate {vc['selected_error_rate_percent']:.2f}%, repair yield {vc['repair_yield_percent']:.2f}%, harm rate {vc['harm_rate_percent']:.2f}%, Net {vc['net']:+d}.
- Random-20 five-seed mean ± sample SD: selected error rate {random_mean['selected_error_rate_percent']:.2f} ± {random_sd['selected_error_rate_percent']:.2f}%, repair yield {random_mean['repair_yield_percent']:.2f} ± {random_sd['repair_yield_percent']:.2f}%, Net {random_mean['net']:.2f} ± {random_sd['net']:.2f}.

## Required diagnostic interpretation

Clean-B3 and G0-20 have nearly identical endpoints; their detailed concentration/yield trade-off is reported in `selector_decomposition_table.csv`. Relative to Instability-20, the +20 Net gap is almost entirely localized to unique selections: overlap contributes +1 Net to Instability-20, while Instability-only contributes +46 versus Clean-only +27. Relative to VC-20, overlap has identical +30 Net, whereas VC-only contributes +72 versus Clean-only +17. VC-20's advantage combines higher error concentration and substantially higher repair yield; it does not arise from harm avoidance because its harm rate is higher.

## Bounded conclusion (≤300 Chinese characters)

Clean-B3 在 locked1341 上仍能集中错误，但错误率低于 Instability-20 与 VC-20，且 Repair Yield 明显更低。Instability-20 的 +20 Net 主要来自其独有样本更高的 W2R；VC-20 的 +55 Net 同时伴随更高错误集中度和更高 Repair Yield，尽管其 Harm Rate 更高。因此差距主要与排序迁移及当前冻结再生成器下的干预效用相关，不能据此认定 Evidence-State 模块失效或获得因果性 repairability 结论。

## Decision answers

- A. Clean-B3 primary issue: **mixed** (limited locked-domain error-risk generalization plus low repair yield; not high harmful-intervention rate).
- B. Support for adding an intervention-utility / repairability-aware module: **PARTIALLY** (supported as a prospective research direction, not as a post-hoc modification).
- C. Modify the current short-paper method body: **NO**.
- D. Better suited as a second innovation point for the master's thesis: **YES**.
"""
    (OUT / "diagnostic_audit_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
