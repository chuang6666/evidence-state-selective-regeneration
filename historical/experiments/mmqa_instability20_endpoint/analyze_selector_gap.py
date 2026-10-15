import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
G0_SCRIPT = ROOT / "experiments/mmqa_g0_20_endpoint/run_g0_20_endpoint.py"


def load_module():
    spec = importlib.util.spec_from_file_location("g0_endpoint", G0_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_set(path):
    return {str(row["qid"]) for row in load_jsonl(path)}


def final_answers(g0, official, path):
    rows = g0.index_rows(g0.load_jsonl(path), path.name)
    return {
        qid: g0.answer_from(official, row, "final_answer")
        for qid, row in rows.items()
    }


def subset_stats(name, selected, initial_scores, final_scores):
    selected = sorted(selected)
    initial_wrong = sum(initial_scores[qid]["em"] == 0 for qid in selected)
    initial_right = len(selected) - initial_wrong
    w2r = sum(
        initial_scores[qid]["em"] == 0 and final_scores[qid]["em"] == 1
        for qid in selected
    )
    r2w = sum(
        initial_scores[qid]["em"] == 1 and final_scores[qid]["em"] == 0
        for qid in selected
    )
    return {
        "name": name,
        "selected": len(selected),
        "initial_wrong_selected": initial_wrong,
        "initial_error_concentration_percent": 100 * initial_wrong / len(selected),
        "initial_right_selected": initial_right,
        "wrong_to_right": w2r,
        "right_to_wrong": r2w,
        "net": w2r - r2w,
        "repair_rate_among_selected_initial_wrong_percent": 100 * w2r / initial_wrong,
        "harm_rate_among_selected_initial_right_percent": (
            100 * r2w / initial_right if initial_right else 0.0
        ),
        "net_per_selected_percent": 100 * (w2r - r2w) / len(selected),
    }


def main():
    g0 = load_module()
    official = g0.import_official_evaluator()
    initial_rows = g0.index_rows(g0.load_jsonl(g0.LOCKED_BASE), "initial")
    gold_all = g0.index_rows(g0.load_jsonl(g0.GOLD), "gold")
    qids = sorted(initial_rows)
    gold = {qid: gold_all[qid]["answers"] for qid in qids}
    initial_answers = {
        qid: g0.answer_from(official, initial_rows[qid], "draft_answer")
        for qid in qids
    }
    initial_scores = g0.score_answers(official, initial_answers, gold, qids)

    paths = {
        "clean_b3": {
            "selection": ROOT / "outputs/mmqa_clean_risk_v2_final_full_dev_locked_selection_v1/locked1341_clean_b3_top269_selection_v1.jsonl",
            "final": ROOT / "outputs/mmqa_clean_risk_v2_final_prediction_merge_before_gold_v1/clean_v2_locked1341_final_predictions_frozen_before_gold_v1.jsonl",
        },
        "instability20": {
            "selection": ROOT / "experiments/mmqa_instability20_endpoint/selected_qids.jsonl",
            "final": ROOT / "experiments/mmqa_instability20_endpoint/final_predictions.jsonl",
        },
        "vc20": {
            "selection": ROOT / "outputs/mmqa_vc20_selection_freeze_v1_1/vc20_qid_tiebreak_selected269_frozen_v1.jsonl",
            "final": ROOT / "outputs/mmqa_vc20_matched_regeneration_union_qwen37plus_v1/frozen_variant_predictions_before_eval_v1/mmqa_vc20_qid_ascending_locked1341_predictions_frozen_before_eval_v1.jsonl",
        },
    }

    selections = {}
    scores = {}
    summaries = {}
    for name, item in paths.items():
        selections[name] = selected_set(item["selection"])
        answers = final_answers(g0, official, item["final"])
        scores[name] = g0.score_answers(official, answers, gold, qids)
        summaries[name] = subset_stats(
            name, selections[name], initial_scores, scores[name]
        )

    overlaps = {}
    names = list(paths)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            common = selections[left] & selections[right]
            overlaps[f"{left}__{right}"] = {
                "intersection": len(common),
                f"{left}_only": len(selections[left] - selections[right]),
                f"{right}_only": len(selections[right] - selections[left]),
                "jaccard": len(common) / len(selections[left] | selections[right]),
            }

    region_effects = {}
    for external in ("instability20", "vc20"):
        clean_only = selections["clean_b3"] - selections[external]
        external_only = selections[external] - selections["clean_b3"]
        common = selections["clean_b3"] & selections[external]
        region_effects[f"clean_b3_vs_{external}"] = {
            "common_evaluated_under_clean": subset_stats(
                "common_clean", common, initial_scores, scores["clean_b3"]
            ),
            "common_evaluated_under_external": subset_stats(
                "common_external", common, initial_scores, scores[external]
            ),
            "clean_only": subset_stats(
                "clean_only", clean_only, initial_scores, scores["clean_b3"]
            ),
            "external_only": subset_stats(
                "external_only", external_only, initial_scores, scores[external]
            ),
        }

    print(json.dumps({
        "selector_summaries": summaries,
        "selection_overlaps": overlaps,
        "region_effects": region_effects,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
