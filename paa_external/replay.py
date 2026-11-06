"""Replay the four PAA external comparisons from frozen score projections."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from evidence_state_repro.mmqa_endpoint_historical import best_score
from evidence_state_repro.offline import ranking
from evidence_state_repro.replay import verify_manifest
from evidence_state_repro.stats_mmqa_historical import exact_mcnemar_p, paired_bootstrap_delta


def read_index(path):
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = row["qid"]
            if qid in result:
                raise ValueError(f"Duplicate QID in {path.name}: {qid}")
            result[qid] = row
    return result


def check(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, context):
    for key, value in actual.items():
        target = expected[key]
        if isinstance(value, dict):
            compare(value, target, f"{context}.{key}")
        else:
            check(math.isclose(value, target, rel_tol=0, abs_tol=1e-12),
                  f"{context}.{key}: {value} != {target}")


def dev_summary(dev, baseline, method, expected):
    rows = []
    check(set(dev) == set(baseline) and len(dev) == 1096, "Dev1096 QID coverage differs")
    for qid, row in dev.items():
        frozen = row["methods"][method]
        if "heldout_fold" in frozen:
            check(frozen["heldout_fold"] == baseline[qid]["fold"], f"OOF fold mismatch: {qid}")
        rows.append({"qid": qid, "scores": {method: frozen["risk"]},
                     "initial_wrong_label": baseline[qid]["initial_wrong_label"],
                     "saved_selected": {method: frozen["selected"]}})
    result = ranking(rows, method)
    compare({key: value for key, value in result.items() if key != "variant"},
            expected, method)
    return result


def selection_key(qid, score):
    # The six original forced cases lead. Suff has a second, unavailable tier.
    tier = 0 if score["forced"] else (1 if score["status"] == "terminal_unavailable" else 2)
    return tier, -score["risk"] if tier == 2 else 0, qid


def locked_summary(locked, method, expected, run_statistics):
    n = len(locked)
    check(n == 1341, "locked1341 QID coverage differs")
    qids = sorted(locked)
    scores = {q: locked[q]["methods"][method] for q in qids}
    order = sorted(qids, key=lambda q: selection_key(q, scores[q]))
    check(sum(row["forced"] for row in scores.values()) == 6, f"{method}: forced count")
    check(all(scores[q]["rank"] == rank for rank, q in enumerate(order, 1)),
          f"{method}: frozen ranking mismatch")
    check(all(scores[q]["selected"] == (i < 269) for i, q in enumerate(order)),
          f"{method}: frozen 269-case membership mismatch")

    initial = [locked[q]["initial"]["em"] for q in qids]
    em = [scores[q]["em"] for q in qids]
    f1 = [scores[q]["f1"] for q in qids]
    w2r = sum(a == 0 and b == 1 for a, b in zip(initial, em))
    r2w = sum(a == 1 and b == 0 for a, b in zip(initial, em))
    check(all((scores[q]["em"], scores[q]["f1"]) ==
              (locked[q]["initial"]["em"], locked[q]["initial"]["f1"]) for q in order[269:]),
          f"{method}: unexpected unselected changes")
    errors = sum(1 - locked[q]["initial"]["em"] for q in order[:269])
    result = {"n": n, "em_count": sum(em), "em": mean(em), "f1": mean(f1),
              "wrong_to_right": w2r, "right_to_wrong": r2w, "net": w2r - r2w,
              "selected_N": 269, "selected_initial_errors": errors}
    compare(result, expected["point"], method)
    result["repair_yield"] = w2r / errors

    if run_statistics:
        b3_em = [locked[q]["b3"]["em"] for q in qids]
        b3_f1 = [locked[q]["b3"]["f1"] for q in qids]
        better = sum(a > b for a, b in zip(em, b3_em))
        worse = sum(a < b for a, b in zip(em, b3_em))
        # Reuse the base release's pure paired functions and the external seeds.
        def interval(values, baseline, protocol):
            computed = paired_bootstrap_delta(values, baseline, protocol["seed"])
            computed["repetitions"] = computed.pop("bootstrap_samples")
            return computed
        expected_stats = expected["paired"]
        paired = {
            "em": {"a_better_discordant": better, "b_better_discordant": worse,
                   "mcnemar_exact_two_sided_p": exact_mcnemar_p(better, worse),
                   "paired_bootstrap": interval(em, b3_em, expected_stats["em"]["paired_bootstrap"])},
            "token_f1": {"paired_bootstrap": interval(f1, b3_f1, expected_stats["token_f1"]["paired_bootstrap"])},
        }
        compare(paired, expected_stats, f"{method}.paired")
        result["paired_vs_b3"] = paired
    return result


def rescore(locked, methods, answers_file, references_file):
    predictions = read_index(ROOT / "frozen_predictions/locked1341.jsonl")
    references = read_index(references_file)
    replacements = read_index(answers_file)
    check(set(predictions) == set(references) == set(locked), "Reference QID coverage differs")
    selection_union = {q for q in locked if any(locked[q]["methods"][m]["selected"] for m in methods)}
    check(set(replacements) == selection_union, "Replacement union differs")
    for qid, row in locked.items():
        reference = references[qid]["answers"]
        check(isinstance(reference, list) and reference and all(isinstance(a, str) for a in reference),
              f"Invalid reference members: {qid}")
        for method in methods:
            saved = row["methods"][method]
            answer = (replacements[qid]["methods"][method] if saved["selected"]
                      else predictions[qid]["methods"]["Initial"])
            em, f1 = best_score(answer, reference)
            compare({"em": em, "f1": f1}, saved, f"{method}.{qid}")
    return len(locked) * len(methods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", action="store_true", help="Verify paired intervals and tests.")
    parser.add_argument("--references", type=Path, help="Local canonical locked1341 reference JSONL.")
    parser.add_argument("--answers", type=Path, help="Authorized local selected-answer JSONL.")
    args = parser.parse_args()
    if (args.references is None) != (args.answers is None):
        parser.error("--references and --answers must be supplied together")

    verified = verify_manifest(ROOT)
    dev = read_index(HERE / "dev1096.jsonl")
    locked = read_index(HERE / "locked1341.jsonl")
    baseline = read_index(ROOT / "frozen_scores/dev1096_oof.jsonl")
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    result = {"verified_manifest_files": verified, "dev": {}, "locked": {}}
    for method, row in expected.items():
        result["dev"][method] = dev_summary(dev, baseline, method, row["dev"])
        result["locked"][method] = locked_summary(locked, method, row["locked"], args.statistics)
    if args.answers is not None:
        result["independent_scores_verified"] = rescore(locked, list(expected), args.answers, args.references)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
