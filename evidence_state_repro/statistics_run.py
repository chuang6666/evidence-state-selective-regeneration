"""Explicit offline reconstruction of specified manuscript CIs and tests.

Historical RNGs, sample multiplicity, QID ordering, pair order and directions
are preserved. Does not retrain, regenerate, change selections or overwrite
frozen inputs. Not a claim that every statistic in all SI tables is covered.
"""
import json
import hashlib
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from . import stats_mmqa_historical as mmqa
from . import stats_controlled_historical as controlled
from . import stats_g0_historical as g0
from . import stats_m3_historical as m3
from .stats_risk_projection import fixed_p
from .offline import safe_output, dump
from .replay import load_rows, verify_manifest


def reproduce_statistics(root, output, scores_dir=None):
    root = Path(root).resolve()
    verify_manifest(root)
    expected = json.loads((root / "provenance/statistics_expectations.json").read_text(encoding="utf-8"))
    score_root = Path(scores_dir).resolve() if scores_dir else root / "frozen_scores"
    if scores_dir:
        receipt = json.loads((score_root / "rescoring_validation.json").read_text(encoding="utf-8"))
        if receipt.get("status") != "PASS_ANSWER_LEVEL_RESCORING":
            raise ValueError("Rescoring validation required")
        for filename, expected_hash in receipt["score_files_sha256"].items():
            if hashlib.sha256((score_root / filename).read_bytes()).hexdigest() != expected_hash:
                raise ValueError("Rescored score file changed: " + filename)
    results = {}

    def add(name, bounds, details, p=None):
        print("Checking " + name, file=sys.stderr, flush=True)
        saved = expected[name]
        match = all(math.isclose(a, b, abs_tol=1e-8, rel_tol=0) for a, b in zip(bounds, saved["ci95_pp"]))
        if "mcnemar_p" in saved:
            match = match and p is not None and math.isclose(p, saved["mcnemar_p"], rel_tol=1e-10, abs_tol=1e-300)
        if not match:
            raise AssertionError(name + ": differs from frozen CI/test; no original values changed; got=" + repr(bounds))
        results[name] = {"ci95_pp": [float(x) for x in bounds], "matches_frozen": match,
                         "details": details, **({"mcnemar_p": p} if p is not None else {})}

    locked = sorted(load_rows(score_root / "locked1341.jsonl"), key=lambda r: r["qid"])

    def vector(method, metric):
        return [float(r["methods"][method][metric]) for r in locked]

    pairs = (("B3_minus_Initial", "B3-Refit", "Initial", mmqa.paired_bootstrap_delta),
             ("B3_minus_VC20", "B3-Refit", "VC-20", mmqa.paired_bootstrap_delta),
             ("G0_minus_B3", "G0-20", "B3-Refit", g0.paired_bootstrap),
             ("Instability_minus_B3", "Instability-20", "B3-Refit", g0.paired_bootstrap))
    for name, left, right, function in pairs:
        for metric, seed in (("em", 20260828), ("f1", 20260829)):
            a, b = vector(left, metric), vector(right, metric)
            computed = function(a, b, seed)
            p = mmqa.exact_mcnemar_p(sum(x > y for x, y in zip(a, b)), sum(y > x for x, y in zip(a, b))) if metric == "em" else None
            add(name + "/" + metric, [computed["ci95_low_pp"], computed["ci95_high_pp"]], computed, p)
    random_methods = sorted(m for m in locked[0]["methods"] if m.startswith("Random-"))
    if len(random_methods) != 5:
        raise ValueError("Five fixed Random assignments required")
    for metric, seed in (("em", 20260828), ("f1", 20260829)):
        a = vector("B3-Refit", metric)
        b = [statistics.mean(float(r["methods"][m][metric]) for m in random_methods) for r in locked]
        computed = mmqa.paired_bootstrap_delta(a, b, seed)
        add("B3_minus_RandomMean/" + metric, [computed["ci95_low_pp"], computed["ci95_high_pp"]], computed)

    data = sorted(load_rows(score_root / "dev1096_controlled.jsonl"), key=lambda r: r["qid"])
    for left, right in (("A4", "A0"), ("A6", "A0"), ("A6", "A4")):
        for metric, seed in (("em", 260817), ("f1", 260818)):
            a = [r["methods"][left][metric] for r in data]
            b = [r["methods"][right][metric] for r in data]
            computed = controlled.paired_bootstrap_delta(a, b, seed)
            p = mmqa.exact_mcnemar_p(sum(x > y for x, y in zip(a, b)), sum(y > x for x, y in zip(a, b))) if metric == "em" else None
            add("controlled_" + left + "_vs_" + right + "/" + metric, computed["ci95_percentage_points"], computed, p)

    data = sorted(load_rows(score_root / "m3vqa13125.jsonl"), key=lambda r: r["qid"])
    for name, right in (("M3_B3_minus_Initial", "Initial"), ("M3_B3_minus_Random", "Random-20")):
        delta = np.array([r["methods"]["B3-Refit"]["accuracy"] - r["methods"][right]["accuracy"] for r in data], dtype=float)
        computed = m3.paired_bootstrap_ci(delta)
        add(name + "/accuracy", [computed["delta_accuracy_pp_95ci_lower"], computed["delta_accuracy_pp_95ci_upper"]], computed)

    formal = sorted(load_rows(root / "frozen_scores/dev1096_oof.jsonl"), key=lambda r: r["qid"])
    explanatory = sorted(load_rows(root / "frozen_scores/dev1096_explanatory_oof.jsonl"), key=lambda r: r["qid"])
    if [r["qid"] for r in formal] != [r["qid"] for r in explanatory]:
        raise ValueError("OOF population mismatch")
    labels = np.array([r["initial_wrong_label"] for r in formal], dtype=int)
    ranks = np.arange(len(formal))
    score = {"B3": np.array([r["scores"]["CLEAN_B3"] for r in formal]),
             "G0": np.array([r["scores"]["G0"] for r in formal])}
    rng = np.random.default_rng(260817)
    diffs = np.empty(10000)
    for j in range(10000):
        ix = rng.integers(0, len(formal), size=len(formal))
        diffs[j] = fixed_p(ranks[ix], labels[ix], score["B3"][ix]) - fixed_p(ranks[ix], labels[ix], score["G0"][ix])
    add("OOF_B3_minus_G0/P20", np.percentile(diffs, [2.5, 97.5]) * 100,
        {"replicates": 10000, "seed": 260817, "reranking": True, "refitting": False})
    rng = np.random.default_rng(260817)  # one continuous stream across the original five-pair order
    score = {v: np.array([r["scores"][v] for r in explanatory]) for v in explanatory[0]["scores"]}
    for left, right in (("A4", "A1"), ("A4", "A2"), ("A5", "A3"), ("S1", "A4"), ("A6", "S2")):
        for j in range(10000):
            ix = rng.integers(0, len(formal), size=len(formal))
            diffs[j] = fixed_p(ranks[ix], labels[ix], score[left][ix]) - fixed_p(ranks[ix], labels[ix], score[right][ix])
        add("component_" + left + "_minus_" + right + "/P20", np.percentile(diffs, [2.5, 97.5]) * 100,
            {"replicates": 10000, "seed": 260817, "continuous_original_pair_order": True})
    output = safe_output(root, output)
    result = {"status": "PASS_SPECIFIED_MANUSCRIPT_CIS_AND_TESTS", "metric_CI_checks": len(results),
              "numpy_version": np.__version__, "comparisons": results,
              "scope": "Specified main-text endpoint CIs/tests, OOF B3-G0 P20 CI and five component P20 CIs, not every supplementary statistic.",
              "score_source": "answer_level_rescored" if scores_dir else "frozen_saved_scores",
              "model_fits": 0, "generation_calls": 0}
    dump(output / "specified_statistics.json", result)
    return result
