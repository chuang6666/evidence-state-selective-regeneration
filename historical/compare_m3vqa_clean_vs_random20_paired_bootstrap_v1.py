from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(r"E:\RAG_Projects\MultimodalEvidenceChain")

CLEAN_DETAILS = (
    ROOT
    / "outputs"
    / "m3vqa_clean_v2_one_time_official_gold_eval_v1"
    / "m3vqa_clean_v2_one_time_official_gold_eval_details_v1.jsonl"
)

RANDOM_DETAILS = (
    ROOT
    / "outputs"
    / "m3vqa_random20_seed20260822_one_time_official_gold_eval_v1"
    / "m3vqa_random20_seed20260822_official_eval_per_qid_v1.jsonl"
)

CLEAN_EVAL_SCRIPT = (
    ROOT
    / "evaluate_m3vqa_clean_v2_one_time_official_gold_v1.py"
)

OUTDIR = (
    ROOT
    / "outputs"
    / "m3vqa_clean_vs_random20_seed20260822_paired_bootstrap_v1"
)

SUMMARY = (
    OUTDIR
    / "m3vqa_clean_vs_random20_paired_bootstrap_summary_v1.json"
)

EXPECTED_N = 13_125
EXPECTED_REPS = 20_000
EXPECTED_SEED = 20_260_823

EXPECTED_CLEAN_ACC = 0.31162652
EXPECTED_RANDOM_ACC = 0.30348404815282

EPS = 1e-8


class ProtocolError(RuntimeError):
    pass


def load_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="strict",
    ) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            row = json.loads(line)

            if not isinstance(row, dict):
                raise ProtocolError(
                    f"{path.name}:{line_no}: not object"
                )

            rows.append(row)

    return rows


def index_unique(rows, label):
    out = {}

    for row in rows:
        qid = str(row.get("qid", "")).strip()

        if not qid:
            raise ProtocolError(
                f"{label}: missing qid"
            )

        if qid in out:
            raise ProtocolError(
                f"{label}: duplicate qid={qid}"
            )

        out[qid] = row

    return out


def load_clean_eval_module():
    spec = importlib.util.spec_from_file_location(
        "clean_eval_frozen",
        CLEAN_EVAL_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise ProtocolError(
            "cannot import frozen Clean evaluator"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_rows = load_jsonl(
        CLEAN_DETAILS
    )

    random_rows = load_jsonl(
        RANDOM_DETAILS
    )

    clean = index_unique(
        clean_rows,
        "clean",
    )

    random20 = index_unique(
        random_rows,
        "random20",
    )

    if (
        len(clean) != EXPECTED_N
        or len(random20) != EXPECTED_N
    ):
        raise ProtocolError(
            "population count mismatch"
        )

    if set(clean) != set(random20):
        raise ProtocolError(
            "Clean/Random QID sets differ"
        )

    qids = sorted(clean)

    clean_scores = np.asarray(
        [
            float(
                clean[qid][
                    "clean_v2_official_accuracy"
                ]
            )
            for qid in qids
        ],
        dtype=np.float64,
    )

    random_scores = np.asarray(
        [
            float(
                random20[qid][
                    "random20_official_accuracy"
                ]
            )
            for qid in qids
        ],
        dtype=np.float64,
    )

    clean_acc = float(
        np.mean(clean_scores)
    )

    random_acc = float(
        np.mean(random_scores)
    )

    delta = (
        clean_scores
        - random_scores
    )

    point_delta_pp = float(
        np.mean(delta) * 100.0
    )

    if (
        abs(
            clean_acc
            - EXPECTED_CLEAN_ACC
        )
        > EPS
    ):
        raise ProtocolError(
            f"Clean Accuracy mismatch: "
            f"{clean_acc}"
        )

    if (
        abs(
            random_acc
            - EXPECTED_RANDOM_ACC
        )
        > EPS
    ):
        raise ProtocolError(
            f"Random Accuracy mismatch: "
            f"{random_acc}"
        )

    # Reuse the exact paired bootstrap implementation
    # already frozen in the Clean evaluator.
    frozen = load_clean_eval_module()

    if (
        int(frozen.BOOTSTRAP_REPS)
        != EXPECTED_REPS
    ):
        raise ProtocolError(
            "bootstrap repetitions mismatch"
        )

    if (
        int(frozen.BOOTSTRAP_SEED)
        != EXPECTED_SEED
    ):
        raise ProtocolError(
            "bootstrap seed mismatch"
        )

    bootstrap = (
        frozen.paired_bootstrap_ci(
            delta
        )
    )

    lower = float(
        bootstrap[
            "delta_accuracy_pp_95ci_lower"
        ]
    )

    upper = float(
        bootstrap[
            "delta_accuracy_pp_95ci_upper"
        ]
    )

    if lower > 0:
        interpretation = (
            "CLEAN_SIGNIFICANTLY_HIGHER_"
            "OFFICIAL_ACCURACY"
        )

    elif upper < 0:
        interpretation = (
            "RANDOM_SIGNIFICANTLY_HIGHER_"
            "OFFICIAL_ACCURACY"
        )

    else:
        interpretation = (
            "NUMERICAL_DIFFERENCE_"
            "CI_INCLUDES_ZERO"
        )

    clean_higher_qids = int(
        np.sum(delta > 1e-12)
    )

    random_higher_qids = int(
        np.sum(delta < -1e-12)
    )

    same_score_qids = (
        EXPECTED_N
        - clean_higher_qids
        - random_higher_qids
    )

    result = {
        "status": (
            "M3VQA_CLEAN_VS_RANDOM20_"
            "PAIRED_BOOTSTRAP_V1_COMPLETE"
        ),
        "N": EXPECTED_N,
        "comparison": (
            "Clean-B3 minus Random-20"
        ),
        "Random20_seed": 20260822,
        "bootstrap_repetitions": (
            EXPECTED_REPS
        ),
        "bootstrap_seed": (
            EXPECTED_SEED
        ),
        "Clean_B3_official_accuracy": (
            clean_acc
        ),
        "Random20_official_accuracy": (
            random_acc
        ),
        "Clean_minus_Random_pp": (
            point_delta_pp
        ),
        "paired_official_accuracy_bootstrap":
            bootstrap,
        "CI_excludes_zero": (
            lower > 0.0
            or upper < 0.0
        ),
        "direction": interpretation,
        "per_qid_score_comparison": {
            "Clean_higher": (
                clean_higher_qids
            ),
            "Random_higher": (
                random_higher_qids
            ),
            "same_score": (
                same_score_qids
            ),
        },
        "protocol": {
            "Gold_loaded": False,
            "API_calls": 0,
            "predictions_modified": False,
            "selection_modified": False,
            "scores_recomputed": False,
            "uses_frozen_per_qid_official_scores":
                True,
        },
    }

    SUMMARY.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("-" * 120)

    print(
        "M3-VQA CLEAN-B3 vs RANDOM-20 "
        "PAIRED OFFICIAL-ACCURACY BOOTSTRAP"
    )

    print("-" * 120)

    print(
        f"N:                         "
        f"{EXPECTED_N}"
    )

    print(
        f"Clean-B3 Accuracy:         "
        f"{clean_acc:.12%}"
    )

    print(
        f"Random-20 Accuracy:        "
        f"{random_acc:.12%}"
    )

    print(
        f"Clean - Random:            "
        f"{point_delta_pp:+.6f} pp"
    )

    print(
        f"Bootstrap reps:            "
        f"{EXPECTED_REPS}"
    )

    print(
        f"Bootstrap seed:            "
        f"{EXPECTED_SEED}"
    )

    print("")
    print(
        "Bootstrap result:"
    )

    print(
        json.dumps(
            bootstrap,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("")
    print(
        f"Clean higher QIDs:         "
        f"{clean_higher_qids}"
    )

    print(
        f"Random higher QIDs:        "
        f"{random_higher_qids}"
    )

    print(
        f"Same-score QIDs:           "
        f"{same_score_qids}"
    )

    print("")
    print(
        f"Interpretation:            "
        f"{interpretation}"
    )

    print(
        f"Summary:                   "
        f"{SUMMARY}"
    )

    print("-" * 120)

    print(
        "STATUS = "
        "M3VQA_CLEAN_VS_RANDOM20_"
        "PAIRED_BOOTSTRAP_V1_COMPLETE"
    )

    print("-" * 120)


if __name__ == "__main__":
    main()