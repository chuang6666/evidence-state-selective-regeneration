from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parent

JOB_FILE = (
    ROOT / "outputs" / "mmqa_natural_transition_state_jobs_v1"
    / "mmqa_natural_transition_state_extraction_jobs_v1.jsonl"
)

OUT_DIR = ROOT / "outputs" / "mmqa_transition_topology_tfidf_oof_v1"
OOF_FILE = OUT_DIR / "mmqa_transition_topology_tfidf_oof_predictions_v1.jsonl"
SUMMARY_FILE = OUT_DIR / "mmqa_transition_topology_tfidf_oof_v1_summary.json"

TRANSITIONS = ["T1", "T2", "T3", "T4"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def label_set(row: dict[str, Any]) -> set[str]:
    raw = row.get("required_transitions")
    if not isinstance(raw, list):
        return set()
    return {str(x).strip() for x in raw if str(x).strip() in TRANSITIONS}


def make_features() -> FeatureUnion:
    # Lightweight, fully local topology baseline.
    # Word n-grams capture operators/semantic phrases; char n-grams help
    # with entity names and phrasing variation in a small dataset.
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=30000,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=30000,
            ),
        ),
    ])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    per_t = {}
    f1s = []
    for j, t in enumerate(TRANSITIONS):
        p, r, f1, support = precision_recall_fscore_support(
            y_true[:, j],
            y_pred[:, j],
            average="binary",
            zero_division=0,
        )
        per_t[t] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "positive_support": int(np.sum(y_true[:, j] == 1)),
            "predicted_positive": int(np.sum(y_pred[:, j] == 1)),
        }
        f1s.append(float(f1))

    exact = float(np.mean(np.all(y_true == y_pred, axis=1)))
    return {
        "exact_topology_match_T1_T4": exact,
        "macro_f1_T1_T4": float(np.mean(f1s)),
        "micro_f1_T1_T4": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_transition": per_t,
    }


def main():
    if not JOB_FILE.is_file():
        raise FileNotFoundError(JOB_FILE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(JOB_FILE)
    rows = [r for r in rows if r.get("qid") and r.get("question")]
    rows.sort(key=lambda x: str(x["qid"]))

    qids = [str(r["qid"]) for r in rows]
    questions = [str(r["question"]) for r in rows]

    y = np.zeros((len(rows), len(TRANSITIONS)), dtype=np.int64)
    for i, row in enumerate(rows):
        labs = label_set(row)
        for j, t in enumerate(TRANSITIONS):
            y[i, j] = 1 if t in labs else 0

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_prob = np.zeros_like(y, dtype=np.float64)
    fold_id = np.full(len(rows), -1, dtype=np.int64)

    for fold, (train_idx, test_idx) in enumerate(kf.split(questions)):
        X_train_text = [questions[i] for i in train_idx]
        X_test_text = [questions[i] for i in test_idx]

        vectorizer = make_features()
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        for j, t in enumerate(TRANSITIONS):
            y_train = y[train_idx, j]

            # Safety for degenerate folds, although current MMQA labels are not degenerate.
            if len(np.unique(y_train)) < 2:
                oof_prob[test_idx, j] = float(y_train[0])
                continue

            clf = LogisticRegression(
                C=2.0,
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42 + fold,
            )
            clf.fit(X_train, y_train)
            oof_prob[test_idx, j] = clf.predict_proba(X_test)[:, 1]

        fold_id[test_idx] = fold

    raw_pred = (oof_prob >= 0.5).astype(np.int64)

    # Schema-constrained view:
    # In the frozen T1-T5 schema used by this project, T3 and T4 are paired
    # for the current MMQA formal mappings (parallel branches + aggregation).
    constrained_pred = raw_pred.copy()
    branch_or_agg = np.maximum(constrained_pred[:, 2], constrained_pred[:, 3])
    constrained_pred[:, 2] = branch_or_agg
    constrained_pred[:, 3] = branch_or_agg

    raw_metrics = evaluate(y, raw_pred)
    constrained_metrics = evaluate(y, constrained_pred)

    output_rows = []
    for i, row in enumerate(rows):
        pred_raw = [t for j, t in enumerate(TRANSITIONS) if raw_pred[i, j] == 1] + ["T5"]
        pred_cons = [t for j, t in enumerate(TRANSITIONS) if constrained_pred[i, j] == 1] + ["T5"]
        gold_proxy = [t for j, t in enumerate(TRANSITIONS) if y[i, j] == 1] + ["T5"]

        output_rows.append({
            "qid": qids[i],
            "fold": int(fold_id[i]),
            "question": questions[i],
            "offline_required_transitions_proxy": gold_proxy,
            "predicted_required_transitions_raw": pred_raw,
            "predicted_required_transitions_schema_constrained": pred_cons,
            "probabilities": {
                t: float(oof_prob[i, j])
                for j, t in enumerate(TRANSITIONS)
            },
            "inference_features": "question_only",
        })

    with OOF_FILE.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_counts = {
        t: int(np.sum(y[:, j]))
        for j, t in enumerate(TRANSITIONS)
    }
    raw_pred_counts = {
        t: int(np.sum(raw_pred[:, j]))
        for j, t in enumerate(TRANSITIONS)
    }
    constrained_pred_counts = {
        t: int(np.sum(constrained_pred[:, j]))
        for j, t in enumerate(TRANSITIONS)
    }

    summary = {
        "n": len(rows),
        "folds": 5,
        "features": "question-only TF-IDF word(1,2)+char(3,5)",
        "labels": (
            "T1-T4 required_transitions from offline MMQA structural mapping. "
            "Used only as training/evaluation supervision; never inference input."
        ),
        "label_positive_counts": label_counts,
        "raw_predicted_positive_counts": raw_pred_counts,
        "schema_constrained_predicted_positive_counts": constrained_pred_counts,
        "raw_oof_metrics": raw_metrics,
        "schema_constrained_oof_metrics": constrained_metrics,
        "guardrail": (
            "Every reported prediction is out-of-fold. Each qid is predicted by a model "
            "that was not trained on that qid. Inference input is question text only. "
            "No official_type, required_transitions, gold answer, gold evidence, draft answer, "
            "or counterfactual supervision is used as an input feature."
        ),
        "api_model_calls": 0,
        "status": "MMQA_TRANSITION_TOPOLOGY_TFIDF_OOF_V1_COMPLETE",
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 120)
    print("MMQA — TRANSITION TOPOLOGY TF-IDF 5-FOLD OOF V1")
    print("=" * 120)
    print("N:", len(rows))
    print("Label positive counts:", label_counts)
    print("-" * 120)
    print("RAW OOF")
    print("Exact topology match:", f"{raw_metrics['exact_topology_match_T1_T4']:.4f}")
    print("Macro-F1 T1-T4:", f"{raw_metrics['macro_f1_T1_T4']:.4f}")
    print("Micro-F1 T1-T4:", f"{raw_metrics['micro_f1_T1_T4']:.4f}")
    for t, x in raw_metrics["per_transition"].items():
        print(
            f"{t}: P={x['precision']:.4f} "
            f"R={x['recall']:.4f} F1={x['f1']:.4f} "
            f"support={x['positive_support']} pred={x['predicted_positive']}"
        )

    print("-" * 120)
    print("SCHEMA-CONSTRAINED OOF")
    print("Exact topology match:", f"{constrained_metrics['exact_topology_match_T1_T4']:.4f}")
    print("Macro-F1 T1-T4:", f"{constrained_metrics['macro_f1_T1_T4']:.4f}")
    print("Micro-F1 T1-T4:", f"{constrained_metrics['micro_f1_T1_T4']:.4f}")
    for t, x in constrained_metrics["per_transition"].items():
        print(
            f"{t}: P={x['precision']:.4f} "
            f"R={x['recall']:.4f} F1={x['f1']:.4f} "
            f"support={x['positive_support']} pred={x['predicted_positive']}"
        )

    print("-" * 120)
    print("API/model calls = 0")
    print("OOF predictions:", OOF_FILE)
    print("Summary:", SUMMARY_FILE)
    print("STATUS = MMQA_TRANSITION_TOPOLOGY_TFIDF_OOF_V1_COMPLETE")


if __name__ == "__main__":
    main()
