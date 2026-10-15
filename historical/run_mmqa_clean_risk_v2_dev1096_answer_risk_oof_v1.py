#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Freeze and execute Clean Risk V2 Dev1096 five-fold OOF risk evaluation.

Two explicit modes are provided:

  --freeze-protocol   Validate and freeze labels/splits/protocol only.
  --run-all           Revalidate the frozen protocol, then run six OOF variants.

No API, raw Gold, locked-1341, final full-Dev fit, or feature mutation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
EXPECTED_N = 1096
N_SPLITS = 5
SPLIT_SEED = 260817
MODEL_SEED = 260817
TOP_FRACTION = 0.20
TOP_K = math.ceil(EXPECTED_N * TOP_FRACTION)
EXPECTED_WRONG = 680
EXPECTED_CORRECT = 416
EXPECTED_FEATURE_CONTRACT_SHA = (
    "9BACED6295AE3F31E023DD9E91395ABC567861AB4DA040A5E0D337B344ED14B1"
)
EXPECTED_FEATURE_MATRIX_SHA = (
    "FA579EE4028FFB2F07EDA664B3547D688733E0022677048783223E763330DCF7"
)

FEATURE_DIR = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_feature_generation_v1"
)
FEATURE_MATRIX_FILE = FEATURE_DIR / "dev1096_clean_risk_feature_matrix_v1.jsonl"
FEATURE_SUMMARY_FILE = FEATURE_DIR / "feature_generation_summary_v1.json"
FEATURE_HASH_FILE = FEATURE_DIR / "SHA256SUMS.json"
CONTRACT_FILE = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_feature_contract_v1"
    / "clean_risk_v2_feature_contract_v1.json"
)

# This already-saved Dev artifact is the sole label/split source.  Only qid,
# fold, and initial_wrong_label are extracted; old risk scores are ignored.
SAVED_LABEL_SPLIT_SOURCE = (
    ROOT / "outputs" / "mmqa_transition_answer_risk_oof_v1"
    / "risk_oof_predictions_v1.jsonl"
)
ORIGINAL_RISK_RUNNER = ROOT / "run_mmqa_transition_answer_risk_oof_v1_1.py"

OUT_DIR = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_answer_risk_oof_v1"
)
PROTOCOL_FILE = OUT_DIR / "clean_v2_oof_evaluation_protocol_v1.json"
LABEL_SPLIT_FILE = OUT_DIR / "dev1096_saved_answer_error_labels_and_shared_folds_v1.jsonl"
PREFLIGHT_FILE = OUT_DIR / "clean_v2_oof_preflight_v1.json"
PROTOCOL_HASH_FILE = OUT_DIR / "PROTOCOL_SHA256SUMS.json"

COMBINED_PREDICTIONS_FILE = OUT_DIR / "dev1096_clean_v2_all_variants_oof_predictions_v1.jsonl"
METRICS_FILE = OUT_DIR / "clean_v2_oof_metrics_v1.json"
FOLD_STATS_FILE = OUT_DIR / "clean_v2_oof_fold_statistics_v1.json"
COEFFICIENTS_FILE = OUT_DIR / "clean_v2_oof_coefficient_summary_v1.json"
SELECTION_FILE = OUT_DIR / "clean_v2_oof_variant_selection_v1.json"
SUMMARY_FILE = OUT_DIR / "clean_v2_oof_summary_v1.json"
HASH_FILE = OUT_DIR / "SHA256SUMS.json"

VARIANTS = [
    ("G0", "G0_GENERAL_ONLY"),
    ("CLEAN_B0", "CLEAN_B0_COARSE_GENERAL"),
    ("CLEAN_B1", "CLEAN_B1_GENERAL_T1"),
    ("CLEAN_B2", "CLEAN_B2_GENERAL_T1_T4"),
    ("CLEAN_B3", "CLEAN_B3_GENERAL_T1_T5"),
    ("CLEAN_B4", "CLEAN_B4_T1_T5_ONLY"),
]
VARIANT_KEYS = [key for _, key in VARIANTS]
SHORT_NAME = {key: short for short, key in VARIANTS}
VARIANT_ORDER = {key: index for index, key in enumerate(VARIANT_KEYS)}
EXPECTED_RAW_KEY_COUNTS = {
    "G0_GENERAL_ONLY": 12,
    "CLEAN_B0_COARSE_GENERAL": 25,
    "CLEAN_B1_GENERAL_T1": 34,
    "CLEAN_B2_GENERAL_T1_T4": 58,
    "CLEAN_B3_GENERAL_T1_T5": 65,
    "CLEAN_B4_T1_T5_ONLY": 40,
}
OPTIONAL_SPARSE_KEYS = {
    "LOC.localized_valid_prob",
    "LOC.localized_threshold",
    "LOC.localized_invalid_severity",
    "LOC.localized_transition",
}


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
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def make_model() -> Pipeline:
    # Exact original V1 model family and fixed hyperparameters.
    return Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", LogisticRegression(
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=MODEL_SEED,
        )),
    ])


def class_balance(indices: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    subset = labels[indices]
    wrong = int(subset.sum())
    n = int(len(subset))
    return {
        "n": n,
        "wrong": wrong,
        "correct": n - wrong,
        "error_rate": float(wrong / n),
    }


def deterministic_ranking(qids: list[str], scores: np.ndarray) -> list[int]:
    # Primary descending score; exact-score ties resolved by ascending QID.
    return sorted(range(len(qids)), key=lambda i: (-float(scores[i]), qids[i]))


def global_metrics(qids: list[str], labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    order = deterministic_ranking(qids, scores)
    selected = order[:TOP_K]
    found = int(labels[selected].sum())
    boundary_score = float(scores[selected[-1]])
    equal_boundary = [i for i in order if float(scores[i]) == boundary_score]
    selected_equal_boundary = [i for i in selected if float(scores[i]) == boundary_score]
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "population_error_rate": float(labels.mean()),
        "top20": {
            "fraction": TOP_FRACTION,
            "k": TOP_K,
            "errors_found": found,
            "precision": float(found / TOP_K),
            "recall": float(found / int(labels.sum())),
            "boundary_score": boundary_score,
            "boundary_equal_score_count": len(equal_boundary),
            "boundary_equal_score_selected_count": len(selected_equal_boundary),
            "tie_break": "risk_score_desc_then_qid_ascending",
        },
    }


def fold_metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "risk_min": float(scores.min()),
        "risk_max": float(scores.max()),
        "risk_mean": float(scores.mean()),
        "risk_std": float(scores.std()),
    }


def coefficient_fold_summary(model: Pipeline, top_n: int = 30) -> tuple[dict[str, Any], dict[str, float]]:
    vectorizer: DictVectorizer = model.named_steps["vec"]
    classifier: LogisticRegression = model.named_steps["clf"]
    names = list(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[0]
    pairs = [(name, float(value)) for name, value in zip(names, coefs)]
    positive = sorted(pairs, key=lambda item: (-item[1], item[0]))[:top_n]
    negative = sorted(pairs, key=lambda item: (item[1], item[0]))[:top_n]
    mapping = dict(pairs)
    summary = {
        "feature_dimension": len(names),
        "intercept": float(classifier.intercept_[0]),
        "coefficient_nonzero_count": int(np.count_nonzero(coefs)),
        "coefficient_l1_norm": float(np.abs(coefs).sum()),
        "coefficient_l2_norm": float(np.linalg.norm(coefs)),
        "coefficient_max_abs": float(np.abs(coefs).max()) if len(coefs) else 0.0,
        "top_positive": [{"feature": name, "coefficient": value} for name, value in positive],
        "top_negative": [{"feature": name, "coefficient": value} for name, value in negative],
    }
    return summary, mapping


def validate_protocol_inputs() -> tuple[
    dict[str, Any], list[str], dict[str, dict[str, Any]], list[dict[str, Any]],
    dict[str, list[str]], dict[str, Any]
]:
    contract_sha = sha256_file(CONTRACT_FILE)
    matrix_sha = sha256_file(FEATURE_MATRIX_FILE)
    if contract_sha != EXPECTED_FEATURE_CONTRACT_SHA:
        raise RuntimeError(f"Feature contract SHA drift: {contract_sha}")
    if matrix_sha != EXPECTED_FEATURE_MATRIX_SHA:
        raise RuntimeError(f"Feature matrix SHA drift: {matrix_sha}")

    feature_summary = load_json(FEATURE_SUMMARY_FILE)
    feature_hashes = load_json(FEATURE_HASH_FILE)
    if feature_summary.get("status") != "MMQA_CLEAN_RISK_V2_DEV1096_FEATURE_GENERATION_V1_COMPLETE":
        raise RuntimeError("Upstream feature generation is not complete")
    if feature_summary.get("answer_risk_training") is not False:
        raise RuntimeError("Unexpected upstream answer-risk training flag")
    recorded_matrix_sha = feature_hashes["generated_files"][FEATURE_MATRIX_FILE.name]["sha256"]
    if recorded_matrix_sha != matrix_sha:
        raise RuntimeError("Feature matrix SHA manifest mismatch")

    contract = load_json(CONTRACT_FILE)
    variant_contract = contract.get("variants")
    if not isinstance(variant_contract, dict) or set(variant_contract) != set(VARIANT_KEYS):
        raise RuntimeError("Variant set differs from frozen protocol request")
    raw_key_counts = {
        key: len(variant_contract[key].get("features", [])) for key in VARIANT_KEYS
    }
    if raw_key_counts != EXPECTED_RAW_KEY_COUNTS:
        raise RuntimeError(f"Contract feature counts drift: {raw_key_counts}")
    whitelists = {
        key: list(variant_contract[key]["features"]) for key in VARIANT_KEYS
    }

    matrix_rows = unique_by_qid(load_jsonl(FEATURE_MATRIX_FILE), "Clean V2 feature matrix")
    saved_rows_raw = load_jsonl(SAVED_LABEL_SPLIT_SOURCE)
    saved_rows = unique_by_qid(saved_rows_raw, "saved Dev label/split source")
    if len(matrix_rows) != EXPECTED_N or set(matrix_rows) != set(saved_rows):
        raise RuntimeError(
            f"Feature/label QID mismatch: feature={len(matrix_rows)} labels={len(saved_rows)}"
        )
    qids = sorted(matrix_rows)

    label_split_rows: list[dict[str, Any]] = []
    labels: list[int] = []
    folds: list[int] = []
    ignored_label_source_fields: set[str] = set()
    feature_schema_failures: list[dict[str, Any]] = []
    for qid in qids:
        saved = saved_rows[qid]
        ignored_label_source_fields.update(
            set(saved) - {"qid", "fold", "initial_wrong_label"}
        )
        label = saved.get("initial_wrong_label")
        fold = saved.get("fold")
        if label not in {0, 1} or not isinstance(fold, int) or fold not in range(1, 6):
            raise RuntimeError(f"Invalid saved label/fold for {qid}: {label}/{fold}")
        labels.append(int(label))
        folds.append(int(fold))
        label_split_rows.append({
            "qid": qid,
            "initial_wrong_label": int(label),
            "heldout_fold": int(fold),
        })

        envelope = matrix_rows[qid]
        if set(envelope) != {"qid", "variants"}:
            feature_schema_failures.append({"qid": qid, "reason": "matrix_envelope"})
            continue
        variants = envelope.get("variants")
        if not isinstance(variants, dict) or set(variants) != set(VARIANT_KEYS):
            feature_schema_failures.append({"qid": qid, "reason": "variant_set"})
            continue
        for variant in VARIANT_KEYS:
            features = variants[variant]
            if not isinstance(features, dict):
                feature_schema_failures.append({"qid": qid, "variant": variant, "reason": "not_object"})
                continue
            allowed = set(whitelists[variant])
            actual = set(features)
            if not actual <= allowed:
                feature_schema_failures.append({
                    "qid": qid, "variant": variant,
                    "reason": "outside_whitelist", "keys": sorted(actual - allowed),
                })
            required_emitted = allowed - OPTIONAL_SPARSE_KEYS
            if not required_emitted <= actual:
                feature_schema_failures.append({
                    "qid": qid, "variant": variant,
                    "reason": "required_contract_key_not_emitted",
                    "keys": sorted(required_emitted - actual),
                })
            if "qid" in features or any(value is None for value in features.values()):
                feature_schema_failures.append({
                    "qid": qid, "variant": variant, "reason": "qid_or_null_feature"
                })
    if feature_schema_failures:
        raise RuntimeError(f"Feature schema preflight failed: {feature_schema_failures[:10]}")

    y = np.asarray(labels, dtype=int)
    saved_fold_array = np.asarray(folds, dtype=int)
    if int(y.sum()) != EXPECTED_WRONG or len(y) - int(y.sum()) != EXPECTED_CORRECT:
        raise RuntimeError("Saved label class balance differs from frozen expectation")

    regenerated_folds = np.zeros(EXPECTED_N, dtype=int)
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED
    )
    for fold, (_, validation_indices) in enumerate(
        splitter.split(np.zeros(EXPECTED_N), y), 1
    ):
        regenerated_folds[validation_indices] = fold
    fold_mismatch_indices = np.flatnonzero(regenerated_folds != saved_fold_array)
    if len(fold_mismatch_indices):
        examples = [
            {
                "qid": qids[int(i)],
                "saved_fold": int(saved_fold_array[i]),
                "regenerated_fold": int(regenerated_folds[i]),
            }
            for i in fold_mismatch_indices[:20]
        ]
        raise RuntimeError(f"Saved fold is not the original fixed split: {examples}")

    fold_class_balance = {}
    for fold in range(1, 6):
        va = np.flatnonzero(saved_fold_array == fold)
        tr = np.flatnonzero(saved_fold_array != fold)
        fold_class_balance[str(fold)] = {
            "train": class_balance(tr, y),
            "validation": class_balance(va, y),
        }

    validation = {
        "contract_sha256": contract_sha,
        "matrix_sha256": matrix_sha,
        "label_source_sha256": sha256_file(SAVED_LABEL_SPLIT_SOURCE),
        "original_risk_runner_sha256": sha256_file(ORIGINAL_RISK_RUNNER),
        "qids": len(qids),
        "wrong": int(y.sum()),
        "correct": int(len(y) - y.sum()),
        "saved_vs_regenerated_fold_mismatches": int(len(fold_mismatch_indices)),
        "fold_class_balance": fold_class_balance,
        "feature_schema_failures": len(feature_schema_failures),
        "label_source_fields_used": ["qid", "fold", "initial_wrong_label"],
        "label_source_fields_ignored": sorted(ignored_label_source_fields),
        "raw_key_counts": raw_key_counts,
    }
    return contract, qids, matrix_rows, label_split_rows, whitelists, validation


def freeze_protocol() -> None:
    contract, qids, _, label_split_rows, whitelists, validation = validate_protocol_inputs()
    model_contract = {
        "pipeline": ["DictVectorizer", "LogisticRegression"],
        "dict_vectorizer": {
            "sparse": True,
            "separator": "=",
            "sort": True,
            "dtype": "numpy.float64",
        },
        "logistic_regression": {
            "solver": "liblinear",
            "penalty": "l2",
            "dual": False,
            "C": 1.0,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "class_weight": "balanced",
            "max_iter": 2000,
            "multi_class": "auto/default binary behavior",
            "random_state": MODEL_SEED,
            "tol": 0.0001,
            "warm_start": False,
        },
        "hyperparameter_search": False,
        "final_full_dev_model_fit": False,
    }
    protocol = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_OOF_EVALUATION_PROTOCOL_V1_FROZEN",
        "scope": "six-variant five-fold held-out OOF answer-error risk evaluation only",
        "variants_in_fixed_order": [
            {
                "short_name": short,
                "feature_matrix_key": key,
                "raw_key_count": len(whitelists[key]),
                "features_sha256": hashlib.sha256(
                    json.dumps(whitelists[key], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).hexdigest().upper(),
            }
            for short, key in VARIANTS
        ],
        "feature_contract": {
            "path": str(CONTRACT_FILE),
            "sha256": validation["contract_sha256"],
            "feature_addition_deletion_or_modification_allowed": False,
        },
        "feature_matrix": {
            "path": str(FEATURE_MATRIX_FILE),
            "sha256": validation["matrix_sha256"],
            "rows": EXPECTED_N,
        },
        "labels": {
            "source": str(SAVED_LABEL_SPLIT_SOURCE),
            "source_sha256": validation["label_source_sha256"],
            "fields_used": validation["label_source_fields_used"],
            "raw_gold_opened": False,
            "semantics": "initial_wrong_label: 1=wrong, 0=correct",
            "expected_wrong": EXPECTED_WRONG,
            "expected_correct": EXPECTED_CORRECT,
        },
        "cross_validation": {
            "type": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": SPLIT_SEED,
            "same_qid_split_for_all_variants": True,
            "saved_fold_assignments_must_equal_regenerated_original_split": True,
            "each_qid_heldout_prediction_count": 1,
        },
        "model": model_contract,
        "metrics": {
            "primary_model_selection_metric": "global_top20_error_precision",
            "reported": ["roc_auc", "average_precision", "global_top20_error_precision", "global_top20_error_recall"],
            "top20_population": EXPECTED_N,
            "top20_k": TOP_K,
            "top20_computation": "sort all 1096 OOF risk scores for each variant, then take first 220",
            "ranking": "risk_score_descending",
            "exact_score_tie_break": "qid_ascending",
            "variant_selection_tie_break": [
                "higher_average_precision",
                "higher_roc_auc",
                "lower_raw_contract_key_count",
                "fixed_variant_order_G0_B0_B1_B2_B3_B4",
            ],
        },
        "reporting": {
            "complete_oof_predictions_per_variant": True,
            "fold_statistics": True,
            "class_balance": True,
            "feature_dimensions": True,
            "coefficient_summary": {
                "per_fold_top_positive": 30,
                "per_fold_top_negative": 30,
                "aggregate_across_fold_models": True,
            },
        },
        "prohibitions": {
            "api_calls": 0,
            "raw_gold_opened": False,
            "locked_1341_opened_or_evaluated": False,
            "feature_contract_modified": False,
            "variant_redesign_after_results": False,
            "hyperparameter_search": False,
            "final_full_dev_model_fit": False,
            "v1_modified": False,
        },
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    preflight = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_OOF_PREFLIGHT_PASS",
        "protocol_frozen": True,
        "training_started": False,
        "validation": validation,
        "all_variants_share_saved_fold_assignment": True,
        "feature_matrix_unchanged": True,
        "saved_labels_only": True,
        "raw_gold_opened": False,
        "locked_1341_opened_or_evaluated": False,
        "external_api_calls": 0,
        "final_full_dev_model_fit": False,
        "v1_modified": False,
        "gate": "PASS: six-variant OOF training/evaluation may run under the frozen protocol",
    }

    frozen_write(LABEL_SPLIT_FILE, jsonl_bytes(label_split_rows))
    frozen_write(PROTOCOL_FILE, json_bytes(protocol))
    frozen_write(PREFLIGHT_FILE, json_bytes(preflight))
    protocol_hashes = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_OOF_PROTOCOL_SHA_FROZEN",
        "generated_files": {
            path.name: file_record(path)
            for path in [LABEL_SPLIT_FILE, PROTOCOL_FILE, PREFLIGHT_FILE]
        },
        "read_only_sources": {
            path.name: file_record(path)
            for path in [FEATURE_MATRIX_FILE, CONTRACT_FILE, SAVED_LABEL_SPLIT_SOURCE, ORIGINAL_RISK_RUNNER, Path(__file__).resolve()]
        },
        "training_started": False,
        "raw_gold_opened": False,
        "locked_1341_opened_or_evaluated": False,
        "external_api_calls": 0,
        "final_full_dev_model_fit": False,
        "v1_modified": False,
    }
    frozen_write(PROTOCOL_HASH_FILE, json_bytes(protocol_hashes))

    print("=" * 120)
    print("MMQA CLEAN RISK V2 — DEV1096 OOF EVALUATION PROTOCOL FREEZE V1")
    print("=" * 120)
    print("Protocol/preflight: PASS")
    print("Contract SHA:", validation["contract_sha256"])
    print("Feature matrix SHA:", validation["matrix_sha256"])
    print("QID/label balance:", validation["qids"], validation["wrong"], validation["correct"])
    print("Saved vs regenerated fold mismatches:", validation["saved_vs_regenerated_fold_mismatches"])
    print("Variants/raw key counts:", validation["raw_key_counts"])
    print("Top-20%: global OOF ranking, k=", TOP_K, "tie=qid_ascending")
    print("Training started: NO")
    print("API/Gold/locked/final-full-fit/V1-modified: 0 / 0 / 0 / 0 / 0")
    print("Protocol:", PROTOCOL_FILE)
    print("Protocol SHA:", PROTOCOL_HASH_FILE)
    print("STATUS =", protocol["status"])


def verify_frozen_protocol() -> tuple[
    dict[str, Any], list[str], dict[str, dict[str, Any]], np.ndarray, np.ndarray
]:
    if not all(path.is_file() for path in [PROTOCOL_FILE, LABEL_SPLIT_FILE, PREFLIGHT_FILE, PROTOCOL_HASH_FILE]):
        raise RuntimeError("Protocol is not frozen; run --freeze-protocol first")
    protocol = load_json(PROTOCOL_FILE)
    preflight = load_json(PREFLIGHT_FILE)
    manifest = load_json(PROTOCOL_HASH_FILE)
    if protocol.get("status") != "MMQA_CLEAN_RISK_V2_DEV1096_OOF_EVALUATION_PROTOCOL_V1_FROZEN":
        raise RuntimeError("Protocol status is not frozen")
    if preflight.get("status") != "MMQA_CLEAN_RISK_V2_DEV1096_OOF_PREFLIGHT_PASS":
        raise RuntimeError("Preflight is not PASS")
    for name, record in manifest["generated_files"].items():
        path = OUT_DIR / name
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Frozen protocol artifact changed: {path}")
    if sha256_file(FEATURE_MATRIX_FILE) != protocol["feature_matrix"]["sha256"]:
        raise RuntimeError("Feature matrix changed after protocol freeze")
    if sha256_file(CONTRACT_FILE) != protocol["feature_contract"]["sha256"]:
        raise RuntimeError("Feature contract changed after protocol freeze")

    matrix = unique_by_qid(load_jsonl(FEATURE_MATRIX_FILE), "feature matrix")
    label_rows = unique_by_qid(load_jsonl(LABEL_SPLIT_FILE), "frozen clean label/split snapshot")
    qids = sorted(matrix)
    if len(qids) != EXPECTED_N or set(matrix) != set(label_rows):
        raise RuntimeError("Frozen training input QID mismatch")
    labels = np.asarray([label_rows[qid]["initial_wrong_label"] for qid in qids], dtype=int)
    folds = np.asarray([label_rows[qid]["heldout_fold"] for qid in qids], dtype=int)
    return protocol, qids, matrix, labels, folds


def run_all() -> None:
    protocol, qids, matrix, labels, folds = verify_frozen_protocol()
    variant_scores: dict[str, np.ndarray] = {
        key: np.full(EXPECTED_N, np.nan, dtype=float) for key in VARIANT_KEYS
    }
    assignment_counts: dict[str, np.ndarray] = {
        key: np.zeros(EXPECTED_N, dtype=int) for key in VARIANT_KEYS
    }
    fold_stats: dict[str, list[dict[str, Any]]] = {key: [] for key in VARIANT_KEYS}
    coefficient_stats: dict[str, dict[str, Any]] = {}
    aggregate_coefficients: dict[str, dict[str, list[float]]] = {
        key: defaultdict(list) for key in VARIANT_KEYS
    }

    for variant in VARIANT_KEYS:
        per_fold_coefficients: list[dict[str, Any]] = []
        for fold in range(1, 6):
            validation_indices = np.flatnonzero(folds == fold)
            training_indices = np.flatnonzero(folds != fold)
            train_x = [matrix[qids[int(i)]]["variants"][variant] for i in training_indices]
            validation_x = [matrix[qids[int(i)]]["variants"][variant] for i in validation_indices]
            model = make_model()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(train_x, labels[training_indices])
            scores = model.predict_proba(validation_x)[:, 1]
            variant_scores[variant][validation_indices] = scores
            assignment_counts[variant][validation_indices] += 1

            coefficient_summary, coefficient_map = coefficient_fold_summary(model)
            for feature_name, value in coefficient_map.items():
                aggregate_coefficients[variant][feature_name].append(value)
            per_fold_coefficients.append({
                "fold": fold,
                **coefficient_summary,
            })
            fold_stats[variant].append({
                "fold": fold,
                "train": class_balance(training_indices, labels),
                "validation": class_balance(validation_indices, labels),
                "validation_metrics": fold_metric(labels[validation_indices], scores),
                "feature_dimension": coefficient_summary["feature_dimension"],
                "n_iter": [int(x) for x in model.named_steps["clf"].n_iter_],
                "fit_warning_count": len(caught),
                "fit_warnings": [str(item.message) for item in caught],
            })

        counts = assignment_counts[variant]
        if not np.all(counts == 1):
            bad = np.flatnonzero(counts != 1)
            raise RuntimeError(
                f"OOF assignment count violation in {variant}: "
                f"{[(qids[int(i)], int(counts[i])) for i in bad[:20]]}"
            )
        if not np.isfinite(variant_scores[variant]).all():
            raise RuntimeError(f"Missing/nonfinite OOF scores in {variant}")

        aggregate_rows = []
        for feature_name, values in aggregate_coefficients[variant].items():
            array = np.asarray(values, dtype=float)
            aggregate_rows.append({
                "feature": feature_name,
                "fold_models_present": len(values),
                "coefficient_mean": float(array.mean()),
                "coefficient_std": float(array.std()),
                "coefficient_min": float(array.min()),
                "coefficient_max": float(array.max()),
                "coefficient_mean_abs": float(np.abs(array).mean()),
            })
        positive = sorted(
            aggregate_rows, key=lambda row: (-row["coefficient_mean"], row["feature"])
        )[:50]
        negative = sorted(
            aggregate_rows, key=lambda row: (row["coefficient_mean"], row["feature"])
        )[:50]
        largest_abs = sorted(
            aggregate_rows, key=lambda row: (-row["coefficient_mean_abs"], row["feature"])
        )[:100]
        coefficient_stats[variant] = {
            "short_name": SHORT_NAME[variant],
            "per_fold": per_fold_coefficients,
            "aggregate_unique_vectorized_features": len(aggregate_rows),
            "aggregate_top_positive_mean": positive,
            "aggregate_top_negative_mean": negative,
            "aggregate_top_mean_absolute": largest_abs,
        }

    metrics = {
        variant: {
            "short_name": SHORT_NAME[variant],
            **global_metrics(qids, labels, variant_scores[variant]),
            "feature_dimension_per_fold": [
                item["feature_dimension"] for item in fold_stats[variant]
            ],
            "oof_prediction_count_min": int(assignment_counts[variant].min()),
            "oof_prediction_count_max": int(assignment_counts[variant].max()),
        }
        for variant in VARIANT_KEYS
    }

    # Apply only the pre-frozen selection rule.
    selected_variant = sorted(
        VARIANT_KEYS,
        key=lambda key: (
            -metrics[key]["top20"]["precision"],
            -metrics[key]["average_precision"],
            -metrics[key]["roc_auc"],
            EXPECTED_RAW_KEY_COUNTS[key],
            VARIANT_ORDER[key],
        ),
    )[0]
    ranking = sorted(
        VARIANT_KEYS,
        key=lambda key: (
            -metrics[key]["top20"]["precision"],
            -metrics[key]["average_precision"],
            -metrics[key]["roc_auc"],
            EXPECTED_RAW_KEY_COUNTS[key],
            VARIANT_ORDER[key],
        ),
    )

    # Complete per-variant OOF prediction files and a combined convenience file.
    generated_prediction_files: list[Path] = []
    rank_maps: dict[str, dict[int, int]] = {}
    selected_sets: dict[str, set[int]] = {}
    for variant in VARIANT_KEYS:
        order = deterministic_ranking(qids, variant_scores[variant])
        rank_maps[variant] = {index: rank + 1 for rank, index in enumerate(order)}
        selected_sets[variant] = set(order[:TOP_K])
        rows = [
            {
                "qid": qids[i],
                "heldout_fold": int(folds[i]),
                "initial_wrong_label": int(labels[i]),
                "oof_risk_score": float(variant_scores[variant][i]),
                "oof_prediction_count": int(assignment_counts[variant][i]),
                "global_rank": rank_maps[variant][i],
                "global_top20_selected": i in selected_sets[variant],
            }
            for i in range(EXPECTED_N)
        ]
        path = OUT_DIR / f"dev1096_{SHORT_NAME[variant].lower()}_oof_predictions_v1.jsonl"
        frozen_write(path, jsonl_bytes(rows))
        generated_prediction_files.append(path)

    combined_rows = []
    for i, qid in enumerate(qids):
        combined_rows.append({
            "qid": qid,
            "heldout_fold": int(folds[i]),
            "initial_wrong_label": int(labels[i]),
            "variant_oof_predictions": {
                SHORT_NAME[variant]: {
                    "risk_score": float(variant_scores[variant][i]),
                    "global_rank": rank_maps[variant][i],
                    "global_top20_selected": i in selected_sets[variant],
                    "oof_prediction_count": int(assignment_counts[variant][i]),
                }
                for variant in VARIANT_KEYS
            },
        })
    frozen_write(COMBINED_PREDICTIONS_FILE, jsonl_bytes(combined_rows))

    metrics_output = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_OOF_METRICS_COMPLETE",
        "primary_model_selection_metric": "global_top20_error_precision",
        "top20_k": TOP_K,
        "global_population": EXPECTED_N,
        "variants": metrics,
    }
    fold_output = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_OOF_FOLD_STATISTICS_COMPLETE",
        "shared_fold_assignment": True,
        "class_balance_overall": {
            "n": EXPECTED_N,
            "wrong": int(labels.sum()),
            "correct": int(EXPECTED_N - labels.sum()),
            "error_rate": float(labels.mean()),
        },
        "variants": fold_stats,
    }
    coefficient_output = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_OOF_COEFFICIENT_SUMMARY_COMPLETE",
        "note": "Five held-out fold models per variant; no final full-Dev model was fitted.",
        "variants": coefficient_stats,
    }
    selection_output = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_OOF_VARIANT_SELECTED_BY_FROZEN_RULE",
        "primary_metric": "global_top20_error_precision",
        "selected_variant": selected_variant,
        "selected_short_name": SHORT_NAME[selected_variant],
        "selected_metrics": metrics[selected_variant],
        "ranking": [
            {
                "rank": rank + 1,
                "variant": variant,
                "short_name": SHORT_NAME[variant],
                "top20_error_precision": metrics[variant]["top20"]["precision"],
                "top20_error_recall": metrics[variant]["top20"]["recall"],
                "average_precision": metrics[variant]["average_precision"],
                "roc_auc": metrics[variant]["roc_auc"],
                "raw_contract_key_count": EXPECTED_RAW_KEY_COUNTS[variant],
            }
            for rank, variant in enumerate(ranking)
        ],
        "variant_redesign_after_results": False,
    }
    summary = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_FIVE_FOLD_OOF_ANSWER_RISK_V1_COMPLETE",
        "protocol_status": protocol["status"],
        "qids": EXPECTED_N,
        "wrong": int(labels.sum()),
        "correct": int(EXPECTED_N - labels.sum()),
        "shared_folds": True,
        "each_qid_exactly_one_heldout_prediction_per_variant": all(
            np.all(assignment_counts[variant] == 1) for variant in VARIANT_KEYS
        ),
        "variants": {
            SHORT_NAME[variant]: {
                "roc_auc": metrics[variant]["roc_auc"],
                "average_precision": metrics[variant]["average_precision"],
                "top20_error_precision": metrics[variant]["top20"]["precision"],
                "top20_error_recall": metrics[variant]["top20"]["recall"],
                "top20_errors_found": metrics[variant]["top20"]["errors_found"],
                "feature_dimension_per_fold": metrics[variant]["feature_dimension_per_fold"],
            }
            for variant in VARIANT_KEYS
        },
        "selected_variant_by_frozen_primary_metric": SHORT_NAME[selected_variant],
        "external_api_calls": 0,
        "raw_gold_opened": False,
        "saved_dev_labels_only": True,
        "locked_1341_opened_or_evaluated": False,
        "feature_contract_modified": False,
        "variant_redesigned_after_results": False,
        "final_full_dev_model_fit": False,
        "v1_modified": False,
        "next_step": "STOP_BEFORE_FINAL_FULL_DEV_MODEL_OR_LOCKED_1341_APPLICATION",
    }
    frozen_write(METRICS_FILE, json_bytes(metrics_output))
    frozen_write(FOLD_STATS_FILE, json_bytes(fold_output))
    frozen_write(COEFFICIENTS_FILE, json_bytes(coefficient_output))
    frozen_write(SELECTION_FILE, json_bytes(selection_output))
    frozen_write(SUMMARY_FILE, json_bytes(summary))

    generated = [
        *generated_prediction_files,
        COMBINED_PREDICTIONS_FILE,
        METRICS_FILE,
        FOLD_STATS_FILE,
        COEFFICIENTS_FILE,
        SELECTION_FILE,
        SUMMARY_FILE,
    ]
    hash_manifest = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_OOF_SHA_FROZEN",
        "generated_files": {path.name: file_record(path) for path in generated},
        "frozen_protocol_files": {
            path.name: file_record(path)
            for path in [PROTOCOL_FILE, LABEL_SPLIT_FILE, PREFLIGHT_FILE, PROTOCOL_HASH_FILE]
        },
        "read_only_sources": {
            path.name: file_record(path)
            for path in [FEATURE_MATRIX_FILE, CONTRACT_FILE, SAVED_LABEL_SPLIT_SOURCE, Path(__file__).resolve()]
        },
        "external_api_calls": 0,
        "raw_gold_opened": False,
        "locked_1341_opened_or_evaluated": False,
        "feature_contract_modified": False,
        "final_full_dev_model_fit": False,
        "v1_modified": False,
    }
    frozen_write(HASH_FILE, json_bytes(hash_manifest))

    print("=" * 140)
    print("MMQA CLEAN RISK V2 — DEV1096 FIVE-FOLD OOF ANSWER-RISK V1")
    print("=" * 140)
    print(f"N={EXPECTED_N} | wrong={int(labels.sum())} | correct={int(EXPECTED_N-labels.sum())} | Top20 k={TOP_K}")
    print(f"{'VARIANT':12s} {'AUC':>9s} {'AP':>9s} {'TOP20-P':>10s} {'TOP20-R':>10s} {'ERRORS':>8s} {'DIMS':>30s}")
    for variant in VARIANT_KEYS:
        item = metrics[variant]
        print(
            f"{SHORT_NAME[variant]:12s} {item['roc_auc']:9.6f} {item['average_precision']:9.6f} "
            f"{item['top20']['precision']:10.6f} {item['top20']['recall']:10.6f} "
            f"{item['top20']['errors_found']:8d} {str(item['feature_dimension_per_fold']):>30s}"
        )
    print("-" * 140)
    print("Selected by frozen primary metric:", SHORT_NAME[selected_variant])
    print("Each QID exactly one held-out prediction per variant: YES")
    print("Final full-Dev fit: NO")
    print("API/Gold/locked/feature-change/variant-redesign/V1-modified: 0 / 0 / 0 / 0 / 0 / 0")
    print("Metrics:", METRICS_FILE)
    print("Fold stats:", FOLD_STATS_FILE)
    print("Coefficients:", COEFFICIENTS_FILE)
    print("SHA:", HASH_FILE)
    print("STATUS =", summary["status"])
    print("NEXT STEP =", summary["next_step"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-protocol", action="store_true")
    mode.add_argument("--run-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze_protocol:
        freeze_protocol()
    elif args.run_all:
        run_all()


if __name__ == "__main__":
    main()
