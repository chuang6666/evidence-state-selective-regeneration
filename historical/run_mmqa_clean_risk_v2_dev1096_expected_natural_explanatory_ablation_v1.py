#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Clean Risk V2 Dev1096 Expected/Natural explanatory ablation V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
EXPECTED_N = 1096
EXPECTED_WRONG = 680
EXPECTED_CORRECT = 416
N_SPLITS = 5
SPLIT_SEED = 260817
MODEL_SEED = 260817
TOP_K = 220
TRANSITIONS = ("T1", "T2", "T3", "T4", "T5")
BOOTSTRAP_SEED = 260817
BOOTSTRAP_REPLICATES = 10_000

INPUTS = {
    "feature_contract": (
        ROOT / "outputs" / "mmqa_clean_risk_v2_feature_contract_v1"
        / "clean_risk_v2_feature_contract_v1.json",
        "9BACED6295AE3F31E023DD9E91395ABC567861AB4DA040A5E0D337B344ED14B1",
    ),
    "feature_matrix": (
        ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_feature_generation_v1"
        / "dev1096_clean_risk_feature_matrix_v1.jsonl",
        "FA579EE4028FFB2F07EDA664B3547D688733E0022677048783223E763330DCF7",
    ),
    "qid_state_manifest": (
        ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_verifier_reconstruction_preflight_v1"
        / "mmqa_clean_v2_dev1096_qid_state_manifest_v1.jsonl",
        "7FC5663DCD5A575F7722B3F7002540017957CC276E79E1D240988E9A092E3843",
    ),
    "label_fold": (
        ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_answer_risk_oof_v1"
        / "dev1096_saved_answer_error_labels_and_shared_folds_v1.jsonl",
        "712195863BA08DAE57E92462B18531C3A80E4D9CF798404A8FD5FF9F1DB4DE88",
    ),
    "oof_protocol": (
        ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_answer_risk_oof_v1"
        / "clean_v2_oof_evaluation_protocol_v1.json",
        "D05D1FD64BBA017B99BAF96E5812D11D3F5E5C4990B063A54FA9CE8540A3CF2A",
    ),
}

OUT_DIR = (
    ROOT / "outputs"
    / "mmqa_clean_risk_v2_dev1096_expected_natural_explanatory_ablation_v1"
)
CONTRACT_OUT = OUT_DIR / "explanatory_ablation_contract_v1.json"
PREFLIGHT_OUT = OUT_DIR / "explanatory_ablation_preflight_v1.json"
FEATURES_OUT = OUT_DIR / "dev1096_explanatory_feature_matrix_v1.jsonl"
PREFLIGHT_SHA_OUT = OUT_DIR / "PREFLIGHT_SHA256SUMS.json"
COMBINED_PREDICTIONS_OUT = OUT_DIR / "dev1096_all_explanatory_variants_oof_predictions_v1.jsonl"
METRICS_OUT = OUT_DIR / "explanatory_ablation_metrics_v1.json"
FOLD_STATS_OUT = OUT_DIR / "explanatory_ablation_fold_statistics_v1.json"
BOOTSTRAP_OUT = OUT_DIR / "explanatory_ablation_pairwise_bootstrap_v1.json"
SUMMARY_OUT = OUT_DIR / "explanatory_ablation_summary_v1.json"
SHA_OUT = OUT_DIR / "SHA256SUMS.json"

G12 = (
    "GEN.question_chars", "GEN.question_tokens", "GEN.draft_chars",
    "GEN.draft_tokens", "GEN.draft_single_token", "GEN.draft_numeric",
    "GEN.evidence_count", "GEN.evidence_mean_chars", "GEN.evidence_max_chars",
    "GEN.modality::image", "GEN.modality::table", "GEN.modality::text",
)
E21 = (
    "TOPO.predicted_required_transitions_raw.__len__",
    "TOPO.predicted_required_transitions_schema_constrained.__len__",
    *(f"{t}.required" for t in TRANSITIONS),
    *(f"TOPO.predicted_required_transitions_raw.__contains__{t}" for t in TRANSITIONS),
    *(f"TOPO.predicted_required_transitions_schema_constrained.__contains__{t}" for t in TRANSITIONS),
    *(f"TOPO.probabilities.{t}" for t in TRANSITIONS[:4]),
)
N6 = (
    "NAT.present_natural_transitions.__len__",
    *(f"NAT.present_natural_transitions.__contains__{t}" for t in TRANSITIONS),
)
M6 = (
    "LOC.missing_transitions.__len__",
    *(f"LOC.missing_transitions.__contains__{t}" for t in TRANSITIONS),
)
V4 = (
    "DIAG.verifier_invalid", "LOC.localized_valid_prob",
    "LOC.localized_threshold", "LOC.localized_invalid_severity",
)
L19 = (
    "LOC.localized_transition",
    *(name for t in TRANSITIONS for name in (
        f"{t}.localized", f"{t}.localized_structural", f"{t}.localized_invalid",
    )),
    "LOC.localized_valid_prob", "LOC.localized_threshold",
    "LOC.localized_invalid_severity",
)

VARIANT_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "S1", "S2")
STATIC_FIELDS = {
    "A0": G12,
    "A1": G12 + E21,
    "A2": G12 + N6,
    "A3": G12 + E21 + N6,
    "A4": G12 + M6,
    "A5": G12 + E21 + N6 + M6,
}
EXPECTED_CONTRACT_COUNTS = {
    "A0": 12, "A1": 33, "A2": 18, "A3": 39, "A4": 18,
    "A5": 45, "A6": 65, "S1": 22, "S2": 46,
}
EXPECTED_A6_MATERIALIZED_COUNTS = {61: 385, 62: 697, 65: 14}
EXPECTED_S1_MATERIALIZED_COUNTS = {19: 1082, 22: 14}
EXPECTED_S2_MATERIALIZED_COUNTS = {46: 1096}

PAIRWISE_COMPARISONS = (
    ("A4", "A1", "mismatch_vs_expected_only"),
    ("A4", "A2", "mismatch_vs_natural_only"),
    ("A5", "A3", "mismatch_increment_given_expected_and_natural"),
    ("S1", "A4", "coarse_verifier_increment"),
    ("A6", "S2", "localization_increment"),
)

FORBIDDEN_FEATURE_FRAGMENTS = (
    "offline_required_transitions_proxy", "TOPO.question", "TOPO.fold",
    "TOPO.inference_features", "gold", "reference", "correct", "incorrect",
    "exact_match", "token_f1", "em_score", "official_type", "question_type",
    "provider", "prompt", "cache", "latency", "runtime", "attempt",
)


def fail(message: str) -> None:
    raise RuntimeError(f"FAIL-STOP: {message}")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        fail(f"missing input file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        fail(f"JSON root is not an object: {path}")
    return obj


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception as exc:
                fail(f"malformed JSONL {path}:{line_no}: {exc}")
            if not isinstance(row, dict):
                fail(f"non-object JSONL row {path}:{line_no}")
            rows.append(row)
    return rows


def unique_by_qid(rows: Iterable[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid", "") or "").strip()
        if not qid:
            fail(f"blank QID in {source}")
        if qid in result:
            fail(f"duplicate QID in {source}: {qid}")
        result[qid] = row
    return result


def json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def frozen_write(path: Path, data: bytes) -> None:
    try:
        path.relative_to(OUT_DIR)
    except ValueError:
        fail(f"refusing to write outside isolated namespace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            fail(f"frozen output exists with different content: {path}")
        return
    path.write_bytes(data)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_transition_list(value: Any, source: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        fail(f"{source} is not a list")
    normalized = tuple(str(item) for item in value)
    if len(set(normalized)) != len(normalized):
        fail(f"duplicate transition in {source}: {normalized}")
    unknown = set(normalized) - set(TRANSITIONS)
    if unknown:
        fail(f"unknown transition in {source}: {sorted(unknown)}")
    return normalized


def project_natural(present_natural_transitions: Sequence[str]) -> dict[str, int]:
    """Pure N6 projection. This function cannot inspect any state-row field."""
    present = validate_transition_list(present_natural_transitions, "present_natural_transitions")
    present_set = set(present)
    return {
        N6[0]: len(present),
        **{N6[index + 1]: int(t in present_set) for index, t in enumerate(TRANSITIONS)},
    }


def project_mismatch(required_but_missing: Sequence[str]) -> dict[str, int]:
    """Pure M6 projection from the frozen joint Expected/Present mismatch list."""
    missing = validate_transition_list(required_but_missing, "required_but_missing")
    missing_set = set(missing)
    return {
        M6[0]: len(missing),
        **{M6[index + 1]: int(t in missing_set) for index, t in enumerate(TRANSITIONS)},
    }


def select_present(source: Mapping[str, Any], fields: Sequence[str], label: str) -> dict[str, Any]:
    missing = [field for field in fields if field not in source]
    if missing:
        fail(f"{label} is missing required fields: {missing}")
    return {field: source[field] for field in fields}


def validate_feature_values(qid: str, variant: str, features: Mapping[str, Any]) -> None:
    for name, value in features.items():
        if value is None:
            fail(f"null feature: {qid}/{variant}/{name}")
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not math.isfinite(float(value)):
                fail(f"NaN/Inf feature: {qid}/{variant}/{name}={value}")
        elif not isinstance(value, str):
            fail(f"unsupported feature value type: {qid}/{variant}/{name}={type(value)}")


def make_model() -> Pipeline:
    return Pipeline([
        ("vec", DictVectorizer(sparse=True, separator="=", sort=True, dtype=np.float64)),
        ("clf", LogisticRegression(
            solver="liblinear", C=1.0, class_weight="balanced", max_iter=2000,
            random_state=MODEL_SEED,
        )),
    ])


def deterministic_ranking(qids: Sequence[str], scores: np.ndarray) -> list[int]:
    return sorted(range(len(qids)), key=lambda i: (-float(scores[i]), qids[i]))


def metric_vector(qids: Sequence[str], labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = deterministic_ranking(qids, scores)
    selected = order[:TOP_K]
    errors = int(labels[selected].sum())
    return {
        "top20_error_precision": float(errors / TOP_K),
        "top20_error_recall": float(errors / int(labels.sum())),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def class_balance(indices: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    y = labels[indices]
    wrong = int(y.sum())
    return {
        "n": int(len(y)), "wrong": wrong, "correct": int(len(y) - wrong),
        "error_rate": float(y.mean()),
    }


def validate_frozen_protocol(protocol: Mapping[str, Any]) -> None:
    cv = protocol.get("cross_validation", {})
    model = protocol.get("model", {})
    vectorizer = model.get("dict_vectorizer", {})
    classifier = model.get("logistic_regression", {})
    metrics = protocol.get("metrics", {})
    expected_checks = {
        "protocol.status": (
            protocol.get("status"),
            "MMQA_CLEAN_RISK_V2_DEV1096_OOF_EVALUATION_PROTOCOL_V1_FROZEN",
        ),
        "cv.type": (cv.get("type"), "StratifiedKFold"),
        "cv.n_splits": (cv.get("n_splits"), N_SPLITS),
        "cv.shuffle": (cv.get("shuffle"), True),
        "cv.random_state": (cv.get("random_state"), SPLIT_SEED),
        "cv.same_split": (cv.get("same_qid_split_for_all_variants"), True),
        "vectorizer.sparse": (vectorizer.get("sparse"), True),
        "vectorizer.separator": (vectorizer.get("separator"), "="),
        "vectorizer.sort": (vectorizer.get("sort"), True),
        "vectorizer.dtype": (vectorizer.get("dtype"), "numpy.float64"),
        "classifier.solver": (classifier.get("solver"), "liblinear"),
        "classifier.penalty": (classifier.get("penalty"), "l2"),
        "classifier.dual": (classifier.get("dual"), False),
        "classifier.C": (classifier.get("C"), 1.0),
        "classifier.fit_intercept": (classifier.get("fit_intercept"), True),
        "classifier.intercept_scaling": (classifier.get("intercept_scaling"), 1),
        "classifier.class_weight": (classifier.get("class_weight"), "balanced"),
        "classifier.max_iter": (classifier.get("max_iter"), 2000),
        "classifier.random_state": (classifier.get("random_state"), MODEL_SEED),
        "classifier.tol": (classifier.get("tol"), 0.0001),
        "classifier.warm_start": (classifier.get("warm_start"), False),
        "model.hyperparameter_search": (model.get("hyperparameter_search"), False),
        "model.final_full_dev_model_fit": (model.get("final_full_dev_model_fit"), False),
        "metrics.primary": (
            metrics.get("primary_model_selection_metric"),
            "global_top20_error_precision",
        ),
        "metrics.population": (metrics.get("top20_population"), EXPECTED_N),
        "metrics.k": (metrics.get("top20_k"), TOP_K),
        "metrics.ranking": (metrics.get("ranking"), "risk_score_descending"),
        "metrics.tie": (metrics.get("exact_score_tie_break"), "qid_ascending"),
    }
    drift = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in expected_checks.items()
        if actual != expected
    }
    required_reported = {
        "roc_auc", "average_precision", "global_top20_error_precision",
        "global_top20_error_recall",
    }
    if set(metrics.get("reported", [])) != required_reported:
        drift["metrics.reported"] = {
            "actual": metrics.get("reported"), "expected": sorted(required_reported)
        }
    if drift:
        fail(f"frozen model/metric protocol drift: {drift}")


def validate_input_allowlist() -> None:
    expected_names = {
        "feature_contract", "feature_matrix", "qid_state_manifest",
        "label_fold", "oof_protocol",
    }
    if set(INPUTS) != expected_names or len(INPUTS) != 5:
        fail(f"input allowlist is not the frozen five-file set: {sorted(INPUTS)}")
    prohibited_path_tokens = ("locked", "m3", "gold", "reference")
    for name, (path, _) in INPUTS.items():
        try:
            path.resolve().relative_to(ROOT.resolve())
        except ValueError:
            fail(f"input is outside the project root: {name}={path}")
        hits = [token for token in prohibited_path_tokens if token in str(path).lower()]
        if hits:
            fail(f"prohibited data-source path in input allowlist: {name}={path}, hits={hits}")


def validate_variant_contracts(a6_fields: Sequence[str]) -> dict[str, list[str]]:
    s2_fields = [field for field in a6_fields if field not in set(L19)]
    fields = {name: list(values) for name, values in STATIC_FIELDS.items()}
    fields["A6"] = list(a6_fields)
    fields["S1"] = list(G12 + M6 + V4)
    fields["S2"] = s2_fields

    for name in VARIANT_ORDER:
        if len(fields[name]) != len(set(fields[name])):
            fail(f"duplicate field in {name} contract")
        if len(fields[name]) != EXPECTED_CONTRACT_COUNTS[name]:
            fail(
                f"{name} contract count drift: {len(fields[name])} != "
                f"{EXPECTED_CONTRACT_COUNTS[name]}"
            )

    a3, a5 = set(fields["A3"]), set(fields["A5"])
    if a5 - a3 != set(M6):
        fail(f"FeatureSet(A5)-FeatureSet(A3) != M6: {sorted(a5-a3)}")
    if a3 - a5:
        fail(f"FeatureSet(A3)-FeatureSet(A5) is not empty: {sorted(a3-a5)}")

    a3_forbidden = [
        field for field in fields["A3"]
        if field in set(M6 + V4 + L19)
        or field.startswith("DIAG.") or field.startswith("LOC.")
        or "localized" in field.lower() or "verifier" in field.lower()
    ]
    if a3_forbidden:
        fail(f"A3 contains mismatch/diagnosis/verifier/localization fields: {a3_forbidden}")

    all_fields = sorted({field for values in fields.values() for field in values})
    forbidden_hits = {
        field: [fragment for fragment in FORBIDDEN_FEATURE_FRAGMENTS
                if fragment.lower() in field.lower()]
        for field in all_fields
    }
    forbidden_hits = {field: hits for field, hits in forbidden_hits.items() if hits}
    if forbidden_hits:
        fail(f"forbidden feature names present: {forbidden_hits}")
    return fields


def preflight_and_build() -> tuple[
    list[str], dict[str, dict[str, dict[str, Any]]], np.ndarray, np.ndarray,
    dict[str, list[str]], dict[str, Any]
]:
    validate_input_allowlist()
    input_records: dict[str, dict[str, Any]] = {}
    for name, (path, expected_sha) in INPUTS.items():
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            fail(f"input SHA drift for {name}: {actual_sha} != {expected_sha}")
        input_records[name] = file_record(path)

    contract = load_json(INPUTS["feature_contract"][0])
    protocol = load_json(INPUTS["oof_protocol"][0])
    validate_frozen_protocol(protocol)
    if contract.get("status") != "MMQA_CLEAN_RISK_V2_G0_B0_B4_FEATURE_CONTRACT_FROZEN":
        fail("feature contract status is not frozen")
    contract_variants = contract.get("variants", {})
    b3_contract = contract_variants.get("CLEAN_B3_GENERAL_T1_T5", {})
    a6_fields = b3_contract.get("features")
    if not isinstance(a6_fields, list) or b3_contract.get("count") != 65:
        fail("frozen Clean-B3 65-field contract is invalid")
    variant_fields = validate_variant_contracts(a6_fields)

    matrix_by_qid = unique_by_qid(
        load_jsonl(INPUTS["feature_matrix"][0]), "frozen Clean V2 feature matrix"
    )
    state_by_qid = unique_by_qid(
        load_jsonl(INPUTS["qid_state_manifest"][0]), "frozen QID state manifest"
    )
    labels_by_qid = unique_by_qid(
        load_jsonl(INPUTS["label_fold"][0]), "frozen label/fold file"
    )
    qid_sets = {"matrix": set(matrix_by_qid), "state": set(state_by_qid),
                "labels": set(labels_by_qid)}
    if any(len(values) != EXPECTED_N for values in qid_sets.values()):
        fail(f"input QID count drift: { {k: len(v) for k, v in qid_sets.items()} }")
    if not (qid_sets["matrix"] == qid_sets["state"] == qid_sets["labels"]):
        fail("matrix/state/label QID sets differ")
    qids = sorted(qid_sets["matrix"])

    labels = np.empty(EXPECTED_N, dtype=int)
    folds = np.empty(EXPECTED_N, dtype=int)
    derived: dict[str, dict[str, dict[str, Any]]] = {}
    m6_cell_mismatches = 0
    m6_row_mismatches = 0
    materialized_counts = {name: Counter() for name in VARIANT_ORDER}
    fold_balance: dict[str, Any] = {}

    for index, qid in enumerate(tqdm(qids, desc="Preflight feature projections", unit="qid")):
        label_row = labels_by_qid[qid]
        if set(label_row) != {"qid", "initial_wrong_label", "heldout_fold"}:
            fail(f"unexpected label/fold fields for {qid}: {sorted(label_row)}")
        label = label_row.get("initial_wrong_label")
        fold = label_row.get("heldout_fold")
        if label not in {0, 1} or not isinstance(fold, int) or fold not in range(1, 6):
            fail(f"invalid label/fold for {qid}: {label}/{fold}")
        labels[index], folds[index] = int(label), int(fold)

        envelope = matrix_by_qid[qid]
        if set(envelope) != {"qid", "variants"}:
            fail(f"feature-matrix envelope drift for {qid}: {sorted(envelope)}")
        matrix_variants = envelope.get("variants")
        expected_matrix_variants = set(contract_variants)
        if not isinstance(matrix_variants, dict) or set(matrix_variants) != expected_matrix_variants:
            fail(f"frozen matrix variant set drift for {qid}")
        g0 = matrix_variants.get("G0_GENERAL_ONLY")
        a6 = matrix_variants.get("CLEAN_B3_GENERAL_T1_T5")
        if not isinstance(g0, dict) or not isinstance(a6, dict):
            fail(f"missing G0/Clean-B3 feature object for {qid}")
        if set(g0) != set(G12):
            fail(f"G12 field drift for {qid}: {sorted(set(g0)^set(G12))}")
        if not set(a6) <= set(a6_fields):
            fail(f"A6 contains fields outside frozen Clean-B3 contract for {qid}")
        missing_non_sparse = set(a6_fields) - {
            "LOC.localized_transition", "LOC.localized_valid_prob",
            "LOC.localized_threshold", "LOC.localized_invalid_severity",
        } - set(a6)
        if missing_non_sparse:
            fail(f"A6 missing non-sparse fields for {qid}: {sorted(missing_non_sparse)}")

        state = state_by_qid[qid]
        # Deliberately pass only the named list into each pure projection function.
        n6 = project_natural(state.get("present_natural_transitions"))
        m6 = project_mismatch(state.get("required_but_missing"))
        old_m6 = select_present(a6, M6, f"frozen A6 M6/{qid}")
        row_mismatch = False
        for field in M6:
            if old_m6[field] != m6[field]:
                m6_cell_mismatches += 1
                row_mismatch = True
        m6_row_mismatches += int(row_mismatch)

        g = select_present(g0, G12, f"G12/{qid}")
        e = select_present(a6, E21, f"E21/{qid}")
        v4 = select_present(a6, (V4[0],), f"coarse verifier flag/{qid}")
        numeric_v4_present = [field for field in V4[1:] if field in a6]
        if len(numeric_v4_present) not in {0, 3}:
            fail(f"partial coarse verifier numeric triple for {qid}: {numeric_v4_present}")
        v4.update({field: a6[field] for field in numeric_v4_present})

        row_variants: dict[str, dict[str, Any]] = {
            "A0": dict(g),
            "A1": {**g, **e},
            "A2": {**g, **n6},
            "A3": {**g, **e, **n6},
            "A4": {**g, **m6},
            "A5": {**g, **e, **n6, **m6},
            "A6": dict(a6),
            "S1": {**g, **m6, **v4},
            "S2": {field: value for field, value in a6.items() if field not in set(L19)},
        }
        for variant, features in row_variants.items():
            allowed = set(variant_fields[variant])
            if not set(features) <= allowed:
                fail(f"{qid}/{variant} emitted fields outside contract: {sorted(set(features)-allowed)}")
            if variant in STATIC_FIELDS and set(features) != allowed:
                fail(f"{qid}/{variant} exact field-set mismatch: {sorted(set(features)^allowed)}")
            if variant == "S2" and set(features) != allowed:
                fail(f"{qid}/S2 exact field-set mismatch: {sorted(set(features)^allowed)}")
            validate_feature_values(qid, variant, features)
            materialized_counts[variant][len(features)] += 1
        derived[qid] = row_variants

    if m6_cell_mismatches or m6_row_mismatches:
        fail(f"M6 parity failed: rows={m6_row_mismatches}, cells={m6_cell_mismatches}")
    if int(labels.sum()) != EXPECTED_WRONG or EXPECTED_N - int(labels.sum()) != EXPECTED_CORRECT:
        fail(f"label balance drift: wrong={int(labels.sum())}, correct={EXPECTED_N-int(labels.sum())}")
    if set(folds.tolist()) != set(range(1, 6)):
        fail(f"held-out fold set drift: {sorted(set(folds.tolist()))}")
    regenerated_folds = np.zeros(EXPECTED_N, dtype=int)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(np.zeros(EXPECTED_N), labels), 1
    ):
        regenerated_folds[validation_indices] = fold
    fold_mismatches = np.flatnonzero(regenerated_folds != folds)
    if len(fold_mismatches):
        examples = [
            {
                "qid": qids[int(i)], "saved": int(folds[i]),
                "regenerated": int(regenerated_folds[i]),
            }
            for i in fold_mismatches[:20]
        ]
        fail(f"saved held-out fold mapping differs from seed-260817 split: {examples}")
    for fold in range(1, 6):
        validation = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        fold_balance[str(fold)] = {
            "train": class_balance(training, labels),
            "validation": class_balance(validation, labels),
        }
        if len(set(labels[validation].tolist())) != 2:
            fail(f"held-out fold {fold} does not contain both classes")

    expected_materialized = {
        **{name: {count: EXPECTED_N} for name, count in EXPECTED_CONTRACT_COUNTS.items()
           if name in STATIC_FIELDS},
        "A6": EXPECTED_A6_MATERIALIZED_COUNTS,
        "S1": EXPECTED_S1_MATERIALIZED_COUNTS,
        "S2": EXPECTED_S2_MATERIALIZED_COUNTS,
    }
    actual_materialized = {
        name: dict(sorted(counter.items())) for name, counter in materialized_counts.items()
    }
    if actual_materialized != expected_materialized:
        fail(f"materialized feature-count distribution drift: {actual_materialized}")

    contract_out = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_EXPECTED_NATURAL_EXPLANATORY_ABLATION_V1_CONTRACT_FROZEN",
        "analysis_type": "post_hoc_explanatory_ablation_not_confirmatory_model_selection",
        "fixed_variant_order": list(VARIANT_ORDER),
        "variants": {
            name: {
                "raw_contract_key_count": len(variant_fields[name]),
                "features": variant_fields[name],
                "materialized_key_count_distribution": {
                    str(k): v for k, v in actual_materialized[name].items()
                },
            }
            for name in VARIANT_ORDER
        },
        "groups": {
            "G12": {"features": list(G12), "source": "frozen G0 surface/general features"},
            "E21": {"features": list(E21), "source": "frozen Expected Topology fields"},
            "N6": {
                "features": list(N6),
                "sole_source": "qid_state_manifest.present_natural_transitions",
                "projection_receives_whole_state_row": False,
            },
            "M6": {
                "features": list(M6),
                "sole_source": "qid_state_manifest.required_but_missing",
                "parity_target": "frozen Clean-B3 six missing fields",
            },
            "V4": {
                "features": list(V4),
                "source": "frozen Clean-B3 coarse verifier-derived fields",
                "numeric_triple_sparse": True,
            },
            "L19": {
                "features": list(L19),
                "operation": "removed from frozen Clean-B3 to form S2",
            },
        },
        "programmatic_nesting_gates": {
            "FeatureSet(A5)-FeatureSet(A3)==M6": True,
            "FeatureSet(A3)-FeatureSet(A5)==empty": True,
            "A3_has_mismatch_diagnosis_verifier_or_localization": False,
        },
        "oof_protocol": {
            "qids": EXPECTED_N,
            "folds": N_SPLITS,
            "saved_heldout_fold_mapping_reused": True,
            "saved_initial_wrong_label_reused": True,
            "split_seed": SPLIT_SEED,
            "model_seed": MODEL_SEED,
            "dict_vectorizer": "DictVectorizer(sparse=True)",
            "classifier": (
                "LogisticRegression(solver='liblinear', C=1.0, "
                "class_weight='balanced', max_iter=2000, random_state=260817)"
            ),
            "top20_k": TOP_K,
            "ranking": "risk_score_descending_then_qid_ascending",
            "reported_metrics": [
                "top20_error_precision", "top20_error_recall",
                "average_precision", "roc_auc",
            ],
            "hyperparameter_search": False,
            "final_full_dev_fit": False,
            "formal_variant_selection": False,
        },
        "pairwise_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "unit": "paired_qid_resampling_with_replacement",
            "pairs": [
                {"left": left, "right": right, "question": question}
                for left, right, question in PAIRWISE_COMPARISONS
            ],
            "used_for_formal_variant_selection": False,
        },
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "input_allowlist": input_records,
        "prohibitions": {
            "external_api_calls": 0,
            "mmqa_locked_1341_gold_access": 0,
            "m3_gold_access": 0,
            "formal_clean_b3_training_or_modification": 0,
            "formal_variant_reselection": 0,
            "existing_output_modification": 0,
        },
    }
    feature_rows = [
        {"qid": qid, "variants": derived[qid]} for qid in qids
    ]
    preflight_out = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_EXPECTED_NATURAL_EXPLANATORY_ABLATION_V1_PREFLIGHT_PASS",
        "training_started": False,
        "input_sha_gate": "PASS_5_OF_5",
        "qids": len(qids),
        "unique_qids": len(set(qids)),
        "wrong": int(labels.sum()),
        "correct": int(EXPECTED_N - labels.sum()),
        "heldout_fold_values": sorted(set(folds.tolist())),
        "saved_vs_regenerated_fold_mismatches": 0,
        "fold_class_balance": fold_balance,
        "m6_parity": {
            "rows_compared": EXPECTED_N,
            "row_mismatches": m6_row_mismatches,
            "cell_mismatches": m6_cell_mismatches,
        },
        "variant_contract_counts": {
            name: len(variant_fields[name]) for name in VARIANT_ORDER
        },
        "variant_materialized_count_distributions": {
            name: {str(k): v for k, v in actual_materialized[name].items()}
            for name in VARIANT_ORDER
        },
        "a5_minus_a3_exactly_m6": True,
        "a3_minus_a5_empty": True,
        "a3_information_isolation": "PASS",
        "model_metric_protocol_parity": "PASS",
        "forbidden_feature_or_source_hits": 0,
        "input_files_read": [str(INPUTS[name][0]) for name in INPUTS],
        "external_api_calls": 0,
        "mmqa_locked_1341_gold_access": 0,
        "m3_gold_access": 0,
        "formal_clean_b3_modified": False,
        "formal_variant_reselection": False,
        "old_output_modified": False,
        "gate": "PASS: --run-all may execute the nine fixed OOF explanatory variants",
    }

    frozen_write(CONTRACT_OUT, json_bytes(contract_out))
    frozen_write(PREFLIGHT_OUT, json_bytes(preflight_out))
    frozen_write(FEATURES_OUT, jsonl_bytes(feature_rows))
    preflight_sha = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_EXPLANATORY_ABLATION_PREFLIGHT_SHA_FROZEN",
        "generated_files": {
            path.name: file_record(path)
            for path in (CONTRACT_OUT, PREFLIGHT_OUT, FEATURES_OUT)
        },
        "read_only_inputs": input_records,
        "script": file_record(Path(__file__).resolve()),
        "training_started": False,
        "external_api_calls": 0,
        "gold_access": 0,
    }
    frozen_write(PREFLIGHT_SHA_OUT, json_bytes(preflight_sha))
    validation = {
        "input_records": input_records,
        "materialized_counts": actual_materialized,
        "fold_class_balance": fold_balance,
        "m6_parity_rows": EXPECTED_N,
        "m6_row_mismatches": 0,
        "m6_cell_mismatches": 0,
    }
    return qids, derived, labels, folds, variant_fields, validation


def detailed_global_metrics(
    qids: Sequence[str], labels: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    metrics = metric_vector(qids, labels, scores)
    order = deterministic_ranking(qids, scores)
    selected = order[:TOP_K]
    boundary = float(scores[selected[-1]])
    equal_boundary = [i for i in order if float(scores[i]) == boundary]
    selected_equal = [i for i in selected if float(scores[i]) == boundary]
    return {
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
        "population_error_rate": float(labels.mean()),
        "top20": {
            "k": TOP_K,
            "errors_found": int(labels[selected].sum()),
            "precision": metrics["top20_error_precision"],
            "recall": metrics["top20_error_recall"],
            "boundary_score": boundary,
            "boundary_equal_score_count": len(equal_boundary),
            "boundary_equal_score_selected_count": len(selected_equal),
            "tie_break": "risk_score_desc_then_qid_ascending",
        },
    }


def fold_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "risk_min": float(scores.min()),
        "risk_max": float(scores.max()),
        "risk_mean": float(scores.mean()),
        "risk_std": float(scores.std()),
    }


def fast_bootstrap_metrics(
    qid_ranks: np.ndarray, labels: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    if len(np.unique(labels)) != 2:
        fail("bootstrap replicate contains fewer than two classes")
    order = np.lexsort((qid_ranks, -scores))
    selected = order[:TOP_K]
    errors_total = int(labels.sum())
    return np.asarray([
        float(labels[selected].sum() / TOP_K),
        float(labels[selected].sum() / errors_total),
        float(average_precision_score(labels, scores)),
        float(roc_auc_score(labels, scores)),
    ])


def paired_bootstrap(
    qids: list[str], labels: np.ndarray, variant_scores: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    metric_names = (
        "top20_error_precision", "top20_error_recall", "average_precision", "roc_auc"
    )
    qid_rank_map = {qid: rank for rank, qid in enumerate(sorted(qids))}
    base_qid_ranks = np.asarray([qid_rank_map[qid] for qid in qids], dtype=int)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output: dict[str, Any] = {}
    total = BOOTSTRAP_REPLICATES * len(PAIRWISE_COMPARISONS)
    progress = tqdm(total=total, desc="Paired bootstrap", unit="rep")
    try:
        for left, right, question in PAIRWISE_COMPARISONS:
            observed_left = fast_bootstrap_metrics(base_qid_ranks, labels, variant_scores[left])
            observed_right = fast_bootstrap_metrics(base_qid_ranks, labels, variant_scores[right])
            differences = np.empty((BOOTSTRAP_REPLICATES, len(metric_names)), dtype=float)
            for replicate in range(BOOTSTRAP_REPLICATES):
                sample = rng.integers(0, EXPECTED_N, size=EXPECTED_N)
                sampled_ranks = base_qid_ranks[sample]
                sampled_labels = labels[sample]
                left_metrics = fast_bootstrap_metrics(
                    sampled_ranks, sampled_labels, variant_scores[left][sample]
                )
                right_metrics = fast_bootstrap_metrics(
                    sampled_ranks, sampled_labels, variant_scores[right][sample]
                )
                differences[replicate] = left_metrics - right_metrics
                progress.update(1)
            output[f"{left}_minus_{right}"] = {
                "left": left,
                "right": right,
                "question": question,
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "metrics": {
                    name: {
                        "observed_difference": float(observed_left[i] - observed_right[i]),
                        "bootstrap_mean_difference": float(differences[:, i].mean()),
                        "percentile_95_ci": [
                            float(np.percentile(differences[:, i], 2.5)),
                            float(np.percentile(differences[:, i], 97.5)),
                        ],
                        "fraction_difference_gt_zero": float(np.mean(differences[:, i] > 0)),
                        "fraction_difference_lt_zero": float(np.mean(differences[:, i] < 0)),
                    }
                    for i, name in enumerate(metric_names)
                },
            }
    finally:
        progress.close()
    return output


def run_all() -> None:
    qids, derived, labels, folds, variant_fields, validation = preflight_and_build()
    scores = {
        variant: np.full(EXPECTED_N, np.nan, dtype=float) for variant in VARIANT_ORDER
    }
    assignments = {
        variant: np.zeros(EXPECTED_N, dtype=int) for variant in VARIANT_ORDER
    }
    fold_stats: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in VARIANT_ORDER
    }

    training_steps = [(variant, fold) for variant in VARIANT_ORDER for fold in range(1, 6)]
    for variant, fold in tqdm(training_steps, desc="Nine-variant OOF training", unit="fold-model"):
        validation_indices = np.flatnonzero(folds == fold)
        training_indices = np.flatnonzero(folds != fold)
        train_x = [derived[qids[int(i)]][variant] for i in training_indices]
        validation_x = [derived[qids[int(i)]][variant] for i in validation_indices]
        model = make_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train_x, labels[training_indices])
        fold_scores = model.predict_proba(validation_x)[:, 1]
        scores[variant][validation_indices] = fold_scores
        assignments[variant][validation_indices] += 1
        vectorizer: DictVectorizer = model.named_steps["vec"]
        classifier: LogisticRegression = model.named_steps["clf"]
        fold_stats[variant].append({
            "fold": fold,
            "train": class_balance(training_indices, labels),
            "validation": class_balance(validation_indices, labels),
            "validation_metrics": fold_metrics(labels[validation_indices], fold_scores),
            "raw_contract_key_count": len(variant_fields[variant]),
            "vectorized_feature_dimension": len(vectorizer.get_feature_names_out()),
            "n_iter": [int(value) for value in classifier.n_iter_],
            "fit_warning_count": len(caught),
            "fit_warnings": [str(item.message) for item in caught],
        })

    for variant in VARIANT_ORDER:
        bad = np.flatnonzero(assignments[variant] != 1)
        if len(bad):
            fail(
                f"OOF assignment violation for {variant}: "
                f"{[(qids[int(i)], int(assignments[variant][i])) for i in bad[:20]]}"
            )
        if not np.isfinite(scores[variant]).all():
            fail(f"missing/NaN/Inf OOF scores for {variant}")

    metrics = {
        variant: {
            **detailed_global_metrics(qids, labels, scores[variant]),
            "raw_contract_key_count": len(variant_fields[variant]),
            "vectorized_feature_dimension_per_fold": [
                row["vectorized_feature_dimension"] for row in fold_stats[variant]
            ],
            "oof_prediction_count_min": int(assignments[variant].min()),
            "oof_prediction_count_max": int(assignments[variant].max()),
        }
        for variant in VARIANT_ORDER
    }

    prediction_files: list[Path] = []
    rank_maps: dict[str, dict[int, int]] = {}
    selected_sets: dict[str, set[int]] = {}
    for variant in tqdm(VARIANT_ORDER, desc="Freeze per-variant predictions", unit="variant"):
        order = deterministic_ranking(qids, scores[variant])
        rank_maps[variant] = {index: rank + 1 for rank, index in enumerate(order)}
        selected_sets[variant] = set(order[:TOP_K])
        rows = [
            {
                "qid": qids[i],
                "heldout_fold": int(folds[i]),
                "initial_wrong_label": int(labels[i]),
                "oof_risk_score": float(scores[variant][i]),
                "oof_prediction_count": int(assignments[variant][i]),
                "global_rank": rank_maps[variant][i],
                "global_top20_selected": i in selected_sets[variant],
            }
            for i in range(EXPECTED_N)
        ]
        path = OUT_DIR / f"dev1096_{variant.lower()}_oof_predictions_v1.jsonl"
        frozen_write(path, jsonl_bytes(rows))
        prediction_files.append(path)

    combined_rows = []
    for i, qid in enumerate(tqdm(qids, desc="Build combined predictions", unit="qid")):
        combined_rows.append({
            "qid": qid,
            "heldout_fold": int(folds[i]),
            "initial_wrong_label": int(labels[i]),
            "variant_oof_predictions": {
                variant: {
                    "risk_score": float(scores[variant][i]),
                    "global_rank": rank_maps[variant][i],
                    "global_top20_selected": i in selected_sets[variant],
                    "oof_prediction_count": int(assignments[variant][i]),
                }
                for variant in VARIANT_ORDER
            },
        })
    frozen_write(COMBINED_PREDICTIONS_OUT, jsonl_bytes(combined_rows))

    bootstrap = paired_bootstrap(qids, labels, scores)
    metrics_out = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_EXPECTED_NATURAL_EXPLANATORY_ABLATION_V1_METRICS_COMPLETE",
        "analysis_type": "post_hoc_explanatory_not_formal_variant_selection",
        "top20_k": TOP_K,
        "ranking": "risk_score_descending_then_qid_ascending",
        "variants": metrics,
    }
    fold_stats_out = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_EXPECTED_NATURAL_EXPLANATORY_ABLATION_V1_FOLD_STATS_COMPLETE",
        "shared_saved_heldout_folds": True,
        "class_balance": {
            "n": EXPECTED_N, "wrong": int(labels.sum()),
            "correct": int(EXPECTED_N - labels.sum()),
            "error_rate": float(labels.mean()),
        },
        "variants": fold_stats,
    }
    bootstrap_out = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_EXPECTED_NATURAL_EXPLANATORY_ABLATION_V1_BOOTSTRAP_COMPLETE",
        "analysis_type": "post_hoc_explanatory",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "pairs": bootstrap,
        "used_for_formal_variant_selection": False,
    }
    summary_out = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_EXPECTED_NATURAL_EXPLANATORY_ABLATION_V1_COMPLETE",
        "qids": EXPECTED_N,
        "wrong": int(labels.sum()),
        "correct": int(EXPECTED_N - labels.sum()),
        "variants_run_in_fixed_order": list(VARIANT_ORDER),
        "each_qid_exactly_one_heldout_prediction_per_variant": all(
            np.all(assignments[variant] == 1) for variant in VARIANT_ORDER
        ),
        "results": {
            variant: {
                "top20_error_precision": metrics[variant]["top20"]["precision"],
                "top20_error_recall": metrics[variant]["top20"]["recall"],
                "average_precision": metrics[variant]["average_precision"],
                "roc_auc": metrics[variant]["roc_auc"],
                "vectorized_feature_dimension_per_fold": (
                    metrics[variant]["vectorized_feature_dimension_per_fold"]
                ),
            }
            for variant in VARIANT_ORDER
        },
        "formal_clean_b3_selection_or_result_modified": False,
        "new_formal_variant_selected": None,
        "hyperparameter_search": False,
        "final_full_dev_model_fit": False,
        "external_api_calls": 0,
        "mmqa_locked_1341_gold_access": 0,
        "m3_gold_access": 0,
        "old_output_modified": False,
        "next_step": "STOP_AFTER_EXPLANATORY_OOF_ABLATION_FOR_HUMAN_REVIEW",
    }
    frozen_write(METRICS_OUT, json_bytes(metrics_out))
    frozen_write(FOLD_STATS_OUT, json_bytes(fold_stats_out))
    frozen_write(BOOTSTRAP_OUT, json_bytes(bootstrap_out))
    frozen_write(SUMMARY_OUT, json_bytes(summary_out))

    generated = [
        CONTRACT_OUT, PREFLIGHT_OUT, FEATURES_OUT, PREFLIGHT_SHA_OUT,
        *prediction_files, COMBINED_PREDICTIONS_OUT, METRICS_OUT,
        FOLD_STATS_OUT, BOOTSTRAP_OUT, SUMMARY_OUT,
    ]
    sha_out = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_EXPECTED_NATURAL_EXPLANATORY_ABLATION_V1_SHA_FROZEN",
        "generated_files": {path.name: file_record(path) for path in generated},
        "read_only_inputs": validation["input_records"],
        "script": file_record(Path(__file__).resolve()),
        "external_api_calls": 0,
        "gold_access": 0,
        "formal_clean_b3_modified": False,
        "old_output_modified": False,
    }
    frozen_write(SHA_OUT, json_bytes(sha_out))

    print("=" * 132)
    print("MMQA CLEAN RISK V2 — DEV1096 EXPECTED/NATURAL EXPLANATORY ABLATION V1")
    print("=" * 132)
    print(f"N={EXPECTED_N} wrong={int(labels.sum())} correct={EXPECTED_N-int(labels.sum())} Top20={TOP_K}")
    print(f"{'VARIANT':8s} {'AUC':>10s} {'AP':>10s} {'TOP20-P':>10s} {'TOP20-R':>10s}")
    for variant in VARIANT_ORDER:
        row = metrics[variant]
        print(
            f"{variant:8s} {row['roc_auc']:10.6f} {row['average_precision']:10.6f} "
            f"{row['top20']['precision']:10.6f} {row['top20']['recall']:10.6f}"
        )
    print("Formal variant reselection: NO")
    print("API/MMQA-locked-Gold/M3-Gold/final-full-fit/old-output-change: 0/0/0/0/0")
    print("Output:", OUT_DIR)
    print("SHA:", SHA_OUT)
    print("STATUS =", summary_out["status"])


def preflight_only() -> None:
    qids, _, labels, folds, variant_fields, validation = preflight_and_build()
    print("=" * 132)
    print("MMQA CLEAN RISK V2 — DEV1096 EXPECTED/NATURAL EXPLANATORY ABLATION V1 PREFLIGHT")
    print("=" * 132)
    print("Input SHA: PASS 5/5")
    print(f"QID={len(qids)} unique={len(set(qids))} wrong={int(labels.sum())} correct={len(labels)-int(labels.sum())}")
    print("Held-out folds:", sorted(set(folds.tolist())))
    print("M6 parity rows/cells mismatched: 0/0")
    print("Variant contract counts:", {name: len(variant_fields[name]) for name in VARIANT_ORDER})
    print("Materialized counts:", validation["materialized_counts"])
    print("A5/A3 nesting: PASS")
    print("A3 isolation: PASS")
    print("Model/metric protocol: PASS")
    print("Training started: NO")
    print("API/MMQA-locked-Gold/M3-Gold/old-output-change: 0/0/0/0")
    print("Output:", OUT_DIR)
    print("STATUS = MMQA_CLEAN_RISK_V2_DEV1096_EXPECTED_NATURAL_EXPLANATORY_ABLATION_V1_PREFLIGHT_PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        preflight_only()
    elif args.run_all:
        run_all()


if __name__ == "__main__":
    main()
