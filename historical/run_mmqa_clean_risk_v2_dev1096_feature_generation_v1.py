#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate the frozen Clean Risk V2 Dev1096 feature matrices.

This runner is intentionally feature-generation only.  It uses explicit,
contract-named fields and never recursively flattens an input object.  It does
not load Gold, locked-1341, a risk label, or a risk model.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
EXPECTED_N = 1096
TRANSITIONS = ("T1", "T2", "T3", "T4", "T5")
EXPECTED_CONTRACT_SHA256 = (
    "9BACED6295AE3F31E023DD9E91395ABC567861AB4DA040A5E0D337B344ED14B1"
)

CONTRACT_DIR = ROOT / "outputs" / "mmqa_clean_risk_v2_feature_contract_v1"
CONTRACT_FILE = CONTRACT_DIR / "clean_risk_v2_feature_contract_v1.json"
CONTRACT_HASH_FILE = CONTRACT_DIR / "SHA256SUMS.json"

JOBS_DIR = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_exact_top6_natural_jobs_v1"
)
BASE_FILE = JOBS_DIR / "mmqa_dev1096_exact_top6_initial_inputs_v1.jsonl"
NATURAL_JOB_FILE = JOBS_DIR / "mmqa_dev1096_exact_top6_natural_state_jobs_v1.jsonl"
JOBS_HASH_FILE = JOBS_DIR / "SHA256SUMS.json"

TOPOLOGY_FILE = (
    ROOT / "outputs" / "mmqa_transition_topology_tfidf_oof_v1"
    / "mmqa_transition_topology_tfidf_oof_predictions_v1.jsonl"
)
STATE_MANIFEST_FILE = (
    ROOT / "outputs"
    / "mmqa_clean_risk_v2_dev1096_verifier_reconstruction_preflight_v1"
    / "mmqa_clean_v2_dev1096_qid_state_manifest_v1.jsonl"
)

LOCALIZATION_DIR = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_localization_v1"
)
LOCALIZATION_FILE = LOCALIZATION_DIR / "dev1096_localization_v1.jsonl"
LOCALIZATION_HASH_FILE = LOCALIZATION_DIR / "SHA256SUMS.json"

OUT_DIR = (
    ROOT / "outputs" / "mmqa_clean_risk_v2_dev1096_feature_generation_v1"
)
PREFLIGHT_FILE = OUT_DIR / "feature_generation_preflight_v1.json"
MATRIX_FILE = OUT_DIR / "dev1096_clean_risk_feature_matrix_v1.jsonl"
SCHEMA_FILE = OUT_DIR / "clean_risk_feature_schema_and_provenance_v1.json"
AUDIT_FILE = OUT_DIR / "feature_schema_parity_audit_v1.json"
SUMMARY_FILE = OUT_DIR / "feature_generation_summary_v1.json"
HASH_FILE = OUT_DIR / "SHA256SUMS.json"

VARIANT_TRANSITIONS = {
    "G0_GENERAL_ONLY": (),
    "CLEAN_B0_COARSE_GENERAL": (),
    "CLEAN_B1_GENERAL_T1": ("T1",),
    "CLEAN_B2_GENERAL_T1_T4": ("T1", "T2", "T3", "T4"),
    "CLEAN_B3_GENERAL_T1_T5": TRANSITIONS,
    "CLEAN_B4_T1_T5_ONLY": TRANSITIONS,
}

EXPECTED_COUNTS = {
    "G0_GENERAL_ONLY": 12,
    "CLEAN_B0_COARSE_GENERAL": 25,
    "CLEAN_B1_GENERAL_T1": 34,
    "CLEAN_B2_GENERAL_T1_T4": 58,
    "CLEAN_B3_GENERAL_T1_T5": 65,
    "CLEAN_B4_T1_T5_ONLY": 40,
}

EXPECTED_DECISIONS = {
    "STRUCTURAL_MISSING", "VERIFIER_INVALID", "CLEAN_OR_UNCERTAIN"
}
EXPECTED_REASONS = {
    "required_by_topology_but_transition_state_missing",
    "present_state_below_conservative_invalid_threshold",
    "no_missing_required_state_and_no_present_state_crossed_conservative_invalid_threshold",
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


def text_value(value: Any) -> str:
    return "" if value is None else str(value).strip()


def text_norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", text_value(value)).lower()
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def required_list(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list):
        raise RuntimeError(f"Topology field is not a list: {key}")
    out = [str(x).upper() for x in value]
    if len(out) != len(set(out)) or any(t not in TRANSITIONS for t in out):
        raise RuntimeError(f"Invalid topology transitions in {key}: {value}")
    return out


def exact_top6_evidence_ids(row: dict[str, Any]) -> list[str]:
    evidence = row.get("retrieved_evidence")
    if not isinstance(evidence, list):
        return []
    return [str(x.get("evidence_id", "") or "") if isinstance(x, dict) else "" for x in evidence]


def general_features(base: dict[str, Any]) -> dict[str, Any]:
    """Explicit G0 builder; this is not a recursive flatten."""
    question = text_value(base.get("question"))
    draft = text_value(base.get("draft_answer"))
    evidence = base.get("retrieved_evidence")
    if not isinstance(evidence, list):
        raise RuntimeError("retrieved_evidence is not a list")

    modality_counts = Counter({"image": 0, "table": 0, "text": 0})
    evidence_lengths: list[int] = []
    for item in evidence:
        if not isinstance(item, dict):
            raise RuntimeError("Non-object retrieved evidence")
        modality = text_value(item.get("modality")).lower()
        if modality not in modality_counts:
            raise RuntimeError(f"Modality outside frozen contract: {modality!r}")
        modality_counts[modality] += 1
        serialized = (text_value(item.get("title")) + " " + text_value(item.get("content"))).strip()
        if serialized:
            evidence_lengths.append(len(serialized))

    q_tokens = text_norm(question).split()
    a_norm = text_norm(draft)
    a_tokens = a_norm.split()
    return {
        "GEN.question_chars": len(question),
        "GEN.question_tokens": len(q_tokens),
        "GEN.draft_chars": len(draft),
        "GEN.draft_tokens": len(a_tokens),
        "GEN.draft_single_token": int(len(a_tokens) == 1),
        "GEN.draft_numeric": int(bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", a_norm))),
        "GEN.evidence_count": len(evidence),
        "GEN.evidence_mean_chars": (
            float(sum(evidence_lengths) / len(evidence_lengths)) if evidence_lengths else 0.0
        ),
        "GEN.evidence_max_chars": max(evidence_lengths) if evidence_lengths else 0,
        "GEN.modality::image": int(modality_counts["image"]),
        "GEN.modality::table": int(modality_counts["table"]),
        "GEN.modality::text": int(modality_counts["text"]),
    }


def build_explicit_candidate(
    base: dict[str, Any], topology: dict[str, Any], localization: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build only contract-approved named values, plus validation metadata."""
    features = general_features(base)
    raw_required = required_list(topology, "predicted_required_transitions_raw")
    schema_required = required_list(
        topology, "predicted_required_transitions_schema_constrained"
    )
    required = set(schema_required)
    missing_raw = localization.get("missing_transitions")
    present_raw = localization.get("present_required_states")
    if not isinstance(missing_raw, list) or not isinstance(present_raw, list):
        raise RuntimeError("Clean localization missing list fields")
    missing = [str(x).upper() for x in missing_raw]
    if len(missing) != len(set(missing)) or any(t not in TRANSITIONS for t in missing):
        raise RuntimeError(f"Invalid missing transition list: {missing_raw}")

    decision = text_value(localization.get("decision")).upper()
    reason = text_value(localization.get("localized_reason"))
    localized = text_value(localization.get("localized_transition")).upper()
    if decision not in EXPECTED_DECISIONS:
        raise RuntimeError(f"Decision outside frozen categories: {decision!r}")
    if reason not in EXPECTED_REASONS:
        raise RuntimeError(f"Reason outside frozen categories: {reason!r}")
    if localized and localized not in TRANSITIONS:
        raise RuntimeError(f"Localized transition outside frozen categories: {localized!r}")

    features.update({
        "DIAG.decision": decision,
        "DIAG.structural_missing": int(decision == "STRUCTURAL_MISSING"),
        "DIAG.verifier_invalid": int(decision == "VERIFIER_INVALID"),
        "DIAG.clean_or_uncertain": int(decision == "CLEAN_OR_UNCERTAIN"),
        "LOC.decision": decision,
        "LOC.localized_reason": reason,
        "LOC.missing_transitions.__len__": len(missing),
        # Clean V2 renamed the old present_states payload to present_required_states.
        # The count semantics are identical and explicitly recorded in provenance.
        "LOC.present_states.__len__": len(present_raw),
        "TOPO.predicted_required_transitions_raw.__len__": len(raw_required),
        "TOPO.predicted_required_transitions_schema_constrained.__len__": len(schema_required),
    })

    optional_localized_numeric = {
        "LOC.localized_valid_prob": localization.get("localized_valid_prob"),
        "LOC.localized_threshold": localization.get("localized_threshold"),
        "LOC.localized_invalid_severity": localization.get("localized_invalid_severity"),
    }
    optional_present: list[str] = []
    for key, value in optional_localized_numeric.items():
        number = finite_number(value)
        if number is not None:
            features[key] = number
            optional_present.append(key)
        elif value is not None:
            raise RuntimeError(f"Non-finite/non-numeric localized value: {key}={value!r}")

    probabilities = topology.get("probabilities")
    if not isinstance(probabilities, dict):
        raise RuntimeError("Topology probabilities is not an object")
    for transition in TRANSITIONS:
        features[f"{transition}.required"] = int(transition in required)
        features[f"{transition}.localized"] = int(localized == transition)
        features[f"{transition}.localized_structural"] = int(
            localized == transition and decision == "STRUCTURAL_MISSING"
        )
        features[f"{transition}.localized_invalid"] = int(
            localized == transition and decision == "VERIFIER_INVALID"
        )
        features[f"LOC.missing_transitions.__contains__{transition}"] = int(
            transition in missing
        )
        features[
            f"TOPO.predicted_required_transitions_raw.__contains__{transition}"
        ] = int(transition in raw_required)
        features[
            f"TOPO.predicted_required_transitions_schema_constrained.__contains__{transition}"
        ] = int(transition in schema_required)
        if transition != "T5":
            probability = finite_number(probabilities.get(transition))
            if probability is None or not 0.0 <= probability <= 1.0:
                raise RuntimeError(f"Invalid topology probability for {transition}")
            features[f"TOPO.probabilities.{transition}"] = probability

    validation = {
        "raw_required": raw_required,
        "schema_required": schema_required,
        "missing": missing,
        "localized": localized,
        "decision": decision,
        "reason": reason,
        "present_required_count": len(present_raw),
        "optional_localized_numeric_present": optional_present,
    }
    return features, validation


def select_variant(
    candidate: dict[str, Any], whitelist: list[str], allowed_transitions: tuple[str, ...],
    localized_transition: str,
) -> dict[str, Any]:
    """Select exact named keys; optional numeric keys remain sparse when absent."""
    out: dict[str, Any] = {}
    for key in whitelist:
        if key == "LOC.localized_transition":
            if localized_transition in allowed_transitions:
                out[key] = localized_transition
            continue
        if key in candidate:
            out[key] = candidate[key]
    return out


def provenance_for_feature(key: str) -> dict[str, Any]:
    if key.startswith("GEN."):
        if key.startswith("GEN.modality::"):
            fields = ["retrieved_evidence[*].modality"]
            semantic = "Count of Exact Top-6 evidence items with the named modality; zero emitted when absent."
        elif key in {"GEN.evidence_count", "GEN.evidence_mean_chars", "GEN.evidence_max_chars"}:
            fields = ["retrieved_evidence[*].title", "retrieved_evidence[*].content"]
            semantic = "Exact Top-6 evidence count or title-plus-content character statistic under the frozen normalization."
        elif key.startswith("GEN.question"):
            fields = ["question"]
            semantic = "Question surface length under the frozen character/token definition."
        else:
            fields = ["draft_answer"]
            semantic = "Frozen Initial answer surface statistic under the frozen normalization."
        return {
            "source_artifact": str(BASE_FILE.relative_to(ROOT)),
            "source_fields": fields,
            "semantic": semantic,
            "deployment_available": True,
            "generation": "deterministic_local",
            "feature_type": "numeric",
        }

    if key.startswith("DIAG."):
        return {
            "source_artifact": str(LOCALIZATION_FILE.relative_to(ROOT)),
            "source_fields": ["decision"],
            "semantic": (
                "Frozen Clean V2 diagnosis category" if key == "DIAG.decision"
                else f"Indicator derived explicitly from frozen diagnosis for {key.split('.', 1)[1]}."
            ),
            "deployment_available": True,
            "generation": "frozen_localization_or_deterministic_indicator",
            "feature_type": "categorical" if key == "DIAG.decision" else "numeric",
        }

    if key.startswith("LOC."):
        if key == "LOC.present_states.__len__":
            fields = ["present_required_states"]
            semantic = "Count of Clean V2 present required states; explicit semantic rename from V1 LOC.present_states.__len__."
            feature_type = "numeric"
            emission = "always"
        elif key == "LOC.missing_transitions.__len__":
            fields = ["missing_transitions"]
            semantic = "Number of required-but-missing transitions under frozen Clean V2 localization."
            feature_type = "numeric"
            emission = "always"
        elif key.startswith("LOC.missing_transitions.__contains__"):
            fields = ["missing_transitions"]
            semantic = "Membership indicator in the frozen required-but-missing transition list."
            feature_type = "numeric"
            emission = "always"
        elif key == "LOC.localized_transition":
            fields = ["localized_transition"]
            semantic = "Categorical localized transition, emitted only when it belongs to the variant transition group."
            feature_type = "categorical"
            emission = "matching_transition_group_only"
        elif key in {"LOC.decision", "LOC.localized_reason"}:
            fields = [key.split(".", 1)[1]]
            semantic = "Frozen Clean V2 localization category."
            feature_type = "categorical"
            emission = "always"
        else:
            fields = [key.split(".", 1)[1]]
            semantic = "Localized aggregate verifier value from the single frozen VERIFIER_INVALID selection; no per-transition verifier expansion."
            feature_type = "numeric"
            emission = "sparse_omit_when_null"
        return {
            "source_artifact": str(LOCALIZATION_FILE.relative_to(ROOT)),
            "source_fields": fields,
            "semantic": semantic,
            "deployment_available": True,
            "generation": "frozen_clean_v2_localization",
            "feature_type": feature_type,
            "emission": emission,
        }

    if key.startswith("TOPO."):
        if ".probabilities." in key:
            transition = key.rsplit(".", 1)[-1]
            fields = [f"probabilities.{transition}"]
            semantic = "Question-only topology model probability under OOF Dev/full-fit deployment semantics."
        elif "schema_constrained" in key:
            fields = ["predicted_required_transitions_schema_constrained"]
            semantic = "Length or membership from schema-constrained question-only Expected Topology."
        else:
            fields = ["predicted_required_transitions_raw"]
            semantic = "Length or membership from raw question-only Expected Topology prediction."
        return {
            "source_artifact": str(TOPOLOGY_FILE.relative_to(ROOT)),
            "source_fields": fields,
            "semantic": semantic,
            "deployment_available": True,
            "generation": "question_only_topology_oof_dev_fullfit_deployment",
            "feature_type": "numeric",
        }

    transition_match = re.fullmatch(r"(T[1-5])\.(required|localized|localized_structural|localized_invalid)", key)
    if transition_match:
        transition, suffix = transition_match.groups()
        if suffix == "required":
            artifact = TOPOLOGY_FILE
            fields = ["predicted_required_transitions_schema_constrained"]
            semantic = f"Indicator that {transition} is required by schema-constrained Expected Topology."
            generation = "deterministic_membership"
        else:
            artifact = LOCALIZATION_FILE
            fields = ["decision", "localized_transition"]
            semantic = f"Explicit indicator for {transition} under frozen diagnosis/localization semantics ({suffix})."
            generation = "deterministic_indicator"
        return {
            "source_artifact": str(artifact.relative_to(ROOT)),
            "source_fields": fields,
            "semantic": semantic,
            "deployment_available": True,
            "generation": generation,
            "feature_type": "numeric",
        }

    raise RuntimeError(f"No provenance rule for contract feature: {key}")


def matches_forbidden(key: str, patterns: list[str]) -> bool:
    lowered = key.lower()
    return any(fnmatch.fnmatchcase(lowered, pattern.lower()) for pattern in patterns)


def main() -> None:
    # Preflight: verify exact contract and all upstream frozen hashes before reading rows.
    contract_sha = sha256_file(CONTRACT_FILE)
    if contract_sha != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError(
            f"Feature contract SHA mismatch: {contract_sha} != {EXPECTED_CONTRACT_SHA256}"
        )
    contract = load_json(CONTRACT_FILE)
    contract_hashes = load_json(CONTRACT_HASH_FILE)
    jobs_hashes = load_json(JOBS_HASH_FILE)
    localization_hashes = load_json(LOCALIZATION_HASH_FILE)

    if contract_hashes["generated_files"][CONTRACT_FILE.name]["sha256"] != contract_sha:
        raise RuntimeError("Contract SHA manifest mismatch")
    expected_source_hashes = {
        BASE_FILE: jobs_hashes["generated_files"][BASE_FILE.name]["sha256"],
        NATURAL_JOB_FILE: jobs_hashes["generated_files"][NATURAL_JOB_FILE.name]["sha256"],
        TOPOLOGY_FILE: jobs_hashes["read_only_source_files"][TOPOLOGY_FILE.name]["sha256"],
        LOCALIZATION_FILE: localization_hashes["files"][LOCALIZATION_FILE.name],
    }
    for path, expected_hash in expected_source_hashes.items():
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Frozen source SHA mismatch: {path}: {actual_hash} != {expected_hash}")

    variants_contract = contract.get("variants")
    if not isinstance(variants_contract, dict):
        raise RuntimeError("Contract variants missing")
    schema_counts = {
        variant: len(variants_contract[variant]["features"])
        for variant in EXPECTED_COUNTS
    }
    declared_counts = {
        variant: variants_contract[variant]["count"] for variant in EXPECTED_COUNTS
    }
    if schema_counts != EXPECTED_COUNTS or declared_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Frozen variant key-count mismatch: schema={schema_counts}, declared={declared_counts}"
        )
    if set(variants_contract) != set(EXPECTED_COUNTS):
        raise RuntimeError("Unexpected/missing variants in frozen contract")

    forbidden_patterns = list(contract["strict_exclusions"]["exact_or_prefix"])
    contract_feature_union: list[str] = []
    for variant in EXPECTED_COUNTS:
        features = variants_contract[variant]["features"]
        if len(features) != len(set(features)):
            raise RuntimeError(f"Duplicate contract feature in {variant}")
        for key in features:
            if key not in contract_feature_union:
                contract_feature_union.append(key)
            if matches_forbidden(key, forbidden_patterns):
                raise RuntimeError(f"Contract itself includes forbidden feature: {key}")

    # Explicit boundary: only aggregate localized verifier values are allowed.
    forbidden_per_transition_verifier_fragments = (
        ".valid_prob", ".threshold", ".reliability_weight", ".invalid_severity",
        ".n_components", ".mean_prob", ".max_prob", ".severity", ".score_available",
    )
    for key in contract_feature_union:
        if key.startswith(TRANSITIONS) and any(
            fragment in key for fragment in forbidden_per_transition_verifier_fragments
        ):
            raise RuntimeError(f"Per-transition verifier numeric leaked into contract: {key}")

    base = unique_by_qid(load_jsonl(BASE_FILE), "Exact Top-6 initial inputs")
    natural_jobs = unique_by_qid(load_jsonl(NATURAL_JOB_FILE), "Exact Top-6 natural jobs")
    topology = unique_by_qid(load_jsonl(TOPOLOGY_FILE), "OOF topology")
    state_manifest = unique_by_qid(load_jsonl(STATE_MANIFEST_FILE), "Clean V2 state manifest")
    localization = unique_by_qid(load_jsonl(LOCALIZATION_FILE), "Clean V2 localization")

    qid_sets = {
        "exact_top6_initial_inputs": set(base),
        "exact_top6_natural_jobs": set(natural_jobs),
        "oof_topology": set(topology),
        "clean_v2_state_manifest": set(state_manifest),
        "clean_v2_localization": set(localization),
    }
    reference_qids = set(base)
    qid_checks = {
        name: {
            "count": len(qids),
            "equals_reference": qids == reference_qids,
            "missing": sorted(reference_qids - qids)[:20],
            "extra": sorted(qids - reference_qids)[:20],
        }
        for name, qids in qid_sets.items()
    }
    if len(reference_qids) != EXPECTED_N or not all(
        item["count"] == EXPECTED_N and item["equals_reference"]
        for item in qid_checks.values()
    ):
        raise RuntimeError(f"QID coverage mismatch: {qid_checks}")

    preflight = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_FEATURE_GENERATION_PREFLIGHT_PASS",
        "feature_contract_sha256": contract_sha,
        "feature_contract_status": contract.get("status"),
        "variant_raw_key_counts": schema_counts,
        "qid_checks": qid_checks,
        "generic_recursive_flatten_used": False,
        "external_api_calls": 0,
        "gold_opened": False,
        "locked_1341_opened_or_evaluated": False,
        "answer_risk_training": False,
        "v1_modified": False,
    }
    frozen_write(PREFLIGHT_FILE, json_bytes(preflight))

    matrix_rows: list[dict[str, Any]] = []
    exact_top6_failures: list[dict[str, Any]] = []
    topology_parity_failures: list[dict[str, Any]] = []
    localization_parity_failures: list[dict[str, Any]] = []
    forbidden_feature_failures: list[dict[str, Any]] = []
    whitelist_failures: list[dict[str, Any]] = []
    null_or_nonfinite_failures: list[dict[str, Any]] = []
    categorical_values: dict[str, set[str]] = defaultdict(set)
    emitted_key_count_histograms: dict[str, Counter[int]] = {
        variant: Counter() for variant in EXPECTED_COUNTS
    }
    emitted_counts_by_key: dict[str, Counter[str]] = {
        variant: Counter() for variant in EXPECTED_COUNTS
    }

    for qid in sorted(reference_qids):
        base_row = base[qid]
        job_row = natural_jobs[qid]
        topo_row = topology[qid]
        state_row = state_manifest[qid]
        loc_row = localization[qid]

        base_ids = exact_top6_evidence_ids(base_row)
        job_ids = exact_top6_evidence_ids(job_row)
        exact_reasons: list[str] = []
        if len(base_ids) != 6 or len(set(base_ids)) != 6 or any(not x for x in base_ids):
            exact_reasons.append("base_not_exactly_6_unique_nonblank_evidence_ids")
        if base_ids != job_ids:
            exact_reasons.append("base_and_natural_job_evidence_id_order_mismatch")
        if text_value(base_row.get("question")) != text_value(job_row.get("question")):
            exact_reasons.append("question_mismatch_between_base_and_natural_job")
        if text_value(base_row.get("draft_answer")) != text_value(job_row.get("draft_answer")):
            exact_reasons.append("draft_mismatch_between_base_and_natural_job")
        if exact_reasons:
            exact_top6_failures.append({"qid": qid, "reasons": exact_reasons})

        raw_required = required_list(topo_row, "predicted_required_transitions_raw")
        schema_required = required_list(
            topo_row, "predicted_required_transitions_schema_constrained"
        )
        state_required = [str(x).upper() for x in state_row.get("predicted_required_transitions_oof", [])]
        topo_probs = topo_row.get("probabilities")
        state_probs = state_row.get("topology_probabilities")
        if schema_required != state_required or topo_probs != state_probs:
            topology_parity_failures.append({
                "qid": qid,
                "schema_required": schema_required,
                "state_required": state_required,
                "probabilities_equal": topo_probs == state_probs,
            })

        missing = [str(x).upper() for x in loc_row.get("missing_transitions", [])]
        state_missing = [str(x).upper() for x in state_row.get("required_but_missing", [])]
        state_present = [str(x).upper() for x in state_row.get("required_and_present", [])]
        loc_present_rows = loc_row.get("present_required_states", [])
        loc_present = [
            str(x.get("transition", "")).upper()
            for x in loc_present_rows if isinstance(x, dict)
        ] if isinstance(loc_present_rows, list) else []
        if missing != state_missing or loc_present != state_present:
            localization_parity_failures.append({
                "qid": qid,
                "missing": missing,
                "state_missing": state_missing,
                "present": loc_present,
                "state_present": state_present,
            })

        candidate, validation = build_explicit_candidate(base_row, topo_row, loc_row)
        variants: dict[str, dict[str, Any]] = {}
        for variant, expected_count in EXPECTED_COUNTS.items():
            whitelist = list(variants_contract[variant]["features"])
            features = select_variant(
                candidate,
                whitelist,
                VARIANT_TRANSITIONS[variant],
                validation["localized"],
            )
            variants[variant] = features
            emitted_key_count_histograms[variant][len(features)] += 1
            emitted_counts_by_key[variant].update(features.keys())

            unexpected = sorted(set(features) - set(whitelist))
            if unexpected:
                whitelist_failures.append({"qid": qid, "variant": variant, "keys": unexpected})
            forbidden = sorted(
                key for key in features if matches_forbidden(key, forbidden_patterns)
            )
            if forbidden:
                forbidden_feature_failures.append({"qid": qid, "variant": variant, "keys": forbidden})
            for key, value in features.items():
                if value is None or (
                    isinstance(value, float) and not math.isfinite(value)
                ):
                    null_or_nonfinite_failures.append({
                        "qid": qid, "variant": variant, "key": key, "value": value
                    })
                if key in {
                    "DIAG.decision", "LOC.decision", "LOC.localized_reason",
                    "LOC.localized_transition",
                }:
                    categorical_values[key].add(str(value))

        # qid is the row envelope/index only and is never included in a feature dictionary.
        matrix_rows.append({"qid": qid, "variants": variants})

    if exact_top6_failures:
        raise RuntimeError(f"Exact Top-6 parity failed: {exact_top6_failures[:5]}")
    if topology_parity_failures:
        raise RuntimeError(f"Topology parity failed: {topology_parity_failures[:5]}")
    if localization_parity_failures:
        raise RuntimeError(f"Localization parity failed: {localization_parity_failures[:5]}")
    if forbidden_feature_failures or whitelist_failures or null_or_nonfinite_failures:
        raise RuntimeError(
            "Feature audit failed: "
            f"forbidden={forbidden_feature_failures[:5]} "
            f"whitelist={whitelist_failures[:5]} "
            f"null={null_or_nonfinite_failures[:5]}"
        )

    categorical_expected = {
        "DIAG.decision": EXPECTED_DECISIONS,
        "LOC.decision": EXPECTED_DECISIONS,
        "LOC.localized_reason": EXPECTED_REASONS,
        "LOC.localized_transition": set(TRANSITIONS),
    }
    categorical_checks = {
        key: {
            "observed": sorted(categorical_values[key]),
            "allowed": sorted(allowed),
            "observed_subset_of_allowed": categorical_values[key] <= allowed,
            "all_allowed_values_observed": categorical_values[key] == allowed,
        }
        for key, allowed in categorical_expected.items()
    }
    if not all(item["observed_subset_of_allowed"] for item in categorical_checks.values()):
        raise RuntimeError(f"Unexpected categorical value: {categorical_checks}")

    provenance = {
        key: provenance_for_feature(key) for key in contract_feature_union
    }
    deployment_unavailable = sorted(
        key for key, item in provenance.items() if item.get("deployment_available") is not True
    )
    if deployment_unavailable or set(provenance) != set(contract_feature_union):
        raise RuntimeError("Feature provenance/deployment-availability audit failed")

    schema = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_FEATURE_SCHEMA_AND_PROVENANCE_FROZEN",
        "feature_contract_sha256": contract_sha,
        "variant_schemas": {
            variant: {
                "raw_key_count": len(variants_contract[variant]["features"]),
                "raw_keys": variants_contract[variant]["features"],
                "transition_scope": list(VARIANT_TRANSITIONS[variant]),
            }
            for variant in EXPECTED_COUNTS
        },
        "feature_provenance": provenance,
        "matrix_envelope": {
            "qid": "row index only; explicitly not a model feature",
            "variants": "sparse raw feature dictionaries selected by explicit whitelist",
        },
        "sparse_semantics": {
            "localized_numeric": "omit key when frozen localization value is null",
            "localized_transition": "emit only when localized transition belongs to the variant transition group",
            "all_other_contract_keys": "emitted for every QID",
        },
        "generic_recursive_flatten_used": False,
    }

    audit = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_FEATURE_SCHEMA_PARITY_AUDIT_PASS",
        "qid_coverage": len(matrix_rows),
        "unique_qids": len({row["qid"] for row in matrix_rows}),
        "variant_raw_key_counts": schema_counts,
        "variant_declared_key_counts": declared_counts,
        "variant_expected_key_counts": EXPECTED_COUNTS,
        "variant_key_count_contract_pass": schema_counts == declared_counts == EXPECTED_COUNTS,
        "per_row_emitted_key_count_histograms": {
            variant: {str(k): v for k, v in sorted(hist.items())}
            for variant, hist in emitted_key_count_histograms.items()
        },
        "per_key_emitted_qid_counts": {
            variant: dict(sorted(counts.items()))
            for variant, counts in emitted_counts_by_key.items()
        },
        "sparse_missing_policy": "localized numeric nulls and nonmatching localized-transition categories are omitted; null is never emitted",
        "null_or_nonfinite_value_violations": len(null_or_nonfinite_failures),
        "exact_top6_parity_failures": len(exact_top6_failures),
        "topology_parity_failures": len(topology_parity_failures),
        "localization_parity_failures": len(localization_parity_failures),
        "whitelist_violations": len(whitelist_failures),
        "forbidden_feature_violations": len(forbidden_feature_failures),
        "per_transition_verifier_numeric_features_added": False,
        "categorical_value_checks": categorical_checks,
        "all_feature_provenance_present": set(provenance) == set(contract_feature_union),
        "all_features_deployment_available": not deployment_unavailable,
        "deployment_unavailable_features": deployment_unavailable,
        "qid_is_feature": False,
        "generic_recursive_flatten_used": False,
        "external_api_calls": 0,
        "gold_opened": False,
        "locked_1341_opened_or_evaluated": False,
        "answer_risk_training": False,
        "v1_modified": False,
    }

    frozen_write(MATRIX_FILE, jsonl_bytes(matrix_rows))
    frozen_write(SCHEMA_FILE, json_bytes(schema))
    frozen_write(AUDIT_FILE, json_bytes(audit))

    summary = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_FEATURE_GENERATION_V1_COMPLETE",
        "feature_contract_sha256": contract_sha,
        "qid_coverage": len(matrix_rows),
        "feature_matrix_rows": len(matrix_rows),
        "variant_raw_key_counts": schema_counts,
        "diagnosis_counts": dict(sorted(Counter(
            text_value(localization[qid].get("decision")) for qid in reference_qids
        ).items())),
        "exact_top6_evidence_count_values": sorted({
            len(base[qid].get("retrieved_evidence", [])) for qid in reference_qids
        }),
        "schema_parity_audit_status": audit["status"],
        "feature_provenance_count": len(provenance),
        "generic_recursive_flatten_used": False,
        "per_transition_verifier_numeric_features_added": False,
        "external_api_calls": 0,
        "gold_opened": False,
        "locked_1341_opened_or_evaluated": False,
        "answer_risk_training": False,
        "v1_modified": False,
        "next_step": "STOP_BEFORE_OOF_ANSWER_RISK_TRAINING",
    }
    frozen_write(SUMMARY_FILE, json_bytes(summary))

    generated = [PREFLIGHT_FILE, MATRIX_FILE, SCHEMA_FILE, AUDIT_FILE, SUMMARY_FILE]
    source_files = [
        CONTRACT_FILE, BASE_FILE, NATURAL_JOB_FILE, TOPOLOGY_FILE,
        STATE_MANIFEST_FILE, LOCALIZATION_FILE, Path(__file__).resolve(),
    ]
    hashes = {
        "status": "MMQA_CLEAN_RISK_V2_DEV1096_FEATURE_GENERATION_SHA_FROZEN",
        "algorithm": "SHA-256",
        "namespace": OUT_DIR.name,
        "generated_files": {
            path.name: {
                "path": str(path), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in generated
        },
        "read_only_sources": {
            path.name: {
                "path": str(path), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_files
        },
        "external_api_calls": 0,
        "gold_opened": False,
        "locked_1341_opened_or_evaluated": False,
        "answer_risk_training": False,
        "v1_modified": False,
    }
    frozen_write(HASH_FILE, json_bytes(hashes))

    print("=" * 120)
    print("MMQA CLEAN RISK V2 — DEV1096 FEATURE GENERATION V1")
    print("=" * 120)
    print("Contract SHA:", contract_sha, "PASS")
    print("QID coverage:", len(matrix_rows), "/", EXPECTED_N)
    print("Variant raw key counts:", schema_counts)
    print("Exact Top-6 parity failures:", len(exact_top6_failures))
    print("Topology/localization parity failures:", len(topology_parity_failures), "/", len(localization_parity_failures))
    print("Whitelist/forbidden/null violations:", len(whitelist_failures), "/", len(forbidden_feature_failures), "/", len(null_or_nonfinite_failures))
    print("Categorical values:", {k: sorted(v) for k, v in categorical_values.items()})
    print("All provenance/deployment availability: PASS")
    print("API/Gold/locked/risk-training/V1-modified: 0 / 0 / 0 / 0 / 0")
    print("Feature matrix:", MATRIX_FILE)
    print("Schema/parity audit:", AUDIT_FILE)
    print("SHA manifest:", HASH_FILE)
    print("STATUS =", summary["status"])
    print("NEXT STEP =", summary["next_step"])


if __name__ == "__main__":
    main()
