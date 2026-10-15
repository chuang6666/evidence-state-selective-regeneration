from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent

BASE_TRAIN_READY = (
    ROOT / "outputs" / "mmqa_dev1100_verifier_train_ready_v1"
    / "mmqa_verifier_train_ready_v1.jsonl"
)
NATURAL_INSTANCES = (
    ROOT / "outputs" / "mmqa_natural_verifier_inference_inputs_v1"
    / "mmqa_natural_verifier_instances_v1.jsonl"
)
V2_ROOT = ROOT / "outputs" / "mmqa_transition_causal_verifier_v2_pairwise"
V2_OOF = V2_ROOT / "oof_predictions_available.jsonl"
V2_NATURAL_TRANSITION_SCORES = (
    ROOT / "outputs" / "mmqa_natural_frozen_verifier_v2_inference_v1"
    / "mmqa_natural_frozen_verifier_transition_scores_v1.jsonl"
)
V2_NATURAL_INSTANCE_SCORES = (
    ROOT / "outputs" / "mmqa_natural_frozen_verifier_v2_inference_v1"
    / "mmqa_natural_frozen_verifier_instance_scores_v1.jsonl"
)

OUT_ROOT = ROOT / "outputs" / "mmqa_transition_causal_verifier_v3_t1_t2_natural_adapter_corrected"
NATURALIZED_FILE = OUT_ROOT / "naturalized_t1_t2_t3_train_ready_v1.jsonl"
NATURALIZED_PAIR_FILE = OUT_ROOT / "naturalized_t1_t2_t3_pairs_v1.jsonl"
BUILD_AUDIT_FILE = OUT_ROOT / "naturalized_supervision_build_audit_v1.json"
OOF_INSTANCE_FILE = OUT_ROOT / "v3_oof_instance_predictions.jsonl"
OOF_PAIR_FILE = OUT_ROOT / "v3_oof_pair_predictions.jsonl"
NATURAL_INSTANCE_SCORE_FILE = OUT_ROOT / "v3_natural_instance_scores.jsonl"
NATURAL_TRANSITION_SCORE_FILE = OUT_ROOT / "v3_natural_transition_scores.jsonl"
SUMMARY_FILE = OUT_ROOT / "verifier_v3_natural_adapter_summary.json"

TRANSITIONS = ["T1", "T2", "T3", "T4", "T5"]
ADAPT_TRANSITIONS = ["T1", "T2"]
T2ID = {t: i for i, t in enumerate(TRANSITIONS)}
A2ID = {t: i for i, t in enumerate(ADAPT_TRANSITIONS)}

EXPECTED_TAGS = {
    "T1": ["QUESTION", "VISUAL_EVIDENCE", "CANDIDATE_GROUNDING_CLAIM"],
    "T2": ["QUESTION", "UPSTREAM_CLAIM", "CANDIDATE_BRIDGE_EVIDENCE", "DOWNSTREAM_CLAIM"],
    "T3": ["QUESTION", "BRANCH_VISUAL_EVIDENCE", "CANDIDATE_BRANCH_CLAIM"],
}

SEED = 42


# ----------------------------
# Generic IO / reproducibility
# ----------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise RuntimeError(f"Malformed JSONL {path}, line {line_no}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Exact V2 serializer / packer
# ----------------------------

def parse_sections(text: str) -> dict[str, str]:
    out = {}
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line.startswith("[") or "]" not in line:
            continue
        tag, content = line.split("]", 1)
        out[tag[1:].strip()] = content.strip()
    return out


def token_clip(tokenizer, text: str, budget: int) -> str:
    # EXACT frozen V2 inference behavior.
    # Important: always tokenize -> truncate -> decode, even when the text
    # is already shorter than the budget. Returning raw text here changes
    # RoBERTa whitespace normalization and breaks probability reproducibility.
    if not text or budget <= 0:
        return ""
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=budget,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)

def focus_evidence(evidence: str, anchors: list[str], max_chars: int = 7000) -> str:
    if len(evidence) <= max_chars:
        return evidence
    low = evidence.casefold()
    for anchor in anchors:
        a = str(anchor or "").strip()
        if not a:
            continue
        pos = low.find(a.casefold())
        if pos >= 0:
            half = max_chars // 2
            start = max(0, pos - half)
            end = min(len(evidence), start + max_chars)
            start = max(0, end - max_chars)
            return evidence[start:end]
    return evidence[:max_chars]


def pack_input(tokenizer, row: dict[str, Any], max_length: int) -> str:
    # Kept byte-for-byte equivalent in logic to V2 trainer.
    sec = parse_sections(row["input_text"])
    t = row["transition_short"]

    def line(tag: str, text: str, budget: int) -> str:
        return f"[{tag}] {token_clip(tokenizer, text, budget)}"

    parts = [f"[TRANSITION] {t}"]

    if t == "T1":
        parts += [
            line("QUESTION", sec.get("QUESTION", ""), 96),
            line("VISUAL_EVIDENCE", sec.get("VISUAL_EVIDENCE", ""), 220),
            line("CANDIDATE_GROUNDING_CLAIM", sec.get("CANDIDATE_GROUNDING_CLAIM", ""), 96),
        ]
    elif t == "T2":
        q = sec.get("QUESTION", "")
        up = sec.get("UPSTREAM_CLAIM", "")
        down = sec.get("DOWNSTREAM_CLAIM", "")
        ev = focus_evidence(sec.get("CANDIDATE_BRIDGE_EVIDENCE", ""), [down, up])
        parts += [
            line("QUESTION", q, 96),
            line("UPSTREAM_CLAIM", up, 96),
            line("DOWNSTREAM_CLAIM", down, 96),
            line("CANDIDATE_BRIDGE_EVIDENCE", ev, 200),
        ]
    elif t == "T3":
        parts += [
            line("QUESTION", sec.get("QUESTION", ""), 96),
            line("BRANCH_VISUAL_EVIDENCE", sec.get("BRANCH_VISUAL_EVIDENCE", ""), 220),
            line("CANDIDATE_BRANCH_CLAIM", sec.get("CANDIDATE_BRANCH_CLAIM", ""), 96),
        ]
    elif t == "T4":
        parts += [
            line("QUESTION", sec.get("QUESTION", ""), 96),
            line("AGGREGATION_OPERATOR", sec.get("AGGREGATION_OPERATOR", ""), 32),
            line("VERIFIED_BRANCH_VALUES", sec.get("VERIFIED_BRANCH_VALUES", ""), 160),
            line("CANDIDATE_AGGREGATION_RESULT", sec.get("CANDIDATE_AGGREGATION_RESULT", ""), 96),
        ]
    elif t == "T5":
        parts += [
            line("QUESTION", sec.get("QUESTION", ""), 96),
            line("VERIFIED_UPSTREAM_CLAIMS", sec.get("VERIFIED_UPSTREAM_CLAIMS", ""), 220),
            line("CANDIDATE_FINAL_ANSWER", sec.get("CANDIDATE_FINAL_ANSWER", ""), 96),
        ]
    else:
        raise ValueError(f"Unknown transition_short={t}")

    packed = "\n".join(parts)
    ids = tokenizer.encode(
        packed,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    return tokenizer.decode(ids, skip_special_tokens=True)


# ----------------------------
# Natural instance conversion
# ----------------------------

def natural_instance_to_sections(row: dict[str, Any]) -> dict[str, str]:
    t = str(row["transition_type"])
    f = row.get("fields") or {}
    q = str(f.get("question", "") or "").strip()

    if t == "T1":
        return {
            "QUESTION": q,
            "VISUAL_EVIDENCE": str(f.get("visual_evidence_text", "") or "").strip(),
            "CANDIDATE_GROUNDING_CLAIM": str(f.get("candidate_grounding_claim", "") or "").strip(),
        }

    if t == "T2":
        return {
            "QUESTION": q,
            "UPSTREAM_CLAIM": str(f.get("upstream_claim", "") or "").strip(),
            "CANDIDATE_BRIDGE_EVIDENCE": str(f.get("candidate_bridge_evidence_text", "") or "").strip(),
            "DOWNSTREAM_CLAIM": str(f.get("downstream_claim", "") or "").strip(),
        }

    if t == "T3":
        return {
            "QUESTION": q,
            "BRANCH_VISUAL_EVIDENCE": str(f.get("branch_visual_evidence_text", "") or "").strip(),
            "CANDIDATE_BRANCH_CLAIM": str(f.get("candidate_branch_claim", "") or "").strip(),
        }

    raise ValueError(t)


def sections_to_input_text(t: str, sec: dict[str, str]) -> str:
    tags = EXPECTED_TAGS[t]
    return "\n".join(f"[{tag}] {str(sec.get(tag, '') or '').strip()}" for tag in tags)


# ----------------------------
# Text similarity for safe matching
# ----------------------------

def norm_tokens(text: Any) -> set[str]:
    import re
    s = str(text or "").casefold()
    toks = re.findall(r"[a-z0-9\u00c0-\u024f\u0400-\u04ff\u4e00-\u9fff]+", s)
    stop = {
        "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at",
        "by", "with", "from", "is", "are", "was", "were", "be", "been",
        "what", "which", "who", "when", "where", "how", "does", "did",
    }
    return {x for x in toks if len(x) >= 2 and x not in stop}


def semantic_overlap(a: Any, b: Any) -> float:
    ta, tb = norm_tokens(a), norm_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    jacc = inter / len(ta | tb)
    containment = inter / min(len(ta), len(tb))
    return max(jacc, containment)


# ----------------------------
# Build naturalized counterfactual pairs
# ----------------------------

def build_base_pairs(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for r in base_rows:
        t = str(r.get("transition_short"))
        if t not in ADAPT_TRANSITIONS:
            continue
        grouped[(t, str(r["pair_id"]))].append(r)

    pairs = []
    malformed = []
    for (t, pid), items in grouped.items():
        if len(items) != 2 or sorted(int(x["label"]) for x in items) != [0, 1]:
            malformed.append((t, pid))
            continue
        pos = next(x for x in items if int(x["label"]) == 1)
        neg = next(x for x in items if int(x["label"]) == 0)
        if str(pos["qid"]) != str(neg["qid"]) or int(pos["fold"]) != int(neg["fold"]):
            malformed.append((t, pid))
            continue

        ps = parse_sections(pos["input_text"])
        ns = parse_sections(neg["input_text"])
        tags = [x for x in EXPECTED_TAGS[t] if x != "QUESTION"]
        changed = [tag for tag in tags if str(ps.get(tag, "")) != str(ns.get(tag, ""))]

        # T1/T2 are expected to be one-component counterfactuals.
        # T3 may be a branch-bundle counterfactual: the visual branch evidence,
        # the branch claim, or both may change together.
        valid_changed = (
            (t in {"T1", "T2"} and len(changed) == 1)
            or (t == "T3" and 1 <= len(changed) <= 2)
        )
        if not valid_changed:
            malformed.append((t, pid, "changed_tags", changed))
            continue

        pairs.append({
            "transition": t,
            "pair_id": pid,
            "qid": str(pos["qid"]),
            "fold": int(pos["fold"]),
            "positive": pos,
            "negative": neg,
            "positive_sections": ps,
            "negative_sections": ns,
            "changed_tags": changed,
            "domain": "controlled",
        })

    if malformed:
        print(
            f"[pair-audit warning] skipped malformed T1/T2/T3 pairs: "
            f"{len(malformed)}; examples={malformed[:5]}"
        )
    return pairs


def choose_natural_instance(
    pair: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    if not candidates:
        return None, 0.0

    t = pair["transition"]
    changed = set(pair["changed_tags"])
    base_pos = pair["positive_sections"]

    best = None
    best_score = -1.0

    for cand in candidates:
        sec = natural_instance_to_sections(cand)
        scores = []

        # Prefer a natural component whose unchanged semantics resemble the
        # controlled positive pair. This is only a matching heuristic; it
        # does not generate labels.
        for tag in EXPECTED_TAGS[t]:
            if tag == "QUESTION" or tag in changed:
                continue
            nv = sec.get(tag, "")
            bv = base_pos.get(tag, "")
            if nv and bv:
                scores.append(semantic_overlap(nv, bv))

        # Question is same-qid but add a tiny check for malformed joins.
        qsim = semantic_overlap(sec.get("QUESTION", ""), base_pos.get("QUESTION", ""))
        if qsim > 0:
            scores.append(qsim)

        score = float(np.mean(scores)) if scores else 0.0
        if score > best_score:
            best_score = score
            best = cand

    return best, best_score


def naturalize_pair(
    pair: dict[str, Any],
    natural_row: dict[str, Any],
    match_score: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    t = pair["transition"]
    changed_tags = set(pair["changed_tags"])
    ps = pair["positive_sections"]
    ns = pair["negative_sections"]
    nat = natural_instance_to_sections(natural_row)

    pos_sec = {}
    neg_sec = {}
    naturalized_unchanged_tags = []

    for tag in EXPECTED_TAGS[t]:
        if tag == "QUESTION":
            q = nat.get("QUESTION") or ps.get("QUESTION", "")
            pos_sec[tag] = q
            neg_sec[tag] = q
            continue

        if tag in changed_tags:
            # Preserve the frozen causal corruption by default.
            pos_sec[tag] = ps.get(tag, "")
            neg_sec[tag] = ns.get(tag, "")
        else:
            # Replace unchanged context by the same-qid natural state.
            nv = nat.get(tag, "")
            if nv:
                pos_sec[tag] = nv
                neg_sec[tag] = nv
                naturalized_unchanged_tags.append(tag)
            else:
                pos_sec[tag] = ps.get(tag, "")
                neg_sec[tag] = ps.get(tag, "")

    # Conservative bundle-aware matching for authentic natural phrasing.
    pos_sims = []
    neg_sims = []
    for tag in changed_tags:
        nv = nat.get(tag, "")
        if not nv:
            continue
        pos_sims.append(semantic_overlap(nv, ps.get(tag, "")))
        neg_sims.append(semantic_overlap(nv, ns.get(tag, "")))

    sim_pos = float(np.mean(pos_sims)) if pos_sims else 0.0
    sim_neg = float(np.mean(neg_sims)) if neg_sims else 0.0

    natural_changed_role = "not_used"
    if sim_pos >= 0.70 and sim_pos >= sim_neg + 0.15:
        for tag in changed_tags:
            nv = nat.get(tag, "")
            if nv:
                pos_sec[tag] = nv
        natural_changed_role = "positive"
    elif sim_neg >= 0.70 and sim_neg >= sim_pos + 0.15:
        for tag in changed_tags:
            nv = nat.get(tag, "")
            if nv:
                neg_sec[tag] = nv
        natural_changed_role = "negative"

    # T3 branch-bundle pairs sometimes change both evidence and claim.
    # If so, there is no unchanged semantic context to naturalize except the
    # question. Keep the pair only when the observed natural branch bundle can
    # be conservatively attached to one labelled side.
    effective = bool(naturalized_unchanged_tags) or natural_changed_role != "not_used"

    nat_instance_id = str(natural_row["instance_id"])
    new_pid = f"naturalized::{pair['pair_id']}::{nat_instance_id}"

    common = {
        "pair_id": new_pid,
        "qid": pair["qid"],
        "fold": pair["fold"],
        "transition_short": t,
        "transition_type": {
            "T1": "T1_visual_grounding",
            "T2": "T2_sequential_bridge",
            "T3": "T3_parallel_branch",
        }[t],
        "source_domain": "naturalized",
        "source_base_pair_id": pair["pair_id"],
        "source_natural_instance_id": nat_instance_id,
        "changed_tags": sorted(changed_tags),
        "naturalized_unchanged_tags": naturalized_unchanged_tags,
        "natural_match_score": match_score,
        "natural_changed_role": natural_changed_role,
        "natural_changed_sim_pos": sim_pos,
        "natural_changed_sim_neg": sim_neg,
        "naturalization_effective": effective,
    }

    pos = {
        **common,
        "instance_id": new_pid + "::positive",
        "label": 1,
        "input_text": sections_to_input_text(t, pos_sec),
    }
    neg = {
        **common,
        "instance_id": new_pid + "::negative",
        "label": 0,
        "input_text": sections_to_input_text(t, neg_sec),
    }

    audit = {
        "pair_id": new_pid,
        "qid": pair["qid"],
        "fold": pair["fold"],
        "transition": t,
        "changed_tags": sorted(changed_tags),
        "naturalized_unchanged_tags": naturalized_unchanged_tags,
        "natural_instance_id": nat_instance_id,
        "match_score": match_score,
        "natural_changed_role": natural_changed_role,
        "natural_changed_sim_pos": sim_pos,
        "natural_changed_sim_neg": sim_neg,
        "naturalization_effective": effective,
    }
    return pos, neg, audit

def build_naturalized_supervision(
    base_rows: list[dict[str, Any]],
    natural_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_pairs = build_base_pairs(base_rows)

    nat_by_qt = defaultdict(list)
    for r in natural_rows:
        t = str(r.get("transition_type"))
        if t in ADAPT_TRANSITIONS:
            nat_by_qt[(str(r["qid"]), t)].append(r)

    naturalized_rows = []
    naturalized_pairs = []
    audits = []
    skip = Counter()
    by_t = Counter()
    changed_tags = Counter()
    natural_changed_roles = Counter()
    match_scores = defaultdict(list)

    for p in base_pairs:
        candidates = nat_by_qt.get((p["qid"], p["transition"]), [])
        if not candidates:
            skip[f"{p['transition']}::no_natural_instance"] += 1
            continue

        nat, score = choose_natural_instance(p, candidates)
        if nat is None:
            skip[f"{p['transition']}::no_match"] += 1
            continue

        pos, neg, audit = naturalize_pair(p, nat, score)
        if not audit["naturalization_effective"]:
            skip[f"{p['transition']}::no_effective_naturalization"] += 1
            continue
        if pos["input_text"] == neg["input_text"]:
            skip[f"{p['transition']}::identical_after_naturalize"] += 1
            continue

        naturalized_rows.extend([pos, neg])
        naturalized_pairs.append({
            "transition": p["transition"],
            "pair_id": pos["pair_id"],
            "qid": p["qid"],
            "fold": p["fold"],
            "domain": "naturalized",
            "positive": pos,
            "negative": neg,
        })
        audits.append(audit)

        by_t[p["transition"]] += 1
        for changed_tag in p["changed_tags"]:
            changed_tags[(p["transition"], changed_tag)] += 1
        natural_changed_roles[(p["transition"], audit["natural_changed_role"])] += 1
        match_scores[p["transition"]].append(float(score))

    # Leakage / pair integrity audit.
    qid_folds = defaultdict(set)
    for r in naturalized_rows:
        qid_folds[str(r["qid"])].add(int(r["fold"]))
    fold_conflicts = {q: sorted(v) for q, v in qid_folds.items() if len(v) != 1}
    if fold_conflicts:
        raise RuntimeError(f"Naturalized qid fold conflicts: {list(fold_conflicts.items())[:10]}")

    build_summary = {
        "base_t1_t2_t3_pairs": len(base_pairs),
        "naturalized_pairs": len(naturalized_pairs),
        "naturalized_binary_instances": len(naturalized_rows),
        "naturalized_pair_counts": dict(by_t),
        "skip_counts": dict(skip),
        "changed_tag_counts": {
            f"{t}::{tag}": n for (t, tag), n in changed_tags.items()
        },
        "natural_changed_role_counts": {
            f"{t}::{role}": n for (t, role), n in natural_changed_roles.items()
        },
        "mean_match_score": {
            t: (float(np.mean(v)) if v else None)
            for t, v in match_scores.items()
        },
        "qid_fold_conflict_count": len(fold_conflicts),
        "label_policy": (
            "Labels are inherited from the already frozen gold-derived causal "
            "counterfactual pairs. T1/T2 preserve their changed component; T3 also supports "
            "branch-bundle counterfactuals. Unchanged context is replaced by same-qid natural "
            "state text. A natural changed component/bundle is attached to a labelled side only "
            "under conservative semantic matching."
        ),
        "inference_leakage": False,
        "api_model_calls": 0,
        "status": "NATURALIZED_T1_T2_T3_SUPERVISION_READY",
    }

    write_jsonl(NATURALIZED_FILE, naturalized_rows)
    write_jsonl(NATURALIZED_PAIR_FILE, [
        {
            "transition": p["transition"],
            "pair_id": p["pair_id"],
            "qid": p["qid"],
            "fold": p["fold"],
            "domain": p["domain"],
            "positive_instance_id": p["positive"]["instance_id"],
            "negative_instance_id": p["negative"]["instance_id"],
        }
        for p in naturalized_pairs
    ])
    BUILD_AUDIT_FILE.write_text(
        json.dumps(
            {
                "summary": build_summary,
                "pair_audits": audits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return naturalized_rows, naturalized_pairs, build_summary



def preflight_reproduce_saved_v2_natural_scores(
    natural_rows: list[dict[str, Any]],
    qid_fold: dict[str, int],
    saved_instance_rows: list[dict[str, Any]],
    args,
    device: torch.device,
) -> dict[str, Any]:
    """
    Recompute frozen V2 probabilities for ALL Natural T1/T2 instances using the
    corrected exact packer, then compare instance-by-instance with the already
    saved V2 inference file. Abort before optimization if the baseline cannot
    be reproduced.
    """
    saved = {
        str(r["instance_id"]): float(r["prob_positive"])
        for r in saved_instance_rows
        if str(r.get("transition")) in ADAPT_TRANSITIONS
    }

    rows = [
        r for r in natural_rows
        if str(r.get("transition_type")) in ADAPT_TRANSITIONS
    ]
    by_fold = defaultdict(list)
    for r in rows:
        qid = str(r["qid"])
        if qid not in qid_fold:
            raise RuntimeError(f"Preflight: qid missing V2 fold: {qid}")
        by_fold[qid_fold[qid]].append(r)

    diffs = []
    per_t = defaultdict(list)
    compared = 0
    missing_saved = []

    for fold in range(5):
        sub = by_fold.get(fold, [])
        if not sub:
            continue

        ckpt = torch.load(
            V2_ROOT / f"fold_{fold}" / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            V2_ROOT / f"fold_{fold}" / "tokenizer",
            local_files_only=True,
            use_fast=True,
        )
        model = TransitionAwareVerifier(str(ckpt["model_name"]))
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        model.eval()

        records = []
        for r in sub:
            tr = natural_row_to_train_ready(r)
            records.append({
                "record_id": str(r["instance_id"]),
                "transition": str(r["transition_type"]),
                "row": tr,
            })

        ds = TextRecordDataset(records, tokenizer, args.max_length)
        dl = DataLoader(
            ds,
            batch_size=args.encode_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=make_text_collate(tokenizer),
            pin_memory=(device.type == "cuda"),
        )

        # IMPORTANT: the saved Frozen-V2 Natural scores were produced with
        # CUDA FP16 autocast. Reproduce the SAME numerical path here before
        # comparing probabilities; otherwise FP32-vs-FP16 drift (~1e-3) makes
        # an unnecessarily strict 1e-4 check fail even with identical inputs.
        use_amp = device.type == "cuda"

        with torch.no_grad():
            for enc, tids, indices in dl:
                enc = {
                    k: v.to(device, non_blocking=True)
                    for k, v in enc.items()
                }
                tids = tids.to(device, non_blocking=True)

                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    ctx = torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=use_amp,
                    )
                else:
                    ctx = torch.cuda.amp.autocast(enabled=use_amp)

                with ctx:
                    _, logits = model.encode_and_logit(
                        enc["input_ids"], enc["attention_mask"], tids
                    )

                probs = torch.sigmoid(logits.float()).cpu().numpy()

                for idx2, prob in zip(indices, probs.tolist()):
                    rec = records[idx2]
                    iid = rec["record_id"]
                    if iid not in saved:
                        missing_saved.append(iid)
                        continue
                    d = abs(float(prob) - saved[iid])
                    diffs.append(d)
                    per_t[rec["transition"]].append(d)
                    compared += 1

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if missing_saved:
        raise RuntimeError(
            f"Preflight: {len(missing_saved)} Natural T1/T2 instances are "
            f"missing from saved V2 scores; examples={missing_saved[:10]}"
        )

    if compared != len(rows):
        raise RuntimeError(
            f"Preflight: compared={compared}, expected={len(rows)}"
        )

    max_abs = float(max(diffs)) if diffs else None
    mean_abs = float(np.mean(diffs)) if diffs else None
    p99_abs = float(np.quantile(diffs, 0.99)) if diffs else None

    result = {
        "compared_instances": compared,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "p99_abs_diff": p99_abs,
        "per_transition": {
            t: {
                "n": len(v),
                "max_abs_diff": float(max(v)) if v else None,
                "mean_abs_diff": float(np.mean(v)) if v else None,
            }
            for t, v in per_t.items()
        },
        "saved_score_precision_path": "cuda_fp16_autocast" if device.type == "cuda" else "cpu_fp32",
        "tolerance": 1e-4,
        "passed": bool(max_abs is not None and max_abs <= 1e-4),
    }

    if not result["passed"]:
        raise RuntimeError(
            "FROZEN V2 REPRODUCIBILITY CHECK FAILED. "
            f"max_abs_diff={max_abs}, mean_abs_diff={mean_abs}. "
            "Optimization aborted before training."
        )

    return result

# ----------------------------
# V2 backbone + frozen heads
# ----------------------------

class TransitionAwareVerifier(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        try:
            self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.backbone = AutoModel.from_pretrained(model_name)

        hidden = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in TRANSITIONS])

    def encode_and_logit(self, input_ids, attention_mask, transition_ids):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0, :]
        all_logits = torch.cat([head(h) for head in self.heads], dim=1)
        logits = all_logits.gather(1, transition_ids.unsqueeze(1)).squeeze(1)
        return h, logits


class ResidualAdapters(nn.Module):
    def __init__(self, hidden: int, bottleneck: int = 64, dropout: float = 0.10):
        super().__init__()
        self.adapters = nn.ModuleList()
        for _ in ADAPT_TRANSITIONS:
            block = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, bottleneck),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, 1),
            )
            # Start exactly at V2 behavior.
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)
            self.adapters.append(block)

    def forward(self, h: torch.Tensor, adapt_ids: torch.Tensor) -> torch.Tensor:
        delta = torch.empty(h.shape[0], device=h.device, dtype=h.dtype)
        for aid in torch.unique(adapt_ids):
            mask = adapt_ids == aid
            delta[mask] = self.adapters[int(aid.item())](h[mask]).squeeze(-1)
        return delta


# ----------------------------
# Pair representation / cache
# ----------------------------

def base_pairs_for_training(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = build_base_pairs(base_rows)
    return [
        {
            "transition": p["transition"],
            "pair_id": p["pair_id"],
            "qid": p["qid"],
            "fold": p["fold"],
            "domain": "controlled",
            "positive": p["positive"],
            "negative": p["negative"],
        }
        for p in pairs
    ]


class TextRecordDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.packed = [
            pack_input(tokenizer, r["row"], max_length)
            for r in records
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            self.packed[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "transition_id": T2ID[rec["transition"]],
            "record_index": idx,
        }


def make_text_collate(tokenizer):
    def collate(batch):
        enc = tokenizer.pad(
            {
                "input_ids": [x["input_ids"] for x in batch],
                "attention_mask": [x["attention_mask"] for x in batch],
            },
            padding=True,
            return_tensors="pt",
        )
        tids = torch.tensor([x["transition_id"] for x in batch], dtype=torch.long)
        indices = [x["record_index"] for x in batch]
        return enc, tids, indices
    return collate


@torch.no_grad()
def encode_records(
    model: TransitionAwareVerifier,
    tokenizer,
    records: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    if not records:
        return {}

    ds = TextRecordDataset(records, tokenizer, max_length)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_text_collate(tokenizer),
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    out = {}

    for enc, tids, indices in dl:
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        tids_dev = tids.to(device, non_blocking=True)
        h, logits = model.encode_and_logit(
            enc["input_ids"],
            enc["attention_mask"],
            tids_dev,
        )
        h = h.float().cpu()
        logits = logits.float().cpu()

        for j, idx in enumerate(indices):
            rec = records[idx]
            out[rec["record_id"]] = {
                "h": h[j].clone(),
                "base_logit": float(logits[j].item()),
                "transition": rec["transition"],
            }

    return out


def make_record_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for p in pairs:
        for role in ("positive", "negative"):
            row = p[role]
            rid = str(row["instance_id"])
            if rid in seen:
                continue
            seen.add(rid)
            records.append({
                "record_id": rid,
                "transition": p["transition"],
                "row": row,
            })
    return records


class EmbeddedPairDataset(Dataset):
    def __init__(self, pairs, cache):
        self.pairs = pairs
        self.cache = cache

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pos = self.cache[str(p["positive"]["instance_id"])]
        neg = self.cache[str(p["negative"]["instance_id"])]
        return {
            "pos_h": pos["h"],
            "neg_h": neg["h"],
            "pos_base": pos["base_logit"],
            "neg_base": neg["base_logit"],
            "adapt_id": A2ID[p["transition"]],
            "transition": p["transition"],
            "domain": p["domain"],
            "pair_id": p["pair_id"],
            "qid": p["qid"],
            "fold": p["fold"],
            "pos_instance_id": p["positive"]["instance_id"],
            "neg_instance_id": p["negative"]["instance_id"],
        }


def embedded_collate(batch):
    return {
        "pos_h": torch.stack([x["pos_h"] for x in batch]),
        "neg_h": torch.stack([x["neg_h"] for x in batch]),
        "pos_base": torch.tensor([x["pos_base"] for x in batch], dtype=torch.float32),
        "neg_base": torch.tensor([x["neg_base"] for x in batch], dtype=torch.float32),
        "adapt_id": torch.tensor([x["adapt_id"] for x in batch], dtype=torch.long),
        "meta": [{
            k: x[k] for k in (
                "transition", "domain", "pair_id", "qid", "fold",
                "pos_instance_id", "neg_instance_id"
            )
        } for x in batch],
    }


# ----------------------------
# Metrics
# ----------------------------

def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else None


def safe_ap(y, p):
    return float(average_precision_score(y, p)) if len(set(y)) == 2 else None


def instance_metrics(rows):
    if not rows:
        return {
            "n": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "f1_positive": None,
            "binary_macro_f1": None,
            "mcc": None,
            "roc_auc": None,
            "average_precision": None,
        }

    y = np.array([int(r["label"]) for r in rows], dtype=int)
    prob = np.array([float(r["prob_positive"]) for r in rows], dtype=float)
    pred = (prob >= 0.5).astype(int)
    pr, rc, f1_pos, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_positive": float(pr),
        "recall_positive": float(rc),
        "f1_positive": float(f1_pos),
        "binary_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": safe_auc(y, prob),
        "average_precision": safe_ap(y, prob),
    }


def pair_metrics(pair_rows):
    if not pair_rows:
        return {"pairs": 0, "pairwise_accuracy": None, "mean_margin": None}
    margins = [float(x["pos_prob"]) - float(x["neg_prob"]) for x in pair_rows]
    return {
        "pairs": len(pair_rows),
        "pairwise_accuracy": float(np.mean(np.asarray(margins) > 0.0)),
        "mean_margin": float(np.mean(margins)),
    }


def summarize_by_domain(instance_rows, pair_rows):
    result = {}
    for domain in ("controlled", "naturalized"):
        ir = [x for x in instance_rows if x["domain"] == domain]
        pr = [x for x in pair_rows if x["domain"] == domain]

        per_t = {}
        macro_f1 = []
        macro_auc = []
        for t in ADAPT_TRANSITIONS:
            sub = [x for x in ir if x["transition"] == t]
            m = instance_metrics(sub)
            per_t[t] = m
            if m["binary_macro_f1"] is not None:
                macro_f1.append(m["binary_macro_f1"])
            if m["roc_auc"] is not None:
                macro_auc.append(m["roc_auc"])

        per_t_pair = {
            t: pair_metrics([x for x in pr if x["transition"] == t])
            for t in ADAPT_TRANSITIONS
        }
        pair_accs = [
            x["pairwise_accuracy"] for x in per_t_pair.values()
            if x["pairwise_accuracy"] is not None
        ]

        result[domain] = {
            "instances": len(ir),
            "pairs": len(pr),
            "overall": instance_metrics(ir),
            "per_transition": per_t,
            "macro_transition_binary_macro_f1": (
                float(np.mean(macro_f1)) if macro_f1 else None
            ),
            "macro_transition_auc": (
                float(np.mean(macro_auc)) if macro_auc else None
            ),
            "pairwise": pair_metrics(pr),
            "per_transition_pairwise": per_t_pair,
            "macro_transition_pairwise_accuracy": (
                float(np.mean(pair_accs)) if pair_accs else None
            ),
        }

    return result


# ----------------------------
# Adapter training / eval
# ----------------------------

@torch.no_grad()
def evaluate_adapter(
    adapter: ResidualAdapters,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    adapter.eval()
    inst_rows = []
    pair_rows = []

    for batch in loader:
        pos_h = batch["pos_h"].to(device)
        neg_h = batch["neg_h"].to(device)
        pos_base = batch["pos_base"].to(device)
        neg_base = batch["neg_base"].to(device)
        aid = batch["adapt_id"].to(device)

        pos_delta = adapter(pos_h, aid)
        neg_delta = adapter(neg_h, aid)
        pos_prob = torch.sigmoid(pos_base + pos_delta).cpu().numpy()
        neg_prob = torch.sigmoid(neg_base + neg_delta).cpu().numpy()

        for meta, pp, pn in zip(batch["meta"], pos_prob.tolist(), neg_prob.tolist()):
            pair_rows.append({
                **meta,
                "pos_prob": float(pp),
                "neg_prob": float(pn),
            })
            inst_rows.append({
                "transition": meta["transition"],
                "domain": meta["domain"],
                "pair_id": meta["pair_id"],
                "qid": meta["qid"],
                "fold": meta["fold"],
                "instance_id": meta["pos_instance_id"],
                "label": 1,
                "prob_positive": float(pp),
            })
            inst_rows.append({
                "transition": meta["transition"],
                "domain": meta["domain"],
                "pair_id": meta["pair_id"],
                "qid": meta["qid"],
                "fold": meta["fold"],
                "instance_id": meta["neg_instance_id"],
                "label": 0,
                "prob_positive": float(pn),
            })

    return inst_rows, pair_rows, summarize_by_domain(inst_rows, pair_rows)


def selection_score(summary: dict[str, Any]) -> float:
    c = summary["controlled"]
    n = summary["naturalized"]

    weighted = [
        (c.get("macro_transition_binary_macro_f1"), 0.35),
        (n.get("macro_transition_binary_macro_f1"), 0.35),
        (c.get("macro_transition_pairwise_accuracy"), 0.15),
        (n.get("macro_transition_pairwise_accuracy"), 0.15),
    ]
    available = [(float(v), w) for v, w in weighted if v is not None]
    if not available:
        return -1.0
    denom = sum(w for _, w in available)
    return float(sum(v * w for v, w in available) / denom)

def training_sampler(pairs: list[dict[str, Any]]) -> WeightedRandomSampler:
    counts = Counter((p["transition"], p["domain"]) for p in pairs)
    weights = [
        1.0 / counts[(p["transition"], p["domain"])]
        for p in pairs
    ]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=max(len(pairs), 64),
        replacement=True,
    )


def train_one_fold(
    fold: int,
    all_pairs: list[dict[str, Any]],
    args,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_dir = OUT_ROOT / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = fold_dir / "best_adapter.pt"
    metrics_out = fold_dir / "best_metrics.json"

    v2_ckpt_path = V2_ROOT / f"fold_{fold}" / "best_model.pt"
    tok_dir = V2_ROOT / f"fold_{fold}" / "tokenizer"

    ckpt = torch.load(v2_ckpt_path, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True, use_fast=True)

    base_model = TransitionAwareVerifier(str(ckpt["model_name"]))
    base_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    base_model.to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    train_pairs = [p for p in all_pairs if int(p["fold"]) != fold]
    valid_pairs = [p for p in all_pairs if int(p["fold"]) == fold]

    if not train_pairs or not valid_pairs:
        raise RuntimeError(f"Fold {fold}: empty train/valid pairs")

    train_records = make_record_rows(train_pairs)
    valid_records = make_record_rows(valid_pairs)

    print(
        f"Fold {fold}: encoding train texts={len(train_records)} "
        f"valid texts={len(valid_records)}"
    )
    train_cache = encode_records(
        base_model, tokenizer, train_records,
        args.max_length, args.encode_batch_size, device
    )
    valid_cache = encode_records(
        base_model, tokenizer, valid_records,
        args.max_length, args.encode_batch_size, device
    )

    hidden = int(base_model.backbone.config.hidden_size)
    adapter = ResidualAdapters(
        hidden=hidden,
        bottleneck=args.bottleneck,
        dropout=args.adapter_dropout,
    ).to(device)

    train_ds = EmbeddedPairDataset(train_pairs, train_cache)
    valid_ds = EmbeddedPairDataset(valid_pairs, valid_cache)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.adapter_batch_size,
        sampler=training_sampler(train_pairs),
        num_workers=0,
        collate_fn=embedded_collate,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=max(args.adapter_batch_size, 64),
        shuffle=False,
        num_workers=0,
        collate_fn=embedded_collate,
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.adapter_lr,
        weight_decay=args.weight_decay,
    )

    best_score = -1.0
    best_epoch = -1
    patience_left = args.patience
    best_summary = None

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        losses = []

        for batch in train_loader:
            pos_h = batch["pos_h"].to(device)
            neg_h = batch["neg_h"].to(device)
            pos_base = batch["pos_base"].to(device)
            neg_base = batch["neg_base"].to(device)
            aid = batch["adapt_id"].to(device)

            pos_delta = adapter(pos_h, aid)
            neg_delta = adapter(neg_h, aid)

            pos_logit = pos_base + pos_delta
            neg_logit = neg_base + neg_delta

            point_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    pos_logit, torch.ones_like(pos_logit)
                )
                + F.binary_cross_entropy_with_logits(
                    neg_logit, torch.zeros_like(neg_logit)
                )
            )
            rank_loss = F.softplus(-(pos_logit - neg_logit)).mean()
            residual_reg = 0.5 * (
                pos_delta.pow(2).mean() + neg_delta.pow(2).mean()
            )

            loss = (
                args.point_weight * point_loss
                + args.rank_weight * rank_loss
                + args.residual_reg * residual_reg
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        _, _, val_summary = evaluate_adapter(adapter, valid_loader, device)
        score = selection_score(val_summary)

        c = val_summary["controlled"]
        n = val_summary["naturalized"]

        print(
            f"Fold {fold} epoch {epoch}: "
            f"loss={np.mean(losses):.4f} "
            f"selection={score:.4f} "
            f"controlled_macroF1={c['macro_transition_binary_macro_f1']} "
            f"naturalized_macroF1={n['macro_transition_binary_macro_f1']} "
            f"controlled_pairAcc={c['macro_transition_pairwise_accuracy']} "
            f"naturalized_pairAcc={n['macro_transition_pairwise_accuracy']}"
        )

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_summary = val_summary
            patience_left = args.patience
            torch.save(
                {
                    "adapter_state_dict": adapter.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "hidden": hidden,
                    "bottleneck": args.bottleneck,
                    "adapt_transitions": ADAPT_TRANSITIONS,
                    "source_v2_checkpoint": str(v2_ckpt_path),
                    "selection_score": score,
                },
                ckpt_out,
            )
            metrics_out.write_text(
                json.dumps(
                    {
                        "fold": fold,
                        "best_epoch": epoch,
                        "selection_score": score,
                        "validation": val_summary,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_epoch < 0:
        raise RuntimeError(f"Fold {fold}: no best adapter saved")

    best = torch.load(ckpt_out, map_location="cpu", weights_only=False)
    adapter.load_state_dict(best["adapter_state_dict"], strict=True)
    adapter.to(device)

    inst, pair, summary = evaluate_adapter(adapter, valid_loader, device)
    for r in inst:
        r["adapter_fold"] = fold
    for r in pair:
        r["adapter_fold"] = fold

    del adapter, base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        {
            "fold": fold,
            "best_epoch": best_epoch,
            "selection_score": best_score,
            "validation": summary,
            "train_pair_counts": dict(Counter(
                f"{p['transition']}::{p['domain']}" for p in train_pairs
            )),
            "valid_pair_counts": dict(Counter(
                f"{p['transition']}::{p['domain']}" for p in valid_pairs
            )),
        },
        inst,
        pair,
    )



def decide_adapter_deployment(
    oof_summary: dict[str, Any],
    v2_reference: dict[str, Any],
    naturalized_pair_counts: Counter,
) -> dict[str, dict[str, Any]]:
    """
    Conservative deployment gate.
    We deploy an adapter only if:
      1) >=20 naturalized held-out pairs exist,
      2) held-out naturalized AUC >= 0.60,
      3) held-out naturalized pair accuracy >= 0.60,
      4) controlled Macro-F1 drops by no more than 0.03 vs frozen V2.
    No Natural answer labels or final-test labels are used.
    """
    decisions = {}
    for t in ADAPT_TRANSITIONS:
        nat_m = oof_summary["naturalized"]["per_transition"][t]
        nat_pw = oof_summary["naturalized"]["per_transition_pairwise"][t]
        ctl_m = oof_summary["controlled"]["per_transition"][t]
        v2_m = v2_reference[t]

        n_pairs = int(naturalized_pair_counts[t])
        nat_auc = nat_m.get("roc_auc")
        nat_pair_acc = nat_pw.get("pairwise_accuracy")
        ctl_f1 = ctl_m.get("binary_macro_f1")
        v2_ctl_f1 = v2_m.get("binary_macro_f1")
        ctl_drop = (
            float(v2_ctl_f1 - ctl_f1)
            if v2_ctl_f1 is not None and ctl_f1 is not None
            else None
        )

        passed = bool(
            n_pairs >= 20
            and nat_auc is not None and nat_auc >= 0.60
            and nat_pair_acc is not None and nat_pair_acc >= 0.60
            and ctl_drop is not None and ctl_drop <= 0.03
        )

        decisions[t] = {
            "deploy_adapter": passed,
            "naturalized_pairs": n_pairs,
            "heldout_naturalized_auc": nat_auc,
            "heldout_naturalized_pair_accuracy": nat_pair_acc,
            "controlled_macro_f1": ctl_f1,
            "frozen_v2_controlled_macro_f1": v2_ctl_f1,
            "controlled_macro_f1_drop": ctl_drop,
            "criteria": {
                "min_naturalized_pairs": 20,
                "min_naturalized_auc": 0.60,
                "min_naturalized_pair_accuracy": 0.60,
                "max_controlled_macro_f1_drop": 0.03,
            },
        }
    return decisions

# ----------------------------
# Natural held-out V3 scoring
# ----------------------------

def build_qid_fold_map(oof_rows: list[dict[str, Any]]) -> dict[str, int]:
    by_qid = defaultdict(set)
    for r in oof_rows:
        by_qid[str(r["qid"])].add(int(r["fold"]))
    bad = {q: sorted(v) for q, v in by_qid.items() if len(v) != 1}
    if bad:
        raise RuntimeError(f"V2 OOF fold conflict: {list(bad.items())[:10]}")
    return {q: next(iter(v)) for q, v in by_qid.items()}


def natural_row_to_train_ready(row: dict[str, Any]) -> dict[str, Any]:
    t = str(row["transition_type"])
    sec = natural_instance_to_sections(row)
    return {
        "instance_id": row["instance_id"],
        "qid": row["qid"],
        "transition_short": t,
        "input_text": sections_to_input_text(t, sec),
    }


@torch.no_grad()
def score_natural_with_v3(
    natural_rows: list[dict[str, Any]],
    qid_fold: dict[str, int],
    deployment: dict[str, dict[str, Any]],
    args,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = [
        r for r in natural_rows
        if str(r.get("transition_type")) in ADAPT_TRANSITIONS
    ]
    by_fold = defaultdict(list)
    for r in rows:
        qid = str(r["qid"])
        if qid not in qid_fold:
            continue
        by_fold[qid_fold[qid]].append(r)

    outputs = []

    for fold in range(5):
        sub = by_fold.get(fold, [])
        if not sub:
            continue

        v2_ckpt = torch.load(
            V2_ROOT / f"fold_{fold}" / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            V2_ROOT / f"fold_{fold}" / "tokenizer",
            local_files_only=True,
            use_fast=True,
        )
        base_model = TransitionAwareVerifier(str(v2_ckpt["model_name"]))
        base_model.load_state_dict(v2_ckpt["model_state_dict"], strict=True)
        base_model.to(device)
        base_model.eval()

        adapter_ckpt = torch.load(
            OUT_ROOT / f"fold_{fold}" / "best_adapter.pt",
            map_location="cpu",
            weights_only=False,
        )
        hidden = int(adapter_ckpt["hidden"])
        adapter = ResidualAdapters(
            hidden=hidden,
            bottleneck=int(adapter_ckpt["bottleneck"]),
            dropout=args.adapter_dropout,
        )
        adapter.load_state_dict(adapter_ckpt["adapter_state_dict"], strict=True)
        adapter.to(device)
        adapter.eval()

        records = []
        for r in sub:
            tr = natural_row_to_train_ready(r)
            records.append({
                "record_id": str(r["instance_id"]),
                "transition": str(r["transition_type"]),
                "row": tr,
            })

        ds = TextRecordDataset(records, tokenizer, args.max_length)
        dl = DataLoader(
            ds,
            batch_size=args.encode_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=make_text_collate(tokenizer),
            pin_memory=(device.type == "cuda"),
        )

        for enc, tids, indices in dl:
            enc = {k: v.to(device) for k, v in enc.items()}
            tids_dev = tids.to(device)
            h, base_logits = base_model.encode_and_logit(
                enc["input_ids"], enc["attention_mask"], tids_dev
            )
            aids = torch.tensor(
                [A2ID[records[i]["transition"]] for i in indices],
                dtype=torch.long,
                device=device,
            )
            raw_delta = adapter(h, aids)
            deploy_mask = torch.tensor(
                [
                    1.0 if deployment[records[i]["transition"]]["deploy_adapter"] else 0.0
                    for i in indices
                ],
                dtype=raw_delta.dtype,
                device=device,
            )
            delta = raw_delta * deploy_mask
            logits = base_logits + delta
            probs = torch.sigmoid(logits).float().cpu().numpy()
            base_probs = torch.sigmoid(base_logits).float().cpu().numpy()
            deltas = delta.float().cpu().numpy()
            raw_deltas = raw_delta.float().cpu().numpy()

            for idx, bp, p, d, rd in zip(
                indices,
                base_probs.tolist(),
                probs.tolist(),
                deltas.tolist(),
                raw_deltas.tolist(),
            ):
                src = sub[idx]
                outputs.append({
                    "instance_id": src["instance_id"],
                    "qid": src["qid"],
                    "fold": fold,
                    "transition": src["transition_type"],
                    "component_index": src.get("component_index"),
                    "v2_prob_positive_recomputed": float(bp),
                    "v3_prob_positive": float(p),
                    "adapter_logit_delta_deployed": float(d),
                    "adapter_logit_delta_raw": float(rd),
                    "adapter_deployed": bool(deployment[str(src["transition_type"])]["deploy_adapter"]),
                    "scoring_protocol": "qid-held-out-v2-plus-heldout-natural-adapter-with-oof-deployment-gate",
                })

        del base_model, adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return outputs


def aggregate_natural_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = defaultdict(list)
    for r in rows:
        by[(str(r["qid"]), str(r["transition"]))].append(r)

    out = []
    for (qid, t), items in sorted(by.items()):
        items = sorted(items, key=lambda x: int(x.get("component_index") or 0))
        v2 = np.asarray([float(x["v2_prob_positive_recomputed"]) for x in items])
        v3 = np.asarray([float(x["v3_prob_positive"]) for x in items])
        out.append({
            "qid": qid,
            "transition": t,
            "fold": int(items[0]["fold"]),
            "n_components": len(items),
            "v2_primary_min_prob": float(v2.min()),
            "v3_primary_min_prob": float(v3.min()),
            "v2_mean_prob": float(v2.mean()),
            "v3_mean_prob": float(v3.mean()),
            "delta_primary_min": float(v3.min() - v2.min()),
            "delta_mean": float(v3.mean() - v2.mean()),
        })
    return out


def natural_shift_summary(v3_rows, old_v2_rows):
    old_map = {
        (str(x["qid"]), str(x["transition"])): x
        for x in old_v2_rows
        if str(x.get("transition")) in ADAPT_TRANSITIONS
    }

    per_t = {}
    for t in ADAPT_TRANSITIONS:
        sub = [x for x in v3_rows if x["transition"] == t]
        vals3 = [float(x["v3_primary_min_prob"]) for x in sub]
        vals2_re = [float(x["v2_primary_min_prob"]) for x in sub]

        old_vals = []
        exact_match_count = 0
        for x in sub:
            k = (str(x["qid"]), t)
            if k in old_map:
                oldp = float(old_map[k]["primary_prob_positive"])
                old_vals.append(oldp)
                if abs(oldp - float(x["v2_primary_min_prob"])) < 1e-5:
                    exact_match_count += 1

        per_t[t] = {
            "qid_transition_n": len(sub),
            "v2_recomputed_median": float(np.median(vals2_re)) if vals2_re else None,
            "v3_median": float(np.median(vals3)) if vals3 else None,
            "median_gain": (
                float(np.median(vals3) - np.median(vals2_re))
                if vals3 else None
            ),
            "v2_recomputed_mean": float(np.mean(vals2_re)) if vals2_re else None,
            "v3_mean": float(np.mean(vals3)) if vals3 else None,
            "mean_gain": (
                float(np.mean(vals3) - np.mean(vals2_re))
                if vals3 else None
            ),
            "v2_below_0_5_rate": (
                float(np.mean(np.asarray(vals2_re) < 0.5))
                if vals2_re else None
            ),
            "v3_below_0_5_rate": (
                float(np.mean(np.asarray(vals3) < 0.5))
                if vals3 else None
            ),
            "old_v2_file_match_count": len(old_vals),
            "old_v2_exact_recompute_match_count": exact_match_count,
        }
    return per_t


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.10)
    parser.add_argument("--adapter-lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--point-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--residual-reg", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--adapter-batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    for p in (
        BASE_TRAIN_READY,
        NATURAL_INSTANCES,
        V2_OOF,
        V2_NATURAL_TRANSITION_SCORES,
        V2_NATURAL_INSTANCE_SCORES,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    for fold in range(5):
        if not (V2_ROOT / f"fold_{fold}" / "best_model.pt").is_file():
            raise FileNotFoundError(V2_ROOT / f"fold_{fold}" / "best_model.pt")
        if not (V2_ROOT / f"fold_{fold}" / "tokenizer").is_dir():
            raise FileNotFoundError(V2_ROOT / f"fold_{fold}" / "tokenizer")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    base_rows = load_jsonl(BASE_TRAIN_READY)
    natural_rows = load_jsonl(NATURAL_INSTANCES)
    v2_oof_rows = load_jsonl(V2_OOF)
    old_v2_nat = load_jsonl(V2_NATURAL_TRANSITION_SCORES)
    old_v2_instance = load_jsonl(V2_NATURAL_INSTANCE_SCORES)

    print("=" * 124)
    print("MMQA — TRANSITION-CAUSAL VERIFIER V3 NATURAL-ADAPTER OPTIMIZATION")
    print("=" * 124)
    print("Goal: repair T1/T2 natural distribution shift; freeze T3/T4/T5.")
    print("Core strategy: frozen V2 encoder+heads + transition-specific residual natural adapters.")
    print("Naturalized labels: inherited from frozen gold-derived T1/T2 counterfactual pairs.")
    print("Leakage control: original qid folds preserved; held-out qids never train their own adapter.")
    print("External API calls = 0")
    print("-" * 124)

    qid_fold = build_qid_fold_map(v2_oof_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Running strict frozen-V2 Natural probability reproducibility preflight...")
    repro = preflight_reproduce_saved_v2_natural_scores(
        natural_rows=natural_rows,
        qid_fold=qid_fold,
        saved_instance_rows=old_v2_instance,
        args=args,
        device=device,
    )
    print(
        "V2 reproducibility: PASS "
        f"n={repro['compared_instances']} "
        f"precision={repro['saved_score_precision_path']} "
        f"max_abs_diff={repro['max_abs_diff']:.8g} "
        f"mean_abs_diff={repro['mean_abs_diff']:.8g}"
    )
    print("-" * 124)

    naturalized_rows, naturalized_pairs, build_summary = build_naturalized_supervision(
        base_rows, natural_rows
    )
    controlled_pairs = base_pairs_for_training(base_rows)

    nat_counts = Counter(p["transition"] for p in naturalized_pairs)
    if nat_counts["T1"] == 0 or nat_counts["T2"] == 0:
        raise RuntimeError(
            "Naturalized T1/T2 supervision is empty. "
            f"Counts={dict(nat_counts)}; inspect {BUILD_AUDIT_FILE}"
        )

    all_pairs = controlled_pairs + naturalized_pairs

    print("Controlled T1/T2 pairs:", len(controlled_pairs))
    print("Naturalized pairs:", len(naturalized_pairs))
    print("Naturalized pair counts:", build_summary["naturalized_pair_counts"])
    print("Naturalized skip counts:", build_summary["skip_counts"])
    print("Mean natural match score:", build_summary["mean_match_score"])
    print("-" * 124)

    fold_records = []
    oof_inst = []
    oof_pair = []

    for fold in range(5):
        rec, inst, pair = train_one_fold(
            fold=fold,
            all_pairs=all_pairs,
            args=args,
            device=device,
        )
        fold_records.append(rec)
        oof_inst.extend(inst)
        oof_pair.extend(pair)

    write_jsonl(
        OOF_INSTANCE_FILE,
        sorted(oof_inst, key=lambda x: (x["fold"], x["qid"], x["domain"], x["transition"], x["instance_id"])),
    )
    write_jsonl(
        OOF_PAIR_FILE,
        sorted(oof_pair, key=lambda x: (x["fold"], x["qid"], x["domain"], x["transition"], x["pair_id"])),
    )

    oof_summary = summarize_by_domain(oof_inst, oof_pair)

    # Frozen V2 controlled reference, restricted to adapted transitions.
    v2_oof_t12 = []
    for r in v2_oof_rows:
        if str(r.get("transition")) in ADAPT_TRANSITIONS:
            x = dict(r)
            x["domain"] = "controlled"
            v2_oof_t12.append(x)

    v2_controlled_per_t = {}
    for t in ADAPT_TRANSITIONS:
        sub = [x for x in v2_oof_t12 if x["transition"] == t]
        v2_controlled_per_t[t] = instance_metrics(sub)

    deployment = decide_adapter_deployment(
        oof_summary=oof_summary,
        v2_reference=v2_controlled_per_t,
        naturalized_pair_counts=nat_counts,
    )

    print("-" * 124)
    print("OOF DEPLOYMENT GATE")
    for t in ADAPT_TRANSITIONS:
        d = deployment[t]
        print(
            f"{t}: deploy={d['deploy_adapter']} "
            f"naturalized_pairs={d['naturalized_pairs']} "
            f"nat_AUC={d['heldout_naturalized_auc']} "
            f"nat_pairAcc={d['heldout_naturalized_pair_accuracy']} "
            f"controlled_F1_drop={d['controlled_macro_f1_drop']}"
        )

    # Held-out Natural scoring on actual extracted states.
    nat_v3_inst = score_natural_with_v3(
        natural_rows=natural_rows,
        qid_fold=qid_fold,
        deployment=deployment,
        args=args,
        device=device,
    )
    nat_v3_trans = aggregate_natural_scores(nat_v3_inst)

    write_jsonl(
        NATURAL_INSTANCE_SCORE_FILE,
        sorted(nat_v3_inst, key=lambda x: (x["fold"], x["qid"], x["transition"], int(x.get("component_index") or 0))),
    )
    write_jsonl(
        NATURAL_TRANSITION_SCORE_FILE,
        sorted(nat_v3_trans, key=lambda x: (x["fold"], x["qid"], x["transition"])),
    )

    shift = natural_shift_summary(nat_v3_trans, old_v2_nat)

    summary = {
        "method": "Corrected Frozen V2 + transition-specific residual natural adapters for T1/T2; T3/T4/T5 frozen",
        "architecture": {
            "frozen_backbone_and_v2_heads": True,
            "candidate_adapted_transitions": ADAPT_TRANSITIONS,
            "deployment_decisions": deployment,
            "unchanged_transitions": ["T3", "T4", "T5"],
            "adapter": f"LayerNorm -> Linear(768,{args.bottleneck}) -> GELU -> Dropout -> Linear({args.bottleneck},1)",
            "adapter_final_layer_zero_initialized": True,
        },
        "frozen_v2_reproducibility_preflight": repro,
        "supervision_build": build_summary,
        "fold_records": fold_records,
        "oof_v3": oof_summary,
        "v2_controlled_reference_t1_t2_t3": v2_controlled_per_t,
        "natural_heldout_shift": shift,
        "evaluation_protocol": {
            "five_fold_qid_cross_fitting": True,
            "naturalized_training_rows_use_same_original_qid_fold": True,
            "actual_natural_inference_uses_heldout_fold_adapter": True,
            "gold_oracle_used_at_inference": False,
            "external_api_calls": 0,
        },
        "interpretation_guardrail": (
            "An upward Natural probability shift is an alignment diagnostic, not by itself proof of correctness. "
            "Primary model-quality evidence is held-out controlled + held-out naturalized pair discrimination."
        ),
        "status": "MMQA_VERIFIER_V3_T1_T2_NATURAL_ADAPTER_CORRECTED_COMPLETE",
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 124)
    print("FINAL V3 OOF")
    print("=" * 124)

    for domain in ("controlled", "naturalized"):
        s = oof_summary[domain]
        print(
            f"{domain}: "
            f"pairs={s['pairs']} "
            f"macroF1={s['macro_transition_binary_macro_f1']} "
            f"macroAUC={s['macro_transition_auc']} "
            f"macroPairAcc={s['macro_transition_pairwise_accuracy']}"
        )
        for t in ADAPT_TRANSITIONS:
            m = s["per_transition"][t]
            pw = s["per_transition_pairwise"][t]
            print(
                f"  {t}: n={m['n']} MacroF1={m['binary_macro_f1']} "
                f"AUC={m['roc_auc']} pairAcc={pw['pairwise_accuracy']}"
            )

    print("-" * 124)
    print("ACTUAL NATURAL HELD-OUT SCORE SHIFT")
    for t in ADAPT_TRANSITIONS:
        s = shift[t]
        print(
            f"{t}: n={s['qid_transition_n']} "
            f"V2_median={s['v2_recomputed_median']} "
            f"V3_median={s['v3_median']} "
            f"median_gain={s['median_gain']} "
            f"V2_below0.5={s['v2_below_0_5_rate']} "
            f"V3_below0.5={s['v3_below_0_5_rate']} "
            f"V2_recompute_exact_matches={s['old_v2_exact_recompute_match_count']}/{s['old_v2_file_match_count']}"
        )

    print("-" * 124)
    print("T3/T4/T5 changed by this optimization: NO")
    print("External API calls = 0")
    print("Naturalized supervision:", NATURALIZED_FILE)
    print("V3 OOF instances:", OOF_INSTANCE_FILE)
    print("V3 Natural transition scores:", NATURAL_TRANSITION_SCORE_FILE)
    print("Summary:", SUMMARY_FILE)
    print("STATUS = MMQA_VERIFIER_V3_T1_T2_NATURAL_ADAPTER_CORRECTED_COMPLETE")


if __name__ == "__main__":
    main()
