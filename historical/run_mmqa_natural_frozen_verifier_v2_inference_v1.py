from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent

INSTANCE_FILE = (
    ROOT / "outputs" / "mmqa_natural_verifier_inference_inputs_v1"
    / "mmqa_natural_verifier_instances_v1.jsonl"
)
MANIFEST_FILE = (
    ROOT / "outputs" / "mmqa_natural_verifier_inference_inputs_v1"
    / "mmqa_natural_qid_state_manifest_v1.jsonl"
)
VERIFIER_ROOT = (
    ROOT / "outputs" / "mmqa_transition_causal_verifier_v2_pairwise"
)
OOF_PRED_FILE = VERIFIER_ROOT / "oof_predictions_available.jsonl"

OUT_DIR = ROOT / "outputs" / "mmqa_natural_frozen_verifier_v2_inference_v1"
INSTANCE_SCORE_FILE = OUT_DIR / "mmqa_natural_frozen_verifier_instance_scores_v1.jsonl"
TRANSITION_SCORE_FILE = OUT_DIR / "mmqa_natural_frozen_verifier_transition_scores_v1.jsonl"
QID_STATE_FILE = OUT_DIR / "mmqa_natural_frozen_verifier_qid_states_v1.jsonl"
SUMMARY_FILE = OUT_DIR / "mmqa_natural_frozen_verifier_v2_inference_v1_summary.json"

TRANSITIONS = ["T1", "T2", "T3", "T4", "T5"]
T2ID = {t: i for i, t in enumerate(TRANSITIONS)}


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
                raise RuntimeError(f"Malformed JSONL line {line_no} in {path}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    """
    EXACTLY mirrors the frozen V2 verifier training packer.
    """
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


class TransitionAwareVerifier(nn.Module):
    """
    EXACT architecture used by the frozen V2 training script:
    shared encoder + five transition-specific linear heads.
    """
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        # The model was already trained locally, so prefer the local HF cache.
        try:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except Exception:
            # Fallback only if the local cache lookup behaves differently
            # in the installed transformers version.
            self.backbone = AutoModel.from_pretrained(model_name)

        hidden = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in TRANSITIONS])

    def forward(self, input_ids, attention_mask, transition_ids):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        h = self.dropout(out.last_hidden_state[:, 0, :])
        all_logits = torch.cat([head(h) for head in self.heads], dim=1)
        logits = all_logits.gather(
            1,
            transition_ids.unsqueeze(1),
        ).squeeze(1)
        return logits


def make_training_format_input(row: dict[str, Any]) -> str:
    """
    The natural-input builder stored sections over multiple lines.
    The frozen training parser expects '[TAG] content' on the SAME line.
    Reconstruct exact training-format sections from the structured fields.
    """
    t = str(row["transition_type"])
    f = row.get("fields")
    if not isinstance(f, dict):
        raise ValueError(f"Missing structured fields for {row.get('instance_id')}")

    q = str(f.get("question", "") or "")

    if t == "T1":
        return "\n".join([
            f"[QUESTION] {q}",
            f"[VISUAL_EVIDENCE] {str(f.get('visual_evidence_text', '') or '')}",
            f"[CANDIDATE_GROUNDING_CLAIM] {str(f.get('candidate_grounding_claim', '') or '')}",
        ])

    if t == "T2":
        return "\n".join([
            f"[QUESTION] {q}",
            f"[UPSTREAM_CLAIM] {str(f.get('upstream_claim', '') or '')}",
            f"[CANDIDATE_BRIDGE_EVIDENCE] {str(f.get('candidate_bridge_evidence_text', '') or '')}",
            f"[DOWNSTREAM_CLAIM] {str(f.get('downstream_claim', '') or '')}",
        ])

    if t == "T3":
        return "\n".join([
            f"[QUESTION] {q}",
            f"[BRANCH_VISUAL_EVIDENCE] {str(f.get('branch_visual_evidence_text', '') or '')}",
            f"[CANDIDATE_BRANCH_CLAIM] {str(f.get('candidate_branch_claim', '') or '')}",
        ])

    if t == "T4":
        vals = f.get("verified_branch_values")
        if not isinstance(vals, list):
            vals = []
        return "\n".join([
            f"[QUESTION] {q}",
            f"[AGGREGATION_OPERATOR] {str(f.get('aggregation_operator', '') or '')}",
            f"[VERIFIED_BRANCH_VALUES] {json.dumps(vals, ensure_ascii=False)}",
            f"[CANDIDATE_AGGREGATION_RESULT] {str(f.get('candidate_aggregation_result', '') or '')}",
        ])

    if t == "T5":
        claims = f.get("verified_upstream_claims")
        if not isinstance(claims, list):
            claims = []
        return "\n".join([
            f"[QUESTION] {q}",
            f"[VERIFIED_UPSTREAM_CLAIMS] {json.dumps(claims, ensure_ascii=False)}",
            f"[CANDIDATE_FINAL_ANSWER] {str(f.get('candidate_final_answer', '') or '')}",
        ])

    raise ValueError(f"Unknown transition type: {t}")


def build_qid_fold_map(oof_rows: list[dict[str, Any]]) -> dict[str, int]:
    by_qid: dict[str, set[int]] = defaultdict(set)

    for row in oof_rows:
        qid = str(row.get("qid", "") or "")
        if not qid:
            continue
        by_qid[qid].add(int(row["fold"]))

    bad = {
        qid: sorted(folds)
        for qid, folds in by_qid.items()
        if len(folds) != 1
    }
    if bad:
        first = list(bad.items())[:10]
        raise RuntimeError(
            f"Verifier OOF qids assigned to multiple folds: {first}"
        )

    return {
        qid: next(iter(folds))
        for qid, folds in by_qid.items()
    }


class NaturalVerifierDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        max_length: int,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.packed = []
        for row in rows:
            training_row = {
                "input_text": make_training_format_input(row),
                "transition_short": row["transition_type"],
            }
            self.packed.append(
                pack_input(
                    tokenizer,
                    training_row,
                    max_length,
                )
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
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
            "transition_id": T2ID[row["transition_type"]],
            "row_index": idx,
        }


def make_collate(tokenizer):
    def collate(batch):
        enc = tokenizer.pad(
            {
                "input_ids": [x["input_ids"] for x in batch],
                "attention_mask": [x["attention_mask"] for x in batch],
            },
            padding=True,
            return_tensors="pt",
        )
        tids = torch.tensor(
            [x["transition_id"] for x in batch],
            dtype=torch.long,
        )
        indices = [x["row_index"] for x in batch]
        return enc, tids, indices
    return collate


def score_fold(
    fold: int,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    device: torch.device,
    fp16: bool,
) -> list[dict[str, Any]]:
    fold_dir = VERIFIER_ROOT / f"fold_{fold}"
    ckpt_path = fold_dir / "best_model.pt"
    tokenizer_dir = fold_dir / "tokenizer"

    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(tokenizer_dir)

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    model_name = str(checkpoint["model_name"])
    transitions = list(checkpoint["transitions"])
    if transitions != TRANSITIONS:
        raise RuntimeError(
            f"Unexpected checkpoint transition order: {transitions}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        use_fast=True,
        local_files_only=True,
    )

    model = TransitionAwareVerifier(
        model_name,
        dropout=0.1,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.to(device)
    model.eval()

    ds = NaturalVerifierDataset(
        rows,
        tokenizer,
        max_length=max_length,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate(tokenizer),
        pin_memory=(device.type == "cuda"),
    )

    out_rows = []
    use_amp = device.type == "cuda" and fp16

    with torch.no_grad():
        for enc, tids, indices in loader:
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
                logits = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    transition_ids=tids,
                )

            logits_np = logits.float().cpu().numpy()
            probs_np = torch.sigmoid(logits.float()).cpu().numpy()

            for local_idx, logit, prob in zip(
                indices,
                logits_np.tolist(),
                probs_np.tolist(),
            ):
                src = rows[local_idx]
                out_rows.append({
                    "instance_id": src["instance_id"],
                    "qid": src["qid"],
                    "fold": fold,
                    "transition": src["transition_type"],
                    "component_index": src.get("component_index"),
                    "logit": float(logit),
                    "prob_positive": float(prob),
                    "pred_at_0_5": int(float(prob) >= 0.5),
                    "topology_probability": src.get("topology_probability"),
                    "checkpoint": str(ckpt_path),
                    "scoring_protocol": "qid-held-out-verifier-fold",
                })

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out_rows


def aggregate_transition_scores(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_qt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_qt[(row["qid"], row["transition"])].append(row)

    out = []
    for (qid, t), items in sorted(
        by_qt.items(),
        key=lambda kv: (kv[0][0], T2ID[kv[0][1]]),
    ):
        items = sorted(
            items,
            key=lambda x: (
                x.get("component_index")
                if x.get("component_index") is not None
                else -1
            ),
        )
        probs = np.asarray(
            [float(x["prob_positive"]) for x in items],
            dtype=float,
        )
        min_idx = int(np.argmin(probs))

        out.append({
            "qid": qid,
            "transition": t,
            "fold": int(items[0]["fold"]),
            "n_components": len(items),
            # Conservative primary score: one bad component can break the transition.
            "primary_prob_positive": float(np.min(probs)),
            "aggregation": "min_component_probability",
            "min_prob": float(np.min(probs)),
            "mean_prob": float(np.mean(probs)),
            "max_prob": float(np.max(probs)),
            "lowest_scoring_instance_id": items[min_idx]["instance_id"],
            "component_instance_ids": [x["instance_id"] for x in items],
        })

    return out


def build_qid_states(
    manifests: list[dict[str, Any]],
    transition_scores: list[dict[str, Any]],
    qid_fold: dict[str, int],
) -> list[dict[str, Any]]:
    score_map = {
        (x["qid"], x["transition"]): x
        for x in transition_scores
    }

    out = []
    for m in manifests:
        qid = str(m["qid"])
        required = m.get("predicted_required_transitions_oof") or []
        present = m.get("required_and_present") or []
        missing = m.get("required_but_missing") or []

        transitions = {}
        for t in required:
            if t in missing:
                transitions[t] = {
                    "state_status": "MISSING_REQUIRED_TRANSITION_STATE",
                    "verifier_prob_positive": None,
                    "verifier_score_available": False,
                }
            elif t in present:
                s = score_map.get((qid, t))
                if s is None:
                    transitions[t] = {
                        "state_status": "PRESENT_BUT_SCORE_MISSING",
                        "verifier_prob_positive": None,
                        "verifier_score_available": False,
                    }
                else:
                    transitions[t] = {
                        "state_status": "PRESENT",
                        "verifier_prob_positive": s["primary_prob_positive"],
                        "verifier_score_available": True,
                        "n_components": s["n_components"],
                        "score_aggregation": s["aggregation"],
                        "min_prob": s["min_prob"],
                        "mean_prob": s["mean_prob"],
                        "max_prob": s["max_prob"],
                    }

        out.append({
            "qid": qid,
            "verifier_fold": qid_fold.get(qid),
            "predicted_required_transitions_oof": required,
            "required_and_present": present,
            "required_but_missing": missing,
            "transitions": transitions,
        })

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    for p in (
        INSTANCE_FILE,
        MANIFEST_FILE,
        OOF_PRED_FILE,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    instances = load_jsonl(INSTANCE_FILE)
    manifests = load_jsonl(MANIFEST_FILE)
    oof_rows = load_jsonl(OOF_PRED_FILE)

    qid_fold = build_qid_fold_map(oof_rows)

    natural_qids = {
        str(x["qid"])
        for x in manifests
    }
    missing_fold_qids = sorted(
        q for q in natural_qids
        if q not in qid_fold
    )
    if missing_fold_qids:
        raise RuntimeError(
            f"Natural qids missing held-out verifier fold: "
            f"{missing_fold_qids[:10]} total={len(missing_fold_qids)}"
        )

    # Attach the qid-held-out verifier fold.
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        qid = str(row["qid"])
        fold = qid_fold[qid]
        x = dict(row)
        x["heldout_verifier_fold"] = fold
        by_fold[fold].append(x)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 120)
    print("MMQA — NATURAL FROZEN VERIFIER V2 INFERENCE V1")
    print("=" * 120)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Natural verifier instances:", len(instances))
    print("Natural qids:", len(natural_qids))
    print("Scoring protocol: each qid uses ONLY its held-out verifier fold")
    print("5-checkpoint ensemble as primary inference: NO")
    print("Reason: avoid qid-level train/test leakage on dev_used")
    print("Input packing: exact frozen V2 training pack_input")
    print("Max length:", args.max_length)
    print("Batch size:", args.batch_size)
    print("FP16:", bool(args.fp16 and device.type == "cuda"))
    print("API/model calls: local frozen model only; external API = 0")
    print("-" * 120)

    all_scores = []
    fold_counts = Counter()

    for fold in range(5):
        rows = sorted(
            by_fold.get(fold, []),
            key=lambda x: (
                str(x["qid"]),
                T2ID[str(x["transition_type"])],
                int(x.get("component_index") or 0),
            ),
        )
        print(f"Fold {fold}: scoring {len(rows)} natural instances...")
        if not rows:
            continue

        fold_scores = score_fold(
            fold=fold,
            rows=rows,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            fp16=args.fp16,
        )
        all_scores.extend(fold_scores)
        fold_counts[fold] = len(fold_scores)
        print(
            f"Fold {fold}: done {len(fold_scores)}/{len(rows)}"
        )

    all_scores = sorted(
        all_scores,
        key=lambda x: (
            x["qid"],
            T2ID[x["transition"]],
            (
                x["component_index"]
                if x["component_index"] is not None
                else -1
            ),
        ),
    )

    if len(all_scores) != len(instances):
        raise RuntimeError(
            f"Scored instances mismatch: "
            f"{len(all_scores)} != {len(instances)}"
        )

    transition_scores = aggregate_transition_scores(all_scores)
    qid_states = build_qid_states(
        manifests,
        transition_scores,
        qid_fold,
    )

    write_jsonl(INSTANCE_SCORE_FILE, all_scores)
    write_jsonl(TRANSITION_SCORE_FILE, transition_scores)
    write_jsonl(QID_STATE_FILE, qid_states)

    inst_counts = Counter(x["transition"] for x in all_scores)
    qt_counts = Counter(x["transition"] for x in transition_scores)
    probs_by_t = defaultdict(list)
    for x in transition_scores:
        probs_by_t[x["transition"]].append(
            float(x["primary_prob_positive"])
        )

    per_t = {}
    for t in TRANSITIONS:
        vals = probs_by_t[t]
        per_t[t] = {
            "instance_scores": int(inst_counts[t]),
            "qid_transition_scores": int(qt_counts[t]),
            "mean_primary_prob": (
                float(np.mean(vals))
                if vals else None
            ),
            "median_primary_prob": (
                float(np.median(vals))
                if vals else None
            ),
            "below_0_5_qid_transitions": (
                int(sum(v < 0.5 for v in vals))
                if vals else 0
            ),
            "below_0_5_rate": (
                float(np.mean(np.asarray(vals) < 0.5))
                if vals else None
            ),
        }

    missing_count = 0
    present_scored_count = 0
    for q in qid_states:
        for x in q["transitions"].values():
            if x["state_status"] == "MISSING_REQUIRED_TRANSITION_STATE":
                missing_count += 1
            elif x["state_status"] == "PRESENT":
                present_scored_count += 1

    summary = {
        "natural_qids": len(natural_qids),
        "natural_verifier_instances_input": len(instances),
        "natural_verifier_instances_scored": len(all_scores),
        "qid_transition_scores": len(transition_scores),
        "required_present_scored_state_slots": present_scored_count,
        "required_missing_state_slots": missing_count,
        "fold_instance_counts": {
            str(k): int(v)
            for k, v in sorted(fold_counts.items())
        },
        "per_transition": per_t,
        "scoring_protocol": {
            "primary": "qid-held-out-verifier-fold",
            "qid_leakage_control": (
                "Each natural qid is scored only by the checkpoint "
                "whose verifier-training fold held that qid out."
            ),
            "transition_component_aggregation": (
                "minimum probability across components"
            ),
            "external_api_calls": 0,
            "max_length": args.max_length,
            "fp16": bool(args.fp16 and device.type == "cuda"),
        },
        "important_caution": (
            "A verifier probability is available only for required+present states. "
            "Required+missing states remain a separate structural signal and are NOT "
            "silently assigned probability 0."
        ),
        "status": "MMQA_NATURAL_FROZEN_VERIFIER_V2_INFERENCE_V1_COMPLETE",
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("-" * 120)
    print("Scored instances:", len(all_scores))
    print("Aggregated qid-transition scores:", len(transition_scores))
    print("Required+present scored state slots:", present_scored_count)
    print("Required+missing state slots:", missing_count)
    print("Fold instance counts:", dict(sorted(fold_counts.items())))
    for t in TRANSITIONS:
        x = per_t[t]
        print(
            f"{t}: instances={x['instance_scores']} "
            f"qid_transition_scores={x['qid_transition_scores']} "
            f"mean_prob={x['mean_primary_prob']} "
            f"median_prob={x['median_primary_prob']} "
            f"below0.5={x['below_0_5_qid_transitions']} "
            f"rate={x['below_0_5_rate']}"
        )
    print("-" * 120)
    print("Instance scores:", INSTANCE_SCORE_FILE)
    print("Transition scores:", TRANSITION_SCORE_FILE)
    print("QID states:", QID_STATE_FILE)
    print("Summary:", SUMMARY_FILE)
    print("STATUS = MMQA_NATURAL_FROZEN_VERIFIER_V2_INFERENCE_V1_COMPLETE")


if __name__ == "__main__":
    main()
