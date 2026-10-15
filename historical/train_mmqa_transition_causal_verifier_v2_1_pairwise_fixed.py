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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
DATA_FILE = (
    ROOT
    / "outputs"
    / "mmqa_dev1100_verifier_train_ready_v1"
    / "mmqa_verifier_train_ready_v1.jsonl"
)
OUT_ROOT = ROOT / "outputs" / "mmqa_transition_causal_verifier_v2_pairwise"

TRANSITIONS = ["T1", "T2", "T3", "T4", "T5"]
T2ID = {t: i for i, t in enumerate(TRANSITIONS)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["transition_short"], r["pair_id"])].append(r)

    pairs = []
    malformed = []
    for (t, pair_id), items in grouped.items():
        if len(items) != 2 or sorted(int(x["label"]) for x in items) != [0, 1]:
            malformed.append((t, pair_id, [x.get("label") for x in items]))
            continue
        pos = next(x for x in items if int(x["label"]) == 1)
        neg = next(x for x in items if int(x["label"]) == 0)
        if pos["qid"] != neg["qid"] or int(pos["fold"]) != int(neg["fold"]):
            malformed.append((t, pair_id, "qid/fold mismatch"))
            continue
        pairs.append({
            "transition": t,
            "pair_id": pair_id,
            "qid": pos["qid"],
            "fold": int(pos["fold"]),
            "positive": pos,
            "negative": neg,
        })

    if malformed:
        raise RuntimeError(f"Malformed pairs found: {malformed[:5]} (total={len(malformed)})")
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pos_text = [pack_input(tokenizer, p["positive"], max_length) for p in pairs]
        self.neg_text = [pack_input(tokenizer, p["negative"], max_length) for p in pairs]

    def __len__(self):
        return len(self.pairs)

    def _encode(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return {
            "pos": self._encode(self.pos_text[idx]),
            "neg": self._encode(self.neg_text[idx]),
            "transition": p["transition"],
            "transition_id": T2ID[p["transition"]],
            "pair_id": p["pair_id"],
            "qid": p["qid"],
            "fold": p["fold"],
            "pos_instance_id": p["positive"]["instance_id"],
            "neg_instance_id": p["negative"]["instance_id"],
        }


def make_collate(tokenizer):
    def collate(batch):
        # First all positives, then all negatives.
        all_ids = [b["pos"]["input_ids"] for b in batch] + [b["neg"]["input_ids"] for b in batch]
        all_mask = [b["pos"]["attention_mask"] for b in batch] + [b["neg"]["attention_mask"] for b in batch]
        enc = tokenizer.pad(
            {"input_ids": all_ids, "attention_mask": all_mask},
            padding=True,
            return_tensors="pt",
        )
        tids = torch.tensor(
            [b["transition_id"] for b in batch] + [b["transition_id"] for b in batch],
            dtype=torch.long,
        )
        meta = [{
            "transition": b["transition"],
            "pair_id": b["pair_id"],
            "qid": b["qid"],
            "fold": b["fold"],
            "pos_instance_id": b["pos_instance_id"],
            "neg_instance_id": b["neg_instance_id"],
        } for b in batch]
        return enc, tids, meta
    return collate


class TransitionAwareVerifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in TRANSITIONS])

    def forward(self, input_ids, attention_mask, transition_ids):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state[:, 0, :])

        # Under CUDA autocast, Linear outputs may be float16 even when h is
        # float32. Avoid in-place assignment into a preallocated float32 tensor:
        # compute all transition heads in one autocast-consistent tensor, then
        # gather the head corresponding to each sample's transition type.
        all_logits = torch.cat([head(h) for head in self.heads], dim=1)  # [B, 5]
        logits = all_logits.gather(
            1, transition_ids.unsqueeze(1)
        ).squeeze(1)
        return logits


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else None


def safe_ap(y, p):
    return float(average_precision_score(y, p)) if len(set(y)) == 2 else None


def instance_metrics(rows):
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


def pairwise_metrics(pair_rows):
    by_t = defaultdict(list)
    margins = []
    wins = 0
    for r in pair_rows:
        margin = float(r["pos_prob"]) - float(r["neg_prob"])
        margins.append(margin)
        wins += int(margin > 0)
        by_t[r["transition"]].append(margin)

    per_t = {}
    for t in TRANSITIONS:
        ms = by_t[t]
        per_t[t] = {
            "pairs": len(ms),
            "pairwise_accuracy": float(sum(m > 0 for m in ms) / len(ms)) if ms else None,
            "mean_margin": float(np.mean(ms)) if ms else None,
        }

    return {
        "pairs": len(pair_rows),
        "pairwise_accuracy": float(wins / len(pair_rows)) if pair_rows else None,
        "mean_margin": float(np.mean(margins)) if margins else None,
        "per_transition": per_t,
    }


def summarize_eval(instance_rows, pair_rows):
    per_t = {}
    macro_binary_f1 = []
    macro_auc = []
    for t in TRANSITIONS:
        sub = [r for r in instance_rows if r["transition"] == t]
        m = instance_metrics(sub)
        per_t[t] = m
        macro_binary_f1.append(m["binary_macro_f1"])
        if m["roc_auc"] is not None:
            macro_auc.append(m["roc_auc"])

    pw = pairwise_metrics(pair_rows)
    macro_pairwise = np.mean([
        pw["per_transition"][t]["pairwise_accuracy"]
        for t in TRANSITIONS
        if pw["per_transition"][t]["pairwise_accuracy"] is not None
    ])

    return {
        "overall": instance_metrics(instance_rows),
        "per_transition": per_t,
        "macro_transition_binary_macro_f1": float(np.mean(macro_binary_f1)),
        "macro_transition_auc": float(np.mean(macro_auc)),
        "pairwise": pw,
        "macro_transition_pairwise_accuracy": float(macro_pairwise),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    instance_rows = []
    pair_rows = []

    for enc, tids, meta in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        tids = tids.to(device)
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            transition_ids=tids,
        )
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        bsz = len(meta)
        pos_probs = probs[:bsz]
        neg_probs = probs[bsz:]

        for m, pp, pn in zip(meta, pos_probs.tolist(), neg_probs.tolist()):
            pair_rows.append({
                **m,
                "pos_prob": float(pp),
                "neg_prob": float(pn),
                "margin": float(pp - pn),
            })
            instance_rows.append({
                "instance_id": m["pos_instance_id"],
                "pair_id": m["pair_id"],
                "qid": m["qid"],
                "fold": m["fold"],
                "transition": m["transition"],
                "label": 1,
                "prob_positive": float(pp),
                "pred": int(pp >= 0.5),
            })
            instance_rows.append({
                "instance_id": m["neg_instance_id"],
                "pair_id": m["pair_id"],
                "qid": m["qid"],
                "fold": m["fold"],
                "transition": m["transition"],
                "label": 0,
                "prob_positive": float(pn),
                "pred": int(pn >= 0.5),
            })

    return instance_rows, pair_rows, summarize_eval(instance_rows, pair_rows)


def train_one_fold(args, fold, all_pairs, tokenizer, device):
    fold_dir = OUT_ROOT / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = fold_dir / "best_metrics.json"
    pred_path = fold_dir / "oof_predictions.jsonl"
    pair_pred_path = fold_dir / "oof_pair_predictions.jsonl"
    state_path = fold_dir / "best_model.pt"

    if metrics_path.is_file() and pred_path.is_file() and pair_pred_path.is_file() and not args.force:
        print(f"Fold {fold}: completed result found; skipping.")
        return (
            json.loads(metrics_path.read_text(encoding="utf-8")),
            load_jsonl(pred_path),
            load_jsonl(pair_pred_path),
        )

    train_pairs = [p for p in all_pairs if p["fold"] != fold]
    valid_pairs = [p for p in all_pairs if p["fold"] == fold]

    train_ds = PairDataset(train_pairs, tokenizer, args.max_length)
    valid_ds = PairDataset(valid_pairs, tokenizer, args.max_length)

    counts = Counter(p["transition"] for p in train_pairs)
    weights = [1.0 / counts[p["transition"]] for p in train_pairs]
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(train_pairs),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed + fold),
    )

    collate = make_collate(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.pair_batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.eval_pair_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = TransitionAwareVerifier(args.model, dropout=args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    opt_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = opt_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    use_amp = device.type == "cuda" and args.fp16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = Counter()

        for step, (enc, tids, _) in enumerate(train_loader, 1):
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            tids = tids.to(device, non_blocking=True)
            bsz = tids.shape[0] // 2

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                )
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

            with autocast_ctx:
                logits = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    transition_ids=tids,
                )
                pos_logits = logits[:bsz]
                neg_logits = logits[bsz:]

                point_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        pos_logits, torch.ones_like(pos_logits)
                    )
                    + F.binary_cross_entropy_with_logits(
                        neg_logits, torch.zeros_like(neg_logits)
                    )
                )
                # Pairwise logistic ranking: positive score should exceed negative score.
                rank_loss = F.softplus(-(pos_logits - neg_logits)).mean()

                loss = (
                    args.point_weight * point_loss
                    + args.rank_weight * rank_loss
                ) / args.grad_accum

            scaler.scale(loss).backward()
            running["point"] += float(point_loss.item())
            running["rank"] += float(rank_loss.item())
            running["total"] += float(
                args.point_weight * point_loss.item()
                + args.rank_weight * rank_loss.item()
            )

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        inst_preds, pair_preds, metrics = evaluate(model, valid_loader, device)
        score = metrics["macro_transition_binary_macro_f1"]

        print(
            f"Fold {fold} | epoch {epoch}/{args.epochs} | "
            f"loss={running['total']/max(1,len(train_loader)):.4f} "
            f"(point={running['point']/max(1,len(train_loader)):.4f}, "
            f"rank={running['rank']/max(1,len(train_loader)):.4f}) | "
            f"macro_binF1={metrics['macro_transition_binary_macro_f1']:.4f} | "
            f"macro_pairAcc={metrics['macro_transition_pairwise_accuracy']:.4f} | "
            f"macro_AUC={metrics['macro_transition_auc']:.4f}"
        )
        for t in TRANSITIONS:
            mt = metrics["per_transition"][t]
            pt = metrics["pairwise"]["per_transition"][t]
            print(
                f"  {t}: n={mt['n']} macroF1={mt['binary_macro_f1']:.4f} "
                f"AUC={mt['roc_auc']:.4f} pairAcc={pt['pairwise_accuracy']:.4f}"
            )

        if score > best_score + 1e-8:
            best_score = score
            patience_left = args.patience

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "transitions": TRANSITIONS,
                    "fold": fold,
                    "epoch": epoch,
                },
                state_path,
            )
            tokenizer.save_pretrained(fold_dir / "tokenizer")
            write_jsonl(pred_path, inst_preds)
            write_jsonl(pair_pred_path, pair_preds)

            record = {
                "fold": fold,
                "best_epoch": epoch,
                "selection_metric": "macro_transition_binary_macro_f1",
                "model": args.model,
                "architecture": "shared_encoder_plus_5_transition_specific_heads",
                "point_weight": args.point_weight,
                "rank_weight": args.rank_weight,
                "train_pair_transition_counts": dict(counts),
                "valid_pair_transition_counts": dict(
                    Counter(p["transition"] for p in valid_pairs)
                ),
                "metrics": metrics,
            }
            metrics_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Fold {fold}: early stopping.")
                break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        json.loads(metrics_path.read_text(encoding="utf-8")),
        load_jsonl(pred_path),
        load_jsonl(pair_pred_path),
    )


def mean_std(vals):
    vals = [float(v) for v in vals]
    if not vals:
        return {"mean": None, "std": None}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--pair-batch-size", type=int, default=2)
    parser.add_argument("--eval-pair-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--point-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.fold not in {-1, 0, 1, 2, 3, 4}:
        raise ValueError("--fold must be -1 or 0..4")
    if not DATA_FILE.is_file():
        raise FileNotFoundError(DATA_FILE)

    set_seed(args.seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 112)
    print("MMQA — TRANSITION-CAUSAL VERIFIER V2 (PAIRWISE + TRANSITION-SPECIFIC HEADS)")
    print("=" * 112)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", args.model)
    print("Selection metric: macro transition binary Macro-F1")
    print("Auxiliary objective: pairwise logistic ranking")
    print("Pair sampling: transition-balanced")
    print("-" * 112)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    rows = load_jsonl(DATA_FILE)
    pairs = build_pairs(rows)

    folds = [args.fold] if args.fold >= 0 else list(range(5))
    metric_records = []
    all_inst = []
    all_pair = []

    for fold in folds:
        rec, inst, pair = train_one_fold(args, fold, pairs, tokenizer, device)
        metric_records.append(rec)
        all_inst.extend(inst)
        all_pair.extend(pair)

    write_jsonl(
        OUT_ROOT / "oof_predictions_available.jsonl",
        sorted(all_inst, key=lambda x: (x["fold"], x["qid"], x["instance_id"])),
    )
    write_jsonl(
        OUT_ROOT / "oof_pair_predictions_available.jsonl",
        sorted(all_pair, key=lambda x: (x["fold"], x["qid"], x["pair_id"])),
    )

    overall_metrics = summarize_eval(all_inst, all_pair)
    summary = {
        "folds_completed": len(metric_records),
        "model": args.model,
        "architecture": "shared_encoder_plus_5_transition_specific_heads",
        "training_objective": {
            "pointwise_bce_weight": args.point_weight,
            "pairwise_logistic_weight": args.rank_weight,
        },
        "oof_available_metrics": overall_metrics,
        "fold_metric_records": metric_records,
        "status": (
            "VERIFIER_V2_5FOLD_COMPLETE"
            if len(metric_records) == 5
            else "VERIFIER_V2_PARTIAL_FOLDS"
        ),
    }
    (OUT_ROOT / "verifier_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("-" * 112)
    print("Completed folds:", len(metric_records))
    print("OOF instances:", len(all_inst))
    print("OOF pairs:", len(all_pair))
    print(
        "OOF macro transition binary Macro-F1:",
        overall_metrics["macro_transition_binary_macro_f1"],
    )
    print(
        "OOF macro transition pairwise accuracy:",
        overall_metrics["macro_transition_pairwise_accuracy"],
    )
    print("Output:", OUT_ROOT)
    print("STATUS =", summary["status"])


if __name__ == "__main__":
    main()
