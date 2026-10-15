from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent

TRAIN_READY = (
    ROOT / "outputs" / "mmqa_dev1100_verifier_train_ready_v1"
    / "mmqa_verifier_train_ready_v1.jsonl"
)
V2_OOF = (
    ROOT / "outputs" / "mmqa_transition_causal_verifier_v2_pairwise"
    / "oof_predictions_available.jsonl"
)
NATURAL_INPUTS = (
    ROOT / "outputs" / "mmqa_natural_verifier_inference_inputs_v1"
    / "mmqa_natural_verifier_instances_v1.jsonl"
)
OLD_NAT_T4 = (
    ROOT / "outputs" / "mmqa_natural_frozen_verifier_v2_inference_v1"
    / "mmqa_natural_frozen_verifier_transition_scores_v1.jsonl"
)

OUT_DIR = ROOT / "outputs" / "mmqa_t4_operator_aware_verifier_v1"
OOF_INSTANCE_FILE = OUT_DIR / "t4_operator_aware_oof_instance_predictions_v1.jsonl"
OOF_PAIR_FILE = OUT_DIR / "t4_operator_aware_oof_pair_predictions_v1.jsonl"
NAT_INSTANCE_FILE = OUT_DIR / "t4_operator_aware_natural_instance_scores_v1.jsonl"
NAT_TRANSITION_FILE = OUT_DIR / "t4_operator_aware_natural_transition_scores_v1.jsonl"
SUMMARY_FILE = OUT_DIR / "t4_operator_aware_verifier_v1_summary.json"

SEED = 42
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
MODES = ["structured", "word", "char", "word_char"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                x = json.loads(s)
            except Exception as e:
                raise RuntimeError(f"Malformed JSONL {path} line {line_no}: {e}") from e
            if isinstance(x, dict):
                rows.append(x)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_sections(text: str) -> dict[str, str]:
    out = {}
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line.startswith("[") or "]" not in line:
            continue
        tag, content = line.split("]", 1)
        out[tag[1:].strip()] = content.strip()
    return out


def flatten_values(x: Any) -> list[str]:
    out = []
    if x is None:
        return out
    if isinstance(x, dict):
        for k, v in x.items():
            out.extend(flatten_values(k))
            out.extend(flatten_values(v))
        return out
    if isinstance(x, (list, tuple, set)):
        for v in x:
            out.extend(flatten_values(v))
        return out
    s = str(x).strip()
    if s:
        out.append(s)
    return out


def parse_branch_values(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple, dict)):
        vals = flatten_values(raw)
        return vals if vals else [str(raw)]

    s = str(raw or "").strip()
    if not s:
        return []

    try:
        obj = json.loads(s)
        vals = flatten_values(obj)
        if vals:
            return vals
    except Exception:
        pass

    # Conservative fallback. Keep the whole string and also expose obvious pieces.
    vals = [s]
    pieces = [p.strip() for p in re.split(r"\s*[|;]\s*", s) if p.strip()]
    if len(pieces) > 1:
        vals.extend(pieces)
    return list(dict.fromkeys(vals))


def norm_text(x: Any) -> str:
    s = str(x or "").casefold()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^\w\.\-%]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def tokens(x: Any) -> set[str]:
    s = norm_text(x)
    return {t for t in s.split() if t}


def numbers(x: Any) -> list[float]:
    s = str(x or "").replace(",", "")
    vals = []
    for m in re.findall(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", s):
        try:
            vals.append(float(m))
        except Exception:
            pass
    return vals


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def jaccard(a: Any, b: Any) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def containment(a: Any, b: Any) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def seqsim(a: Any, b: Any) -> float:
    aa, bb = norm_text(a), norm_text(b)
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


MAX_CUES = (
    "more", "most", "larger", "largest", "greater", "greatest", "higher", "highest",
    "later", "latest", "newer", "newest", "older", "oldest", "longer", "longest",
    "after", "maximum", "max",
)
MIN_CUES = (
    "less", "least", "smaller", "smallest", "lower", "lowest",
    "earlier", "earliest", "younger", "youngest", "shorter", "shortest",
    "before", "minimum", "min",
)
INTERSECT_CUES = ("both", "common", "intersection", "intersect", "shared", "same")
COUNT_CUES = ("how many", "number of", "count")
DIFF_CUES = ("difference", "differ", "gap", "how much")
COMPARE_CUES = ("compare", "which", "than", "between")


def contains_any(s: str, cues: tuple[str, ...]) -> bool:
    return any(c in s for c in cues)


def record_features(question: str, operator: str, branches_raw: Any, candidate: str):
    branches = parse_branch_values(branches_raw)
    q = norm_text(question)
    op = norm_text(operator)
    cand = norm_text(candidate)
    bnorm = [norm_text(x) for x in branches if norm_text(x)]

    cand_t = tokens(cand)
    branch_t = [tokens(x) for x in bnorm]
    union_t = set().union(*branch_t) if branch_t else set()
    inter_t = set.intersection(*branch_t) if branch_t else set()

    exacts = [int(cand == b and cand != "") for b in bnorm]
    jacs = [jaccard(cand, b) for b in bnorm]
    conts = [containment(cand, b) for b in bnorm]
    seqs = [seqsim(cand, b) for b in bnorm]

    cand_nums = numbers(candidate)
    branch_nums_per = [numbers(b) for b in branches]
    branch_nums = [x for xs in branch_nums_per for x in xs]

    cand_num = cand_nums[0] if cand_nums else None
    uniq_branch_nums = sorted(set(branch_nums))
    num_eq_any = 0.0
    num_is_max = 0.0
    num_is_min = 0.0
    num_nearest_norm = 0.0
    num_rank_norm = 0.0

    if cand_num is not None and uniq_branch_nums:
        num_eq_any = float(any(abs(cand_num - b) <= 1e-8 for b in uniq_branch_nums))
        num_is_max = float(abs(cand_num - max(uniq_branch_nums)) <= 1e-8)
        num_is_min = float(abs(cand_num - min(uniq_branch_nums)) <= 1e-8)
        span = max(uniq_branch_nums) - min(uniq_branch_nums)
        nearest = min(abs(cand_num - b) for b in uniq_branch_nums)
        num_nearest_norm = nearest / (abs(span) + 1.0)
        if len(uniq_branch_nums) > 1:
            rank = sum(b <= cand_num for b in uniq_branch_nums) - 1
            num_rank_norm = rank / (len(uniq_branch_nums) - 1)

    qop = q + " " + op
    cue_max = float(contains_any(qop, MAX_CUES))
    cue_min = float(contains_any(qop, MIN_CUES))
    cue_inter = float(contains_any(qop, INTERSECT_CUES))
    cue_count = float(contains_any(qop, COUNT_CUES))
    cue_diff = float(contains_any(qop, DIFF_CUES))
    cue_compare = float(contains_any(qop, COMPARE_CUES))

    max_jac = max(jacs) if jacs else 0.0
    max_cont = max(conts) if conts else 0.0
    max_seq = max(seqs) if seqs else 0.0

    dense = np.array([
        len(branches),
        len(cand_t),
        len(union_t),
        len(inter_t),
        float(sum(exacts)),
        float(exacts[0]) if exacts else 0.0,
        float(exacts[-1]) if exacts else 0.0,
        max_jac,
        float(np.mean(jacs)) if jacs else 0.0,
        max_cont,
        float(np.mean(conts)) if conts else 0.0,
        max_seq,
        float(np.mean(seqs)) if seqs else 0.0,
        safe_ratio(len(cand_t & union_t), len(cand_t)),
        safe_ratio(len(cand_t & inter_t), len(cand_t)),
        safe_ratio(len(inter_t), len(union_t)),
        float(bool(cand_t) and cand_t.issubset(union_t)),
        float(bool(cand_t) and cand_t.issubset(inter_t)),
        float(len(cand_nums)),
        float(len(branch_nums)),
        num_eq_any,
        num_is_max,
        num_is_min,
        num_nearest_norm,
        num_rank_norm,
        cue_max,
        cue_min,
        cue_inter,
        cue_count,
        cue_diff,
        cue_compare,
        cue_max * num_is_max,
        cue_max * num_is_min,
        cue_min * num_is_min,
        cue_min * num_is_max,
        cue_inter * safe_ratio(len(cand_t & inter_t), len(cand_t)),
        cue_count * float(cand_num is not None),
    ], dtype=np.float64)

    relation_tokens = [
        f"branch_count_{min(len(branches), 5)}",
        "cand_exact_any" if sum(exacts) else "cand_not_exact_any",
        "cand_num_eq_any" if num_eq_any else "cand_num_not_eq_any",
        "cand_num_is_max" if num_is_max else "cand_num_not_max",
        "cand_num_is_min" if num_is_min else "cand_num_not_min",
        "cue_max" if cue_max else "no_cue_max",
        "cue_min" if cue_min else "no_cue_min",
        "cue_inter" if cue_inter else "no_cue_inter",
        "cue_count" if cue_count else "no_cue_count",
        "cue_diff" if cue_diff else "no_cue_diff",
        "cue_compare" if cue_compare else "no_cue_compare",
        "cand_subset_union" if (cand_t and cand_t.issubset(union_t)) else "cand_not_subset_union",
        "cand_subset_intersection" if (cand_t and cand_t.issubset(inter_t)) else "cand_not_subset_intersection",
    ]

    canonical = (
        f"OPERATOR {op} "
        f"QUESTION {q} "
        f"BRANCHES {' [BR] '.join(bnorm)} "
        f"CANDIDATE {cand} "
        f"REL {' '.join(relation_tokens)}"
    )

    return dense, canonical


def train_row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    sec = parse_sections(row["input_text"])
    return {
        "instance_id": str(row["instance_id"]),
        "pair_id": str(row["pair_id"]),
        "qid": str(row["qid"]),
        "fold": int(row["fold"]),
        "label": int(row["label"]),
        "question": sec.get("QUESTION", ""),
        "operator": sec.get("AGGREGATION_OPERATOR", ""),
        "branches": sec.get("VERIFIED_BRANCH_VALUES", ""),
        "candidate": sec.get("CANDIDATE_AGGREGATION_RESULT", ""),
    }


def natural_row_to_record(row: dict[str, Any], qid_fold: dict[str, int]) -> dict[str, Any]:
    f = row.get("fields") or {}
    qid = str(row["qid"])
    return {
        "instance_id": str(row["instance_id"]),
        "qid": qid,
        "fold": int(qid_fold[qid]),
        "question": str(f.get("question", "") or ""),
        "operator": str(f.get("aggregation_operator", "") or ""),
        "branches": f.get("verified_branch_values", []),
        "candidate": str(f.get("candidate_aggregation_result", "") or ""),
        "component_index": row.get("component_index"),
    }


def build_pair_index(records: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    by_pair = defaultdict(list)
    for i, r in enumerate(records):
        by_pair[r["pair_id"]].append((i, r))
    pairs = []
    bad = []
    for pid, items in by_pair.items():
        if len(items) != 2:
            bad.append((pid, len(items)))
            continue
        pos = [(i, r) for i, r in items if r["label"] == 1]
        neg = [(i, r) for i, r in items if r["label"] == 0]
        if len(pos) != 1 or len(neg) != 1:
            bad.append((pid, "labels"))
            continue
        pi, pr = pos[0]
        ni, nr = neg[0]
        if pr["qid"] != nr["qid"] or pr["fold"] != nr["fold"]:
            bad.append((pid, "qid/fold"))
            continue
        pairs.append((pi, ni, {
            "pair_id": pid,
            "qid": pr["qid"],
            "fold": pr["fold"],
        }))
    if bad:
        raise RuntimeError(f"Malformed T4 pairs: {bad[:10]} total={len(bad)}")
    return pairs


def qid_fold_map_from_oof(rows):
    by = defaultdict(set)
    for r in rows:
        qid = str(r.get("qid", ""))
        if qid:
            by[qid].add(int(r["fold"]))
    bad = {q: sorted(v) for q, v in by.items() if len(v) != 1}
    if bad:
        raise RuntimeError(f"OOF qid fold conflicts: {list(bad.items())[:10]}")
    return {q: next(iter(v)) for q, v in by.items()}


class FeatureBuilder:
    def __init__(self, mode: str):
        self.mode = mode
        self.word = None
        self.char = None
        self.scaler = StandardScaler()

    def fit(self, records):
        dense = []
        texts = []
        for r in records:
            d, t = record_features(
                r["question"], r["operator"], r["branches"], r["candidate"]
            )
            dense.append(d)
            texts.append(t)
        dense = np.vstack(dense)
        self.scaler.fit(dense)

        if self.mode in ("word", "word_char"):
            self.word = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                max_features=12000,
                sublinear_tf=True,
                lowercase=True,
            )
            self.word.fit(texts)

        if self.mode in ("char", "word_char"):
            self.char = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=20000,
                sublinear_tf=True,
                lowercase=True,
            )
            self.char.fit(texts)
        return self

    def transform(self, records):
        dense = []
        texts = []
        for r in records:
            d, t = record_features(
                r["question"], r["operator"], r["branches"], r["candidate"]
            )
            dense.append(d)
            texts.append(t)

        dense = self.scaler.transform(np.vstack(dense))
        blocks = [csr_matrix(dense)]

        if self.word is not None:
            blocks.append(self.word.transform(texts))
        if self.char is not None:
            blocks.append(self.char.transform(texts))

        return hstack(blocks, format="csr")


def build_pairwise_matrix(X, pairs):
    rows = []
    y = []
    for pi, ni, _ in pairs:
        diff = X[pi] - X[ni]
        rows.append(diff)
        y.append(1)
        rows.append(-diff)
        y.append(0)

    if not rows:
        return csr_matrix((0, X.shape[1])), np.asarray(y, dtype=int)

    # Each diff is one training sample with shape (1, n_features).
    # We must stack them vertically into (2 * n_pairs, n_features).
    Xpair = vstack(rows, format="csr")
    ypair = np.asarray(y, dtype=int)

    if Xpair.shape[0] != len(ypair):
        raise RuntimeError(
            f"Pairwise matrix shape mismatch: Xpair={Xpair.shape}, y={ypair.shape}"
        )

    return Xpair, ypair


def fit_ranker(records, pairs, mode, C):
    fb = FeatureBuilder(mode).fit(records)
    X = fb.transform(records)
    Xpair, ypair = build_pairwise_matrix(X, pairs)

    if Xpair.shape[0] != 2 * len(pairs):
        raise RuntimeError(
            f"Unexpected pairwise sample count: rows={Xpair.shape[0]}, pairs={len(pairs)}"
        )

    ranker = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver="liblinear",
        fit_intercept=False,
        max_iter=4000,
        random_state=SEED,
    )
    ranker.fit(Xpair, ypair)

    raw = ranker.decision_function(X).reshape(-1, 1)
    y = np.asarray([r["label"] for r in records], dtype=int)
    calibrator = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )
    calibrator.fit(raw, y)

    return fb, ranker, calibrator


def predict_model(fb, ranker, calibrator, records):
    X = fb.transform(records)
    raw = ranker.decision_function(X)
    prob = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    return raw, prob


def metric_dict(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    pr, rc, f1p, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_positive": float(pr),
        "recall_positive": float(rc),
        "f1_positive": float(f1p),
        "binary_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


def pairwise_accuracy(records, pairs, probs):
    margins = []
    for pi, ni, meta in pairs:
        margins.append(float(probs[pi] - probs[ni]))
    return {
        "pairs": len(margins),
        "pairwise_accuracy": float(np.mean(np.asarray(margins) > 0.0)),
        "mean_margin": float(np.mean(margins)),
    }


def remap_pairs(subrecords):
    # Build pair indices local to subrecords.
    return build_pair_index(subrecords)


def inner_select(train_records):
    groups = np.asarray([r["qid"] for r in train_records])
    unique_groups = np.unique(groups)
    n_splits = min(3, len(unique_groups))
    if n_splits < 2:
        return "structured", 1.0, []

    splitter = GroupKFold(n_splits=n_splits)
    candidates = []
    yall = np.asarray([r["label"] for r in train_records], dtype=int)

    for mode in MODES:
        for C in C_GRID:
            fold_scores = []
            for tri, vai in splitter.split(train_records, yall, groups):
                tr = [train_records[i] for i in tri]
                va = [train_records[i] for i in vai]
                tr_pairs = remap_pairs(tr)
                va_pairs = remap_pairs(va)
                if not tr_pairs or not va_pairs:
                    continue
                try:
                    fb, ranker, cal = fit_ranker(tr, tr_pairs, mode, C)
                    _, p = predict_model(fb, ranker, cal, va)
                    y = [r["label"] for r in va]
                    auc = roc_auc_score(y, p)
                    pacc = pairwise_accuracy(va, va_pairs, p)["pairwise_accuracy"]
                    score = 0.5 * float(auc) + 0.5 * float(pacc)
                    fold_scores.append(score)
                except Exception:
                    continue

            mean_score = float(np.mean(fold_scores)) if fold_scores else -1.0
            candidates.append({
                "mode": mode,
                "C": C,
                "inner_score": mean_score,
                "fold_scores": fold_scores,
            })

    candidates.sort(key=lambda x: x["inner_score"], reverse=True)
    best = candidates[0]
    return best["mode"], best["C"], candidates[:10]


def v2_t4_reference(oof_rows):
    sub = [r for r in oof_rows if str(r.get("transition")) == "T4"]
    if not sub:
        return None
    y = [int(r["label"]) for r in sub]
    p = [float(r["prob_positive"]) for r in sub]
    m = metric_dict(y, p)
    by_pair = defaultdict(dict)
    for r in sub:
        by_pair[str(r["pair_id"])][int(r["label"])] = float(r["prob_positive"])
    margins = []
    for _, d in by_pair.items():
        if 1 in d and 0 in d:
            margins.append(d[1] - d[0])
    m["pairwise_accuracy"] = float(np.mean(np.asarray(margins) > 0.0))
    m["mean_pair_margin"] = float(np.mean(margins))
    return m


def main():
    for p in (TRAIN_READY, V2_OOF, NATURAL_INPUTS, OLD_NAT_T4):
        if not p.is_file():
            raise FileNotFoundError(p)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ready = load_jsonl(TRAIN_READY)
    oof_rows = load_jsonl(V2_OOF)
    natural_inputs = load_jsonl(NATURAL_INPUTS)
    old_nat = load_jsonl(OLD_NAT_T4)

    t4_rows = [
        train_row_to_record(r)
        for r in train_ready
        if str(r.get("transition_short")) == "T4"
    ]
    if len(t4_rows) != 356:
        print(f"[warning] expected 356 T4 binary rows, found {len(t4_rows)}")

    pairs = build_pair_index(t4_rows)
    qid_fold = qid_fold_map_from_oof(oof_rows)

    natural_t4 = [
        natural_row_to_record(r, qid_fold)
        for r in natural_inputs
        if str(r.get("transition_type")) == "T4" and str(r.get("qid")) in qid_fold
    ]

    print("=" * 122)
    print("MMQA — T4 OPERATOR-AWARE STRUCTURED VERIFIER V1")
    print("=" * 122)
    print("Controlled T4 binary rows:", len(t4_rows))
    print("Controlled T4 pairs:", len(pairs))
    print("Natural T4 instances:", len(natural_t4))
    print("Protocol: outer 5-fold by frozen qid fold; model/C selected only inside outer-train folds.")
    print("Features: operator-aware relational features + optional word/char TF-IDF.")
    print("Training objective: pairwise ranking on positive-negative feature differences + probability calibration.")
    print("Gold/oracle at Natural inference: NO")
    print("External API calls = 0")
    print("-" * 122)

    oof_pred = []
    oof_pairs = []
    nat_pred = []
    fold_records = []

    for fold in range(5):
        tr = [r for r in t4_rows if r["fold"] != fold]
        va = [r for r in t4_rows if r["fold"] == fold]
        tr_pairs = remap_pairs(tr)
        va_pairs = remap_pairs(va)

        mode, C, inner_top = inner_select(tr)
        fb, ranker, cal = fit_ranker(tr, tr_pairs, mode, C)

        raw_va, p_va = predict_model(fb, ranker, cal, va)
        y_va = [r["label"] for r in va]
        metrics = metric_dict(y_va, p_va)
        pw = pairwise_accuracy(va, va_pairs, p_va)

        fold_records.append({
            "fold": fold,
            "selected_mode": mode,
            "selected_C": C,
            "inner_top_candidates": inner_top,
            "validation_metrics": metrics,
            "validation_pairwise": pw,
            "train_rows": len(tr),
            "valid_rows": len(va),
        })

        for r, raw, prob in zip(va, raw_va.tolist(), p_va.tolist()):
            oof_pred.append({
                "instance_id": r["instance_id"],
                "pair_id": r["pair_id"],
                "qid": r["qid"],
                "fold": fold,
                "transition": "T4",
                "label": r["label"],
                "rank_score": float(raw),
                "prob_positive": float(prob),
                "selected_mode": mode,
                "selected_C": C,
            })

        by_pair_local = defaultdict(dict)
        for r, prob in zip(va, p_va.tolist()):
            by_pair_local[r["pair_id"]][r["label"]] = (r, float(prob))
        for pid, d in by_pair_local.items():
            if 1 in d and 0 in d:
                pr, pp = d[1]
                nr, npb = d[0]
                oof_pairs.append({
                    "pair_id": pid,
                    "qid": pr["qid"],
                    "fold": fold,
                    "pos_prob": pp,
                    "neg_prob": npb,
                    "margin": pp - npb,
                    "correct": int(pp > npb),
                })

        nat_fold = [r for r in natural_t4 if r["fold"] == fold]
        if nat_fold:
            raw_n, p_n = predict_model(fb, ranker, cal, nat_fold)
            for r, raw, prob in zip(nat_fold, raw_n.tolist(), p_n.tolist()):
                nat_pred.append({
                    "instance_id": r["instance_id"],
                    "qid": r["qid"],
                    "fold": fold,
                    "transition": "T4",
                    "component_index": r.get("component_index"),
                    "rank_score": float(raw),
                    "prob_positive": float(prob),
                    "selected_mode": mode,
                    "selected_C": C,
                })

        print(
            f"Fold {fold}: mode={mode} C={C} "
            f"AUC={metrics['roc_auc']:.4f} "
            f"MacroF1={metrics['binary_macro_f1']:.4f} "
            f"PairAcc={pw['pairwise_accuracy']:.4f}"
        )

    oof_pred = sorted(oof_pred, key=lambda x: (x["fold"], x["qid"], x["pair_id"], x["label"]))
    oof_pairs = sorted(oof_pairs, key=lambda x: (x["fold"], x["qid"], x["pair_id"]))
    nat_pred = sorted(nat_pred, key=lambda x: (x["fold"], x["qid"], int(x.get("component_index") or 0)))

    y = [x["label"] for x in oof_pred]
    p = [x["prob_positive"] for x in oof_pred]
    final_metrics = metric_dict(y, p)
    final_pair = {
        "pairs": len(oof_pairs),
        "pairwise_accuracy": float(np.mean([x["correct"] for x in oof_pairs])),
        "mean_margin": float(np.mean([x["margin"] for x in oof_pairs])),
    }

    # Natural T4 is one component per qid in the current state, but aggregate generically.
    by_qid = defaultdict(list)
    for x in nat_pred:
        by_qid[x["qid"]].append(x)

    nat_transition = []
    for qid, items in sorted(by_qid.items()):
        probs = np.asarray([x["prob_positive"] for x in items], dtype=float)
        nat_transition.append({
            "qid": qid,
            "transition": "T4",
            "fold": int(items[0]["fold"]),
            "n_components": len(items),
            "primary_prob_positive": float(probs.min()),
            "mean_prob": float(probs.mean()),
            "max_prob": float(probs.max()),
            "aggregation": "min_component_probability",
        })

    old_t4 = {
        str(x["qid"]): float(x["primary_prob_positive"])
        for x in old_nat
        if str(x.get("transition")) == "T4"
    }
    new_vals = [x["primary_prob_positive"] for x in nat_transition]
    old_vals = [old_t4[x["qid"]] for x in nat_transition if x["qid"] in old_t4]

    v2_ref = v2_t4_reference(oof_rows)

    summary = {
        "method": "operator-aware structured pairwise ranker",
        "controlled_t4_rows": len(t4_rows),
        "controlled_t4_pairs": len(pairs),
        "natural_t4_instances": len(natural_t4),
        "fold_records": fold_records,
        "oof_metrics": final_metrics,
        "oof_pairwise": final_pair,
        "v2_t4_reference": v2_ref,
        "improvement_over_v2": {
            "auc_delta": (
                final_metrics["roc_auc"] - v2_ref["roc_auc"]
                if v2_ref else None
            ),
            "macro_f1_delta": (
                final_metrics["binary_macro_f1"] - v2_ref["binary_macro_f1"]
                if v2_ref else None
            ),
            "pairwise_accuracy_delta": (
                final_pair["pairwise_accuracy"] - v2_ref["pairwise_accuracy"]
                if v2_ref else None
            ),
        },
        "natural_heldout": {
            "qid_transition_n": len(nat_transition),
            "old_v2_median": float(np.median(old_vals)) if old_vals else None,
            "new_operator_aware_median": float(np.median(new_vals)) if new_vals else None,
            "old_v2_below_0_5_rate": float(np.mean(np.asarray(old_vals) < 0.5)) if old_vals else None,
            "new_below_0_5_rate": float(np.mean(np.asarray(new_vals) < 0.5)) if new_vals else None,
        },
        "guardrail": (
            "Natural T4 scores are inference-time diagnostic scores only. "
            "Model quality is judged primarily by held-out controlled OOF AUC/Macro-F1/pairwise accuracy."
        ),
        "gold_oracle_used_at_natural_inference": False,
        "external_api_calls": 0,
        "status": "MMQA_T4_OPERATOR_AWARE_VERIFIER_V1_COMPLETE",
    }

    write_jsonl(OOF_INSTANCE_FILE, oof_pred)
    write_jsonl(OOF_PAIR_FILE, oof_pairs)
    write_jsonl(NAT_INSTANCE_FILE, nat_pred)
    write_jsonl(NAT_TRANSITION_FILE, nat_transition)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 122)
    print("FINAL T4 OOF")
    print(
        f"Operator-aware: AUC={final_metrics['roc_auc']:.6f} "
        f"MacroF1={final_metrics['binary_macro_f1']:.6f} "
        f"PairAcc={final_pair['pairwise_accuracy']:.6f}"
    )
    if v2_ref:
        print(
            f"Frozen V2 ref: AUC={v2_ref['roc_auc']:.6f} "
            f"MacroF1={v2_ref['binary_macro_f1']:.6f} "
            f"PairAcc={v2_ref['pairwise_accuracy']:.6f}"
        )
        print(
            f"Delta: AUC={summary['improvement_over_v2']['auc_delta']:+.6f} "
            f"MacroF1={summary['improvement_over_v2']['macro_f1_delta']:+.6f} "
            f"PairAcc={summary['improvement_over_v2']['pairwise_accuracy_delta']:+.6f}"
        )
    print("-" * 122)
    print("NATURAL T4 HELD-OUT")
    print("n =", summary["natural_heldout"]["qid_transition_n"])
    print("Old V2 median =", summary["natural_heldout"]["old_v2_median"])
    print("New operator-aware median =", summary["natural_heldout"]["new_operator_aware_median"])
    print("Old V2 below0.5 =", summary["natural_heldout"]["old_v2_below_0_5_rate"])
    print("New below0.5 =", summary["natural_heldout"]["new_below_0_5_rate"])
    print("-" * 122)
    print("External API calls = 0")
    print("OOF instances:", OOF_INSTANCE_FILE)
    print("OOF pairs:", OOF_PAIR_FILE)
    print("Natural T4 scores:", NAT_TRANSITION_FILE)
    print("Summary:", SUMMARY_FILE)
    print("STATUS = MMQA_T4_OPERATOR_AWARE_VERIFIER_V1_COMPLETE")


if __name__ == "__main__":
    main()
