"""Mechanically extracted frozen scoring bodies; no provider imports.
Historical identifiers/docstrings are retained, not endorsed as current benchmark equivalence.
See docs/SCORING_VERSIONS.md and provenance/scoring_functions.json.
"""
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from collections import Counter
import json, re, string, unicodedata


def safe_text(x: Any) -> str:
    return "" if x is None else str(x).strip()

def norm(x: Any) -> str:
    s = unicodedata.normalize("NFKC", safe_text(x)).casefold()
    s = s.replace("-", " ")
    s = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", s)
    s = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\s\.\-%']", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def tokens(x: Any) -> list[str]:
    return [t for t in norm(x).split() if t]

def token_f1(a: str, b: str) -> float:
    aa, bb = tokens(a), tokens(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    ac, bc = Counter(aa), Counter(bb)
    common = sum((ac & bc).values())
    if common == 0:
        return 0.0
    p = common / len(aa)
    r = common / len(bb)
    return 2 * p * r / (p + r)

def extract_answers(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, (int, float, bool)):
        return [str(x)]
    out = []
    if isinstance(x, list):
        for v in x:
            out.extend(extract_answers(v))
    elif isinstance(x, dict):
        for k in ("answers", "answer", "gold_answers", "gold_answer", "text", "value"):
            if k in x:
                out.extend(extract_answers(x[k]))
    return out

def score_one(pred: str, golds: list[str]) -> tuple[int, float]:
    em = int(any(norm(pred) == norm(g) for g in golds))
    f1 = max(token_f1(pred, g) for g in golds)
    return em, f1
