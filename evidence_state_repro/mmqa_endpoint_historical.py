"""Mechanically extracted frozen scoring bodies; no provider imports.
Historical identifiers/docstrings are retained, not endorsed as current benchmark equivalence.
See docs/SCORING_VERSIONS.md and provenance/scoring_functions.json.
"""
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from collections import Counter
import json, re, string, unicodedata


def stringify_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(stringify_answer(item) for item in value).strip()
    if isinstance(value, dict):
        for key in ("answer", "text", "value"):
            if key in value:
                return stringify_answer(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)

def extract_gold_answers(row: Mapping[str, Any]) -> List[str]:
    raw_answers = row.get("answers", row.get("answer", []))
    if not isinstance(raw_answers, list):
        raw_answers = [raw_answers]
    answers: List[str] = []
    for item in raw_answers:
        if isinstance(item, dict):
            value = item.get("answer", item.get("text", item.get("value", "")))
        else:
            value = item
        if isinstance(value, list):
            answers.extend(stringify_answer(x) for x in value)
        else:
            answers.append(stringify_answer(value))
    # Keep an explicit empty answer if the dataset truly provides one.
    return answers or [""]

def normalize_answer(text: Any) -> str:
    value = stringify_answer(text).lower()
    # Preserve the exact order used by the original locked evaluator:
    # articles first, then ASCII punctuation, then whitespace normalization.
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(ch for ch in value if ch not in string.punctuation)
    return " ".join(value.split())

def exact_match(prediction: Any, gold: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))

def token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def best_score(prediction: Any, gold_answers: Sequence[str]) -> Tuple[int, float]:
    em = int(max(exact_match(prediction, gold) for gold in gold_answers))
    f1 = max(token_f1(prediction, gold) for gold in gold_answers)
    return em, f1
