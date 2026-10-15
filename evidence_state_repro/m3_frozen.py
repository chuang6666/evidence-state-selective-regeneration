"""Mechanically extracted frozen scoring bodies; no provider imports.
Historical identifiers/docstrings are retained, not endorsed as current benchmark equivalence.
See docs/SCORING_VERSIONS.md and provenance/scoring_functions.json.
"""
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from collections import Counter
import json, re, string, unicodedata
import ast


DIGIT_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "entailment": "yes",
    "true": "yes",
    "contradiction": "no",
    "false": "no",
}

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't",
    "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
    "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's",
    "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}

class ProtocolError(RuntimeError):
    pass

def safe(value: Any) -> str:
    return "" if value is None else str(value).strip()

def normalize_answer(text: Any) -> str:
    """Exact normalization order from the official M3-VQA run_eval.py."""
    value = safe(text).lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation + "‘’´`_"))
    value = re.sub(r"\b(the answer is|a|an|the)\b", " ", value)
    words: List[str] = []
    for word in value.split():
        word = DIGIT_MAP.get(word, word)
        word = CONTRACTIONS.get(word, word)
        words.append(word)
    return " ".join(" ".join(words).split())

def convert_to_list(value: Any) -> List[Any]:
    """Safe equivalent of the official answer-string list conversion."""
    if isinstance(value, list):
        return value
    text = safe(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    if "[" in text and "]" in text:
        content = text[text.find("[") + 1 : text.find("]")]
    elif "[" in text:
        content = text[text.find("[") + 1 :]
    elif "]" in text:
        content = text[: text.find("]")]
    else:
        content = text
    return [item.strip() for item in content.split(",") if item.strip()]

def validate_answer_evals(value: Any, qid: str) -> List[List[Any]]:
    if not isinstance(value, list) or not value:
        raise ProtocolError(f"{qid}: answer_evals is not a nonempty list")
    result: List[List[Any]] = []
    for index, eval_set in enumerate(value):
        if not isinstance(eval_set, list) or not eval_set:
            raise ProtocolError(f"{qid}: answer_evals[{index}] is not a nonempty list")
        result.append(eval_set)
    return result

def official_accuracy(predicted_answers: Sequence[Any], answer_evals: Sequence[Sequence[Any]]) -> float:
    """Replicate official M3-VQA evaluate_predictions()."""
    if not predicted_answers:
        return 0.0
    normalized_pred = [normalize_answer(answer) for answer in predicted_answers]
    normalized_evals = [
        [normalize_answer(answer) for answer in eval_set]
        for eval_set in answer_evals
    ]
    flattened = {answer for eval_set in normalized_evals for answer in eval_set}
    if not (set(normalized_pred) & flattened):
        return 0.0
    keys: List[List[int]] = []
    for answer in normalized_pred:
        indices = [
            index
            for index, eval_set in enumerate(normalized_evals)
            if answer in eval_set
        ]
        if indices:
            keys.append(indices)
    matched = min(len(keys), len({index for indices in keys for index in indices}))
    denominator = len(normalized_pred) + len(normalized_evals) - matched
    return matched / denominator if denominator else 0.0
