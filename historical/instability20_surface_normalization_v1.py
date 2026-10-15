#!/usr/bin/env python3
"""Frozen deterministic answer normalization for Instability-20.

Version: surface-normalization-v1

The procedure deliberately performs character-level canonicalization only.  It
does not use Gold, semantic models, synonym maps, entity aliases, or numerical
value conversion.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Any


NORMALIZATION_VERSION = "instability20-surface-normalization-v1"

# Conservative unit tokens for which whitespace after a numeric character is
# typographical rather than semantic.  Longest alternatives are listed first.
_UNIT_TOKEN = (
    r"(?:ghz|mhz|khz|hz|km|cm|mm|kg|mg|lbs|lb|oz|ml|cl|kw|mw|"
    r"ms|min|hrs|hr|m|g|l|w|v|s|h)"
)
_NUMBER_UNIT_SPACE = re.compile(
    rf"(?<=\d)\s+(?=(?:°\s*[cfk]|℃|℉|%|‰|‱|{_UNIT_TOKEN})(?:\b|$))",
    flags=re.IGNORECASE,
)
_DEGREE_LETTER_SPACE = re.compile(r"(?<=°)\s+(?=[cfk](?:\b|$))", re.IGNORECASE)


def _unicode_space_canonicalize(value: str) -> str:
    """Use NFC and map every Unicode whitespace code point to ASCII space."""
    value = unicodedata.normalize("NFC", value)
    value = "".join(" " if character.isspace() else character for character in value)
    return " ".join(value.split())


def canonicalize_surface(value: Any) -> str:
    """Apply the frozen surface-only rules before MMQA answer preprocessing."""
    text = "" if value is None else str(value)
    text = _unicode_space_canonicalize(text)
    text = _NUMBER_UNIT_SPACE.sub("", text)
    text = _DEGREE_LETTER_SPACE.sub("", text)
    return text.strip()


def normalize_instability_answer(value: Any) -> str:
    """Surface canonicalization followed by the frozen MMQA normalizer."""
    text = canonicalize_surface(value).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(character for character in text if character not in string.punctuation)
    return " ".join(text.split())


def frozen_rule_specification() -> dict[str, object]:
    return {
        "version": NORMALIZATION_VERSION,
        "ordered_steps": [
            "Unicode NFC normalization",
            "map every Unicode whitespace code point to ASCII space",
            "trim leading/trailing whitespace and collapse repeated whitespace",
            "remove whitespace between a digit and a frozen common unit/symbol token",
            "remove whitespace between degree sign and C/F/K",
            "apply the existing locked1341 MMQA lowercase/article/ASCII-punctuation/whitespace preprocessing",
        ],
        "numeric_unit_tokens": [
            "°C", "°F", "°K", "℃", "℉", "%", "‰", "‱",
            "GHz", "MHz", "kHz", "Hz", "km", "cm", "mm", "m",
            "kg", "mg", "g", "lbs", "lb", "oz", "mL", "cL", "L",
            "kW", "MW", "W", "V", "ms", "s", "min", "h", "hr", "hrs",
        ],
        "semantic_operations": [],
        "uses_gold": False,
        "uses_embeddings_or_model_judge": False,
        "changes_numeric_values": False,
        "changes_entity_tokens_or_list_members": False,
    }

