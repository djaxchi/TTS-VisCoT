"""Answer normalization utilities for multiple-choice and open-ended outputs."""

from __future__ import annotations

import re
from typing import Optional

_LABELS = {"A", "B", "C", "D"}

_TRAILING_PUNCT = re.compile(r"[.!?,;:]+$")
_LEADING_ARTICLE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def normalize_open_ended_answer(raw_output: str) -> Optional[str]:
    """Normalize a free-text VQA answer for fuzzy comparison.

    Lowercases, strips leading articles (a/an/the), strips trailing punctuation.

    Args:
        raw_output: Raw model output string.

    Returns:
        Normalized answer string, or ``None`` if empty after normalization.
    """
    if not raw_output:
        return None
    text = raw_output.strip().lower()
    text = _TRAILING_PUNCT.sub("", text).strip()
    text = _LEADING_ARTICLE.sub("", text).strip()
    return text if text else None

_EXPLICIT_PATTERNS = [
    re.compile(r"^\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\s*([\.!?])?\s*$", re.IGNORECASE),
    re.compile(r"\bOPTION\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"\b(?:ANSWER|FINAL ANSWER|CHOICE)\s*(?:IS|:)?\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"[\(\[]\s*([ABCD])\s*[\)\]]", re.IGNORECASE),
]

_STANDALONE_CHOICE = re.compile(r"(?<![A-Z0-9])([ABCD])(?![A-Z0-9])", re.IGNORECASE)


def normalize_answer(raw_output: str) -> Optional[str]:
    """Normalize raw model output to one of ``A``/``B``/``C``/``D``.

    Returns ``None`` when no unambiguous normalized label can be extracted.
    """
    if not raw_output:
        return None

    text = raw_output.strip()
    upper = text.upper()

    for pattern in _EXPLICIT_PATTERNS:
        match = pattern.search(upper)
        if match:
            label = match.group(1).upper()
            return label if label in _LABELS else None

    tokens = [m.group(1).upper() for m in _STANDALONE_CHOICE.finditer(upper)]
    unique = set(tokens)
    if len(unique) == 1:
        label = next(iter(unique))
        return label if label in _LABELS else None

    return None
