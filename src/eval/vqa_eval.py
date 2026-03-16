"""VQA-style answer normalisation and evaluation.

This module implements the standard VQA evaluation protocol used by
datasets like GQA and TextVQA — distinct from the A/B/C/D multiple-choice
normalisation in :mod:`src.voting.normalize`.

Reference: VQA v2 evaluation script (https://visualqa.org/evaluation.html)
"""

from __future__ import annotations

import re
import string
from typing import List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}

# Punctuation translation table: removes all string.punctuation chars.
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def vqa_normalize(text: str) -> str:
    """Normalise a free-form VQA answer for comparison.

    Steps applied in order:
    1. Strip leading/trailing whitespace.
    2. Lowercase.
    3. Remove punctuation.
    4. Remove leading articles (a / an / the).
    5. Collapse multiple spaces to one.

    Args:
        text: Raw answer string.

    Returns:
        Normalised answer string.
    """
    if not text:
        return ""
    text = text.strip().lower()
    text = text.translate(_PUNCT_TABLE)
    # Remove leading article (word boundary, at start of string only).
    text = re.sub(r"^(a|an|the)\s+", "", text)
    # Collapse runs of whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_vqa(prediction: str, references: List[str]) -> bool:
    """Check whether *prediction* matches any of the *references*.

    Both prediction and each reference are normalised with
    :func:`vqa_normalize` before comparison.

    Args:
        prediction: Model answer string.
        references: List of acceptable ground-truth answer strings.

    Returns:
        ``True`` if the normalised prediction equals any normalised reference,
        ``False`` otherwise (including when either input is empty).
    """
    if not prediction or not references:
        return False
    norm_pred = vqa_normalize(prediction)
    if not norm_pred:
        return False
    return any(norm_pred == vqa_normalize(ref) for ref in references)
