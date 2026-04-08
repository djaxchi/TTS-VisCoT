"""Open-ended TTS helpers for self-consistency style evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from PIL import Image

from src.eval.vqa_eval import vqa_normalize
from src.models.base import BaseVisualCoTModel


def generate_oe_question_variants(question: str, n: int = 3) -> List[str]:
    """Create up to five deterministic open-ended question variants."""
    if n < 1 or n > 5:
        raise ValueError("n must be between 1 and 5")

    base = question.strip()
    frames = [
        base,
        f"{base}\n\nAnswer concisely using one short phrase.",
        f"{base}\n\nFocus only on what is visible in the image.",
        f"{base}\n\nUse the most direct factual answer.",
        f"{base}\n\nIf uncertain, provide the most likely visual answer.",
    ]
    return frames[:n]


def vote_open_ended(raw_answers: List[str]) -> str:
    """Majority vote over normalized open-ended answers with first-seen tie break."""
    normalized_answers = [vqa_normalize(a) for a in raw_answers]
    valid = [a for a in normalized_answers if a]
    if not valid:
        return ""

    counts = Counter(valid)
    top_count = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == top_count}
    return next(answer for answer in valid if answer in tied)


def run_oe_tts(
    image: Image.Image,
    question: str,
    model: BaseVisualCoTModel,
    *,
    n_candidates: int = 3,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Run open-ended TTS by querying deterministic prompt variants and voting."""
    queries = generate_oe_question_variants(question, n=n_candidates)
    raw_answers: List[str] = []
    for query in queries:
        chain = model.predict(
            image,
            query,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        raw_answers.append(str(chain.get("answer", "") or ""))

    normalized_answers = [vqa_normalize(a) for a in raw_answers]
    winner = vote_open_ended(raw_answers)
    valid = [a for a in normalized_answers if a]
    agreement_rate = 0.0
    if winner and valid:
        agreement_rate = sum(1 for answer in valid if answer == winner) / len(valid)

    return {
        "winner": winner,
        "raw_answers": raw_answers,
        "normalized_answers": normalized_answers,
        "n_candidates": n_candidates,
        "agreement_rate": agreement_rate,
    }