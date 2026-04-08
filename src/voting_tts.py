"""Voting utilities for test-time scaling candidate aggregation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence


@dataclass
class VoteStats:
    winning_answer: Optional[str]
    vote_counts: Dict[str, int]
    agreement_rate: float
    vote_margin: int
    valid_votes: int
    total_candidates: int
    answer_entropy: float  # Shannon entropy (bits) over normalized answers; 0 = consensus


def majority_vote(answers: Sequence[Optional[str]]) -> Optional[str]:
    """Return majority-vote winner over normalized answers."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return ranked[0][0]


def compute_vote_stats(answers: Sequence[Optional[str]]) -> VoteStats:
    """Compute winner, agreement rate, and margin diagnostics."""
    valid = [a for a in answers if a is not None]
    counts = Counter(valid)

    if not counts:
        return VoteStats(
            winning_answer=None,
            vote_counts={},
            agreement_rate=0.0,
            vote_margin=0,
            valid_votes=0,
            total_candidates=len(answers),
            answer_entropy=0.0,
        )

    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    top_answer, top_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0

    total = len(valid)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

    return VoteStats(
        winning_answer=top_answer,
        vote_counts=dict(counts),
        agreement_rate=top_count / total,
        vote_margin=top_count - second_count,
        valid_votes=total,
        total_candidates=len(answers),
        answer_entropy=entropy,
    )


def weighted_vote(
    answers: Sequence[Optional[str]],
    weights: Sequence[float],
) -> tuple[Optional[str], Dict[str, float]]:
    """Return weighted winner and per-answer scores.

    Args:
        answers: Candidate normalized answers.
        weights: Candidate weights aligned with ``answers``.
    """
    if len(answers) != len(weights):
        raise ValueError("answers and weights must have the same length")

    scores: Dict[str, float] = {}
    for ans, w in zip(answers, weights):
        if ans is None:
            continue
        scores[ans] = scores.get(ans, 0.0) + float(w)

    if not scores:
        return None, {}

    top = max(scores.values())
    tied = {a for a, s in scores.items() if s == top}
    winner = next((a for a in answers if a in tied), None)
    return winner, scores
