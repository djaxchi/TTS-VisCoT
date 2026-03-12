"""Majority-vote aggregation for multiple model predictions."""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .normalize import normalize_answer


@dataclass
class VoteResult:
    """Result of voting across multiple predictions.

    Attributes:
        consensus_answer: The answer with the most votes.
        vote_counts: Raw vote counts per answer.
        agreement_rate: Fraction of votes cast for the consensus.
        vote_margin: Normalised gap between top-1 and top-2 vote counts.
        dispersion_score: Normalised entropy (0 = full consensus, 1 = uniform).
        confidence: Combined signal: agreement_rate × (1 − dispersion_score).
    """

    consensus_answer: str
    vote_counts: Dict[str, int]
    agreement_rate: float
    vote_margin: float
    dispersion_score: float
    confidence: float


def majority_vote(predictions: List[str], normalize: bool = True) -> VoteResult:
    """Perform majority voting on a list of model predictions.

    Args:
        predictions: Raw or pre-normalised model answers.
        normalize: If ``True``, run each prediction through
            :func:`~src.voting.normalize.normalize_answer` before voting.

    Returns:
        :class:`VoteResult` containing the consensus answer and statistics.
    """
    if not predictions:
        return VoteResult(
            consensus_answer="",
            vote_counts={},
            agreement_rate=0.0,
            vote_margin=0.0,
            dispersion_score=1.0,
            confidence=0.0,
        )

    if normalize:
        predictions = [normalize_answer(p) for p in predictions]

    vote_counts = Counter(predictions)
    total_votes = len(predictions)

    consensus_answer, max_votes = vote_counts.most_common(1)[0]

    agreement_rate = max_votes / total_votes

    if len(vote_counts) >= 2:
        top_two = vote_counts.most_common(2)
        vote_margin = (top_two[0][1] - top_two[1][1]) / total_votes
    elif len(vote_counts) == 1:
        vote_margin = 1.0
    else:
        vote_margin = 0.0

    if len(vote_counts) > 1:
        probs = np.array([c / total_votes for c in vote_counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(vote_counts))
        dispersion_score = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        dispersion_score = 0.0

    confidence = agreement_rate * (1 - dispersion_score)

    return VoteResult(
        consensus_answer=consensus_answer,
        vote_counts=dict(vote_counts),
        agreement_rate=agreement_rate,
        vote_margin=vote_margin,
        dispersion_score=dispersion_score,
        confidence=confidence,
    )
