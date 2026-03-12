"""Voting / aggregation systems for TTS-VisCoT."""

from .base import BaseVotingSystem
from .majority import majority_vote, VoteResult
from .bbox_consensus import (
    BoundingBoxPrediction,
    compute_iou,
    match_boxes,
    consensus_boxes,
)
from .normalize import normalize_answer

_VOTING_REGISTRY: dict[str, type[BaseVotingSystem]] = {}


def register_voting(name: str):
    """Decorator to register a voting system by name."""

    def decorator(cls: type[BaseVotingSystem]):
        _VOTING_REGISTRY[name] = cls
        return cls

    return decorator


def get_voting_system(name: str) -> type[BaseVotingSystem]:
    """Retrieve a registered voting system class by name."""
    if name not in _VOTING_REGISTRY:
        raise KeyError(f"Unknown voting system '{name}'. Available: {list(_VOTING_REGISTRY)}")
    return _VOTING_REGISTRY[name]


__all__ = [
    "BaseVotingSystem",
    "VoteResult",
    "majority_vote",
    "BoundingBoxPrediction",
    "compute_iou",
    "match_boxes",
    "consensus_boxes",
    "normalize_answer",
    "register_voting",
    "get_voting_system",
]
