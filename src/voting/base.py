"""Base class for voting / aggregation systems."""

from abc import ABC, abstractmethod
from typing import List


class BaseVotingSystem(ABC):
    """Abstract base class that all voting strategies must implement."""

    @abstractmethod
    def aggregate(self, predictions: List[str]) -> str:
        """Aggregate a list of predictions into a single consensus answer.

        Args:
            predictions: Raw model predictions (before normalisation).

        Returns:
            The consensus answer string.
        """
        ...
