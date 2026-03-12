"""Abstract base class for augmentation strategies."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAugmentation(ABC):
    """Interface shared by all augmentation implementations."""

    @abstractmethod
    def __call__(self, input: Any) -> Any:
        """Apply the augmentation and return the result."""
        ...
