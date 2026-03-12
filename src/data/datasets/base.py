"""Abstract base class for all datasets in TTS-VisCoT."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseDataset(ABC):
    """Minimal interface that every dataset implementation must satisfy.

    Concrete subclasses must implement :meth:`__len__` and
    :meth:`__getitem__`, and should accept the parameters below in their
    ``__init__``.

    Args:
        split: Dataset split identifier (e.g. ``"train"``, ``"test"``).
        max_samples: If set, only the first *max_samples* examples are
            exposed through :meth:`__len__` and :meth:`__getitem__`.
        cache_dir: Optional local directory for caching downloaded data.
    """

    def __init__(
        self,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = cache_dir

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of examples in this split."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the example at position *idx* as a plain dict."""
        ...
