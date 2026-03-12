"""Dataset registry and public API."""

from .base import BaseDataset
from .treebench import TreeBenchDataset, TreeBenchExample, BoundingBox

_DATASET_REGISTRY: dict[str, type[BaseDataset]] = {
    "treebench": TreeBenchDataset,
}


def register_dataset(name: str):
    """Decorator to register a dataset class by name."""

    def decorator(cls: type[BaseDataset]):
        _DATASET_REGISTRY[name] = cls
        return cls

    return decorator


def get_dataset(name: str) -> type[BaseDataset]:
    """Retrieve a registered dataset class by name.

    Args:
        name: Registered dataset name (e.g. ``"treebench"``).

    Returns:
        The dataset class (not an instance).

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(_DATASET_REGISTRY)}")
    return _DATASET_REGISTRY[name]


__all__ = [
    "BaseDataset",
    "TreeBenchDataset",
    "TreeBenchExample",
    "BoundingBox",
    "register_dataset",
    "get_dataset",
]
