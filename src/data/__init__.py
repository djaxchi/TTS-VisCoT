"""Data loading layer: datasets and augmentation strategies."""

from .datasets import get_dataset, register_dataset
from .augmentation import generate_views, AugmentationConfig

__all__ = ["get_dataset", "register_dataset", "generate_views", "AugmentationConfig"]
