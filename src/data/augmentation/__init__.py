"""Augmentation registry and public API."""

from .base import BaseAugmentation
from .image_aug import AugmentationConfig, AugmentationSet, ImageAugmentor
from .text_aug import TextAugmentor
from .views import generate_views

_AUGMENTATION_REGISTRY: dict[str, type[BaseAugmentation]] = {}


def register_augmentation(name: str):
    """Decorator to register an augmentation class by name."""

    def decorator(cls: type[BaseAugmentation]):
        _AUGMENTATION_REGISTRY[name] = cls
        return cls

    return decorator


def get_augmentation(name: str) -> type[BaseAugmentation]:
    """Retrieve a registered augmentation class by name."""
    if name not in _AUGMENTATION_REGISTRY:
        raise KeyError(
            f"Unknown augmentation '{name}'. Available: {list(_AUGMENTATION_REGISTRY)}"
        )
    return _AUGMENTATION_REGISTRY[name]


__all__ = [
    "BaseAugmentation",
    "AugmentationConfig",
    "AugmentationSet",
    "ImageAugmentor",
    "TextAugmentor",
    "generate_views",
    "register_augmentation",
    "get_augmentation",
]
