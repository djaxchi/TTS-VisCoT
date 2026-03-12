"""Image augmentation for test-time scaling.

Provides label-preserving image transformations via albumentations so that
multiple visual views of the same example can be generated at inference time.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import albumentations as A
import numpy as np
from PIL import Image


class AugmentationSet(Enum):
    """Pre-defined image augmentation sets.

    Attributes:
        SET_A: Safe photometric: brightness / contrast / gamma + light blur.
        SET_B: SET_A plus mild Gaussian blur and noise.
        SET_C: Mild sharpening / denoising.
    """

    SET_A = "brightness_contrast_jpeg"
    SET_B = "set_a_plus_blur_noise"
    SET_C = "sharpen_denoise"


@dataclass
class AugmentationConfig:
    """Top-level augmentation configuration.

    Attributes:
        image_sets: Which :class:`AugmentationSet` values to apply.
        text_paraphrases: Number of text paraphrases to generate per example.
        seed: Random seed for reproducibility (``None`` = unseeded).
    """

    image_sets: List[AugmentationSet] = field(
        default_factory=lambda: [AugmentationSet.SET_A, AugmentationSet.SET_B]
    )
    text_paraphrases: int = 3
    seed: Optional[int] = 42


class ImageAugmentor:
    """Apply label-preserving image augmentations for test-time scaling.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def get_augmentation_pipeline(self, aug_set: AugmentationSet) -> A.Compose:
        """Return an albumentations pipeline for the requested set.

        Args:
            aug_set: The augmentation set to build.

        Returns:
            An :class:`albumentations.Compose` pipeline.
        """
        if aug_set == AugmentationSet.SET_A:
            return A.Compose(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.8
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.Blur(blur_limit=3, p=0.3),
                ]
            )
        if aug_set == AugmentationSet.SET_B:
            return A.Compose(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.8
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.4),
                    A.GaussNoise(p=0.4),
                ]
            )
        if aug_set == AugmentationSet.SET_C:
            return A.Compose(
                [
                    A.Sharpen(alpha=(0.2, 0.4), lightness=(0.8, 1.2), p=0.6),
                    A.MedianBlur(blur_limit=3, p=0.3),
                ]
            )
        # Identity fallback
        return A.Compose([])

    def augment(self, image: Image.Image, aug_set: AugmentationSet) -> Image.Image:
        """Augment a single PIL image.

        Args:
            image: Input image.
            aug_set: Which augmentation pipeline to apply.

        Returns:
            Augmented PIL image of the same size.
        """
        img_np = np.array(image)
        pipeline = self.get_augmentation_pipeline(aug_set)
        augmented = pipeline(image=img_np)
        return Image.fromarray(augmented["image"])

    def generate_image_variants(
        self,
        image: Image.Image,
        sets: Optional[List[AugmentationSet]] = None,
    ) -> List[Tuple[Image.Image, str]]:
        """Generate multiple augmented copies of an image.

        Args:
            image: Original PIL image.
            sets: Augmentation sets to apply; defaults to SET_A and SET_B.

        Returns:
            List of ``(augmented_image, variant_name)`` tuples starting with
            the original as ``("original")``.
        """
        if sets is None:
            sets = [AugmentationSet.SET_A, AugmentationSet.SET_B]

        variants: List[Tuple[Image.Image, str]] = [(image, "original")]
        for aug_set in sets:
            variants.append((self.augment(image, aug_set), aug_set.value))
        return variants
