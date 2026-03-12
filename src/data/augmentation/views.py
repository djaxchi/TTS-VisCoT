"""Multi-view generation: combines image and text augmentations.

The :func:`generate_views` function is the main entry point used by
the TTS methods to produce multiple views of a single example at inference
time.
"""

from typing import Any, Dict, List, Optional

from PIL import Image

from .image_aug import AugmentationConfig, ImageAugmentor
from .text_aug import TextAugmentor


def generate_views(
    image: Image.Image,
    question: str,
    options: Dict[str, str],
    config: Optional[AugmentationConfig] = None,
) -> List[Dict[str, Any]]:
    """Generate multiple (image, question) view pairs for test-time scaling.

    Image variants and text paraphrases are *paired* (not a full Cartesian
    product) to keep the number of views manageable.

    Args:
        image: Original PIL image.
        question: Original question string.
        options: Answer options dict (e.g. ``{"A": "...", "B": "..."}``) —
            kept unchanged across all views.
        config: Augmentation configuration; uses the default
            :class:`~src.data.augmentation.image_aug.AugmentationConfig` if
            ``None``.

    Returns:
        A list of view dicts, each containing:

        - ``"image"``: augmented :class:`PIL.Image.Image`
        - ``"question"``: paraphrased question string
        - ``"options"``: the original options dict (shared reference)
        - ``"variant_id"``: human-readable identifier for this view
    """
    if config is None:
        config = AugmentationConfig()

    image_aug = ImageAugmentor(seed=config.seed)
    text_aug = TextAugmentor(seed=config.seed)

    image_variants = image_aug.generate_image_variants(image, config.image_sets)
    text_variants = text_aug.generate_text_variants(question, config.text_paraphrases)

    views: List[Dict[str, Any]] = [
        {
            "image": image,
            "question": question,
            "options": options,
            "variant_id": "original",
        }
    ]

    max_extra = max(len(image_variants), len(text_variants)) - 1
    for i in range(1, max_extra + 1):
        img, img_name = image_variants[i % len(image_variants)]
        txt, txt_name = text_variants[i % len(text_variants)]
        views.append(
            {
                "image": img,
                "question": txt,
                "options": options,
                "variant_id": f"{img_name}_{txt_name}",
            }
        )

    return views
