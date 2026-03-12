"""Baseline single-view inference (no test-time scaling)."""

from typing import Any, Dict

from PIL import Image

from src.models.base import BaseVisualCoTModel


def run_baseline(
    model: BaseVisualCoTModel,
    image: Image.Image,
    query: str,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Run a single greedy inference pass.

    This is the simplest possible strategy: call the model once and return
    the raw chain dict.  No augmentation, no voting.

    Args:
        model: Any :class:`~src.models.base.BaseVisualCoTModel` instance.
        image: Input PIL image.
        query: Question string.
        temperature: Decoding temperature (0 = greedy).
        max_new_tokens: Token budget for the answer.

    Returns:
        A chain dict with at least ``"bbox_raw"``, ``"coords"``, and
        ``"answer"``.
    """
    return model.predict(image, query, temperature=temperature, max_new_tokens=max_new_tokens)
