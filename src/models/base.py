"""Abstract base class for all Visual CoT models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from PIL import Image


class BaseVisualCoTModel(ABC):
    """Interface that every VLM wrapper must implement.

    Subclasses expose a single :meth:`generate` method that accepts one or
    more images and a text prompt and returns a list of chain dicts.  A
    *chain dict* must contain at least:

    - ``"bbox_raw"`` (str)   — raw bounding-box prediction from turn 1
    - ``"coords"`` (list)    — parsed ``[x1, y1, x2, y2]`` coordinates
    - ``"answer"`` (str)     — final answer string from turn 2

    The list wrapper allows callers to request ``n`` independent samples in a
    single call (needed by the TTS sampling strategy).
    """

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        query: str,
        *,
        n: int = 1,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run inference and return *n* chain dicts.

        Args:
            image: Input image.
            query: Question / instruction string.
            n: Number of independent samples to draw.
            temperature: Sampling temperature (0 = greedy).
            max_new_tokens: Token budget for the final answer.
            **kwargs: Additional model-specific parameters.

        Returns:
            List of *n* chain dicts, each with keys
            ``"bbox_raw"``, ``"coords"``, and ``"answer"``.
        """
        ...

    def predict(
        self,
        image: Image.Image,
        query: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Convenience wrapper: return a single chain dict (greedy decoding).

        Args:
            image: Input image.
            query: Question string.
            temperature: Sampling temperature.
            max_new_tokens: Token budget.

        Returns:
            A single chain dict.
        """
        return self.generate(
            image, query, n=1, temperature=temperature, max_new_tokens=max_new_tokens
        )[0]
