"""VisualCoTModel — wraps the deepcs233/VisCoT-7b-336 HuggingFace checkpoint.

The implementation mirrors the 2-turn pipeline in
:mod:`experiments.run_viscot` but exposes it as a proper
:class:`~src.models.base.BaseVisualCoTModel` so that TTS methods can call
:meth:`generate` with ``n > 1`` to sample multiple chains.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from src.utils.logging import get_logger
from .base import BaseVisualCoTModel

logger = get_logger(__name__)

DEFAULT_MODEL_ID = "deepcs233/VisCoT-7b-336"
CONV_MODE = "llava_v1"
BBOX_SUFFIX = (
    "\nPlease provide the bounding box coordinate of the region "
    "this question is about."
)
_BBOX_RE = re.compile(
    r"\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?"
)


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_bbox(text: str) -> Optional[List[float]]:
    m = _BBOX_RE.search(text)
    return [float(m.group(i)) for i in range(1, 5)] if m else None


def _crop_region(image: Image.Image, coords: List[float]) -> Image.Image:
    """Square-crop a bounding-box region, matching VisCoT's model_cot_loader."""
    w, h = image.size
    x_min, y_min, x_max, y_max = coords
    if max(coords) <= 1.0:
        x_min, x_max, y_min, y_max = x_min * w, x_max * w, y_min * h, y_max * h
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half = max((x_max - x_min) / 2.0, (y_max - y_min) / 2.0, 112.0)
    cx = max(half, min(w - half, cx))
    cy = max(half, min(h - half, cy))
    region = (
        max(0, int(cx - half)),
        max(0, int(cy - half)),
        min(w, int(cx + half)),
        min(h, int(cy + half)),
    )
    return image.crop(region)


class VisualCoTModel(BaseVisualCoTModel):
    """Two-turn Visual Chain-of-Thought model (VisCoT-7b-336).

    Lazily loads the underlying LLaVA weights on the first call to
    :meth:`generate`.

    Args:
        model_path: HuggingFace model ID or local checkpoint directory.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_ID) -> None:
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._use_im_start_end: bool = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Download and initialise the model weights (idempotent)."""
        if self._model is not None:
            return

        try:
            from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
        except ImportError as exc:
            raise ImportError(
                "The 'llava' package from Visual-CoT is required.\n"
                "Install with: pip install git+https://github.com/deepcs233/Visual-CoT.git"
            ) from exc

        device = _get_device()
        logger.info(f"Loading VisCoT from '{self.model_path}' on {device} …")
        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        if "llava" not in model_name.lower():
            model_name = "llava_" + model_name

        map_device = "cpu" if device == "mps" else device
        tokenizer, model, image_processor, _ = load_pretrained_model(
            self.model_path, None, model_name,
            device_map=map_device, device=map_device,
        )
        if device == "mps":
            model = model.to(device)

        self._tokenizer = tokenizer
        self._model = model
        self._image_processor = image_processor
        self._use_im_start_end = model.config.mm_use_im_start_end
        self._device = device

    # ------------------------------------------------------------------
    # BaseVisualCoTModel interface
    # ------------------------------------------------------------------

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
        """Run *n* independent VisCoT chains for *image* and *query*.

        Args:
            image: Input PIL image.
            query: Question string.
            n: Number of chains to sample.
            temperature: Sampling temperature.
            max_new_tokens: Token budget for the final answer turn.

        Returns:
            List of *n* chain dicts containing ``"bbox_raw"``, ``"coords"``,
            and ``"answer"``.
        """
        self._load()
        return [self._run_chain(image, query, temperature, max_new_tokens) for _ in range(n)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_chain(
        self,
        image: Image.Image,
        query: str,
        temperature: float,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        def _image_tok() -> str:
            if self._use_im_start_end:
                return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            return DEFAULT_IMAGE_TOKEN

        def _gen(conv, images: List[Image.Image], max_tok: int) -> str:
            dtype = torch.float16 if self._device != "cpu" else torch.float32
            img_tensor = process_images(images, self._image_processor, self._model.config)
            if isinstance(img_tensor, list):
                img_tensor = torch.stack(img_tensor)
            img_tensor = img_tensor.to(dtype=dtype, device=self._device)
            input_ids = (
                tokenizer_image_token(
                    conv.get_prompt(), self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self._device)
            )
            stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            with torch.inference_mode():
                out_ids = self._model.generate(
                    input_ids,
                    images=img_tensor,
                    do_sample=(temperature > 0.0),
                    temperature=temperature,
                    max_new_tokens=max_tok,
                    use_cache=True,
                )
            decoded = self._tokenizer.batch_decode(
                out_ids[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0].strip()
            if decoded.endswith(stop):
                decoded = decoded[: -len(stop)].strip()
            return decoded

        # Turn 1: bbox
        t1_user = _image_tok() + "\n" + query.strip() + BBOX_SUFFIX
        conv1 = conv_templates[CONV_MODE].copy()
        conv1.append_message(conv1.roles[0], t1_user)
        conv1.append_message(conv1.roles[1], None)
        bbox_raw = _gen(conv1, [image], 64)

        coords = _parse_bbox(bbox_raw) or [0.0, 0.0, 1.0, 1.0]
        cropped = _crop_region(image, coords)

        # Turn 2: answer
        t2_inner = (
            "Please answer the question based on the original image "
            f"and local detail image.\n{query.strip()}"
        )
        t2_user = _image_tok() + "\n" + t2_inner
        conv2 = conv_templates[CONV_MODE].copy()
        conv2.append_message(conv2.roles[0], t1_user)
        conv2.append_message(conv2.roles[1], bbox_raw)
        conv2.append_message(conv2.roles[0], t2_user)
        conv2.append_message(conv2.roles[1], None)
        answer = _gen(conv2, [image, cropped], max_new_tokens)

        return {"bbox_raw": bbox_raw, "coords": coords, "answer": answer}
