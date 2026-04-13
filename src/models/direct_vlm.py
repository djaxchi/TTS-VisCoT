"""DirectVLMModel — Qwen2.5-VL-3B-Instruct with single-turn direct answering.

No chain-of-thought, no tool use.  Used as a no-CoT baseline against
VisCoT and DeepEyesV2-RL.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from PIL import Image

from src.models.base import BaseVisualCoTModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

_SYSTEM_PROMPT = "You are a helpful visual question answering assistant. Answer concisely."


class DirectVLMModel(BaseVisualCoTModel):
    """Single-turn Qwen2.5-VL model that answers without any CoT.

    Args:
        model_id: HuggingFace model ID or local path.
        load_in_8bit: Whether to quantise weights to 8-bit.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        load_in_8bit: bool = True,
    ) -> None:
        self.model_id = model_id
        self.load_in_8bit = load_in_8bit
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        # RTX 5060 Ti (Ada Lovelace) has issues with bitsandbytes 8-bit — use
        # float16 directly.  ~6 GB VRAM vs ~3 GB for 8-bit, but stable.
        quant_config = None
        logger.info("Loading DirectVLM from '{}' (float16, no quantisation)…", self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quant_config,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,   # reduced from 1280 to limit visual tokens / VRAM
        )
        logger.info("DirectVLM loaded.")

    def generate(
        self,
        image: Image.Image,
        query: str,
        *,
        n: int = 1,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self._load()
        return [self._answer(image, query, temperature, max_new_tokens) for _ in range(n)]

    def _answer(
        self,
        image: Image.Image,
        query: str,
        temperature: float,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info
        import base64, io as _io

        buf = _io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_uri = f"data:image;base64,{b64}"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": query},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "use_cache": True}
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        answer = self._processor.batch_decode(
            out_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        import gc
        del inputs, out_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"bbox_raw": None, "coords": [], "answer": answer, "cot_steps": [], "tool_results": []}
