"""GRITModel — GRIT-20-Qwen2.5-VL-3B with grounded reasoning CoT.

GRIT (Grounded Reasoning with Images and Text) prompts the model to:
  1. Think in <think>...</think> while outputting bbox coordinates as JSON.
  2. Re-reason in <rethink>...</rethink>.
  3. Give the final answer after <answer>...</answer>.

Paper: https://arxiv.org/abs/2505.15879
HF checkpoint: yfan1997/GRIT-20-Qwen2.5-VL-3B
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from src.models.base import BaseVisualCoTModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_ID = "yfan1997/GRIT-20-Qwen2.5-VL-3B"

_SYSTEM_PROMPT = (
    "First, think between <think> and </think> while output necessary coordinates "
    "needed to answer the question in JSON with key 'bbox_2d'. "
    "Then, based on the thinking contents and coordinates, rethink between "
    "<rethink></rethink> and then answer the question after <answer>."
)

# Matches <answer>TEXT</answer> or <answer>TEXT (end of string)
_ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_RETHINK_RE = re.compile(r"<rethink>(.*?)</rethink>", re.DOTALL)


def _parse_grit_answer(text: str) -> Optional[str]:
    """Extract the content of the first <answer>...</answer> tag, or None.

    Args:
        text: Raw model output string.

    Returns:
        Stripped answer string, or ``None`` if no tag found.
    """
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


class GRITModel(BaseVisualCoTModel):
    """GRIT grounded-reasoning model (Qwen2.5-VL-3B backbone).

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
        self.system_prompt = _SYSTEM_PROMPT
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        # Use bfloat16: more numerically stable than float16 on H100 (avoids
        # NaN/inf in sampling). RTX 5060 Ti also supports bfloat16.
        import torch as _torch
        quant_config = None
        logger.info("Loading GRITModel from '{}' (bfloat16, no quantisation)…", self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=_torch.bfloat16,
            device_map="auto",
            quantization_config=quant_config,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        logger.info("GRITModel loaded.")

    def generate(
        self,
        image: Image.Image,
        query: str,
        *,
        n: int = 1,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
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
        import base64
        import io as _io

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

        from transformers import GenerationConfig

        # Build an explicit GenerationConfig so transformers doesn't merge/override
        # with the model's own generation_config.json (which has do_sample=False).
        # Passing temperature as a bare kwarg triggers a "not valid and may be ignored"
        # warning in newer transformers when the resolved config has do_sample=False.
        gen_config = GenerationConfig(
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            max_new_tokens=max_new_tokens,
        )

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, generation_config=gen_config, use_cache=True)

        prompt_len = inputs["input_ids"].shape[1]
        # Decode without skipping special tokens so that <answer>...</answer>
        # tags (registered as special tokens) are preserved for parsing.
        # We manually strip only chat-template EOS markers afterwards.
        raw_output = self._processor.batch_decode(
            out_ids[:, prompt_len:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        # Remove common EOS / padding special tokens that are not answer tags.
        for tok in ("<|im_end|>", "<|endoftext|>", "<|end|>", "<pad>", "<eos>"):
            raw_output = raw_output.replace(tok, "")

        del inputs, out_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        answer = _parse_grit_answer(raw_output) or raw_output

        # Collect CoT steps (think + rethink) for token counting.
        cot_steps: List[str] = []
        m_think = _THINK_RE.search(raw_output)
        if m_think:
            cot_steps.append(m_think.group(1).strip())
        m_rethink = _RETHINK_RE.search(raw_output)
        if m_rethink:
            cot_steps.append(m_rethink.group(1).strip())

        return {
            "bbox_raw": None,
            "coords": [],
            "answer": answer,
            "raw_output": raw_output,  # full generated text including <think>/<answer> tags
            "cot_steps": cot_steps,
            "tool_results": [],
        }
