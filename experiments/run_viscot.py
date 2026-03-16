#!/usr/bin/env python3
"""
Run deepcs233/VisCoT-7b-336 on a single image + query.

VisCoT uses a 2-turn chain-of-thought pipeline:
  Turn 1  →  model predicts a bounding box for the relevant image region
  Turn 2  →  model answers using the original image + the cropped region

ALL intermediate outputs are printed: the bbox prediction, the crop
coordinates, and the final answer.

Requirements (install once):
    pip install git+https://github.com/deepcs233/Visual-CoT.git

Usage:
    python experiments/run_viscot.py \\
        --image path/to/image.jpg \\
        --query "What colour is the car on the left?"

    # Use a URL as image source
    python experiments/run_viscot.py \\
        --image "https://example.com/photo.jpg" \\
        --query "Describe the scene."

    # Use a local checkpoint
    python experiments/run_viscot.py \\
        --model-path ./checkpoints/VisCoT-7b-336 \\
        --image photo.jpg \\
        --query "What is the person doing?"
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import TextStreamer

# ---------------------------------------------------------------------------
# Guard: llava package from the Visual-CoT repo
# ---------------------------------------------------------------------------
try:
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
except ImportError:
    print(
        "\n[ERROR] The 'llava' package from Visual-CoT is required.\n"
        "Install it with:\n\n"
        "    pip install git+https://github.com/deepcs233/Visual-CoT.git\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "deepcs233/VisCoT-7b-336"
CONV_MODE = "llava_v1"


def _get_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Suffix appended to the first-turn query to elicit a bounding box.
BBOX_SUFFIX = (
    "\nPlease provide the bounding box coordinate of the region "
    "this question is about."
)

DIVIDER = "=" * 64


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(source: str) -> Image.Image:
    """Load a PIL Image from a local path or an HTTP(S) URL."""
    if source.startswith("http://") or source.startswith("https://"):
        headers = {"User-Agent": "TTS-VisCoT/1.0 (research project; python-requests)"}
        response = requests.get(source, timeout=30, headers=headers)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(source).convert("RGB")


def crop_region(image: Image.Image, coords: list[float]) -> Image.Image:
    """
    Square-crop the bounding box from *image* the same way VisCoT does
    in ``model_cot_loader.py``.

    coords are normalised [0, 1] floats [x_min, y_min, x_max, y_max].
    """
    w, h = image.size
    x_min, y_min, x_max, y_max = coords

    # Denormalise if the values look like ratios
    if max(coords) <= 1.0:
        x_min, x_max = x_min * w, x_max * w
        y_min, y_max = y_min * h, y_max * h

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half = max((x_max - x_min) / 2.0, (y_max - y_min) / 2.0, 112.0)

    # Clamp centre so the square fits inside the image
    cx = max(half, min(w - half, cx))
    cy = max(half, min(h - half, cy))

    region = (
        max(0, int(cx - half)),
        max(0, int(cy - half)),
        min(w, int(cx + half)),
        min(h, int(cy + half)),
    )
    return image.crop(region)


def annotate_image(image: Image.Image, coords: list[float]) -> Image.Image:
    """Return a copy of *image* with the bounding box drawn on it."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = image.size
    x_min, y_min, x_max, y_max = coords
    if max(coords) <= 1.0:
        x_min, x_max = x_min * w, x_max * w
        y_min, y_max = y_min * h, y_max * h
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    return out


# ---------------------------------------------------------------------------
# Bounding-box parsing
# ---------------------------------------------------------------------------

_BBOX_RE = re.compile(
    r"\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?"
)


def parse_bbox(text: str) -> list[float] | None:
    """Extract [x1, y1, x2, y2] from text like '[0.12, 0.34, 0.56, 0.78]'."""
    m = _BBOX_RE.search(text)
    if m:
        return [float(m.group(i)) for i in range(1, 5)]
    return None


# ---------------------------------------------------------------------------
# Single generation turn
# ---------------------------------------------------------------------------

def generate(
    model,
    tokenizer,
    image_processor,
    conv,
    images: list[Image.Image],
    temperature: float,
    max_new_tokens: int,
    stream: bool = False,
) -> str:
    """Run one generation step and return the decoded output string."""
    prompt = conv.get_prompt()

    device = _get_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    # Build image tensor — process_images handles both single and multiple images
    image_tensor = process_images(images, image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor)
    if image_tensor.ndim == 5:
        image_tensor = image_tensor.squeeze(0)
    image_tensor = image_tensor.to(dtype=dtype, device=device)

    input_ids = (
        tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(device)
    )

    stop_str = (
        conv.sep
        if conv.sep_style != SeparatorStyle.TWO
        else conv.sep2
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            streamer=streamer,
        )

    n_input = input_ids.shape[1]
    diff = (input_ids != output_ids[:, :n_input]).sum().item()
    if diff > 0:
        print(f"  [!] {diff} output token(s) differ from input tokens (normal for LLaVA)")

    decoded = tokenizer.batch_decode(
        output_ids[:, n_input:], skip_special_tokens=True
    )[0].strip()
    if decoded.endswith(stop_str):
        decoded = decoded[: -len(stop_str)].strip()
    return decoded


# ---------------------------------------------------------------------------
# VisCoT 2-turn pipeline
# ---------------------------------------------------------------------------

def build_user_turn(query: str, use_im_start_end: bool, suffix: str = "") -> str:
    """Wrap a user query with the correct image token(s)."""
    image_tok = (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if use_im_start_end
        else DEFAULT_IMAGE_TOKEN
    )
    return image_tok + "\n" + query.strip() + suffix


def run_viscot(args: argparse.Namespace) -> dict:
    device = _get_device()
    print(f"\nLoading model from: {args.model_path}")
    print(f"Using device: {device}")
    print("(First run will download ~14 GB of weights — this takes a while.)\n")

    disable_torch_init()
    # Suppress the noisy "max_new_tokens and max_length both set" warning
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    model_name = get_model_name_from_path(args.model_path)
    # VisCoT weights use the LLaVA architecture; the builder routes on the name
    # containing "llava", so we ensure that here.
    if "llava" not in model_name.lower():
        model_name = "llava_" + model_name

    # MPS (Apple Silicon) doesn't support device_map="auto"; use "cpu" for
    # from_pretrained then move the full model to MPS afterwards.
    builder_device_map = "cpu" if device == "mps" else device
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, model_name,
        device_map=builder_device_map,
        device=builder_device_map,
    )
    if device == "mps":
        model = model.to(device)
    use_im_start_end: bool = model.config.mm_use_im_start_end

    # ── Load image ────────────────────────────────────────────────────────────
    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    print(f"Image size: {image.size[0]}×{image.size[1]}")

    # ── Turn 1 — bounding box prediction ─────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("TURN 1 — Bounding Box Prediction")
    print(DIVIDER)

    t1_user = build_user_turn(args.query, use_im_start_end, suffix=BBOX_SUFFIX)
    conv1 = conv_templates[CONV_MODE].copy()
    conv1.append_message(conv1.roles[0], t1_user)
    conv1.append_message(conv1.roles[1], None)

    print(f"User:  {args.query}{BBOX_SUFFIX}")
    if args.stream:
        print("Model: ", end="", flush=True)
    bbox_output = generate(
        model, tokenizer, image_processor, conv1,
        images=[image], temperature=args.temperature, max_new_tokens=64,
        stream=args.stream,
    )
    if not args.stream:
        print(f"Model: {bbox_output}")

    coords = parse_bbox(bbox_output)
    if coords:
        print(f"Parsed bbox: x_min={coords[0]:.3f}  y_min={coords[1]:.3f}  "
              f"x_max={coords[2]:.3f}  y_max={coords[3]:.3f}")
    else:
        print("WARNING: could not parse a bounding box from the output. "
              "Using the full image for Turn 2.")
        coords = [0.0, 0.0, 1.0, 1.0]

    # Always show the annotated image with bounding box
    ann = annotate_image(image, coords)
    ann.show(title="VisCoT — Bounding Box Prediction")
    if args.save_annotated:
        out_path = Path(args.save_annotated)
        ann.save(out_path)
        print(f"Annotated image saved to: {out_path}")

    # ── Turn 2 — answer with original + cropped region ────────────────────────
    print(f"\n{DIVIDER}")
    print("TURN 2 — Final Answer (original image + cropped region)")
    print(DIVIDER)

    cropped = crop_region(image, coords)
    print(f"Cropped region size: {cropped.size[0]}×{cropped.size[1]}")

    # Turn-2 user message: second <image> token refers to the cropped region
    t2_inner = (
        "Please answer the question based on the original image "
        f"and local detail image.\n{args.query.strip()}"
    )
    t2_user = build_user_turn(t2_inner, use_im_start_end)

    # Rebuild conversation with both turns
    conv2 = conv_templates[CONV_MODE].copy()
    conv2.append_message(conv2.roles[0], t1_user)
    conv2.append_message(conv2.roles[1], bbox_output)
    conv2.append_message(conv2.roles[0], t2_user)
    conv2.append_message(conv2.roles[1], None)

    print(f"User:  Please answer the question based on the original image "
          f"and local detail image.\n       {args.query}")
    if args.stream:
        print("Model: ", end="", flush=True)
    final_answer = generate(
        model, tokenizer, image_processor, conv2,
        images=[image, cropped], temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        stream=args.stream,
    )
    if not args.stream:
        print(f"Model: {final_answer}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("SUMMARY")
    print(DIVIDER)
    print(f"Query        : {args.query}")
    print(f"BBox output  : {bbox_output}")
    print(f"Coords       : {coords}")
    print(f"Final answer : {final_answer}")
    print(DIVIDER + "\n")

    return {
        "query": args.query,
        "bbox_raw": bbox_output,
        "coords": coords,
        "final_answer": final_answer,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run VisCoT-7b-336 on a single image + query.\n"
            "Prints ALL outputs: bounding box (Turn 1) + final answer (Turn 2)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to a local image file or an HTTP(S) URL.",
    )
    parser.add_argument(
        "--query", required=True,
        help="Question to ask about the image.",
    )
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID or local path. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature (0 = greedy). Default: 0.2",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max tokens for the final answer. Default: 512",
    )
    parser.add_argument(
        "--save-annotated", metavar="PATH", default=None,
        help="If set, save a copy of the input image with the predicted "
             "bounding box drawn on it to this path.",
    )
    parser.add_argument(
        "--stream", action="store_true", default=False,
        help="Stream tokens to stdout in real time as they are generated.",
    )
    args = parser.parse_args()

    run_viscot(args)


if __name__ == "__main__":
    main()
