#!/usr/bin/env python3
"""
Run DeepEyesV2 (honglyhly/DeepEyesV2_7B_1031) on a single image + query.

Instrumented runner — shows:
  • Live token streaming per reasoning turn
  • Per-turn timing and token count
  • Tool call detection (code blocks) and execution output
  • Full summary table: turns × time × tokens × tok/s

Usage:
    python experiments/run_deepeyes_v2.py \\
        --image path/to/image.jpg \\
        --query "What colour is the car on the left?"
"""

from __future__ import annotations

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Force UTF-8 output on Windows so box-drawing chars and em-dashes render.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
import requests
from PIL import Image

# ── ANSI colours (fall back gracefully on Windows without VT) ────────────────
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED   = "\033[31m"
BLUE  = "\033[34m"

W = 72  # display width

def _bar(char: str = "─", width: int = W) -> str:
    return char * width

def _header(title: str) -> str:
    pad = (W - len(title) - 2) // 2
    return f"{BOLD}{'═' * pad} {title} {'═' * (W - pad - len(title) - 2)}{RESET}"


# ── Instrumented subclass ────────────────────────────────────────────────────

class InstrumentedDeepEyesV2:
    """Wraps DeepEyesV2Model with live streaming, timing, and token counting."""

    def __init__(self, model_id: str, max_turns: int, load_in_8bit: bool) -> None:
        from src.models.deepeyes_v2 import DeepEyesV2Model
        self._model = DeepEyesV2Model(
            model_id=model_id, max_turns=max_turns, load_in_8bit=load_in_8bit
        )
        self.turn_stats: List[Dict[str, Any]] = []

    def load(self) -> None:
        print(f"\n{CYAN}Loading model weights…{RESET}")
        t0 = time.perf_counter()
        self._model._load()
        print(f"{GREEN}Model ready in {time.perf_counter() - t0:.1f}s{RESET}\n")

    # ── monkey-patch _call_model to add streaming ──────────────────────────

    def _make_streaming_call_model(self):
        """Return a replacement for _call_model that streams tokens live."""
        outer = self

        def streaming_call_model(
            messages: List[Dict[str, Any]],
            temperature: float,
            max_new_tokens: int,
        ) -> str:
            from qwen_vl_utils import process_vision_info
            from transformers import TextIteratorStreamer

            m = outer._model

            text = m._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = m._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(m._model.device)

            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
            }
            if temperature > 0.0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
            else:
                generate_kwargs["do_sample"] = False

            streamer = TextIteratorStreamer(
                m._processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generate_kwargs["streamer"] = streamer

            # Run generation in a background thread so we can stream.
            def _gen():
                with torch.inference_mode():
                    m._model.generate(**inputs, **generate_kwargs)

            thread = Thread(target=_gen, daemon=True)

            turn_idx = len(outer.turn_stats) + 1
            print(f"\n{BOLD}{CYAN}{_bar('─')}")
            print(f"  TURN {turn_idx}  (generating…)")
            print(f"{_bar('─')}{RESET}")

            start = time.perf_counter()
            thread.start()

            token_texts: List[str] = []
            for chunk in streamer:
                print(chunk, end="", flush=True)
                token_texts.append(chunk)
            thread.join()

            elapsed = time.perf_counter() - start
            response = "".join(token_texts).strip()

            # Token count via tokenizer (accurate).
            n_tokens = len(
                m._processor.tokenizer.encode(response, add_special_tokens=False)
            )
            tok_per_s = n_tokens / elapsed if elapsed > 0 else 0.0

            outer.turn_stats.append({
                "turn": turn_idx,
                "time_s": elapsed,
                "tokens": n_tokens,
                "tok_per_s": tok_per_s,
            })

            print(f"\n{DIM}[turn {turn_idx} | {elapsed:.2f}s | {n_tokens} tok | {tok_per_s:.1f} tok/s]{RESET}")
            return response

        return streaming_call_model

    def predict(
        self,
        image: Image.Image,
        query: str,
        temperature: float,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        # Patch _call_model before running.
        self._model._call_model = self._make_streaming_call_model()
        return self._model.predict(image, query, temperature=temperature, max_new_tokens=max_new_tokens)


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(source: str) -> Image.Image:
    if source.startswith("http://") or source.startswith("https://"):
        headers = {"User-Agent": "TTS-VisCoT/1.0"}
        resp = requests.get(source, timeout=30, headers=headers)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(source).convert("RGB")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_deepeyes_v2(args: argparse.Namespace) -> dict:
    print(_header("DeepEyesV2 — Instrumented Inference"))
    print(f"  Model      : {CYAN}{args.model_id}{RESET}")
    print(f"  Max turns  : {args.max_turns}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens : {args.max_new_tokens} / turn")
    print(f"  8-bit quant: {not args.no_8bit}")
    print(_bar())

    runner = InstrumentedDeepEyesV2(
        model_id=args.model_id,
        max_turns=args.max_turns,
        load_in_8bit=not args.no_8bit,
    )
    runner.load()

    print(f"  Image: {args.image}")
    image = load_image(args.image)
    print(f"  Size : {image.size[0]}×{image.size[1]}\n")

    print(_header("Agentic Reasoning Loop"))
    print(f"  {BOLD}Query:{RESET} {args.query}\n")

    t_total_start = time.perf_counter()
    chain = runner.predict(
        image,
        args.query,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    t_total = time.perf_counter() - t_total_start

    cot_steps: list = chain.get("cot_steps", [])
    tool_results: list = chain.get("tool_results", [])

    # ── Tool-call summary ──────────────────────────────────────────────────
    print(f"\n{_header('Tool Call Log')}")
    tool_idx = 0
    for i, step in enumerate(cot_steps):
        has_code = "<code>" in step or "```python" in step
        has_answer = "<answer>" in step
        tag = f"{YELLOW}[CODE]{RESET}" if has_code else (f"{GREEN}[ANSWER]{RESET}" if has_answer else f"{DIM}[TEXT]{RESET}")
        print(f"  Turn {i+1:>2}  {tag}")
        if has_code and tool_idx < len(tool_results):
            tr = tool_results[tool_idx]
            print(f"           {DIM}↳ stdout: {tr[:120]}{'…' if len(tr)>120 else ''}{RESET}")
            tool_idx += 1

    # ── Per-turn stats table ───────────────────────────────────────────────
    print(f"\n{_header('Per-Turn Statistics')}")
    print(f"  {'Turn':>4}  {'Time (s)':>8}  {'Tokens':>7}  {'Tok/s':>7}  {'Cumul (s)':>9}")
    print(f"  {_bar('─', W-2)}")
    cumul = 0.0
    for s in runner.turn_stats:
        cumul += s["time_s"]
        print(
            f"  {s['turn']:>4}  {s['time_s']:>8.2f}  {s['tokens']:>7}  "
            f"{s['tok_per_s']:>7.1f}  {cumul:>9.2f}"
        )

    total_tokens = sum(s["tokens"] for s in runner.turn_stats)
    overall_tps  = total_tokens / t_total if t_total > 0 else 0.0

    print(f"  {_bar('─', W-2)}")
    print(f"  {'TOTAL':>4}  {t_total:>8.2f}  {total_tokens:>7}  {overall_tps:>7.1f}")

    # ── Final answer ───────────────────────────────────────────────────────
    print(f"\n{_header('Result')}")
    print(f"  {BOLD}Turns taken  :{RESET} {len(cot_steps)}")
    print(f"  {BOLD}Code calls   :{RESET} {len(tool_results)}")
    print(f"  {BOLD}Total time   :{RESET} {t_total:.2f}s")
    print(f"  {BOLD}Total tokens :{RESET} {total_tokens}")
    print(f"  {BOLD}Final answer :{RESET} {GREEN}{chain['answer']}{RESET}")
    print(_bar("═") + "\n")

    if args.save_output:
        import json
        out_path = Path(args.save_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query": args.query,
            "answer": chain["answer"],
            "cot_steps": cot_steps,
            "tool_results": tool_results,
            "turn_stats": runner.turn_stats,
            "total_time_s": t_total,
            "total_tokens": total_tokens,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"Output saved → {out_path}")

    return {"query": args.query, "answer": chain["answer"],
            "cot_steps": cot_steps, "tool_results": tool_results}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DeepEyesV2 with live token streaming and timing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", required=True,
                        help="Local image path or HTTP(S) URL.")
    parser.add_argument("--query", required=True,
                        help="Question to ask about the image.")
    parser.add_argument("--model-id", default="honglyhly/DeepEyesV2_7B_1031",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--no-8bit", action="store_true",
                        help="Disable 8-bit quantisation (needs ≥40 GB VRAM).")
    parser.add_argument("--save-output", metavar="PATH", default=None,
                        help="Save full JSON output to this path.")
    args = parser.parse_args()
    run_deepeyes_v2(args)


if __name__ == "__main__":
    main()
