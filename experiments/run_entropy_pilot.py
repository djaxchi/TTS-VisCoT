"""Entropy pilot: answer diversity + token-level generation entropy.

Runs 3 open-ended questions (one per task) through all 3 models.
Per model × question collects:
  - 10 independent answers at temperature=0.7  → answer entropy (free-text)
  - per-generation mean token entropy from logit distributions → internal uncertainty
  - wall-clock generation time

Usage:
    python experiments/run_entropy_pilot.py
    python experiments/run_entropy_pilot.py --model qwen   # single model
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets.viscot_benchmark import load_task
from src.eval.stochasticity import compute_entropy
from src.utils.logging import get_logger
from src.utils_normalize import normalize_open_ended_answer

logger = get_logger(__name__)

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "entropy_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Open-ended questions (MCQ options stripped)
# ---------------------------------------------------------------------------

OPEN_QUESTIONS = {
    "vqa": {
        "question_id": "test_Accounting_42",
        "question": (
            "Prices of zero-coupon bonds reveal the following pattern of forward rates "
            "(see the image). In addition to the zero-coupon bond, investors also may "
            "purchase a 3-year bond making annual payments of $60 with par value $1,000. "
            "Under the expectations hypothesis, what is the expected realized compound "
            "yield of the coupon bond? Give a specific percentage value."
        ),
        "gt": "6.66%",
        "source": "hard_bench",   # loaded via load_task
        "task_key": "vqa",
    },
    "ocr": {
        "question_id": "106",
        "question": 'What is the status of "Keep-on display"?',
        "gt": "off",
        "source": "hard_bench",
        "task_key": "ocr",
    },
    "counting": {
        "question_id": "131131002",
        "question": "How many cats are in the image?",
        "gt": "2",
        "source": "local",        # loaded directly from disk
        "image_path": "data/benchmark/images/vqa2/131131.jpg",
    },
}

N_SAMPLES   = 10
TEMPERATURE = 0.7
MODELS = {
    "qwen":     ("src.models.direct_vlm", "DirectVLMModel"),
    "grit":     ("src.models.grit",       "GRITModel"),
    "deepeyes": ("src.models.deepeyes_v2","DeepEyesV2Model"),
}

# ---------------------------------------------------------------------------
# Shared token-entropy helper
# All three models use Qwen2.5-VL weights — same processor / model interface.
# ---------------------------------------------------------------------------

def _generate_with_scores(
    model: Any,
    messages: List[Dict],
    temperature: float,
    max_new_tokens: int,
    extra_gen_kwargs: Optional[Dict] = None,
) -> Tuple[str, float]:
    """Single forward pass returning (decoded_text, mean_token_entropy_bits).

    Works for any model that exposes ._model and ._processor (all three models here).
    Aggregates token-level entropy across all generated tokens.
    """
    from qwen_vl_utils import process_vision_info

    text = model._processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = model._processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model._model.device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    if temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    if extra_gen_kwargs:
        gen_kwargs.update(extra_gen_kwargs)

    with torch.inference_mode():
        out = model._model.generate(**inputs, **gen_kwargs)

    # Decode (strip prompt tokens)
    prompt_len = inputs["input_ids"].shape[1]
    trimmed = out.sequences[:, prompt_len:]
    decoded = model._processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # Token-level entropy: out.scores is a tuple of len n_generated_tokens,
    # each element is (batch=1, vocab_size) raw logits.
    token_entropies: List[float] = []
    for raw_score in out.scores:
        probs = F.softmax(raw_score[0].float(), dim=-1)
        h = -(probs * torch.log2(probs + 1e-10)).sum().item()
        token_entropies.append(h)
    mean_tok_h = sum(token_entropies) / len(token_entropies) if token_entropies else 0.0

    del inputs, out, trimmed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return decoded, mean_tok_h


# ---------------------------------------------------------------------------
# Per-model runners
# Each returns a list of dicts with keys:
#   answer, mean_tok_entropy, gen_time_s
# ---------------------------------------------------------------------------

def _run_direct_vlm(model: Any, image: Any, question: str) -> List[Dict]:
    import base64, io as _io
    buf = _io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    image_uri = f"data:image;base64,{b64}"

    messages = [
        {"role": "system", "content": "You are a helpful visual question answering assistant. Answer concisely."},
        {"role": "user", "content": [
            {"type": "image", "image": image_uri},
            {"type": "text", "text": question},
        ]},
    ]

    results = []
    for i in range(N_SAMPLES):
        t0 = time.perf_counter()
        answer, tok_h = _generate_with_scores(model, messages, TEMPERATURE, max_new_tokens=128)
        elapsed = time.perf_counter() - t0
        print(f"    sample {i+1:2d}: answer={answer[:60]!r:60s}  tok_H={tok_h:.2f}b  t={elapsed:.1f}s")
        results.append({"answer": answer, "mean_tok_entropy": tok_h, "gen_time_s": elapsed})
    return results


def _run_grit(model: Any, image: Any, question: str) -> List[Dict]:
    import base64, io as _io
    buf = _io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    image_uri = f"data:image;base64,{b64}"

    messages = [
        {"role": "system", "content": model.system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image_uri},
            {"type": "text", "text": question},
        ]},
    ]

    results = []
    for i in range(N_SAMPLES):
        t0 = time.perf_counter()
        raw, tok_h = _generate_with_scores(
            model, messages, TEMPERATURE, max_new_tokens=512,
        )
        elapsed = time.perf_counter() - t0
        # Parse <answer> tag if present
        import re
        m = re.search(r"<answer>(.*?)(?:</answer>|$)", raw, re.DOTALL)
        answer = m.group(1).strip() if m else raw
        print(f"    sample {i+1:2d}: answer={answer[:60]!r:60s}  tok_H={tok_h:.2f}b  t={elapsed:.1f}s")
        results.append({"answer": answer, "raw": raw, "mean_tok_entropy": tok_h, "gen_time_s": elapsed})
    return results


def _run_deepeyes(model: Any, image: Any, question: str) -> List[Dict]:
    """Run DeepEyesV2 — multi-turn. Token entropy averaged across all turns."""
    import base64, io as _io, re, numpy as np
    from src.models.deepeyes_v2 import (
        USER_PROMPT_TEMPLATE, SYSTEM_PROMPT, RETURN_CODE_USER_PROMPT,
        RETURN_SEARCH_PROMPT, _parse_answer, _extract_code_block,
        _extract_tool_call, _fix_python_indentation, _execute_code,
    )
    import PIL

    pil_image = image.convert("RGB")
    buf = _io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    image_uri = f"data:image/png;base64,{b64}"
    formatted_query = USER_PROMPT_TEMPLATE.format(question=question)

    results = []
    for i in range(N_SAMPLES):
        exec_ns: Dict[str, Any] = {
            "np": np, "PIL": PIL, "image_1": pil_image, "image": pil_image,
            "math": __import__("math"), "collections": __import__("collections"),
        }
        messages: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": formatted_query},
            ]},
        ]
        all_tok_entropies: List[float] = []
        final_answer = ""
        t0 = time.perf_counter()

        for turn in range(model.max_turns):
            response, tok_h = _generate_with_scores(
                model, messages, TEMPERATURE, max_new_tokens=4096,
                extra_gen_kwargs={"repetition_penalty": 1.05},
            )
            all_tok_entropies.append(tok_h)
            print(f"    sample {i+1:2d} turn {turn+1}: tok_H={tok_h:.2f}b  snippet={response[:80]!r}")

            answer = _parse_answer(response)
            if answer is not None:
                final_answer = answer
                break

            tool_call = _extract_tool_call(response)
            if tool_call is not None:
                stub = RETURN_SEARCH_PROMPT
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": [{"type": "text", "text": stub}]})
                continue

            code = _extract_code_block(response)
            if code is not None:
                code = _fix_python_indentation(code)
                stdout, stderr, figures = _execute_code(code, exec_ns)
                image_placeholder = "Images:\n" + "<image>" * len(figures) if figures else ""
                tool_text = RETURN_CODE_USER_PROMPT.format(
                    stdout=stdout, stderr=stderr, image=image_placeholder
                ).strip()
                tool_content: List[Dict] = [{"type": "text", "text": tool_text}]
                for fig_bytes in figures:
                    fig_b64 = base64.b64encode(fig_bytes).decode()
                    tool_content.append({"type": "image", "image": f"data:image/png;base64,{fig_b64}"})
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": tool_content})
            else:
                final_answer = response
                break

        elapsed = time.perf_counter() - t0
        mean_tok_h = sum(all_tok_entropies) / len(all_tok_entropies) if all_tok_entropies else 0.0
        n_turns = len(all_tok_entropies)
        print(f"    sample {i+1:2d} DONE: answer={final_answer[:60]!r}  mean_tok_H={mean_tok_h:.2f}b  turns={n_turns}  t={elapsed:.1f}s")
        results.append({
            "answer": final_answer,
            "mean_tok_entropy": mean_tok_h,
            "n_turns": n_turns,
            "gen_time_s": elapsed,
        })
    return results


_MODEL_RUNNERS = {
    "qwen":     _run_direct_vlm,
    "grit":     _run_grit,
    "deepeyes": _run_deepeyes,
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_model(model_key: str) -> None:
    import importlib
    mod_path, cls_name = MODELS[model_key]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    runner = _MODEL_RUNNERS[model_key]

    print(f"\n{'='*70}")
    print(f"MODEL: {model_key.upper()}  ({cls_name})")
    print(f"{'='*70}")

    model = cls()
    model._load()

    all_rows = []
    for task, spec in OPEN_QUESTIONS.items():
        print(f"\n--- Task: {task}  qid={spec['question_id']} ---")
        print(f"Question : {spec['question'][:100]}...")
        print(f"GT       : {spec['gt']}")
        print(f"Running {N_SAMPLES} samples at temperature={TEMPERATURE}...")

        # Load image
        if spec.get("source") == "local":
            from PIL import Image as _PILImage
            img_path = Path(__file__).resolve().parents[1] / spec["image_path"]
            image = _PILImage.open(img_path).convert("RGB")
        else:
            examples = load_task(spec["task_key"], n=100)
            ex = next((e for e in examples if str(e["question_id"]) == str(spec["question_id"])), None)
            if ex is None:
                print(f"  WARNING: could not find qid={spec['question_id']}, skipping.")
                continue
            image = ex["image"]

        samples = runner(model, image, spec["question"])

        # Compute metrics
        answers = [s["answer"] for s in samples]
        norm_answers = [normalize_open_ended_answer(a) for a in answers]
        ans_entropy = compute_entropy(norm_answers)
        tok_entropies = [s["mean_tok_entropy"] for s in samples]
        gen_times = [s["gen_time_s"] for s in samples]
        mean_tok_h = sum(tok_entropies) / len(tok_entropies)
        mean_gen_t = sum(gen_times) / len(gen_times)

        row = {
            "model": model_key,
            "task": task,
            "question_id": spec["question_id"],
            "gt": spec["gt"],
            "answers": answers,
            "norm_answers": norm_answers,
            "answer_entropy_bits": ans_entropy,
            "mean_token_entropy_bits": mean_tok_h,
            "token_entropies_per_sample": tok_entropies,
            "gen_times_s": gen_times,
            "mean_gen_time_s": mean_gen_t,
        }
        all_rows.append(row)

        # Print task summary
        print(f"\n  >> Task summary ({task}):")
        print(f"     Answers        : {norm_answers}")
        print(f"     Answer entropy : {ans_entropy:.3f} bits  (max={math.log2(N_SAMPLES):.2f})")
        print(f"     Mean tok. H    : {mean_tok_h:.3f} bits")
        print(f"     Mean gen. time : {mean_gen_t:.1f}s")

    # Save
    out_path = OUT_DIR / f"entropy_{model_key}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved -> {out_path}")

    # Free GPU memory before next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize() -> None:
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Model':<12} {'Task':<10} {'Ans H (bits)':>14} {'Tok H (bits)':>14} {'Gen time (s)':>14}")
    print("-" * 68)
    for model_key in MODELS:
        path = OUT_DIR / f"entropy_{model_key}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                print(f"{r['model']:<12} {r['task']:<10} "
                      f"{r['answer_entropy_bits']:>14.3f} "
                      f"{r['mean_token_entropy_bits']:>14.3f} "
                      f"{r['mean_gen_time_s']:>14.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                        help="Run a single model (default: all sequentially)")
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else list(MODELS.keys())
    for mk in models_to_run:
        run_model(mk)

    summarize()


if __name__ == "__main__":
    main()
