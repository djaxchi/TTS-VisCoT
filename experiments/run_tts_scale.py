#!/usr/bin/env python3
"""TTS scale validation: 9-candidate voting on ALL hard_bench questions (100/100/92).

This script extends run_tts_hard_bench.py by adding per-candidate logprob
extraction.  After generating each candidate answer, a scoring forward pass
extracts the model's option logprobs at the answer position.  This data
enables post-hoc confidence-weighted voting without additional GPU time.

─── Differences from run_tts_hard_bench.py ────────────────────────────────

  1. Each candidate dict includes an ``option_logprobs`` field (dict mapping
     option letters A-J to log-probability floats), or {} if extraction fails.
  2. Output goes to a NEW directory (results/tts_scale/ or results/tts_scale_t0/)
     to avoid mixing with the 30-question runs.
  3. Resume is on by default — safe to restart after a crash.

─── Usage ────────────────────────────────────────────────────────────────────

  # Run both models, standard recipe (T=0.7 + augmentations)
  python experiments/run_tts_scale.py --recipe standard

  # Run one model, T=0 ablation
  python experiments/run_tts_scale.py --recipe t0 --model grit

  # Run single task
  python experiments/run_tts_scale.py --recipe t0 --model qwen3b --task vqa
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── UTF-8 fix ───────────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

TASKS = ["vqa", "ocr", "counting"]
N_QUESTIONS = {"vqa": 100, "ocr": 100, "counting": 92}

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qwen3b": {
        "label": "Qwen2.5-VL (3B)",
        "module": "src.models.direct_vlm",
        "class": "DirectVLMModel",
        "kwargs": {"model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "load_in_8bit": False},
        "max_pixels_override": 1024 * 28 * 28,
        "max_new_tokens": 256,
        "model_type": "direct_vlm",
    },
    "grit": {
        "label": "GRIT (3B)",
        "module": "src.models.grit",
        "class": "GRITModel",
        "kwargs": {"model_id": "yfan1997/GRIT-20-Qwen2.5-VL-3B", "load_in_8bit": False},
        "max_pixels_override": 1280 * 28 * 28,
        "max_new_tokens": 512,
        "model_type": "grit",
    },
}

# ── Candidate recipes ────────────────────────────────────────────────────────

CANDIDATES_STANDARD: List[Tuple[str, str, float]] = [
    ("original",            "original",             0.7),
    ("original",            "hardcoded_paraphrase",  0.7),
    ("original",            "original",             0.0),  # Study B greedy
    ("edge_enhance",        "original",             0.7),
    ("grayscale",           "original",             0.7),
    ("jpeg_recompress",     "original",             0.7),
    ("brightness_contrast", "original",             0.7),
    ("rotation_90",         "original",             0.7),
    ("edge_enhance",        "hardcoded_paraphrase",  0.7),
]
GREEDY_IDX_STANDARD = 2

CANDIDATES_T0: List[Tuple[str, str, float]] = [
    ("original",            "original",             0.0),  # Study B greedy
    ("original",            "hardcoded_paraphrase",  0.0),
    ("original",            "original",             0.7),  # single stochastic draw
    ("edge_enhance",        "original",             0.0),
    ("grayscale",           "original",             0.0),
    ("jpeg_recompress",     "original",             0.0),
    ("brightness_contrast", "original",             0.0),
    ("rotation_90",         "original",             0.0),
    ("edge_enhance",        "hardcoded_paraphrase",  0.0),
]
GREEDY_IDX_T0 = 0

CANDIDATE_AUG_SUBS_STANDARD: Dict[str, Dict[int, str]] = {
    "ocr": {7: "jpeg_recompress"},
}
CANDIDATE_AUG_SUBS_T0: Dict[str, Dict[int, str]] = {}

# ── Active recipe (set in main) ─────────────────────────────────────────────
CANDIDATES: List[Tuple[str, str, float]] = CANDIDATES_STANDARD
GREEDY_IDX: int = GREEDY_IDX_STANDARD
CANDIDATE_AUG_SUBS: Dict[str, Dict[int, str]] = CANDIDATE_AUG_SUBS_STANDARD
OUT_DIR: Path = _PROJECT_ROOT / "results" / "tts_scale"

# ── Answer normalisation ────────────────────────────────────────────────────

from src.utils_normalize import normalize_answer, normalize_open_ended_answer


def _normalise(raw: str, task: str) -> Optional[str]:
    if task == "ocr":
        return normalize_open_ended_answer(raw)
    return normalize_answer(raw)


def _is_correct(pred: Optional[str], gt: str, task: str, answers_all: List[str]) -> bool:
    if pred is None:
        return False
    if task == "ocr":
        all_norm = [normalize_open_ended_answer(a) for a in answers_all]
        return pred in all_norm
    return pred.upper() == gt.upper()


def _majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


# ── Image & text augmentation ───────────────────────────────────────────────

def _build_image_variants(image: Any) -> Dict[str, Any]:
    from src.augmentation.image import ImageVariationConfig, generate_image_variant_specs
    config = ImageVariationConfig(
        preset="strong",
        enable_brightness_contrast=True,
        enable_jpeg_recompress=True,
        enable_grayscale=True,
        enable_edge_enhance=True,
        enable_binary_bw=False,
        enable_rotation=True,
        rotation_degrees=(90,),
    )
    specs = generate_image_variant_specs(image, config=config)
    return {k: v["image"] for k, v in specs.items()}


def _build_text_variants(question: str, choices: Dict[str, str]) -> Dict[str, str]:
    from src.augmentation.text import generate_prompt_variants
    variants = generate_prompt_variants(question=question, choices=choices, mode="rule")
    return {
        "original": variants["original"]["prompt"],
        "hardcoded_paraphrase": variants["hardcoded_paraphrase"]["prompt"],
    }


# ── Logprob extraction ──────────────────────────────────────────────────────

def _extract_logprobs(
    model_obj: Any,
    image: Any,
    prompt: str,
    raw_answer: str,
    model_type: str,
) -> Dict[str, float]:
    """Extract option-letter logprobs via a scoring forward pass.

    For direct VLMs: logits at the first generated token position.
    For GRIT: logits at the position after the <answer> tag in the CoT output.

    Returns dict mapping option letters (A-J) to log-probabilities, or {} on failure.
    """
    from src.eval.tts_eval import _extract_option_stats_at_prefix

    assistant_prefix = None
    system_prompt = None

    if model_type == "grit" and "<answer>" in raw_answer:
        # Use everything up to and including <answer> as the assistant prefix
        idx = raw_answer.index("<answer>") + len("<answer>")
        assistant_prefix = raw_answer[:idx]
    elif model_type == "direct_vlm":
        # DirectVLM: system prompt used during generation
        from src.models.direct_vlm import _SYSTEM_PROMPT
        system_prompt = _SYSTEM_PROMPT

    try:
        stats = _extract_option_stats_at_prefix(
            model_obj,
            image,
            prompt,
            topk=10,
            assistant_prefix=assistant_prefix,
            system_prompt=system_prompt,
        )
        # Extract the logprobs dict from the nested structure
        logprobs_list = stats.get("option_logprobs", [])
        if logprobs_list:
            return logprobs_list[0].get("logprobs", {})
    except Exception as e:
        logger.warning("Logprob extraction failed: {}", e)

    return {}


# ── Checkpoint / resume ─────────────────────────────────────────────────────

def _cache_path(model_key: str) -> Path:
    return OUT_DIR / f"{model_key}_results.jsonl"


def load_done_ids(model_key: str, task: str) -> set:
    path = _cache_path(model_key)
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("task") == task:
                    done.add(str(row["question_id"]))
            except Exception:
                pass
    return done


def _append_result(row: Dict[str, Any], model_key: str) -> None:
    with _cache_path(model_key).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Single-question inference ───────────────────────────────────────────────

def _run_question(
    model_obj: Any,
    image: Any,
    image_variants: Dict[str, Any],
    text_variants: Dict[str, str],
    task: str,
    model_key: str,
) -> tuple[List[Dict[str, Any]], bool]:
    """Run all 9 candidates + logprob scoring for one question.

    Returns (candidate_results, cuda_errored).
    """
    cfg = MODEL_CONFIGS[model_key]
    max_new_tokens = cfg["max_new_tokens"]
    model_type = cfg["model_type"]

    results = []
    cuda_errored = False
    task_subs = CANDIDATE_AUG_SUBS.get(task, {})

    for i, (img_aug, text_var, temp) in enumerate(CANDIDATES):
        img_aug = task_subs.get(i, img_aug)
        img = image_variants.get(img_aug, image_variants["original"])
        prompt = text_variants.get(text_var, text_variants["original"])

        # ── Generate ────────────────────────────────────────────────────
        t0 = time.time()
        try:
            outs = model_obj.generate(
                img, prompt, n=1,
                temperature=temp,
                max_new_tokens=max_new_tokens,
            )
            raw = outs[0]["answer"] if outs else ""
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"\n  [CUDA error on candidate {i} ({img_aug}+{text_var})]", flush=True)
                cuda_errored = True
                break
            raw = f"ERROR: {e}"
        gen_time = time.time() - t0

        norm = _normalise(raw, task)

        # ── Logprob scoring (MCQ tasks only) ────────────────────────────
        option_logprobs: Dict[str, float] = {}
        if task in ("vqa", "counting") and not cuda_errored:
            try:
                option_logprobs = _extract_logprobs(
                    model_obj, img, prompt, raw, model_type
                )
            except Exception as e:
                if "CUDA" in str(e) or "cuda" in str(e):
                    print(f"\n  [CUDA error during logprob extraction, candidate {i}]", flush=True)
                    cuda_errored = True
                logger.warning("Logprob extraction error candidate {}: {}", i, e)

        results.append({
            "candidate_idx": i,
            "image_aug": img_aug,
            "text_variant": text_var,
            "temperature": temp,
            "raw_answer": raw,
            "answer": norm,
            "time_s": round(gen_time, 2),
            "option_logprobs": option_logprobs,
        })

        if cuda_errored:
            break

    return results, cuda_errored


# ── Model load / unload ─────────────────────────────────────────────────────

def _load_model(model_key: str) -> Any:
    import importlib
    from transformers import AutoProcessor
    cfg = MODEL_CONFIGS[model_key]
    mod = importlib.import_module(cfg["module"])
    cls = getattr(mod, cfg["class"])
    model = cls(**cfg["kwargs"])
    model._load()
    if "max_pixels_override" in cfg and hasattr(model, "_processor"):
        mp = cfg["max_pixels_override"]
        print(f"  [config] reloading processor with max_pixels={mp} ({mp//(28*28)} tiles)", flush=True)
        model._processor = AutoProcessor.from_pretrained(
            cfg["kwargs"]["model_id"],
            min_pixels=256 * 28 * 28,
            max_pixels=mp,
        )
    return model


def _unload_model(model: Any) -> None:
    import torch
    try:
        model._model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ── Per-model runner ────────────────────────────────────────────────────────

def run_model(model_key: str, tasks: List[str]) -> None:
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{'='*70}", flush=True)
    print(f"  {cfg['label']}", flush=True)
    print(f"{'='*70}", flush=True)

    from src.data.datasets.viscot_benchmark import load_task as _load_task

    print("\nPreloading images into CPU memory...", flush=True)
    all_questions: Dict[str, List[Dict[str, Any]]] = {}
    all_images: Dict[Tuple[str, str], Any] = {}

    for task in tasks:
        examples = _load_task(task)
        all_questions[task] = examples
        for ex in examples:
            all_images[(task, str(ex["question_id"]))] = ex["image"]
        print(f"  {task}: {len(examples)} questions", flush=True)

    print(f"\nLoading {cfg['label']}...", flush=True)
    t_load = time.time()
    model_obj = _load_model(model_key)
    print(f"  Model loaded in {time.time() - t_load:.0f}s", flush=True)

    try:
        for task in tasks:
            examples = all_questions[task]
            done_ids = load_done_ids(model_key, task)
            todo = [ex for ex in examples if str(ex["question_id"]) not in done_ids]

            print(f"\n  {task.upper()}: {len(done_ids)} done, {len(todo)} to go", flush=True)
            if not todo:
                print(f"  All done — skipping.", flush=True)
                continue

            t_task = time.time()
            for q_idx, ex in enumerate(todo):
                qid = str(ex["question_id"])
                question = ex["question"]
                gt = ex.get("answer", "")
                answers_all = ex.get("answers_all", [gt])
                image = all_images[(task, qid)]

                # Parse choices (MCQ)
                choices: Dict[str, str] = {}
                if task in ("vqa", "counting"):
                    import re as _re
                    for m in _re.finditer(
                        r"\b([A-J])[.:)]\s*(.+?)(?=\s*[,\n]?\s*[A-J][.:)]|[,\n]|$)",
                        question,
                    ):
                        choices[m.group(1).upper()] = m.group(2).strip().rstrip(",")

                image_variants = _build_image_variants(image)
                text_variants = _build_text_variants(question, choices)

                # Progress
                elapsed = time.time() - t_task
                if q_idx > 0:
                    eta = elapsed / q_idx * (len(todo) - q_idx)
                    eta_str = f"ETA {eta/60:.0f}min"
                else:
                    eta_str = "ETA ---"
                print(
                    f"\r  {task} [{q_idx+1}/{len(todo)}] {elapsed/60:.0f}min {eta_str}  qid={qid}      ",
                    end="", flush=True,
                )

                # Run candidates + logprobs
                candidates, cuda_errored = _run_question(
                    model_obj, image, image_variants, text_variants, task, model_key
                )

                all_answers = [c["answer"] for c in candidates]
                greedy = all_answers[GREEDY_IDX] if len(all_answers) > GREEDY_IDX else None

                row = {
                    "model": model_key,
                    "task": task,
                    "question_id": qid,
                    "question": question,
                    "gt_answer": gt,
                    "answers_all": answers_all,
                    "candidates": candidates,
                    "cuda_errored": cuda_errored,
                    "vote_9": _majority_vote(all_answers[:9]),
                    "greedy": greedy,
                    "correct_9": _is_correct(_majority_vote(all_answers[:9]), gt, task, answers_all),
                    "correct_greedy": _is_correct(greedy, gt, task, answers_all),
                    "correct_any": any(_is_correct(c["answer"], gt, task, answers_all) for c in candidates),
                }
                _append_result(row, model_key)
                done_ids.add(qid)

                if cuda_errored:
                    raise RuntimeError(
                        f"CUDA error on {model_key}/{task}/qid={qid}. "
                        "Partial row saved. Restart to continue."
                    )

            print(flush=True)
            _print_task_summary(model_key, task)

    finally:
        print(f"\nUnloading {cfg['label']}...", flush=True)
        _unload_model(model_obj)


# ── Summary ─────────────────────────────────────────────────────────────────

def _print_task_summary(model_key: str, task: str) -> None:
    path = _cache_path(model_key)
    if not path.exists():
        return
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("task") == task and not r.get("cuda_errored", False):
                    rows.append(r)
            except Exception:
                pass
    if not rows:
        return
    n = len(rows)
    g  = sum(r["correct_greedy"] for r in rows) / n * 100
    a9 = sum(r["correct_9"]      for r in rows) / n * 100
    o9 = sum(r["correct_any"]    for r in rows) / n * 100
    print(f"\n  {task.upper()} (n={n}): greedy={g:.1f}%  @9={a9:.1f}% (gain={a9-g:+.1f}pp)  oracle@9={o9:.1f}%", flush=True)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TTS scale validation (100 questions/task)")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), default=None)
    parser.add_argument("--task", choices=TASKS, default=None)
    parser.add_argument("--recipe", choices=["standard", "t0"], default="standard")
    args = parser.parse_args()

    global CANDIDATES, GREEDY_IDX, CANDIDATE_AUG_SUBS, OUT_DIR
    if args.recipe == "t0":
        CANDIDATES = CANDIDATES_T0
        GREEDY_IDX = GREEDY_IDX_T0
        CANDIDATE_AUG_SUBS = CANDIDATE_AUG_SUBS_T0
        OUT_DIR = _PROJECT_ROOT / "results" / "tts_scale_t0"
    else:
        CANDIDATES = CANDIDATES_STANDARD
        GREEDY_IDX = GREEDY_IDX_STANDARD
        CANDIDATE_AUG_SUBS = CANDIDATE_AUG_SUBS_STANDARD
        OUT_DIR = _PROJECT_ROOT / "results" / "tts_scale"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else list(MODEL_CONFIGS)
    tasks = [args.task] if args.task else TASKS

    total = sum(N_QUESTIONS[t] for t in tasks) * len(models) * len(CANDIDATES)
    print(f"\n  Recipe: {args.recipe}", flush=True)
    print(f"  Models: {models}", flush=True)
    print(f"  Tasks:  {tasks}", flush=True)
    print(f"  Total inferences: ~{total:,} (+ logprob scoring passes)", flush=True)
    print(f"  Output: {OUT_DIR}", flush=True)

    t0 = time.time()
    for model_key in models:
        run_model(model_key, tasks)

    print(f"\n  Total wall time: {(time.time()-t0)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
