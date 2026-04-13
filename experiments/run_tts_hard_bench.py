#!/usr/bin/env python3
"""TTS + Study B: 9-candidate majority voting on all hard_bench questions.

─── Experiment design ────────────────────────────────────────────────────────

This script simultaneously runs:

  • Study B (accuracy baseline) — candidate 2 (greedy T=0.0, original image,
    original question) gives the single-call accuracy for each model × task.

  • TTS (9-candidate majority voting) — all 9 candidates are aggregated via
    majority vote at @1, @3, @5, @9.

Both GRIT (3B visual CoT) and Qwen2.5-VL-3B (no CoT) are evaluated on all
hard_bench tasks: VQA (MMMU-Pro), OCR (OCRBench v2), Counting (MMStar).

─── Candidate recipe (modified from prior TTS runs) ─────────────────────────

  Old recipe removed: model_paraphrase (not available for hard_bench) was
  replaced by a greedy baseline candidate. edge_enhance + model_paraphrase
  was replaced by edge_enhance + hardcoded_paraphrase.

  # | Image              | Text                | Temperature
  --|--------------------|---------------------|------------
  0 | original           | original            | 0.7
  1 | original           | hardcoded_paraphrase| 0.7
  2 | original           | original            | 0.0  ← Study B greedy baseline
  3 | edge_enhance       | original            | 0.7
  4 | grayscale          | original            | 0.7
  5 | jpeg_recompress    | original            | 0.7
  6 | brightness_contrast| original            | 0.7
  7 | rotation_90        | original            | 0.7
  8 | edge_enhance       | hardcoded_paraphrase| 0.7

─── Output ───────────────────────────────────────────────────────────────────

  results/tts_hard_bench/
    {model}_results.jsonl   — one row per question, appended as completed
    summary.json            — accuracy table after all questions processed

  Each row: question_id, task, gt_answer, answers_all (OCR),
    candidates[9] {image_aug, text_variant, temperature, raw_answer, answer},
    vote_1, vote_3, vote_5, vote_9,
    correct_1, correct_3, correct_5, correct_9, correct_greedy

─── Usage ────────────────────────────────────────────────────────────────────

  python experiments/run_tts_hard_bench.py
  python experiments/run_tts_hard_bench.py --model qwen3b
  python experiments/run_tts_hard_bench.py --model grit --task vqa
  python experiments/run_tts_hard_bench.py --resume     # default: always resumes
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

# ── Windows console UTF-8 fix ────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

# ── ANSI colours ─────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN = "\033[36m"; GREEN = "\033[32m"; RED = "\033[31m"; YELLOW = "\033[33m"
W = 80

# ── Config ───────────────────────────────────────────────────────────────────

OUT_DIR = _PROJECT_ROOT / "results" / "tts_hard_bench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["vqa", "ocr", "counting"]
N_QUESTIONS = {"vqa": 100, "ocr": 100, "counting": 92}

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qwen3b": {
        "label": "Qwen2.5-VL (3B)",
        "module": "src.models.direct_vlm",
        "class": "DirectVLMModel",
        "kwargs": {"model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "load_in_8bit": False},
        # Override max_pixels after load to handle large OCR images.
        # 512*28*28 (the DirectVLM default) is too low for OCRBench-v2 images
        # and causes a CUDA device-side assert on the visual token embeddings.
        # 1024*28*28 matches GRIT and is stable.
        "max_pixels_override": 1024 * 28 * 28,
        "max_new_tokens": 256,
        "model_type": "direct_vlm",
    },
    "grit": {
        "label": "GRIT (3B)",
        "module": "src.models.grit",
        "class": "GRITModel",
        "kwargs": {"model_id": "yfan1997/GRIT-20-Qwen2.5-VL-3B", "load_in_8bit": False},
        # GRIT natively loads at 1280 tiles; set explicitly to match and avoid
        # processor re-init mismatch on the second task after VQA.
        "max_pixels_override": 1280 * 28 * 28,
        "max_new_tokens": 512,
        "model_type": "grit",
    },
}

# ── Candidate recipes ─────────────────────────────────────────────────────────
#
# "standard" recipe: diversity comes from temperature stochasticity (T=0.7) +
#   image augmentations.  Candidate 2 is the greedy (T=0.0) Study B baseline.
#
# "t0" recipe (ablation): diversity comes from image augmentations ONLY — all
#   candidates use T=0.0 (deterministic).  The previously-greedy slot (idx 2)
#   is repurposed as a single T=0.7 stochastic draw.  Candidate 0 (original
#   image, original text, T=0.0) serves as the Study B greedy baseline.
#
# Tuple: (image_aug, text_variant, temperature)

CANDIDATES_STANDARD: List[Tuple[str, str, float]] = [
    ("original",            "original",             0.7),  # 0
    ("original",            "hardcoded_paraphrase",  0.7),  # 1
    ("original",            "original",             0.0),  # 2 ← Study B greedy
    ("edge_enhance",        "original",             0.7),  # 3
    ("grayscale",           "original",             0.7),  # 4
    ("jpeg_recompress",     "original",             0.7),  # 5
    ("brightness_contrast", "original",             0.7),  # 6
    ("rotation_90",         "original",             0.7),  # 7
    ("edge_enhance",        "hardcoded_paraphrase",  0.7),  # 8
]
GREEDY_IDX_STANDARD = 2  # candidate index used as Study B greedy baseline

CANDIDATES_T0: List[Tuple[str, str, float]] = [
    ("original",            "original",             0.0),  # 0 ← Study B greedy
    ("original",            "hardcoded_paraphrase",  0.0),  # 1
    ("original",            "original",             0.7),  # 2 ← single stochastic draw
    ("edge_enhance",        "original",             0.0),  # 3
    ("grayscale",           "original",             0.0),  # 4
    ("jpeg_recompress",     "original",             0.0),  # 5
    ("brightness_contrast", "original",             0.0),  # 6
    ("rotation_90",         "original",             0.0),  # 7  T=0 → argmax, no multinomial → safe
    ("edge_enhance",        "hardcoded_paraphrase",  0.0),  # 8
]
GREEDY_IDX_T0 = 0  # candidate 0 is original/original/T=0 → greedy baseline

# Active recipe — overwritten in main() based on --recipe flag
CANDIDATES: List[Tuple[str, str, float]] = CANDIDATES_STANDARD
GREEDY_IDX: int = GREEDY_IDX_STANDARD
N_CANDIDATES = len(CANDIDATES)

# Task-level augmentation substitutions for candidates that trigger CUDA errors.
# rotation_90 at T=0.7 causes torch.multinomial to fail on OCR images (NaN logits
# after rotation).  Substituted with jpeg_recompress in the standard (T=0.7) recipe.
# In the T=0 recipe, rotation_90 uses argmax (no multinomial call) so it is safe.
CANDIDATE_AUG_SUBS_STANDARD: Dict[str, Dict[int, str]] = {
    "ocr": {7: "jpeg_recompress"},
}
CANDIDATE_AUG_SUBS_T0: Dict[str, Dict[int, str]] = {}  # rotation_90 safe at T=0

# Active substitution map — overwritten in main()
CANDIDATE_AUG_SUBS: Dict[str, Dict[int, str]] = CANDIDATE_AUG_SUBS_STANDARD


# ── Formatting helpers ────────────────────────────────────────────────────────

def _bar(c: str = "─", w: int = W) -> str:
    return c * w


def _hdr(title: str) -> str:
    pad = (W - len(title) - 2) // 2
    return f"{BOLD}{'═' * pad} {title} {'═' * (W - pad - len(title) - 2)}{RESET}"


def _progress(done: int, total: int, elapsed: float, prefix: str = "") -> str:
    pct = done / total * 100 if total else 0
    bar_w = 30
    filled = int(bar_w * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_w - filled)
    if done > 0:
        eta = elapsed / done * (total - done)
        eta_str = f"ETA {eta/3600:.1f}h"
    else:
        eta_str = "ETA ---"
    return f"{prefix}[{bar}] {done}/{total} ({pct:.0f}%) {elapsed/60:.0f}min  {eta_str}"


# ── Answer normalisation ──────────────────────────────────────────────────────

from src.utils_normalize import normalize_answer, normalize_open_ended_answer


def _normalise(raw: str, task: str) -> Optional[str]:
    if task == "ocr":
        return normalize_open_ended_answer(raw)
    return normalize_answer(raw)


def _is_correct(pred: Optional[str], gt: str, task: str, answers_all: List[str]) -> bool:
    if pred is None:
        return False
    if task == "ocr":
        # OCR accepts any answer from answers_all
        all_norm = [normalize_open_ended_answer(a) for a in answers_all]
        return pred in all_norm
    return pred.upper() == gt.upper()


# ── Majority voting ───────────────────────────────────────────────────────────

def _majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """Return plurality answer, None excluded. Ties: first seen."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    return counts.most_common(1)[0][0]


# ── Image augmentation ────────────────────────────────────────────────────────

def _build_image_variants(image: Any) -> Dict[str, Any]:
    """Build all image augmentation variants needed by the candidate recipe."""
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
    # Return image-per-transform-id dict
    return {k: v["image"] for k, v in specs.items()}


# ── Text variant generation ───────────────────────────────────────────────────

def _build_text_variants(question: str, choices: Dict[str, str]) -> Dict[str, str]:
    """Return prompts for 'original' and 'hardcoded_paraphrase'."""
    from src.augmentation.text import generate_prompt_variants
    variants = generate_prompt_variants(question=question, choices=choices, mode="rule")
    return {
        "original": variants["original"]["prompt"],
        "hardcoded_paraphrase": variants["hardcoded_paraphrase"]["prompt"],
    }


# ── Checkpoint / resume ───────────────────────────────────────────────────────

def _cache_path(model_key: str) -> Path:
    return OUT_DIR / f"{model_key}_results.jsonl"


def load_done_ids(model_key: str, task: str) -> set:
    """Return set of question_ids already completed for this model+task."""
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


# ── Single-question inference ─────────────────────────────────────────────────

def _run_question(
    model_obj: Any,
    image_variants: Dict[str, Any],
    text_variants: Dict[str, str],
    task: str,
    model_key: str,
) -> tuple[List[Dict[str, Any]], bool]:
    """Run all 9 candidates for one question.

    Returns (candidate_results, cuda_errored).
    On CUDA error: stops early, returns whatever candidates completed, and
    sets cuda_errored=True so the caller can save the partial row and re-raise.
    """
    cfg = MODEL_CONFIGS[model_key]
    max_new_tokens = cfg["max_new_tokens"]

    results = []
    cuda_errored = False
    task_subs = CANDIDATE_AUG_SUBS.get(task, {})
    for i, (img_aug, text_var, temp) in enumerate(CANDIDATES):
        # Apply task-specific augmentation substitution if defined
        img_aug = task_subs.get(i, img_aug)
        img = image_variants.get(img_aug, image_variants["original"])
        prompt = text_variants.get(text_var, text_variants["original"])

        t0 = time.time()
        try:
            outs = model_obj.generate(
                img,
                prompt,
                n=1,
                temperature=temp,
                max_new_tokens=max_new_tokens,
            )
            raw = outs[0]["answer"] if outs else ""
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                # GPU is now in a bad state — stop collecting candidates.
                # Return partial results so the caller can save the row before crashing.
                print(f"\n  [CUDA error on candidate {i} ({img_aug}+{text_var}) — saving partial row]",
                      flush=True)
                cuda_errored = True
                break
            raw = f"ERROR: {e}"
        elapsed = time.time() - t0

        norm = _normalise(raw, task)
        results.append({
            "candidate_idx": i,
            "image_aug": img_aug,
            "text_variant": text_var,
            "temperature": temp,
            "raw_answer": raw,
            "answer": norm,
            "time_s": round(elapsed, 2),
        })
    return results, cuda_errored


# ── Model load / unload ───────────────────────────────────────────────────────

def _load_model(model_key: str) -> Any:
    import importlib
    from transformers import AutoProcessor
    cfg = MODEL_CONFIGS[model_key]
    mod = importlib.import_module(cfg["module"])
    cls = getattr(mod, cfg["class"])
    model = cls(**cfg["kwargs"])
    model._load()
    # Apply max_pixels override if specified (e.g. to handle large OCR images
    # without triggering a CUDA device-side assert on visual token embeddings).
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
    try:
        del model._model
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass  # CUDA device-side assert may still be pending — ignore at teardown
        try:
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(0)
            print(f"  [VRAM after unload: {free/1e9:.1f}/{total/1e9:.1f} GB free]", flush=True)
        except Exception:
            pass


# ── Per-model runner ──────────────────────────────────────────────────────────

def run_model(model_key: str, tasks: List[str], max_per_task: Optional[int] = None) -> None:
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n{_hdr(cfg['label'])}", flush=True)

    # Preload all images first (before GPU model loads, to avoid OOM)
    from src.data.datasets.viscot_benchmark import load_task as _load_task

    print("\nPreloading all images into CPU memory...", flush=True)
    all_questions: Dict[str, List[Dict[str, Any]]] = {}
    all_images: Dict[Tuple[str, str], Any] = {}

    for task in tasks:
        examples = _load_task(task)
        if max_per_task:
            examples = examples[:max_per_task]
        all_questions[task] = examples
        for ex in examples:
            all_images[(task, str(ex["question_id"]))] = ex["image"]
        print(f"  {task}: {len(examples)} questions loaded", flush=True)

    print(f"  Total images preloaded: {len(all_images)}", flush=True)

    # Load model
    print(f"\nLoading {cfg['label']}...", flush=True)
    t_load = time.time()
    model_obj = _load_model(model_key)
    print(f"  Model loaded in {time.time() - t_load:.0f}s", flush=True)

    try:
        for task in tasks:
            examples = all_questions[task]
            if max_per_task:
                examples = examples[:max_per_task]
            done_ids = load_done_ids(model_key, task)
            todo = [ex for ex in examples if str(ex["question_id"]) not in done_ids]

            print(f"\n{_bar()}", flush=True)
            print(f"  {BOLD}{task.upper()}{RESET}  {len(done_ids)} done, {len(todo)} to go", flush=True)

            if not todo:
                print(f"  {GREEN}All done — skipping.{RESET}", flush=True)
                continue

            total_q = len(examples)
            t_task_start = time.time()

            for q_idx, ex in enumerate(todo):
                qid = str(ex["question_id"])
                question = ex["question"]
                gt = ex.get("answer", "")
                answers_all = ex.get("answers_all", [gt])
                image = all_images[(task, qid)]

                # Parse choices from question (MCQ tasks embed A/B/C/D in question text).
                # Handles three formats:
                #   MMMU-Pro:  "A. text" or "A) text"
                #   MMStar:    "A: text, B: text, ..."  (comma-separated colon format)
                choices: Dict[str, str] = {}
                if task in ("vqa", "counting"):
                    import re as _re
                    # Unified pattern: A/B/.../J followed by . : or ) then value text
                    # Lookahead stops at the next choice letter or end of string/line.
                    for m in _re.finditer(
                        r"\b([A-J])[.:)]\s*(.+?)(?=\s*[,\n]?\s*[A-J][.:)]|[,\n]|$)",
                        question,
                    ):
                        choices[m.group(1).upper()] = m.group(2).strip().rstrip(",")

                # Build variants
                image_variants = _build_image_variants(image)
                text_variants = _build_text_variants(question, choices)

                # Progress display
                global_done = len(done_ids) + q_idx
                elapsed = time.time() - t_task_start
                print(
                    f"\r  {_progress(q_idx, len(todo), elapsed, f'{task} ')}  qid={qid}",
                    end="", flush=True,
                )

                # Run all 9 candidates (may return partial results on CUDA error)
                candidates, cuda_errored = _run_question(
                    model_obj, image_variants, text_variants, task, model_key
                )

                # Voting — use however many candidates we got (may be < 9 on CUDA error)
                all_answers = [c["answer"] for c in candidates]
                vote_1  = _majority_vote(all_answers[:1])
                vote_3  = _majority_vote(all_answers[:3])
                vote_5  = _majority_vote(all_answers[:5])
                vote_9  = _majority_vote(all_answers[:9])
                # GREEDY_IDX = Study B greedy candidate; may be absent if CUDA error hit before it
                greedy  = all_answers[GREEDY_IDX] if len(all_answers) > GREEDY_IDX else None

                correct_1      = _is_correct(vote_1,  gt, task, answers_all)
                correct_3      = _is_correct(vote_3,  gt, task, answers_all)
                correct_5      = _is_correct(vote_5,  gt, task, answers_all)
                correct_9      = _is_correct(vote_9,  gt, task, answers_all)
                correct_greedy = _is_correct(greedy,  gt, task, answers_all)
                correct_any    = any(_is_correct(c["answer"], gt, task, answers_all) for c in candidates)

                row = {
                    "model": model_key,
                    "task": task,
                    "question_id": qid,
                    "question": question,
                    "gt_answer": gt,
                    "answers_all": answers_all,
                    "candidates": candidates,
                    "cuda_errored": cuda_errored,
                    "vote_1":  vote_1,
                    "vote_3":  vote_3,
                    "vote_5":  vote_5,
                    "vote_9":  vote_9,
                    "greedy":  greedy,
                    "correct_1":      correct_1,
                    "correct_3":      correct_3,
                    "correct_5":      correct_5,
                    "correct_9":      correct_9,
                    "correct_greedy": correct_greedy,
                    "correct_any":    correct_any,
                }
                # Always save the row first (partial row on CUDA error is still checkpointed)
                _append_result(row, model_key)
                done_ids.add(qid)

                if cuda_errored:
                    raise RuntimeError(
                        f"CUDA error on {model_key}/{task}/qid={qid} — GPU poisoned. "
                        "Partial row saved to checkpoint. Restart to continue from next question."
                    )

            # Final newline after progress bar
            print(flush=True)

            # Task summary
            _print_task_summary(model_key, task)

    finally:
        print(f"\nUnloading {cfg['label']}...", flush=True)
        _unload_model(model_obj)
        del model_obj


# ── Summary printer ───────────────────────────────────────────────────────────

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
    acc_greedy = sum(r["correct_greedy"] for r in rows) / n * 100
    acc_1      = sum(r["correct_1"]      for r in rows) / n * 100
    acc_3      = sum(r["correct_3"]      for r in rows) / n * 100
    acc_5      = sum(r["correct_5"]      for r in rows) / n * 100
    acc_9      = sum(r["correct_9"]      for r in rows) / n * 100
    oracle_9   = sum(r["correct_any"]    for r in rows) / n * 100
    print(f"\n  {BOLD}{task.upper()} summary ({n} questions):{RESET}", flush=True)
    print(f"    greedy(T=0)  = {acc_greedy:.1f}%   [Study B baseline]", flush=True)
    print(f"    @1  (T=0.7)  = {acc_1:.1f}%", flush=True)
    print(f"    @3           = {acc_3:.1f}%", flush=True)
    print(f"    @5           = {acc_5:.1f}%", flush=True)
    print(f"    @9           = {acc_9:.1f}%   gain = {acc_9 - acc_greedy:+.1f}pp vs greedy", flush=True)
    print(f"    oracle@9     = {oracle_9:.1f}%", flush=True)


def _print_model_summary(model_key: str) -> None:
    path = _cache_path(model_key)
    if not path.exists():
        return
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if not r.get("cuda_errored", False):
                    rows.append(r)
            except Exception:
                pass
    if not rows:
        return

    print(f"\n{_hdr('FINAL SUMMARY — ' + MODEL_CONFIGS[model_key]['label'])}", flush=True)
    print(f"  {'Task':<12} {'greedy':>8} {'@1':>8} {'@3':>8} {'@5':>8} {'@9':>8} {'oracle@9':>10} {'gain@9':>8}", flush=True)
    print(f"  {_bar('-', 74)}", flush=True)

    by_task: Dict[str, List[Dict]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    for task in TASKS:
        task_rows = by_task.get(task, [])
        if not task_rows:
            continue
        n = len(task_rows)
        g  = sum(r["correct_greedy"] for r in task_rows) / n * 100
        a1 = sum(r["correct_1"]      for r in task_rows) / n * 100
        a3 = sum(r["correct_3"]      for r in task_rows) / n * 100
        a5 = sum(r["correct_5"]      for r in task_rows) / n * 100
        a9 = sum(r["correct_9"]      for r in task_rows) / n * 100
        o9 = sum(r["correct_any"]    for r in task_rows) / n * 100
        print(
            f"  {task:<12} {g:>7.1f}% {a1:>7.1f}% {a3:>7.1f}% {a5:>7.1f}% "
            f"{a9:>7.1f}% {o9:>9.1f}% {a9 - g:>+7.1f}pp",
            flush=True,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTS + Study B on hard_bench")
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default=None,
                   help="Which model to run (default: both, in order)")
    p.add_argument("--task", choices=TASKS, default=None,
                   help="Single task to run (default: all)")
    p.add_argument("--max-per-task", type=int, default=None,
                   help="Cap questions per task (default: all available)")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip already-computed questions (default: always True)")
    p.add_argument(
        "--recipe", choices=["standard", "t0"], default="standard",
        help=(
            "Candidate recipe to use. "
            "'standard': 8×T=0.7 stochastic + 1×T=0.0 greedy (candidate 2). "
            "'t0': 8×T=0.0 deterministic (image diversity only) + 1×T=0.7 stochastic (candidate 2). "
            "Results are written to separate output directories."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Apply recipe selection ────────────────────────────────────────────────
    global CANDIDATES, GREEDY_IDX, N_CANDIDATES, CANDIDATE_AUG_SUBS, OUT_DIR
    if args.recipe == "t0":
        CANDIDATES        = CANDIDATES_T0
        GREEDY_IDX        = GREEDY_IDX_T0
        CANDIDATE_AUG_SUBS = CANDIDATE_AUG_SUBS_T0
        OUT_DIR = _PROJECT_ROOT / "results" / "tts_hard_bench_t0"
    else:
        CANDIDATES        = CANDIDATES_STANDARD
        GREEDY_IDX        = GREEDY_IDX_STANDARD
        CANDIDATE_AUG_SUBS = CANDIDATE_AUG_SUBS_STANDARD
        OUT_DIR = _PROJECT_ROOT / "results" / "tts_hard_bench"
    N_CANDIDATES = len(CANDIDATES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models_to_run = [args.model] if args.model else list(MODEL_CONFIGS)
    tasks_to_run  = [args.task]  if args.task  else TASKS
    max_per_task  = args.max_per_task

    # ETA estimate
    n_per_task = {t: min(N_QUESTIONS[t], max_per_task) if max_per_task else N_QUESTIONS[t]
                  for t in tasks_to_run}
    total_calls = sum(n_per_task.values()) * len(models_to_run) * N_CANDIDATES
    print(f"\n{_hdr('TTS Hard Bench — Experiment Start')}", flush=True)
    print(f"  Recipe : {args.recipe}", flush=True)
    print(f"  Models : {models_to_run}", flush=True)
    print(f"  Tasks  : {tasks_to_run}", flush=True)
    print(f"  Candidates/question : {N_CANDIDATES}", flush=True)
    print(f"  Greedy baseline idx : {GREEDY_IDX}", flush=True)
    print(f"  Total inference calls: {total_calls:,}", flush=True)
    print(f"  Estimated runtime: see CLAUDE.md for GPU estimates", flush=True)
    print(f"  Output: {OUT_DIR}", flush=True)
    print(f"  Checkpoint/resume: enabled — rerun to continue after crash", flush=True)

    t_global = time.time()
    for model_key in models_to_run:
        run_model(model_key, tasks_to_run, max_per_task=max_per_task)
        _print_model_summary(model_key)

    elapsed = time.time() - t_global
    print(f"\n{_hdr('ALL DONE')}", flush=True)
    print(f"  Total wall time: {elapsed/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
