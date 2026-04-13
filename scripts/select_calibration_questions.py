"""Calibration question selection for the entropy pilot.

Two-pass approach to find questions in the productive accuracy range [20%, 70%]
for BOTH the 3B and 7B baselines — so both Study A and Study B operate on the
same question set at a meaningful difficulty level.

Pass 1 — Qwen2.5-VL-3B: run n=5 at T=0.7 on all available questions,
          keep those where accuracy in [20%, 70%].
Pass 2 — Qwen2.5-VL-7B: re-run the Pass-1 survivors, keep the intersection.

OCR extra filter: at least 1/5 samples must match an entry in answers_all.

Stops per task once TARGET_PER_TASK questions are selected.

Output:
    results/calibration/selected_questions.jsonl
        One row per selected question with fields:
          task, question_id, question, answer, answers_all (OCR only),
          acc_3b, acc_7b, image_id, image_source

Usage:
    python scripts/select_calibration_questions.py
    python scripts/select_calibration_questions.py --task vqa   # single task
    python scripts/select_calibration_questions.py --dry-run    # no GPU, shows filtering logic
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets.viscot_benchmark import load_task
from src.utils.logging import get_logger
from src.utils_normalize import normalize_answer, normalize_open_ended_answer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TASKS = ["vqa", "ocr", "counting"]
TARGET_PER_TASK = 10   # desired final questions per task
N_CALIB = 5            # samples per question for calibration
TEMPERATURE = 0.7
ACC_LOW = 0.20         # inclusive lower bound
ACC_HIGH = 0.70        # inclusive upper bound

MODEL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = OUT_DIR / "inference_cache.jsonl"

# ---------------------------------------------------------------------------
# Inference cache  (keyed by model × task × question_id)
# ---------------------------------------------------------------------------

InferenceCache = Dict[Tuple[str, str, str], Dict[str, Any]]


def load_cache() -> InferenceCache:
    """Load all previously computed inference results from CACHE_PATH.

    Returns a dict keyed by (model_key, task, str(question_id)).
    Each value is the full row dict (answers, acc, question, gt, …).
    """
    cache: InferenceCache = {}
    if not CACHE_PATH.exists():
        return cache
    with CACHE_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["model"], row["task"], str(row["question_id"]))
            cache[key] = row
    print(f"  [cache] loaded {len(cache)} cached inference results from {CACHE_PATH}", flush=True)
    return cache


def _cache_key(model_key: str, task: str, question_id: Any) -> Tuple[str, str, str]:
    return (model_key, task, str(question_id))


def _append_cache(row: Dict[str, Any]) -> None:
    """Append one result row to the cache file (one JSON line)."""
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Answer correctness helpers
# ---------------------------------------------------------------------------

def _check_correct(raw_answer: str, gt: str, task: str,
                   answers_all: Optional[List[str]] = None) -> bool:
    """Return True if raw_answer matches the ground truth for this task.

    MCQ tasks (vqa, counting): extract letter via normalize_answer, compare to gt.
    OCR: normalize both sides, check against all acceptable answers in answers_all.
    """
    if task == "ocr":
        norm = normalize_open_ended_answer(raw_answer)
        candidates = answers_all if answers_all else [gt]
        return any(normalize_open_ended_answer(c) == norm for c in candidates)
    else:
        extracted = normalize_answer(raw_answer)
        return extracted is not None and extracted.upper() == gt.strip().upper()


def _accuracy(answers: List[str], gt: str, task: str,
               answers_all: Optional[List[str]] = None) -> float:
    """Fraction of answers that are correct."""
    if not answers:
        return 0.0
    correct = sum(_check_correct(a, gt, task, answers_all) for a in answers)
    return correct / len(answers)


def _in_range(acc: float) -> bool:
    return ACC_LOW <= acc <= ACC_HIGH


def _min_correct() -> int:
    return math.ceil(ACC_LOW * N_CALIB)


def _max_correct() -> int:
    return math.floor(ACC_HIGH * N_CALIB)

# ---------------------------------------------------------------------------
# Per-question runner (uses model.generate())
# ---------------------------------------------------------------------------

def _run_question(
    model: Any,
    example: Dict[str, Any],
    task: str,
) -> Tuple[List[str], float]:
    """Run N_CALIB samples and return (answers, accuracy)."""
    results = model.generate(
        example["image"],
        example["question"],
        n=N_CALIB,
        temperature=TEMPERATURE,
        max_new_tokens=256,
    )
    answers = [r["answer"] for r in results]
    acc = _accuracy(
        answers,
        example["answer"],
        task,
        example.get("answers_all"),
    )
    return answers, acc


def _run_question_cached(
    model: Any,
    model_key: str,
    example: Dict[str, Any],
    task: str,
    cache: InferenceCache,
) -> Tuple[List[str], float, bool]:
    """Return (answers, accuracy, cache_hit).

    Checks the cache first; on a miss runs inference and appends the result
    to CACHE_PATH immediately so partial runs are recoverable.
    Stored fields: model, task, question_id, question, gt, answers_all, answers, acc.
    """
    key = _cache_key(model_key, task, example["question_id"])
    if key in cache:
        cached = cache[key]
        return cached["answers"], cached["acc"], True

    answers, acc = _run_question(model, example, task)

    row: Dict[str, Any] = {
        "model": model_key,
        "task": task,
        "question_id": example["question_id"],
        "question": example["question"],
        "gt": example["answer"],
        "answers_all": example.get("answers_all"),
        "image_id": example["image_id"],
        "image_source": example["image_source"],
        "answers": answers,
        "acc": acc,
    }
    cache[key] = row
    _append_cache(row)
    return answers, acc, False


# ---------------------------------------------------------------------------
# OCR extra filter
# ---------------------------------------------------------------------------

def _ocr_has_any_match(answers: List[str], answers_all: List[str]) -> bool:
    """At least one answer must match an entry in answers_all."""
    for raw in answers:
        norm = normalize_open_ended_answer(raw)
        if any(normalize_open_ended_answer(c) == norm for c in answers_all):
            return True
    return False


# ---------------------------------------------------------------------------
# Pass helpers
# ---------------------------------------------------------------------------

def _pass_one(
    model: Any,
    model_key: str,
    task: str,
    examples: List[Dict[str, Any]],
    cache: InferenceCache,
    max_survivors: int = TARGET_PER_TASK * 2,
) -> List[Dict[str, Any]]:
    """Run Qwen-3B on examples until max_survivors found; return those that pass the filter.

    Results are served from cache when available and written to cache on miss.
    """
    survivors: List[Dict[str, Any]] = []
    print(f"\n  [Pass 1 / {task.upper()}] scanning (stop at {max_survivors} survivors)…", flush=True)

    pbar = tqdm(examples, desc=f"Pass1/{task}", unit="q", dynamic_ncols=True)
    for ex in pbar:
        if len(survivors) >= max_survivors:
            pbar.set_postfix_str(f"reached {max_survivors} survivors — stopping early")
            pbar.close()
            break

        qid = ex["question_id"]
        answers, acc, hit = _run_question_cached(model, model_key, ex, task, cache)
        n_correct = round(acc * N_CALIB)
        hit_flag = " [cache]" if hit else ""

        # OCR extra filter
        if task == "ocr":
            answers_all = ex.get("answers_all", [ex["answer"]])
            has_match = _ocr_has_any_match(answers, answers_all)
            ocr_flag = f"  ocr_match={'YES' if has_match else 'NO'}"
        else:
            has_match = True
            ocr_flag = ""

        passed = _in_range(acc) and has_match
        status = "PASS" if passed else "skip"
        pbar.set_postfix_str(f"qid={qid} acc={n_correct}/{N_CALIB} {status}{hit_flag}")
        print(f"    qid={qid}  acc_3b={n_correct}/{N_CALIB}={acc:.2f}{ocr_flag}{hit_flag}  -> {status}", flush=True)

        if passed:
            survivors.append({**ex, "_acc_3b": acc, "_answers_3b": answers})

    print(f"  [Pass 1 / {task.upper()}] {len(survivors)} survivors -> Pass 2", flush=True)
    return survivors


def _pass_two(
    model: Any,
    model_key: str,
    task: str,
    survivors: List[Dict[str, Any]],
    cache: InferenceCache,
) -> List[Dict[str, Any]]:
    """Run Qwen-7B on Pass-1 survivors; return the intersection, capped at TARGET.

    Results are served from cache when available and written to cache on miss.
    """
    selected: List[Dict[str, Any]] = []
    print(f"\n  [Pass 2 / {task.upper()}] checking {len(survivors)} survivors with 7B…", flush=True)

    pbar = tqdm(survivors, desc=f"Pass2/{task}", unit="q", dynamic_ncols=True)
    for ex in pbar:
        if len(selected) >= TARGET_PER_TASK:
            pbar.set_postfix_str(f"reached {TARGET_PER_TASK} — stopping")
            pbar.close()
            break

        qid = ex["question_id"]
        answers, acc, hit = _run_question_cached(model, model_key, ex, task, cache)
        n_correct = round(acc * N_CALIB)
        hit_flag = " [cache]" if hit else ""
        passed = _in_range(acc)
        status = "SELECT" if passed else "skip"
        pbar.set_postfix_str(f"qid={qid} acc={n_correct}/{N_CALIB} {status}{hit_flag}")
        print(f"    qid={qid}  acc_3b={ex['_acc_3b']:.2f}  acc_7b={n_correct}/{N_CALIB}={acc:.2f}{hit_flag}  -> {status}", flush=True)

        if passed:
            selected.append({
                "task": task,
                "question_id": ex["question_id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "answers_all": ex.get("answers_all"),
                "image_id": ex["image_id"],
                "image_source": ex["image_source"],
                "acc_3b": ex["_acc_3b"],
                "acc_7b": acc,
                "calib_answers_3b": ex["_answers_3b"],
                "calib_answers_7b": answers,
            })

    print(f"  [Pass 2 / {task.upper()}] selected {len(selected)} questions.", flush=True)
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(tasks_to_run: List[str], dry_run: bool = False) -> None:
    if dry_run:
        _dry_run_demo(tasks_to_run)
        return

    import torch
    from src.models.direct_vlm import DirectVLMModel

    # ── Load inference cache ──────────────────────────────────────────────
    cache = load_cache()

    # ── Load questions once (no GPU needed) ──────────────────────────────
    task_examples: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks_to_run:
        print(f"\nLoading {task} examples…", flush=True)
        task_examples[task] = load_task(task)  # all available rows
        print(f"  {len(task_examples[task])} examples loaded.", flush=True)

    # ── Pass 1: Qwen-3B ──────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"PASS 1 — Qwen2.5-VL-3B ({MODEL_3B})", flush=True)
    print(f"  n={N_CALIB} samples, T={TEMPERATURE}", flush=True)
    print(f"  accuracy range: [{ACC_LOW:.0%}, {ACC_HIGH:.0%}]", flush=True)
    print(f"  = {_min_correct()}-{_max_correct()} correct out of {N_CALIB}", flush=True)
    print(f"{'='*60}", flush=True)

    model_3b = DirectVLMModel(model_id=MODEL_3B)
    model_3b._load()

    task_survivors: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks_to_run:
        task_survivors[task] = _pass_one(model_3b, "qwen3b", task, task_examples[task], cache)

    del model_3b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Save survivors as final output ───────────────────────────────────
    out_path = OUT_DIR / "selected_questions.jsonl"
    all_selected = []
    with out_path.open("w", encoding="utf-8") as f:
        for task, rows in task_survivors.items():
            for row in rows:
                record = {
                    "task": task,
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "answers_all": row.get("answers_all"),
                    "image_id": row["image_id"],
                    "image_source": row["image_source"],
                    "acc_3b": row["_acc_3b"],
                    "calib_answers_3b": row["_answers_3b"],
                }
                all_selected.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("CALIBRATION SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for task in tasks_to_run:
        rows = task_survivors.get(task, [])
        if rows:
            mean_3b = sum(r["_acc_3b"] for r in rows) / len(rows)
            print(f"  {task:<10}  {len(rows):2d} survivors  mean_acc_3b={mean_3b:.2f}", flush=True)
        else:
            print(f"  {task:<10}   0 survivors — consider loosening thresholds", flush=True)

    print(f"\nSaved {len(all_selected)} selected questions -> {out_path}", flush=True)


def _dry_run_demo(tasks_to_run: List[str]) -> None:
    """Print filtering logic without loading any models."""
    print("DRY RUN — no models loaded")
    print(f"  Tasks        : {tasks_to_run}")
    print(f"  Target       : {TARGET_PER_TASK} per task")
    print(f"  N_CALIB      : {N_CALIB} samples at T={TEMPERATURE}")
    print(f"  Accuracy range: [{ACC_LOW:.0%}, {ACC_HIGH:.0%}]"
          f"  = {_min_correct()}–{_max_correct()} correct out of {N_CALIB}")
    print(f"  OCR filter   : at least 1/{N_CALIB} samples must match answers_all")
    print(f"  Output       : {OUT_DIR / 'selected_questions.jsonl'}")

    for task in tasks_to_run:
        examples = load_task(task, n=5)
        print(f"\n  Sample from {task} (showing first 5 rows):")
        for ex in examples:
            print(f"    qid={ex['question_id']}  gt={ex['answer']!r}"
                  f"  answers_all={ex.get('answers_all', 'N/A')!r:.60s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="Run a single task (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without loading models")
    args = parser.parse_args()

    tasks_to_run = [args.task] if args.task else TASKS
    main(tasks_to_run, dry_run=args.dry_run)
