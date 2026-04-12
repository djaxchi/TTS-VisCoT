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
MODEL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    task: str,
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run Qwen-3B on all examples; return those that pass the accuracy filter."""
    survivors: List[Dict[str, Any]] = []
    print(f"\n  [Pass 1 / {task.upper()}] scanning {len(examples)} questions with 3B…")

    for ex in examples:
        qid = ex["question_id"]
        answers, acc = _run_question(model, ex, task)
        n_correct = round(acc * N_CALIB)

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
        print(f"    qid={qid}  acc_3b={n_correct}/{N_CALIB}={acc:.2f}{ocr_flag}  → {status}")

        if passed:
            survivors.append({**ex, "_acc_3b": acc, "_answers_3b": answers})

    print(f"  [Pass 1 / {task.upper()}] {len(survivors)} survivors → Pass 2")
    return survivors


def _pass_two(
    model: Any,
    task: str,
    survivors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run Qwen-7B on Pass-1 survivors; return the intersection, capped at TARGET."""
    selected: List[Dict[str, Any]] = []
    print(f"\n  [Pass 2 / {task.upper()}] checking {len(survivors)} survivors with 7B…")

    for ex in survivors:
        if len(selected) >= TARGET_PER_TASK:
            print(f"  [Pass 2 / {task.upper()}] reached {TARGET_PER_TASK} — stopping.")
            break

        qid = ex["question_id"]
        answers, acc = _run_question(model, ex, task)
        n_correct = round(acc * N_CALIB)
        passed = _in_range(acc)
        status = "SELECT" if passed else "skip"
        print(f"    qid={qid}  acc_3b={ex['_acc_3b']:.2f}  acc_7b={n_correct}/{N_CALIB}={acc:.2f}  → {status}")

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

    print(f"  [Pass 2 / {task.upper()}] selected {len(selected)} questions.")
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

    all_selected: List[Dict[str, Any]] = []

    # ── Load questions once (no GPU needed) ──────────────────────────────
    task_examples: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks_to_run:
        print(f"\nLoading {task} examples…")
        task_examples[task] = load_task(task)  # all available rows
        print(f"  {len(task_examples[task])} examples loaded.")

    # ── Pass 1: Qwen-3B ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PASS 1 — Qwen2.5-VL-3B ({MODEL_3B})")
    print(f"  n={N_CALIB} samples, T={TEMPERATURE}")
    print(f"  accuracy range: [{ACC_LOW:.0%}, {ACC_HIGH:.0%}]")
    print(f"  = {_min_correct()}–{_max_correct()} correct out of {N_CALIB}")
    print(f"{'='*60}")

    model_3b = DirectVLMModel(model_id=MODEL_3B)
    model_3b._load()

    task_survivors: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks_to_run:
        task_survivors[task] = _pass_one(model_3b, task, task_examples[task])

    del model_3b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save Pass-1 intermediate results (allows resuming)
    p1_path = OUT_DIR / "calibration_pass1.jsonl"
    with p1_path.open("w", encoding="utf-8") as f:
        for task, rows in task_survivors.items():
            for row in rows:
                # strip PIL image before serialising
                serialisable = {k: v for k, v in row.items()
                                if k != "image" and not k.startswith("_acc")}
                serialisable["_acc_3b"] = row["_acc_3b"]
                serialisable["_answers_3b"] = row["_answers_3b"]
                f.write(json.dumps(serialisable, ensure_ascii=False) + "\n")
    print(f"\nPass-1 intermediate results saved → {p1_path}")

    # ── Pass 2: Qwen-7B ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PASS 2 — Qwen2.5-VL-7B ({MODEL_7B})")
    print(f"{'='*60}")

    model_7b = DirectVLMModel(model_id=MODEL_7B)
    model_7b._load()

    for task in tasks_to_run:
        selected = _pass_two(model_7b, task, task_survivors[task])
        all_selected.extend(selected)
        if len(selected) < TARGET_PER_TASK:
            print(f"\n  WARNING: only {len(selected)}/{TARGET_PER_TASK} questions "
                  f"found for {task}. Consider loosening ACC_HIGH or using more questions.")

    del model_7b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Save final selection ──────────────────────────────────────────────
    out_path = OUT_DIR / "selected_questions.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*60}")
    by_task: Dict[str, List] = {}
    for row in all_selected:
        by_task.setdefault(row["task"], []).append(row)
    for task in tasks_to_run:
        rows = by_task.get(task, [])
        if rows:
            mean_3b = sum(r["acc_3b"] for r in rows) / len(rows)
            mean_7b = sum(r["acc_7b"] for r in rows) / len(rows)
            print(f"  {task:<10}  {len(rows):2d} questions  "
                  f"mean_acc_3b={mean_3b:.2f}  mean_acc_7b={mean_7b:.2f}")
        else:
            print(f"  {task:<10}   0 questions — consider loosening thresholds")

    print(f"\nSaved {len(all_selected)} selected questions → {out_path}")


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
