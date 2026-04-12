"""Study A — entropy delta: GRIT (3B, visual CoT) vs Qwen2.5-VL-3B (no CoT).

Both models share Qwen2.5-VL-3B base weights.  The only variable is the
visual CoT fine-tuning applied to GRIT.

For each selected question × model:
  - n=10 independent samples at temperature=0.7
  - Final answer extracted via the model's generate() (GRIT uses <answer> tags,
    Qwen returns the direct output)
  - Shannon entropy computed over normalised answers

Delta:
    Δ_entropy = entropy(GRIT) − entropy(Qwen3B)

A positive Δ on ≥ 2/3 tasks is the go criterion for the full TTS run.

Reads:  results/calibration/selected_questions.jsonl
Writes: results/study_a/entropy_results.jsonl
        results/study_a/summary.json

Usage:
    python experiments/run_study_a_entropy.py
    python experiments/run_study_a_entropy.py --task ocr
    python experiments/run_study_a_entropy.py --model grit   # single model
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.stochasticity import compute_entropy
from src.utils.logging import get_logger
from src.utils_normalize import normalize_answer, normalize_open_ended_answer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TASKS = ["vqa", "ocr", "counting"]
N_SAMPLES = 10
TEMPERATURE = 0.7

MODELS = {
    "qwen3b": ("src.models.direct_vlm", "DirectVLMModel",
               {"model_id": "Qwen/Qwen2.5-VL-3B-Instruct"}),
    "grit":   ("src.models.grit", "GRITModel", {}),
}

CALIB_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "calibration" / "selected_questions.jsonl"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "study_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Answer normalisation — consistent across both models
# ---------------------------------------------------------------------------

def _normalise(raw: str, task: str) -> Optional[str]:
    """Extract and normalise the final answer for entropy computation.

    MCQ tasks (vqa, counting): extract letter A-J via normalize_answer.
    OCR: lowercase + strip via normalize_open_ended_answer.
    """
    if task == "ocr":
        return normalize_open_ended_answer(raw)
    return normalize_answer(raw)  # returns uppercase letter or None


# ---------------------------------------------------------------------------
# Load calibration questions
# ---------------------------------------------------------------------------

def load_selected(tasks: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Load selected questions from the calibration output, grouped by task."""
    if not CALIB_PATH.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIB_PATH}\n"
            "Run scripts/select_calibration_questions.py first."
        )

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with CALIB_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task"] in tasks:
                by_task[row["task"]].append(row)

    for task in tasks:
        n = len(by_task[task])
        if n == 0:
            logger.warning("No questions found for task '{}' in {}", task, CALIB_PATH)
        else:
            logger.info("Loaded {} questions for task '{}'", n, task)

    return dict(by_task)


# ---------------------------------------------------------------------------
# Image loader (re-fetches from hard_bench cache)
# ---------------------------------------------------------------------------

def _load_image(row: Dict[str, Any]):
    """Load PIL image for a selected question row via the hard_bench loader."""
    from src.data.datasets.viscot_benchmark import load_task
    task = row["task"]
    target_qid = str(row["question_id"])

    # Use offset=0 scan — images are cached on disk so this is fast.
    examples = load_task(task)
    for ex in examples:
        if str(ex["question_id"]) == target_qid:
            return ex["image"]
    raise ValueError(
        f"Image not found for question_id={target_qid!r} task={task!r}. "
        "Run scripts/prepare_hard_bench.py to rebuild the dataset."
    )


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------

def _run_question(
    model: Any,
    image: Any,
    question: str,
    task: str,
) -> List[Optional[str]]:
    """Return N_SAMPLES normalised answers for one question."""
    results = model.generate(
        image,
        question,
        n=N_SAMPLES,
        temperature=TEMPERATURE,
        max_new_tokens=512,
    )
    raw_answers = [r["answer"] for r in results]
    return [_normalise(a, task) for a in raw_answers]


# ---------------------------------------------------------------------------
# Model loader / unloader
# ---------------------------------------------------------------------------

def _load_model(model_key: str) -> Any:
    import importlib
    mod_path, cls_name, kwargs = MODELS[model_key]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    model = cls(**kwargs)
    model._load()
    return model


def _unload_model(model: Any) -> None:
    import torch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model(
    model_key: str,
    by_task: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Run model_key on all selected questions; return per-question result rows."""
    print(f"\n{'='*60}")
    print(f"MODEL: {model_key.upper()}")
    print(f"  n={N_SAMPLES} samples, T={TEMPERATURE}")
    print(f"{'='*60}")

    model = _load_model(model_key)
    rows: List[Dict[str, Any]] = []

    for task, questions in by_task.items():
        print(f"\n--- Task: {task.upper()} ({len(questions)} questions) ---")
        for q in questions:
            qid = q["question_id"]
            print(f"  qid={qid}  gt={q['answer']!r}", end="", flush=True)

            image = _load_image(q)
            norm_answers = _run_question(model, image, q["question"], task)
            entropy = compute_entropy(norm_answers)

            n_valid = sum(1 for a in norm_answers if a is not None)
            n_unique = len(set(a for a in norm_answers if a is not None))
            print(f"  unique={n_unique}/{n_valid}  H={entropy:.3f}b")

            rows.append({
                "model": model_key,
                "task": task,
                "question_id": qid,
                "question": q["question"],
                "gt": q["answer"],
                "norm_answers": norm_answers,
                "n_unique": n_unique,
                "n_valid": n_valid,
                "entropy_bits": entropy,
            })

    _unload_model(model)
    return rows


def main(tasks: List[str], models_to_run: List[str]) -> None:
    by_task = load_selected(tasks)
    if not by_task:
        print("No questions loaded — exiting.")
        return

    all_rows: List[Dict[str, Any]] = []

    for model_key in models_to_run:
        rows = run_model(model_key, by_task)
        all_rows.extend(rows)

        # Checkpoint after each model
        ckpt_path = OUT_DIR / f"entropy_{model_key}.jsonl"
        with ckpt_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nCheckpoint saved → {ckpt_path}")

    # ── Save full results ────────────────────────────────────────────────
    out_path = OUT_DIR / "entropy_results.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Compute and print delta summary ─────────────────────────────────
    if set(models_to_run) == {"qwen3b", "grit"}:
        _print_summary(all_rows, tasks)


def _print_summary(rows: List[Dict[str, Any]], tasks: List[str]) -> None:
    """Compute per-task mean entropy and Δ = GRIT − Qwen3B."""
    # Aggregate: task → model → list of per-question entropies
    entropy_by: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        entropy_by[row["task"]][row["model"]].append(row["entropy_bits"])

    summary: Dict[str, Any] = {}
    print(f"\n{'='*60}")
    print("STUDY A — ENTROPY DELTA SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<12} {'Qwen3B H':>10} {'GRIT H':>10} {'Δ':>10}  {'Δ > 0?':>8}")
    print("-" * 56)

    go_count = 0
    for task in tasks:
        q_vals = entropy_by[task].get("qwen3b", [])
        g_vals = entropy_by[task].get("grit", [])

        if not q_vals or not g_vals:
            print(f"  {task:<10}  (incomplete — skipping)")
            continue

        mean_q = sum(q_vals) / len(q_vals)
        mean_g = sum(g_vals) / len(g_vals)
        delta = mean_g - mean_q
        pos = delta > 0
        if pos:
            go_count += 1

        print(f"  {task:<10}  {mean_q:>10.3f}  {mean_g:>10.3f}  {delta:>+10.3f}  {'YES ✓' if pos else 'NO  ✗':>8}")

        summary[task] = {
            "mean_entropy_qwen3b": mean_q,
            "mean_entropy_grit": mean_g,
            "delta": delta,
            "n_questions": len(q_vals),
        }

    print("-" * 56)
    n_tasks = len([t for t in tasks if t in entropy_by and
                   "qwen3b" in entropy_by[t] and "grit" in entropy_by[t]])
    print(f"\n  Δ > 0 on {go_count}/{n_tasks} tasks")

    verdict = "GO" if go_count >= 2 else "NO-GO"
    print(f"  Go criterion (Δ > 0 on ≥ 2/3): {verdict}")
    if verdict == "NO-GO":
        print("  → Stochasticity increase not confirmed. Review hypothesis framing.")
        print("    Check: is GRIT extracting final answers correctly?")
        print("    Check: are questions in the right difficulty range?")

    summary["go_count"] = go_count
    summary["verdict"] = verdict

    summary_path = OUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_path = OUT_DIR / "entropy_results.jsonl"
    print(f"\nFull results → {out_path}")
    print(f"Summary      → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="Run a single task (default: all)")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                        help="Run a single model (default: both)")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS
    models_to_run = [args.model] if args.model else list(MODELS.keys())

    main(tasks, models_to_run)
