#!/usr/bin/env python3
"""Run DeepEyesV2 on N TreeBench questions and report accuracy.

Loads the first --n questions from HaochenWang/TreeBench (test split),
runs DeepEyesV2 on each, normalizes the answer to a MCQ letter (A/B/C/D),
and saves per-question results to a JSONL checkpoint file so the job can
be resumed if interrupted.

Usage:
    python experiments/run_deepeyes_treebench.py
    python experiments/run_deepeyes_treebench.py --n 10
    python experiments/run_deepeyes_treebench.py --n 10 --output results/treebench/deepeyes_10.jsonl

SLURM (Narval):
    sbatch scripts/slurm/deepeyes_treebench.sh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[32m"; RED = "\033[31m"; CYAN = "\033[36m"
W = 80

DEFAULT_N = 10
DEFAULT_OUTPUT = _PROJECT_ROOT / "results" / "treebench" / "deepeyes_treebench.jsonl"


# ---------------------------------------------------------------------------
# Public helpers (tested in tests/test_run_deepeyes_treebench.py)
# ---------------------------------------------------------------------------


def save_prediction(path: Path, row: Dict[str, Any]) -> None:
    """Append *row* as a JSON line to *path*, creating parent dirs as needed.

    Args:
        path: Destination JSONL file (appended, not overwritten).
        row:  Dict with at least ``question_id``, ``correct_answer``,
              ``predicted_letter``, and ``correct`` keys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    """Load all predictions from a JSONL checkpoint file.

    Returns an empty list if the file does not exist.

    Args:
        path: Path to the JSONL checkpoint.

    Returns:
        List of prediction dicts, one per non-blank line.
    """
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_accuracy(predictions: List[Dict[str, Any]]) -> float:
    """Compute fraction of correctly answered questions.

    Args:
        predictions: List of prediction dicts with a boolean ``correct`` field.

    Returns:
        Accuracy in [0, 1], or 0.0 for an empty list.
    """
    if not predictions:
        return 0.0
    return sum(bool(p["correct"]) for p in predictions) / len(predictions)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _bar(c: str = "─", w: int = W) -> str:
    return c * w


def _header(title: str) -> str:
    pad = (W - len(title) - 2) // 2
    return f"{BOLD}{'═' * pad} {title} {'═' * (W - pad - len(title) - 2)}{RESET}"


# ---------------------------------------------------------------------------
# Answer normalization for MCQ
# ---------------------------------------------------------------------------


def _normalize_mcq(raw: str) -> Optional[str]:
    """Extract an A/B/C/D letter from DeepEyesV2's answer string.

    Delegates to :func:`src.utils_normalize.normalize_answer`.
    If that returns ``None``, tries to extract the first bare letter.

    Args:
        raw: Raw answer string from the model.

    Returns:
        Uppercase letter in ``{A, B, C, D}``, or ``None`` if extraction fails.
    """
    from src.utils_normalize import normalize_answer
    return normalize_answer(raw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DeepEyesV2 on N TreeBench questions and report accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"Number of questions to evaluate (default: {DEFAULT_N}).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="JSONL checkpoint file (appended; skips already-completed question_ids).",
    )
    parser.add_argument(
        "--load-in-8bit", action=argparse.BooleanOptionalAction, default=True,
        help="Load model in 8-bit (default: true). Use --no-load-in-8bit for full precision.",
    )
    args = parser.parse_args()

    output_path: Path = args.output

    print(_header("DeepEyesV2 × TreeBench"))
    print(f"  Questions : {args.n}")
    print(f"  Checkpoint: {output_path}")
    print(f"  8-bit     : {args.load_in_8bit}")
    print()

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(_header("Loading TreeBench"))
    from src.data.datasets.treebench import TreeBenchDataset
    dataset = TreeBenchDataset(split="test", max_samples=args.n)
    dataset.load()
    total = len(dataset)
    print(f"  Loaded {total} questions from HaochenWang/TreeBench (test split).")
    print()

    # ── Resume from checkpoint ────────────────────────────────────────────────
    done_rows = load_predictions(output_path)
    done_ids = {r["question_id"] for r in done_rows}
    if done_ids:
        print(f"  {DIM}Resuming — {len(done_ids)} question(s) already done.{RESET}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print(_header("Loading DeepEyesV2"))
    from src.models.deepeyes_v2 import DeepEyesV2Model
    model = DeepEyesV2Model(load_in_8bit=args.load_in_8bit)
    model._load()
    print()

    # ── Inference loop ────────────────────────────────────────────────────────
    print(_header(f"Inference  (0 / {total})"))
    results: List[Dict[str, Any]] = list(done_rows)

    for idx in range(total):
        example = dataset[idx]
        qid = example["image_id"]

        if qid in done_ids:
            print(f"  {DIM}[{idx+1:>3}/{total}] {qid} — skipped (checkpoint){RESET}")
            continue

        question = example["question"]
        options: Dict[str, str] = example["options"]
        correct_answer: str = example["correct_answer"]

        # Append options to the question so the model sees them.
        opts_text = "\n".join(f"  {k}. {v}" for k, v in sorted(options.items()))
        full_question = f"{question}\n\nOptions:\n{opts_text}"

        print(f"  [{idx+1:>3}/{total}] {qid}  correct={correct_answer}")
        print(f"  {DIM}Q: {question[:80]}{'…' if len(question) > 80 else ''}{RESET}")

        t0 = time.time()
        chain = model.generate(example["image"], full_question, n=1)[0]
        elapsed = time.time() - t0

        raw_answer: str = chain["answer"]
        predicted_letter: Optional[str] = _normalize_mcq(raw_answer)
        is_correct = predicted_letter == correct_answer if predicted_letter else False

        mark = f"{GREEN}✓{RESET}" if is_correct else f"{RED}✗{RESET}"
        print(
            f"  {mark}  raw={raw_answer!r:.40}  "
            f"pred={predicted_letter}  expected={correct_answer}  "
            f"({elapsed:.1f}s  turns={len(chain['cot_steps'])})"
        )

        row: Dict[str, Any] = {
            "question_id": qid,
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "raw_answer": raw_answer,
            "predicted_letter": predicted_letter,
            "correct": is_correct,
            "elapsed_s": round(elapsed, 2),
            "cot_steps": len(chain["cot_steps"]),
            "tool_results": len(chain["tool_results"]),
        }
        save_prediction(output_path, row)
        results.append(row)
        done_ids.add(qid)
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(_header("Results"))
    acc = compute_accuracy(results)
    n_correct = sum(bool(r["correct"]) for r in results)
    n_total = len(results)
    print(f"  Accuracy : {acc:.1%}  ({n_correct} / {n_total})")
    print(f"  Output   : {output_path}")
    print()

    # Per-question summary
    print(f"  {'#':<5} {'ID':<15} {'Expected':<10} {'Predicted':<10} {'Result'}")
    print(f"  {_bar('─', W - 4)}")
    for i, r in enumerate(results, 1):
        mark = "✓" if r["correct"] else "✗"
        print(
            f"  {i:<5} {r['question_id']:<15} "
            f"{r['correct_answer']:<10} "
            f"{str(r['predicted_letter']):<10} "
            f"{mark}"
        )
    print(f"\n  {BOLD}Final accuracy: {acc:.1%}{RESET}\n")


if __name__ == "__main__":
    main()
