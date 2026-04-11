"""Step 1 — Stochasticity Pilot.

For each model × task × question (N=10, temperature=0.7) compute the
Shannon entropy of the answer distribution to test whether CoT depth
correlates with output diversity.

Usage
-----
Run one model at a time (each takes 1-3h on a single A100):

    python experiments/run_stochasticity_pilot.py --model qwen
    python experiments/run_stochasticity_pilot.py --model grit
    python experiments/run_stochasticity_pilot.py --model deepeyes

After all three finish, generate figures:

    python experiments/run_stochasticity_pilot.py --summarize

Results are written (with checkpoint/resume) to:
    results/stochasticity/entropy_<model>_<task>.jsonl
    results/stochasticity/entropy_summary.json
    results/stochasticity/entropy_bar_chart.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets.viscot_benchmark import load_task
from src.eval.stochasticity import compute_entropy, entropy_summary
from src.utils.logging import get_logger
from src.utils_normalize import normalize_answer

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASKS = ["vqa", "ocr", "counting"]
N_QUESTIONS = 10   # questions per task
N_SAMPLES = 10     # independent samples per question
TEMPERATURE = 0.7

MODELS = {
    "qwen": {
        "class": "DirectVLMModel",
        "module": "src.models.direct_vlm",
        "label": "Qwen2.5-VL-3B",
    },
    "grit": {
        "class": "GRITModel",
        "module": "src.models.grit",
        "label": "GRIT-3B",
    },
    "deepeyes": {
        "class": "DeepEyesV2Model",
        "module": "src.models.deepeyes_v2",
        "label": "DeepEyesV2-7B",
    },
}

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "stochasticity"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_key: str):
    """Dynamically import and instantiate the model class for *model_key*."""
    import importlib
    cfg = MODELS[model_key]
    mod = importlib.import_module(cfg["module"])
    cls = getattr(mod, cfg["class"])
    return cls()


def _result_path(model_key: str, task: str) -> Path:
    return OUT_DIR / f"entropy_{model_key}_{task}.jsonl"


def _load_checkpoint(model_key: str, task: str) -> set[str]:
    """Return set of question_ids already processed (for resume)."""
    path = _result_path(model_key, task)
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            done.add(row["question_id"])
    return done


def _append_row(model_key: str, task: str, row: dict) -> None:
    path = _result_path(model_key, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_model(model_key: str) -> None:
    """Run the stochasticity pilot for *model_key* across all tasks."""
    logger.info("Loading model '{}'…", model_key)
    model = _load_model(model_key)

    for task in TASKS:
        logger.info("Task: {}", task)
        examples = load_task(task, n=N_QUESTIONS)
        done_ids = _load_checkpoint(model_key, task)
        if done_ids:
            logger.info(
                "  Resuming — {} / {} questions already done.", len(done_ids), len(examples)
            )

        for ex in examples:
            qid = ex["question_id"]
            if qid in done_ids:
                continue

            image = ex["image"]
            question = ex["question"]
            gt = ex["answer"]

            logger.info("  [{}/{}] qid={}", task, N_QUESTIONS, qid)
            chains = model.generate(
                image,
                question,
                n=N_SAMPLES,
                temperature=TEMPERATURE,
            )

            raw_answers = [c["answer"] for c in chains]
            norm_answers = [normalize_answer(a) for a in raw_answers]
            h = compute_entropy(norm_answers)

            row = {
                "question_id": qid,
                "task": task,
                "model": model_key,
                "correct_answer": gt,
                "raw_answers": raw_answers,
                "norm_answers": norm_answers,
                "entropy": h,
            }
            _append_row(model_key, task, row)
            logger.info("    entropy={:.3f}  answers={}", h, norm_answers)


# ---------------------------------------------------------------------------
# Summarize + figure
# ---------------------------------------------------------------------------

def summarize() -> None:
    """Aggregate all JSONL files into a summary JSON and bar-chart figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    all_rows: list[dict] = []
    for model_key in MODELS:
        for task in TASKS:
            path = _result_path(model_key, task)
            if not path.exists():
                logger.warning("Missing: {}", path)
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    all_rows.append(json.loads(line))

    # Mean entropy per (model, task)
    from collections import defaultdict
    buckets: dict[tuple, list] = defaultdict(list)
    for row in all_rows:
        buckets[(row["model"], row["task"])].append(row["entropy"])

    summary: dict = {}
    for (model_key, task), vals in buckets.items():
        summary.setdefault(model_key, {})[task] = sum(vals) / len(vals)

    summary_path = OUT_DIR / "entropy_summary.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary -> {}", summary_path)

    # Bar chart: x=task, grouped bars per model
    model_keys = list(MODELS.keys())
    task_labels = TASKS
    x = np.arange(len(task_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model_key in enumerate(model_keys):
        heights = [summary.get(model_key, {}).get(t, 0.0) for t in task_labels]
        ax.bar(x + i * width, heights, width, label=MODELS[model_key]["label"])

    ax.set_xlabel("Task")
    ax.set_ylabel("Mean Shannon Entropy (bits)")
    ax.set_title("Answer Entropy by Model and Task\n(higher = more diverse outputs)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_labels)
    ax.legend()
    ax.set_ylim(0, 3.5)  # max possible = log2(10) ≈ 3.32 bits

    chart_path = OUT_DIR / "entropy_bar_chart.png"
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart -> {}", chart_path)

    # Print go/no-go verdict
    print("\n=== GO / NO-GO ===")
    print(f"{'Task':<12} {'Qwen':>10} {'GRIT':>10} {'DeepEyes':>10}  Rank OK?")
    rank_ok = 0
    for task in TASKS:
        h_qwen = summary.get("qwen", {}).get(task, float("nan"))
        h_grit = summary.get("grit", {}).get(task, float("nan"))
        h_deep = summary.get("deepeyes", {}).get(task, float("nan"))
        ok = h_deep > h_grit > h_qwen
        rank_ok += int(ok)
        print(f"{task:<12} {h_qwen:>10.3f} {h_grit:>10.3f} {h_deep:>10.3f}  {'OK' if ok else 'FAIL'}")
    verdict = "GO" if rank_ok >= 2 else "NO-GO"
    print(f"\nVerdict: {verdict} ({rank_ok}/3 tasks rank correctly)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stochasticity pilot experiment.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Run inference for this model.",
    )
    group.add_argument(
        "--summarize",
        action="store_true",
        help="Aggregate results and generate figures (no inference).",
    )
    args = parser.parse_args()

    if args.summarize:
        summarize()
    else:
        run_model(args.model)


if __name__ == "__main__":
    main()
