"""
TTS Scaling plots — accuracy vs number of candidates (n) for Qwen vs GRIT.

One subplot per benchmark task (VQA, Counting, OCR, TreeBench).
X-axis: n  (baseline + TTS majority@1,3,5,7,9)
Y-axis: accuracy

Run from repo root:
    python scripts/plot_tts_scaling.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/comparison"

QWEN_COLOR = "#4C72B0"
GRIT_COLOR = "#DD8452"

plt.rcParams.update({
    "figure.dpi": 180,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── run groups ────────────────────────────────────────────────────────────────
# Each entry: (label, qwen_preds_per_task, grit_preds_per_task)
# TreeBench round 3 has no Qwen run → None means skip that bar.
ROUNDS = [
    {
        "label": "Round 1",
        "Qwen": {
            "VQA":       [ROOT / "results/tts_eval/qwen_vqa/predictions.jsonl"],
            "Counting":  [ROOT / "results/tts_eval/qwen_counting/predictions.jsonl"],
            "OCR":       [ROOT / "results/tts_eval/qwen_ocr/predictions.jsonl"],
            "TreeBench": [ROOT / "results/tts_eval/qwen_run1/predictions.jsonl"],
        },
        "GRIT": {
            "VQA":       [ROOT / "results/tts_eval/grit_vqa/predictions.jsonl"],
            "Counting":  [ROOT / "results/tts_eval/grit_counting/predictions.jsonl"],
            "OCR":       [ROOT / "results/tts_eval/grit_ocr/predictions.jsonl"],
            "TreeBench": [ROOT / "results/tts_eval/grit_run1/predictions.jsonl"],
        },
    },
    {
        "label": "Round 2",
        "Qwen": {
            "VQA":       [ROOT / "results/tts_eval/qwen_vqa2/predictions.jsonl"],
            "Counting":  [ROOT / "results/tts_eval/qwen_counting2/predictions.jsonl"],
            "OCR":       [ROOT / "results/tts_eval/qwen_ocr2/predictions.jsonl"],
            "TreeBench": [ROOT / "results/tts_eval/qwen_run2/predictions.jsonl"],
        },
        "GRIT": {
            "VQA":       [ROOT / "results/tts_eval/grit_vqa2/predictions.jsonl"],
            "Counting":  [ROOT / "results/tts_eval/grit_counting2/predictions.jsonl"],
            "OCR":       [ROOT / "results/tts_eval/grit_ocr2/predictions.jsonl"],
            "TreeBench": [ROOT / "results/tts_eval/grit_run2/predictions.jsonl"],
        },
    },
    {
        "label": "Round 3",
        "Qwen": {
            "VQA":       [ROOT / "results/tts_eval/qwen_vqa1/predictions.jsonl"],
            "Counting":  [ROOT / "results/tts_eval/qwen_counting1/predictions.jsonl"],
            "OCR":       [ROOT / "results/tts_eval/qwen_ocr1/predictions.jsonl"],
            "TreeBench": [],   # no qwen_run3
        },
        "GRIT": {
            "VQA":       [ROOT / "results/tts_eval/grit_vqa1/predictions.jsonl"],
            "Counting":  [ROOT / "results/tts_eval/grit_counting1/predictions.jsonl"],
            "OCR":       [ROOT / "results/tts_eval/grit_ocr1/predictions.jsonl"],
            "TreeBench": [ROOT / "results/tts_eval/grit_run3/predictions.jsonl"],
        },
    },
]

TASKS    = ["VQA", "Counting", "OCR", "TreeBench"]
N_VALUES = [1, 3, 5, 7, 9]


# ── loaders ───────────────────────────────────────────────────────────────────
def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _load_combined(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if p.exists():
            rows.extend(_load_jsonl(p))
    return rows


# ── accuracy helpers ──────────────────────────────────────────────────────────
def _baseline_acc(preds: list[dict]) -> float:
    """Accuracy of the single-run baseline (original image + original prompt)."""
    return sum(1 for p in preds if p["baseline"]["is_correct"]) / len(preds)


def _majority_n_acc(arts: list[dict], n: int) -> float:
    """
    Majority@n accuracy: for each question take the first n candidates
    (sorted by candidate_id), majority-vote their normalized answers, and
    check if the winning answer is correct.
    """
    by_q: dict[str, list[dict]] = defaultdict(list)
    for a in arts:
        by_q[a["sample_id"]].append(a)

    correct = 0
    total   = 0
    for cands in by_q.values():
        # sort deterministically by candidate_id (int)
        sorted_cands = sorted(cands, key=lambda c: int(c["candidate_id"]))
        subset       = sorted_cands[:n]
        winner       = Counter(c["normalized_answer"] for c in subset).most_common(1)[0][0]
        # winning answer is correct if any candidate with that answer has is_correct=True
        if any(c["normalized_answer"] == winner and c["is_correct"] for c in subset):
            correct += 1
        total += 1

    return correct / total if total else 0.0


# ── helpers ───────────────────────────────────────────────────────────────────
def _acc(paths: list[Path]) -> float | None:
    preds = _load_combined(paths)
    if not preds:
        return None
    return sum(1 for p in preds if p["tts"]["is_correct"]) / len(preds)


def _draw_bars(ax: plt.Axes, qwen_acc: list, grit_acc: list, title: str) -> None:
    x     = np.arange(len(TASKS))
    width = 0.32

    for offset, accs, color, label in [
        (-width / 2, qwen_acc, QWEN_COLOR, "Qwen2.5-VL 3B"),
        ( width / 2, grit_acc, GRIT_COLOR, "GRIT 3B"),
    ]:
        # filter out None entries (missing runs)
        xs = [x[i] + offset for i, v in enumerate(accs) if v is not None]
        ys = [v              for v in accs               if v is not None]
        bars = ax.bar(xs, ys, width, label=label, color=color, alpha=0.90)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=9, framealpha=0.3)


# ── main figure ───────────────────────────────────────────────────────────────
def figure_all_rounds() -> None:
    fig, axes = plt.subplots(1, len(ROUNDS), figsize=(9 * len(ROUNDS), 5), sharey=True)

    for ax, rnd in zip(axes, ROUNDS):
        qwen_acc = [_acc(rnd["Qwen"][t]) for t in TASKS]
        grit_acc = [_acc(rnd["GRIT"][t]) for t in TASKS]
        _draw_bars(ax, qwen_acc, grit_acc, rnd["label"])

    fig.tight_layout()
    out = OUT_DIR / "figD_tts_all_rounds.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def figure_session_03_17() -> None:
    """Single bar chart for the 03-17 session: grit_run2 + qwen_run1 for TreeBench."""
    runs = {
        "Qwen": {
            "VQA":       ROOT / "results/tts_eval/qwen_vqa/predictions.jsonl",
            "Counting":  ROOT / "results/tts_eval/qwen_counting/predictions.jsonl",
            "OCR":       ROOT / "results/tts_eval/qwen_ocr/predictions.jsonl",
            "TreeBench": ROOT / "results/tts_eval/qwen_run1/predictions.jsonl",
        },
        "GRIT": {
            "VQA":       ROOT / "results/tts_eval/grit_vqa/predictions.jsonl",
            "Counting":  ROOT / "results/tts_eval/grit_counting/predictions.jsonl",
            "OCR":       ROOT / "results/tts_eval/grit_ocr/predictions.jsonl",
            "TreeBench": ROOT / "results/tts_eval/grit_run2/predictions.jsonl",
        },
    }

    tasks = ["VQA", "Counting", "OCR", "TreeBench"]
    qwen_acc = [_acc([runs["Qwen"][t]]) for t in tasks]
    grit_acc = [_acc([runs["GRIT"][t]]) for t in tasks]

    x     = np.arange(len(tasks))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_q = ax.bar(x - width / 2, qwen_acc, width, label="Qwen2.5-VL 3B", color=QWEN_COLOR, alpha=0.90)
    bars_g = ax.bar(x + width / 2, grit_acc, width, label="GRIT 3B",        color=GRIT_COLOR, alpha=0.90)

    for bars in (bars_q, bars_g):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=10, framealpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "figE_session_03_17.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    figure_session_03_17()
    print("Done.")
