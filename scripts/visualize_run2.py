"""Visualise run2_judged.json: accuracy, LLM-judge accuracy, and compute time.

Usage:
    python scripts/visualize_run2.py
    python scripts/visualize_run2.py --input  results/comparison/run2_judged.json
    python scripts/visualize_run2.py --output results/comparison/run2_plot.png
    python scripts/visualize_run2.py --dpi 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "VisCoT (7B)",
    "Qwen2.5-VL (7B, no CoT)",
    "DeepEyesV2-RL (7B)",
    "GRIT (3B)",
]
MODEL_SHORT = [
    "VisCoT\n(7B)",
    "Qwen2.5-VL\n(7B, no CoT)",
    "DeepEyesV2-RL\n(7B)",
    "GRIT\n(3B)",
]
TASKS = ["vqa", "counting", "ocr"]
TASK_LABELS = ["VQA", "Counting", "OCR"]

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]   # one colour per model
BG      = "#F7F7F7"
GRID_C  = "#E0E0E0"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _agg(entries: list[dict]) -> dict:
    """Aggregate a list of entries into summary statistics."""
    n = len(entries)
    if n == 0:
        return dict(exact=float("nan"), judge=float("nan"),
                    mean_s=float("nan"), total_s=0.0, n=0)
    exact = sum(bool(e["correct"])   for e in entries) / n
    judge = sum(bool(e["llm_judge"]) for e in entries) / n
    mean_s  = np.mean([e["elapsed_s"] for e in entries])
    total_s = sum(e["elapsed_s"] for e in entries)
    return dict(exact=exact, judge=judge, mean_s=mean_s, total_s=total_s, n=n)


def build_stats(data: dict) -> dict:
    """Return stats[model][task | 'overall']."""
    stats: dict = {}
    for model in MODEL_ORDER:
        mdata = data.get(model, {})
        stats[model] = {}
        all_entries: list = []
        for task in TASKS:
            entries = mdata.get(task, [])
            stats[model][task] = _agg(entries)
            all_entries.extend(entries)
        stats[model]["overall"] = _agg(all_entries)
    return stats


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax: plt.Axes, title: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID_C, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#AAAAAA")
    ax.spines["bottom"].set_color("#AAAAAA")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, labelpad=6)


def _bar_accuracy_panel(
    ax: plt.Axes,
    stats: dict,
    task_key: str,
    title: str,
) -> None:
    """Draw grouped bars (exact match + llm judge) for one task panel."""
    n_models = len(MODEL_ORDER)
    x = np.arange(n_models)
    w = 0.32

    for i, model in enumerate(MODEL_ORDER):
        s = stats[model][task_key]
        # Exact match bar
        ax.bar(x[i] - w / 2, s["exact"], width=w,
               color=PALETTE[i], alpha=0.92, zorder=3,
               linewidth=0.8, edgecolor="white")
        # LLM-judge bar (same colour, hatched)
        ax.bar(x[i] + w / 2, s["judge"], width=w,
               color=PALETTE[i], alpha=0.50, zorder=3,
               hatch="////", linewidth=0.8, edgecolor="white")
        # Value labels
        for val, offset in [(s["exact"], -w / 2), (s["judge"], +w / 2)]:
            if not np.isnan(val):
                ax.text(x[i] + offset, val + 0.018, f"{val:.0%}",
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color="#333333")

    _style_ax(ax, title=title, ylabel="Accuracy")
    ax.set_ylim(0, 1.13)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_SHORT, fontsize=9)
    ax.tick_params(axis="x", length=0)


def _time_panel(ax: plt.Axes, stats: dict) -> None:
    """Horizontal lollipop chart of mean elapsed time per model × task."""
    n_tasks = len(TASKS)
    n_models = len(MODEL_ORDER)
    y_step = n_tasks + 1.5          # gap between model groups
    task_offsets = np.arange(n_tasks) - (n_tasks - 1) / 2  # centred

    ytick_positions: list[float] = []
    ytick_labels:    list[str]   = []

    for mi, model in enumerate(MODEL_ORDER):
        group_centre = mi * y_step
        ytick_positions.append(group_centre)
        ytick_labels.append(MODEL_SHORT[mi])

        for ti, task in enumerate(TASKS):
            y = group_centre + task_offsets[ti]
            val = stats[model][task]["mean_s"]
            if np.isnan(val):
                continue
            color = PALETTE[mi]
            alpha = 0.55 + 0.15 * ti  # 0.55 / 0.70 / 0.85

            # Stem line
            ax.plot([0, val], [y, y], color=color, lw=1.6, alpha=alpha, zorder=2)
            # Dot
            ax.scatter([val], [y], color=color, s=80, zorder=4, alpha=min(alpha + 0.1, 1.0))
            # Task label on the dot
            ax.text(val * 1.08, y, TASK_LABELS[ti],
                    va="center", fontsize=8, color=color,
                    fontweight="bold", alpha=min(alpha + 0.1, 1.0))
            # Time label
            unit = "s" if val < 60 else "min"
            disp = val if val < 60 else val / 60
            ax.text(val * 1.08 + 7, y, f"({disp:.1f}{unit})",
                    va="center", fontsize=7.5, color="#666666")

    ax.set_xscale("log")
    ax.set_xlabel("Mean inference time per sample (log scale)", fontsize=10)
    ax.set_xlim(left=0.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f}s" if v < 60 else f"{v/60:.1f}min"
    ))
    _style_ax(ax, title="Computation Time per Sample")
    ax.grid(axis="x", color=GRID_C, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/comparison/run2_judged.json")
    parser.add_argument("--output", default="results/comparison/run2_plot.png")
    parser.add_argument("--dpi",    type=int, default=180)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = build_stats(data)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 11), facecolor="white")
    fig.suptitle(
        "Model Comparison — Accuracy & Compute Cost",
        fontsize=17, fontweight="bold", y=0.98,
    )
    gs = GridSpec(
        2, 4,
        figure=fig,
        height_ratios=[1, 1.1],
        hspace=0.52,
        wspace=0.38,
        left=0.06, right=0.97, top=0.91, bottom=0.08,
    )

    # Row 0: four accuracy panels (VQA, Counting, OCR, Overall)
    panels = [
        (gs[0, 0], "vqa",     "VQA"),
        (gs[0, 1], "counting","Counting"),
        (gs[0, 2], "ocr",     "OCR"),
        (gs[0, 3], "overall", "Overall"),
    ]
    for spec, key, title in panels:
        ax = fig.add_subplot(spec)
        _bar_accuracy_panel(ax, stats, key, title)
        if key != "vqa":
            ax.set_ylabel("")

    # Row 1: computation time (spans all 4 columns)
    ax_time = fig.add_subplot(gs[1, :])
    _time_panel(ax_time, stats)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = []
    for i, model in enumerate(MODEL_ORDER):
        legend_handles.append(Patch(facecolor=PALETTE[i], alpha=0.92, label=model))
    legend_handles += [
        Patch(facecolor="#888888", alpha=0.92, label="Exact-match accuracy"),
        Patch(facecolor="#888888", alpha=0.50, hatch="////",
              label="LLM-judge accuracy"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        fontsize=8.8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, 0.955),
    )

    # ── Annotation: agreement between exact match and LLM judge ──────────────
    agree_counts = {}
    for model in MODEL_ORDER:
        mdata = data.get(model, {})
        total = agree = 0
        for task in TASKS:
            for e in mdata.get(task, []):
                total += 1
                if bool(e["correct"]) == bool(e["llm_judge"]):
                    agree += 1
        agree_counts[model] = agree / total if total else float("nan")

    annot_lines = ["Agreement (exact vs LLM-judge):  "]
    for model, short in zip(MODEL_ORDER, MODEL_SHORT):
        annot_lines.append(f"{model.split('(')[0].strip()}: {agree_counts[model]:.0%}   ")
    fig.text(
        0.5, 0.005,
        "  |  ".join(annot_lines),
        ha="center", va="bottom",
        fontsize=8, color="#555555",
        style="italic",
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
