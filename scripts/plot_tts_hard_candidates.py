#!/usr/bin/env python3
"""Candidate-level correctness analysis for TTS_Hard.json.

For each question × candidate, checks whether the candidate's normalized
answer is correct.  Produces two figures:

  1. Heatmap grid (questions × 9 candidates)
       green  = candidate correct
       red    = candidate wrong
       last column shows majority_9 outcome

  2. Summary bar chart per model × task
       - baseline    : 0% (all questions had baseline_correct=False)
       - majority_9  : fraction where majority vote was correct
       - oracle_9    : fraction where ≥1 candidate was correct
                       (upper bound achievable with any voting mechanism)

Usage:
    python scripts/plot_tts_hard_candidates.py
    python scripts/plot_tts_hard_candidates.py --input results/tts/TTS_Hard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

INPUT_PATH  = _PROJECT_ROOT / "results" / "tts" / "TTS_Hard.json"
OUTPUT_DIR  = _PROJECT_ROOT / "results" / "tts"

TASK_LABELS    = ["vqa", "counting", "ocr"]
TASK_DISPLAY   = {"vqa": "VQA", "counting": "Counting", "ocr": "OCR"}
MODEL_COLORS   = ["#4C72B0", "#DD8452"]


# ---------------------------------------------------------------------------
# Per-candidate correctness
# ---------------------------------------------------------------------------

def _is_correct(norm_answer: Optional[str], references: List[str]) -> bool:
    """Check if a normalized candidate answer matches any reference."""
    from src.eval.vqa_eval import evaluate_vqa
    if not norm_answer:
        return False
    return evaluate_vqa(norm_answer, references)


def _candidate_correctness(entry: Dict[str, Any]) -> List[bool]:
    """Return a bool list of length 9: True if candidate[i] was correct."""
    refs  = entry["references"]
    norms = entry.get("candidate_answers_normalized", [])
    return [_is_correct(n, refs) for n in norms]


# ---------------------------------------------------------------------------
# Figure 1 — heatmap grid
# ---------------------------------------------------------------------------

def _plot_heatmap(
    entries: List[Dict[str, Any]],
    model_label: str,
    task: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping heatmap.")
        return

    n_q   = len(entries)
    n_c   = 9
    grid  = np.zeros((n_q, n_c), dtype=float)

    for i, entry in enumerate(entries):
        corr = _candidate_correctness(entry)
        for j, c in enumerate(corr[:n_c]):
            grid[i, j] = 1.0 if c else 0.0

    majority_correct = [bool(entry.get("correct", False)) for entry in entries]

    fig_h = max(4, n_q * 0.35 + 2)
    fig, (ax_grid, ax_maj) = plt.subplots(
        1, 2,
        figsize=(11, fig_h),
        gridspec_kw={"width_ratios": [n_c, 1], "wspace": 0.05},
    )

    # ── Heatmap ──────────────────────────────────────────────────────────────
    cmap = plt.cm.colors.ListedColormap(["#d73027", "#1a9850"])  # type: ignore[attr-defined]
    ax_grid.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax_grid.set_xticks(range(n_c))
    ax_grid.set_xticklabels([f"C{i+1}" for i in range(n_c)], fontsize=8)
    ax_grid.set_yticks(range(n_q))
    ax_grid.set_yticklabels([entry["question_id"][:12] for entry in entries], fontsize=6)
    ax_grid.set_xlabel("Candidate", fontsize=9)
    ax_grid.set_ylabel("Question", fontsize=9)
    ax_grid.set_title(
        f"{model_label}  —  {TASK_DISPLAY.get(task, task.upper())}\n"
        "Candidate correctness  (green=✓  red=✗)",
        fontsize=10, fontweight="bold",
    )

    # Draw grid lines
    for x in np.arange(-0.5, n_c, 1):
        ax_grid.axvline(x, color="white", linewidth=0.5)
    for y in np.arange(-0.5, n_q, 1):
        ax_grid.axhline(y, color="white", linewidth=0.5)

    # Count per question (right edge labels)
    for i, entry in enumerate(entries):
        cnt = int(sum(_candidate_correctness(entry)[:n_c]))
        ax_grid.text(
            n_c - 0.5, i, f" {cnt}/9",
            va="center", ha="left", fontsize=6, color="black",
        )

    # ── Majority column ───────────────────────────────────────────────────────
    maj_grid = np.array([[1.0 if c else 0.0] for c in majority_correct])
    ax_maj.imshow(maj_grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax_maj.set_xticks([0])
    ax_maj.set_xticklabels(["Maj@9"], fontsize=8)
    ax_maj.set_yticks([])
    ax_maj.set_title("Vote", fontsize=9)
    for y in np.arange(-0.5, n_q, 1):
        ax_maj.axhline(y, color="white", linewidth=0.5)

    # Legend
    green_p = mpatches.Patch(color="#1a9850", label="Correct")
    red_p   = mpatches.Patch(color="#d73027", label="Wrong")
    ax_grid.legend(handles=[green_p, red_p], loc="lower right", fontsize=8,
                   bbox_to_anchor=(1.0, -0.12), ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"  Heatmap saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — summary bar chart
# ---------------------------------------------------------------------------

def _plot_summary(
    all_stats: Dict[str, Dict[str, Dict[str, float]]],
    model_labels: List[str],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping summary plot.")
        return

    plt.rcParams.update({
        "figure.dpi":        150,
        "font.size":         11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    n_tasks = len(TASK_LABELS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4.5), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    fig.suptitle(
        "TTS on Baseline Failures — Majority@9 vs Oracle@9\n"
        "(oracle = correct answer present in ≥1 candidate)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    X_KEYS   = ["majority_9", "oracle_9"]
    X_LABELS = ["Majority@9\n(actual)", "Oracle@9\n(upper bound)"]
    x     = np.arange(len(X_KEYS))
    width = 0.35

    for ax, task in zip(axes, TASK_LABELS):
        for model_idx, (ml, color) in enumerate(zip(model_labels, MODEL_COLORS)):
            stats = all_stats.get(ml, {}).get(task, {})
            if not stats:
                continue
            accs   = [stats.get(k, 0.0) for k in X_KEYS]
            short  = ml.split(" ")[0]
            offset = (model_idx - 0.5) * width
            bars   = ax.bar(x + offset, accs, width, label=short, color=color, alpha=0.88)
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.01,
                    f"{h:.0%}",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_title(TASK_DISPLAY.get(task, task.upper()), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Recovery rate" if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Summary plot saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def _print_table(
    all_stats: Dict[str, Dict[str, Dict[str, float]]],
    model_labels: List[str],
) -> None:
    W = 100
    col_w = 42
    print("\n" + "═" * W)
    print(f"  {'TTS Hard — Candidate Correctness Analysis':^{W-4}}")
    print("═" * W)
    header = f"  {'Model':<24}" + "".join(f"{TASK_DISPLAY.get(t,t):>{col_w}}" for t in TASK_LABELS)
    print(header)
    print("  " + "─" * (W - 4))

    for ml in model_labels:
        parts = [f"  {ml:<24}"]
        for task in TASK_LABELS:
            s = all_stats.get(ml, {}).get(task, {})
            if not s:
                parts.append(f"{'N/A':>{col_w}}")
                continue
            n        = s["n"]
            maj      = s["majority_9"]
            oracle   = s["oracle_9"]
            gap      = oracle - maj
            cell = f"n={int(n)}  maj={maj:.0%}  oracle={oracle:.0%}  gap={gap:+.0%}"
            parts.append(f"{cell:>{col_w}}")
        print("".join(parts))

    print("  " + "─" * (W - 4))
    print(
        "\n  maj    = fraction of failure-questions recovered by majority@9\n"
        "  oracle = fraction where correct answer appeared in ≥1 candidate\n"
        "  gap    = votes lost to noise (oracle − majority)\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Candidate-level correctness analysis of TTS_Hard.json."
    )
    parser.add_argument("--input", default=str(INPUT_PATH),
                        help="TTS_Hard.json path.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help="Directory for output plots.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run run_tts_hard.py first.")
        sys.exit(1)

    data: Dict[str, Any] = json.loads(input_path.read_text(encoding="utf-8"))
    model_labels = list(data.keys())

    all_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for ml in model_labels:
        all_stats[ml] = {}
        for task in TASK_LABELS:
            entries: List[Dict[str, Any]] = data.get(ml, {}).get(task, [])
            if not entries:
                continue

            n          = len(entries)
            n_majority = sum(bool(e.get("correct", False)) for e in entries)
            n_oracle   = sum(
                any(_candidate_correctness(e)) for e in entries
            )

            all_stats[ml][task] = {
                "n":          float(n),
                "majority_9": n_majority / n,
                "oracle_9":   n_oracle   / n,
            }

            # Per-task heatmap
            safe_label = ml.replace("/", "-").replace(" ", "_").replace("(", "").replace(")", "")
            heatmap_path = output_dir / f"tts_hard_candidates_{safe_label}_{task}.png"
            print(f"\n{ml} / {task.upper()}: {n} questions")
            print(f"  majority@9 = {n_majority}/{n} ({n_majority/n:.0%})")
            print(f"  oracle@9   = {n_oracle}/{n}   ({n_oracle/n:.0%})")
            _plot_heatmap(entries, ml, task, heatmap_path)

    _print_table(all_stats, model_labels)
    _plot_summary(all_stats, model_labels, output_dir / "tts_hard_summary.png")


if __name__ == "__main__":
    main()
