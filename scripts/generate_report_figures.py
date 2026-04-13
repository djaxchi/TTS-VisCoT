"""Generate all figures for REPORT.md.

Reads results from results/tts_hard_bench/ and results/tts_hard_bench_t0/,
plus hardcoded Study A entropy data, and produces 5 figures:

  fig5  — Run 1: Greedy vs @9 vs Oracle@9 (grouped bars, both models)
  fig6  — Run 1 vs Run 2: TTS gain comparison (grouped bars, both models)
  fig7  — Run 1 vs Run 2: delta chart per model x task
  fig8  — Study A: GRIT vs Qwen3B entropy (paired bars)
  fig9  — Oracle gap: @9 vs Oracle@9 (horizontal bars)

Output: results/report_figures/fig{5..9}_*.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("results/report_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (matches entropy pilot figures) ──────────────────────────
C_QWEN = "#5B9BD5"   # blue
C_GRIT = "#ED7D31"   # orange
C_DEEP = "#70AD47"   # green (for future use)

TASKS = ["VQA", "OCR", "Counting"]

# ── Hardcoded accuracy tables from the report ───────────────────────────────
# Run 1 (standard recipe, mixed T=0 + T=0.7)
run1 = {
    "Qwen3B": {
        "greedy":   [30.0,  3.3, 36.7],
        "@9":       [20.0,  3.3, 36.7],
        "oracle@9": [53.3,  6.7, 93.3],
    },
    "GRIT": {
        "greedy":   [10.0, 26.7, 33.3],
        "@9":       [13.3, 30.0, 30.0],
        "oracle@9": [63.3, 40.0, 73.3],
    },
}

# Run 2 (T=0 ablation, image-only diversity)
run2 = {
    "Qwen3B": {
        "greedy":   [30.0, 30.0, 36.7],
        "@9":       [30.0, 30.0, 30.0],
        "oracle@9": [50.0, 36.7, 70.0],
    },
    "GRIT": {
        "greedy":   [10.0, 26.7, 33.3],
        "@9":       [13.3, 30.0, 33.3],
        "oracle@9": [56.7, 40.0, 66.7],
    },
}

# Study A entropy (bits)
study_a = {
    "Qwen3B": [1.210, 1.881, 1.023],
    "GRIT":   [0.951, 1.410, 0.791],
}


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 5 — Run 1: Greedy vs @9 vs Oracle@9
# ════════════════════════════════════════════════════════════════════════════
def fig5() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bar_w = 0.25
    x = np.arange(len(TASKS))

    for ax, model in zip(axes, ["Qwen3B", "GRIT"]):
        d = run1[model]
        b1 = ax.bar(x - bar_w, d["greedy"],   bar_w, label="Greedy",   color="#4472C4")
        b2 = ax.bar(x,         d["@9"],        bar_w, label="@9 (vote)", color="#ED7D31")
        b3 = ax.bar(x + bar_w, d["oracle@9"],  bar_w, label="Oracle@9", color="#A5A5A5")

        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(TASKS)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{model}", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Figure 5 — Run 1: Greedy vs Majority@9 vs Oracle@9\n"
                 "(9 candidates, mixed temperature + image augmentations, 30 questions/task)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, "fig5_run1_accuracy.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 6 — Side-by-side Run 1 vs Run 2 TTS gain
# ════════════════════════════════════════════════════════════════════════════
def fig6() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bar_w = 0.30
    x = np.arange(len(TASKS))

    for ax, model in zip(axes, ["Qwen3B", "GRIT"]):
        gain1 = [run1[model]["@9"][i] - run1[model]["greedy"][i] for i in range(3)]
        gain2 = [run2[model]["@9"][i] - run2[model]["greedy"][i] for i in range(3)]

        b1 = ax.bar(x - bar_w / 2, gain1, bar_w, label="Run 1 (T=0.7 + aug)", color="#4472C4")
        b2 = ax.bar(x + bar_w / 2, gain2, bar_w, label="Run 2 (T=0 + aug)", color="#ED7D31")

        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                va = "bottom" if h >= 0 else "top"
                offset = 0.5 if h >= 0 else -0.5
                ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                        f"{h:+.1f}", ha="center", va=va, fontsize=9, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(TASKS)
        ax.set_ylabel("TTS gain (pp): @9 minus greedy")
        ax.set_title(f"{model}", fontsize=13, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.set_ylim(-15, 10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Figure 6 — TTS gain: Run 1 (mixed diversity) vs Run 2 (image-only diversity)\n"
                 "(positive = TTS helps, negative = TTS hurts; 30 questions/task)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, "fig6_run1_vs_run2_tts_gain.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 7 — Delta comparison: Run 2 gain minus Run 1 gain
# ════════════════════════════════════════════════════════════════════════════
def fig7() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_w = 0.30
    labels = [f"{m}\n{t}" for m in ["Qwen3B", "GRIT"] for t in TASKS]
    x = np.arange(len(labels))

    deltas = []
    colors = []
    for model in ["Qwen3B", "GRIT"]:
        for i in range(3):
            gain1 = run1[model]["@9"][i] - run1[model]["greedy"][i]
            gain2 = run2[model]["@9"][i] - run2[model]["greedy"][i]
            d = gain2 - gain1
            deltas.append(d)
            colors.append(C_QWEN if model == "Qwen3B" else C_GRIT)

    bars = ax.bar(x, deltas, bar_w * 1.5, color=colors)
    for bar, d in zip(bars, deltas):
        va = "bottom" if d >= 0 else "top"
        offset = 0.3 if d >= 0 else -0.3
        ax.text(bar.get_x() + bar.get_width() / 2, d + offset,
                f"{d:+.1f}", ha="center", va=va, fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Change in TTS gain (pp): Run 2 minus Run 1")
    ax.set_ylim(-12, 15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_QWEN, label="Qwen3B (direct)"),
                       Patch(color=C_GRIT, label="GRIT (visual CoT)")],
              loc="upper right", fontsize=10)

    fig.suptitle("Figure 7 — Effect of removing temperature: change in TTS gain (Run 2 minus Run 1)\n"
                 "(positive = removing temperature helped; 30 questions/task)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, "fig7_delta_run2_minus_run1.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 8 — Study A: GRIT vs Qwen3B entropy
# ════════════════════════════════════════════════════════════════════════════
def fig8() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_w = 0.30
    x = np.arange(len(TASKS))

    b1 = ax.bar(x - bar_w / 2, study_a["Qwen3B"], bar_w, label="Qwen3B (direct)", color=C_QWEN)
    b2 = ax.bar(x + bar_w / 2, study_a["GRIT"],   bar_w, label="GRIT (visual CoT)", color=C_GRIT)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Add delta annotations
    for i in range(3):
        delta = study_a["GRIT"][i] - study_a["Qwen3B"][i]
        mid_x = x[i]
        max_h = max(study_a["Qwen3B"][i], study_a["GRIT"][i])
        ax.text(mid_x, max_h + 0.15, f"{delta:+.2f}",
                ha="center", va="bottom", fontsize=9, color="#C00000", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Mean answer entropy (bits)")
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Figure 8 — Study A: Answer entropy (GRIT vs Qwen3B)\n"
                 "(10 questions/task, 10 draws each at T=0.7; red = GRIT minus Qwen3B)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, "fig8_study_a_entropy.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 9 — Oracle gap: @9 vs Oracle@9
# ════════════════════════════════════════════════════════════════════════════
def fig9() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    # Sorted by gap descending (from Run 1 data)
    entries = []
    for model in ["Qwen3B", "GRIT"]:
        for i, task in enumerate(TASKS):
            at9 = run1[model]["@9"][i]
            oracle = run1[model]["oracle@9"][i]
            gap = oracle - at9
            entries.append((f"{model} — {task}", at9, oracle, gap, model))

    entries.sort(key=lambda e: e[3], reverse=True)

    labels = [e[0] for e in entries]
    at9s = [e[1] for e in entries]
    oracles = [e[2] for e in entries]
    gaps = [e[3] for e in entries]

    y = np.arange(len(labels))
    bar_h = 0.35

    ax.barh(y + bar_h / 2, oracles, bar_h, label="Oracle@9", color="#A5A5A5", zorder=2)
    ax.barh(y - bar_h / 2, at9s, bar_h, label="Majority@9", color="#4472C4", zorder=2)

    # Gap annotations
    for i in range(len(entries)):
        ax.annotate(f"  gap: {gaps[i]:.1f}pp",
                    xy=(oracles[i], y[i] + bar_h / 2),
                    va="center", fontsize=9, color="#C00000", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 110)
    ax.legend(loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.suptitle("Figure 9 — The oracle gap: correct answers exist but voting fails to select them\n"
                 "(Run 1, 9 candidates, 30 questions/task)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, "fig9_oracle_gap.png")


if __name__ == "__main__":
    print("Generating report figures...")
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    print("Done.")
