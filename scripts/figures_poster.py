"""Poster figures — TTS effect on visual-thinking models.

TTS is defined (in this work) as pass@9: correct if any of 9 candidates
matches ground truth. Voting-strategy exploration is noted as a limitation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "results/figures/poster"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(REPO / "results/analysis/scale_results.json") as f:
    DATA = json.load(f)
BASIC = DATA["basic"]

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

COL_BASE = "#8da0cb"   # no TTS
COL_TTS  = "#e4572e"   # with TTS
COL_GRIT = "#e4572e"
COL_QWEN = "#4b6cb7"


def _save(fig, name: str):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure A — Headline: TTS gain across all 4 configs (overall)
# ─────────────────────────────────────────────────────────────────────────────
def fig_headline_gain():
    configs = ["qwen3b_t0", "qwen3b_standard", "grit_t0", "grit_standard"]
    labels  = ["Qwen3B\nT=0", "Qwen3B\nT=0.7", "GRIT\nT=0", "GRIT\nT=0.7"]
    base = np.array([BASIC[c]["overall"]["greedy"] * 100 for c in configs])
    tts  = np.array([BASIC[c]["overall"]["oracle"] * 100 for c in configs])
    gain = tts - base

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COL_QWEN, COL_QWEN, COL_GRIT, COL_GRIT]
    bars = ax.bar(labels, gain, color=colors, edgecolor="black", linewidth=0.7)
    for b, g, t, base_v in zip(bars, gain, tts, base):
        ax.text(b.get_x() + b.get_width()/2, g + 0.6,
                f"+{g:.1f}pp",
                ha="center", fontsize=13, fontweight="bold")
        ax.text(b.get_x() + b.get_width()/2, g / 2,
                f"{base_v:.0f}% → {t:.0f}%",
                ha="center", color="white", fontsize=10)
    ax.set_ylabel("TTS gain  (pp)")
    ax.set_title("Accuracy lift from TTS (pass@9) — full hard_bench (n=292)")
    ax.set_ylim(0, max(gain) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "A_headline_gain")


# ─────────────────────────────────────────────────────────────────────────────
# Figure B — Per-task breakdown: baseline vs TTS
# ─────────────────────────────────────────────────────────────────────────────
def fig_per_task():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    configs = ["qwen3b_standard", "grit_standard"]
    config_labels = ["Qwen3B", "GRIT"]
    config_colors = [COL_QWEN, COL_GRIT]

    for ax, task in zip(axes, ["vqa", "ocr", "counting"]):
        x = np.arange(len(configs))
        w = 0.35
        base = [BASIC[c][task]["greedy"] * 100 for c in configs]
        tts  = [BASIC[c][task]["oracle"] * 100 for c in configs]
        ax.bar(x - w/2, base, w, label="Without TTS", color=COL_BASE, edgecolor="black", linewidth=0.6)
        ax.bar(x + w/2, tts,  w, label="With TTS",    color=COL_TTS,  edgecolor="black", linewidth=0.6)
        for i, (bv, tv) in enumerate(zip(base, tts)):
            ax.text(i - w/2, bv + 1.2, f"{bv:.0f}", ha="center", fontsize=10)
            ax.text(i + w/2, tv + 1.2, f"{tv:.0f}", ha="center", fontsize=10)
            ax.text(i, max(bv, tv) + 6, f"+{tv-bv:.0f}pp",
                    ha="center", fontsize=11, fontweight="bold", color="#333")
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels)
        ax.set_title(f"{task.upper()}  (n={BASIC[configs[0]][task]['n']})")
        ax.set_ylim(0, 95)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Per-task effect of TTS — standard recipe (T=0.7 + image augmentation)",
                 fontsize=14)
    fig.tight_layout()
    _save(fig, "B_per_task")


# ─────────────────────────────────────────────────────────────────────────────
# Figure C — Temperature amplifies TTS
# ─────────────────────────────────────────────────────────────────────────────
def fig_temperature_effect():
    models = ["qwen3b", "grit"]
    model_labels = ["Qwen3B", "GRIT"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    w = 0.35
    t0   = [(BASIC[f"{m}_t0"]["overall"]["oracle"] - BASIC[f"{m}_t0"]["overall"]["greedy"]) * 100 for m in models]
    std  = [(BASIC[f"{m}_standard"]["overall"]["oracle"] - BASIC[f"{m}_standard"]["overall"]["greedy"]) * 100 for m in models]
    ax.bar(x - w/2, t0,  w, label="T=0 (augmentation only)", color="#a0a0c0", edgecolor="black", linewidth=0.6)
    ax.bar(x + w/2, std, w, label="T=0.7 + augmentation",    color=COL_TTS,   edgecolor="black", linewidth=0.6)
    for i, (tv, sv) in enumerate(zip(t0, std)):
        ax.text(i - w/2, tv + 0.5, f"+{tv:.1f}pp", ha="center", fontsize=11, fontweight="bold")
        ax.text(i + w/2, sv + 0.5, f"+{sv:.1f}pp", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=13)
    ax.set_ylabel("TTS gain  (pp)")
    ax.set_title("Sampling temperature amplifies TTS — overall (n=292)")
    ax.set_ylim(0, max(max(t0), max(std)) * 1.2)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "C_temperature_effect")


# ─────────────────────────────────────────────────────────────────────────────
# Figure D — GRIT vs Qwen head-to-head, standard recipe
# ─────────────────────────────────────────────────────────────────────────────
def fig_grit_vs_qwen():
    tasks = ["vqa", "ocr", "counting", "overall"]
    task_labels = ["VQA", "OCR", "Counting", "Overall"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(tasks))
    w = 0.18
    qwen_base = [BASIC["qwen3b_standard"][t]["greedy"] * 100 for t in tasks]
    qwen_tts  = [BASIC["qwen3b_standard"][t]["oracle"] * 100 for t in tasks]
    grit_base = [BASIC["grit_standard"][t]["greedy"] * 100 for t in tasks]
    grit_tts  = [BASIC["grit_standard"][t]["oracle"] * 100 for t in tasks]

    ax.bar(x - 1.5*w, qwen_base, w, label="Qwen3B — no TTS", color=COL_QWEN, alpha=0.45, edgecolor="black", linewidth=0.5)
    ax.bar(x - 0.5*w, qwen_tts,  w, label="Qwen3B — with TTS", color=COL_QWEN, edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.5*w, grit_base, w, label="GRIT — no TTS",   color=COL_GRIT, alpha=0.45, edgecolor="black", linewidth=0.5)
    ax.bar(x + 1.5*w, grit_tts,  w, label="GRIT — with TTS", color=COL_GRIT, edgecolor="black", linewidth=0.5)
    for i, t in enumerate(tasks):
        ax.text(i - 0.5*w, qwen_tts[i]  + 1.3, f"+{qwen_tts[i] - qwen_base[i]:.0f}", ha="center", fontsize=9, color=COL_QWEN, fontweight="bold")
        ax.text(i + 1.5*w, grit_tts[i]  + 1.3, f"+{grit_tts[i] - grit_base[i]:.0f}", ha="center", fontsize=9, color=COL_GRIT, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Visual-thinking (GRIT) vs direct (Qwen3B) — standard recipe")
    ax.set_ylim(0, 92)
    ax.legend(loc="upper left", ncol=2, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "D_grit_vs_qwen")


if __name__ == "__main__":
    print(f"Writing figures to {FIG_DIR}/")
    fig_headline_gain()
    fig_per_task()
    fig_temperature_effect()
    fig_grit_vs_qwen()
    print("\nDone.")
