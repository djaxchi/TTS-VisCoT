"""Generate figures for 292-question TTS scale results.

Produces multiple figure variants so the user can pick which best tells the story.
All figures saved to results/figures/scale/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "results/figures/scale"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(REPO / "results/analysis/scale_results.json") as f:
    DATA = json.load(f)

BASIC = DATA["basic"]     # {config: {task: {greedy, vote, oracle, n}}}
VOTING = DATA["voting"]   # voting strategies output

COLORS = {
    "greedy": "#8da0cb",
    "vote":   "#fc8d62",
    "oracle": "#66c2a5",
}
MODEL_COLORS = {
    "qwen3b": "#4b6cb7",
    "grit":   "#e4572e",
}


def _save(fig, name: str):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Headline grouped bars — greedy/vote/oracle, all 4 configs, overall
# ─────────────────────────────────────────────────────────────────────────────
def fig1_headline_overall():
    configs = ["qwen3b_standard", "qwen3b_t0", "grit_standard", "grit_t0"]
    labels  = ["Qwen3B\nstandard", "Qwen3B\nT=0", "GRIT\nstandard", "GRIT\nT=0"]
    greedy = [BASIC[c]["overall"]["greedy"] * 100 for c in configs]
    vote   = [BASIC[c]["overall"]["vote"]   * 100 for c in configs]
    oracle = [BASIC[c]["overall"]["oracle"] * 100 for c in configs]

    x = np.arange(len(configs))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, greedy, w, label="Greedy",    color=COLORS["greedy"])
    ax.bar(x,     vote,   w, label="Vote@9",    color=COLORS["vote"])
    ax.bar(x + w, oracle, w, label="Oracle@9",  color=COLORS["oracle"])
    for i, (g, v, o) in enumerate(zip(greedy, vote, oracle)):
        ax.text(i - w, g + 0.8, f"{g:.1f}", ha="center", fontsize=8)
        ax.text(i,     v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
        ax.text(i + w, o + 0.8, f"{o:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Fig 1 — Overall accuracy across 292 questions\n(Greedy vs Vote@9 vs Oracle@9)")
    ax.set_ylim(0, 75)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig1_headline_overall")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: TTS gain (vote − greedy) per config per task
# ─────────────────────────────────────────────────────────────────────────────
def fig2_tts_gain_by_task():
    configs = ["qwen3b_standard", "qwen3b_t0", "grit_standard", "grit_t0"]
    labels  = ["Qwen3B std", "Qwen3B T=0", "GRIT std", "GRIT T=0"]
    tasks = ["vqa", "ocr", "counting", "overall"]

    x = np.arange(len(configs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, task in enumerate(tasks):
        deltas = [(BASIC[c][task]["vote"] - BASIC[c][task]["greedy"]) * 100 for c in configs]
        ax.bar(x + (i - 1.5) * w, deltas, w, label=task.upper())
        for j, d in enumerate(deltas):
            ax.text(x[j] + (i - 1.5) * w, d + (0.1 if d >= 0 else -0.3),
                    f"{d:+.1f}", ha="center", fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ accuracy (pp): Vote@9 − Greedy")
    ax.set_title("Fig 2 — TTS gain per task (negative = voting hurts)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig2_tts_gain_by_task")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Oracle gap — the untapped ceiling
# ─────────────────────────────────────────────────────────────────────────────
def fig3_oracle_gap():
    configs = ["qwen3b_standard", "qwen3b_t0", "grit_standard", "grit_t0"]
    labels  = ["Qwen3B std", "Qwen3B T=0", "GRIT std", "GRIT T=0"]
    greedy = np.array([BASIC[c]["overall"]["greedy"] * 100 for c in configs])
    vote   = np.array([BASIC[c]["overall"]["vote"]   * 100 for c in configs])
    oracle = np.array([BASIC[c]["overall"]["oracle"] * 100 for c in configs])

    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(configs))
    ax.barh(y, oracle, color=COLORS["oracle"], alpha=0.35, label="Oracle@9 (ceiling)")
    ax.barh(y, vote,   color=COLORS["vote"],   alpha=0.85, label="Vote@9 (actual)")
    ax.barh(y, greedy, color=COLORS["greedy"], alpha=0.85, label="Greedy (baseline)")
    for i, (g, v, o) in enumerate(zip(greedy, vote, oracle)):
        ax.text(o + 0.5, i, f"gap {o - v:.1f}pp", va="center", fontsize=9, color="#555")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Fig 3 — Oracle gap: room above majority voting, unclosed by any strategy tested")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "fig3_oracle_gap")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Standard vs T=0 comparison (test the "image diversity helps GRIT" claim)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_standard_vs_t0():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axes, ["qwen3b", "grit"]):
        configs = [f"{model}_standard", f"{model}_t0"]
        labels  = ["standard (T=0.7 + aug)", "T=0 ablation (aug only)"]
        greedy = [BASIC[c]["overall"]["greedy"] * 100 for c in configs]
        vote   = [BASIC[c]["overall"]["vote"]   * 100 for c in configs]
        oracle = [BASIC[c]["overall"]["oracle"] * 100 for c in configs]
        x = np.arange(len(configs))
        w = 0.27
        ax.bar(x - w, greedy, w, label="Greedy",   color=COLORS["greedy"])
        ax.bar(x,     vote,   w, label="Vote@9",   color=COLORS["vote"])
        ax.bar(x + w, oracle, w, label="Oracle@9", color=COLORS["oracle"])
        for i, (g, v, o) in enumerate(zip(greedy, vote, oracle)):
            ax.text(i - w, g + 0.8, f"{g:.1f}", ha="center", fontsize=8)
            ax.text(i,     v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
            ax.text(i + w, o + 0.8, f"{o:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{model.upper()}")
        ax.set_ylim(0, 75)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Fig 4 — Standard vs T=0 ablation (isolating image-diversity contribution)")
    fig.tight_layout()
    _save(fig, "fig4_standard_vs_t0")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Per-task delta heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig5_delta_heatmap():
    configs = ["qwen3b_standard", "qwen3b_t0", "grit_standard", "grit_t0"]
    labels  = ["Qwen3B std", "Qwen3B T=0", "GRIT std", "GRIT T=0"]
    tasks = ["vqa", "ocr", "counting", "overall"]
    matrix = np.array([
        [(BASIC[c][t]["vote"] - BASIC[c][t]["greedy"]) * 100 for t in tasks]
        for c in configs
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(labels)
    for i in range(len(configs)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center",
                    color="white" if abs(matrix[i, j]) > 4 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Δ (pp)")
    ax.set_title("Fig 5 — TTS delta (Vote@9 − Greedy), per config × task")
    _save(fig, "fig5_delta_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Voting strategies comparison (all 6)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_voting_strategies():
    strategies = ["greedy", "plurality", "greedy_tiebreak", "greedy_unless_supermaj",
                  "consistency_filter", "logprob_sum", "logprob_mean", "oracle"]
    labels = ["Greedy", "Plurality", "Greedy\n+tiebreak", "GreedyUnless\nSupermaj",
              "Consistency\nfilter", "Logprob\nsum", "Logprob\nmean", "Oracle@9"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(strategies))
    w = 0.38
    q = [VOTING["qwen3b_standard"][s] * 100 for s in strategies]
    g = [VOTING["grit_standard"][s]   * 100 for s in strategies]
    ax.bar(x - w/2, q, w, label="Qwen3B std", color=MODEL_COLORS["qwen3b"])
    ax.bar(x + w/2, g, w, label="GRIT std",   color=MODEL_COLORS["grit"])
    for i, (qi, gi) in enumerate(zip(q, g)):
        ax.text(i - w/2, qi + 0.6, f"{qi:.1f}", ha="center", fontsize=7)
        ax.text(i + w/2, gi + 0.6, f"{gi:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.axhline(VOTING["qwen3b_standard"]["greedy"] * 100,
               color=MODEL_COLORS["qwen3b"], ls="--", alpha=0.5)
    ax.axhline(VOTING["grit_standard"]["greedy"] * 100,
               color=MODEL_COLORS["grit"], ls="--", alpha=0.5)
    ax.set_title("Fig 6 — Voting strategies at 292 questions (standard recipe)\n"
                 "All sampling-based strategies underperform greedy; logprob voting handicapped on VQA (A-D captured only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 65)
    _save(fig, "fig6_voting_strategies")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Counting-only voting strategies (clean 4-option case)
# ─────────────────────────────────────────────────────────────────────────────
def fig7_voting_counting_only():
    strategies = ["greedy", "plurality", "logprob_sum", "logprob_mean", "oracle"]
    labels = ["Greedy", "Plurality", "Logprob sum", "Logprob mean", "Oracle@9"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(strategies))
    w = 0.38
    q = [VOTING["qwen3b_standard_counting"][s] * 100 for s in strategies]
    g = [VOTING["grit_standard_counting"][s]   * 100 for s in strategies]
    ax.bar(x - w/2, q, w, label="Qwen3B std", color=MODEL_COLORS["qwen3b"])
    ax.bar(x + w/2, g, w, label="GRIT std",   color=MODEL_COLORS["grit"])
    for i, (qi, gi) in enumerate(zip(q, g)):
        ax.text(i - w/2, qi + 1, f"{qi:.1f}", ha="center", fontsize=8)
        ax.text(i + w/2, gi + 1, f"{gi:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Fig 7 — Counting-only (MMStar, clean 4-option), standard recipe\n"
                 "Fair logprob comparison: all 4 options captured")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 90)
    _save(fig, "fig7_voting_counting_only")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Per-task grouped bars (broken out so each task is visible)
# ─────────────────────────────────────────────────────────────────────────────
def fig8_per_task_breakdown():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    configs = ["qwen3b_standard", "qwen3b_t0", "grit_standard", "grit_t0"]
    labels  = ["Qwen3B\nstd", "Qwen3B\nT=0", "GRIT\nstd", "GRIT\nT=0"]
    for ax, task in zip(axes, ["vqa", "ocr", "counting"]):
        greedy = [BASIC[c][task]["greedy"] * 100 for c in configs]
        vote   = [BASIC[c][task]["vote"]   * 100 for c in configs]
        oracle = [BASIC[c][task]["oracle"] * 100 for c in configs]
        x = np.arange(len(configs))
        w = 0.27
        ax.bar(x - w, greedy, w, label="Greedy",   color=COLORS["greedy"])
        ax.bar(x,     vote,   w, label="Vote@9",   color=COLORS["vote"])
        ax.bar(x + w, oracle, w, label="Oracle@9", color=COLORS["oracle"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{task.upper()}  (n={BASIC[configs[0]][task]['n']})")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 90)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Fig 8 — Per-task breakdown across 4 configs")
    fig.tight_layout()
    _save(fig, "fig8_per_task_breakdown")


if __name__ == "__main__":
    print(f"Writing figures to {FIG_DIR}/")
    fig1_headline_overall()
    fig2_tts_gain_by_task()
    fig3_oracle_gap()
    fig4_standard_vs_t0()
    fig5_delta_heatmap()
    fig6_voting_strategies()
    fig7_voting_counting_only()
    fig8_per_task_breakdown()
    print("\nDone.")
