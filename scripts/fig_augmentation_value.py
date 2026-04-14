"""Figure: image augmentation adds unique oracle coverage for GRIT but not Qwen.

Uses the T=0 ablation so temperature noise is eliminated — any diversity
in the candidate pool comes purely from image augmentations.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "results/figures/poster"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

COL_QWEN = "#4b6cb7"
COL_GRIT = "#e4572e"
COL_ORIG = "#aaaaaa"


def norm(s): return (s or "").strip().lower()


def is_correct(ans, row):
    a = norm(ans)
    if not a:
        return False
    return any(a == norm(x) for x in (row.get("answers_all") or [row["gt_answer"]]))


def load(p): return [json.loads(l) for l in open(p)]


def per_slot_marginal(rows):
    """For each slot: accuracy and #questions where it's the only correct candidate."""
    slot_meta = {}
    slot_acc = defaultdict(int)
    slot_marginal = defaultdict(int)
    slot_n = defaultdict(int)
    for r in rows:
        cands = {c["candidate_idx"]: c for c in r["candidates"]}
        correct_idxs = {i for i, c in cands.items() if is_correct(c["answer"], r)}
        for i, c in cands.items():
            if i not in slot_meta:
                slot_meta[i] = (c["image_aug"], c["text_variant"], c["temperature"])
            slot_n[i] += 1
            if i in correct_idxs:
                slot_acc[i] += 1
                if len(correct_idxs) == 1:
                    slot_marginal[i] += 1
    n = len(rows)
    return {
        i: {
            "aug": slot_meta[i][0],
            "text": slot_meta[i][1],
            "temp": slot_meta[i][2],
            "acc": slot_acc[i] / slot_n[i],
            "marginal": slot_marginal[i],
            "marginal_pct": slot_marginal[i] / n,
            "n": slot_n[i],
        }
        for i in sorted(slot_meta)
    }


# ── Slot labels ───────────────────────────────────────────────────────────────
def slot_label(meta):
    aug_short = {
        "original": "Original",
        "edge_enhance": "Edge",
        "grayscale": "Gray",
        "jpeg_recompress": "JPEG",
        "brightness_contrast": "Bright.",
        "rotation_90": "Rot.90",
    }
    txt = " + para" if meta["text"] == "hardcoded_paraphrase" else ""
    return f"{aug_short.get(meta['aug'], meta['aug'])}{txt}"


# ─────────────────────────────────────────────────────────────────────────────
# Figure E — Augmentation diversity: how many slots recover each failed question?
# Conditioned on original image failing — so x-axis is "number of augmented
# slots that recover this question" (1=unique rescue, 7=any augmentation works).
# Qwen spikes at 7 (redundant), GRIT spikes at 1 (complementary).
# ─────────────────────────────────────────────────────────────────────────────
def fig_slot_marginal():
    from collections import Counter

    qwen_rows = load(REPO / "results/tts_scale_t0/qwen3b_results.jsonl")
    grit_rows = load(REPO / "results/tts_scale_t0/grit_results.jsonl")

    aug_idxs = [1, 3, 4, 5, 6, 7, 8]
    orig_idx = 2

    def recovery_distribution(rows):
        slot_recovered = {}
        failed_qids = set()
        for r in rows:
            cands = {c["candidate_idx"]: c for c in r["candidates"]}
            if orig_idx in cands and not is_correct(cands[orig_idx]["answer"], r):
                failed_qids.add(r["question_id"])
                for idx in aug_idxs:
                    if idx in cands and is_correct(cands[idx]["answer"], r):
                        slot_recovered.setdefault(idx, set()).add(r["question_id"])
        union = set().union(*slot_recovered.values()) if slot_recovered else set()
        dist = Counter()
        for qid in union:
            cnt = sum(1 for s in slot_recovered.values() if qid in s)
            dist[cnt] += 1
        return dist, len(failed_qids)

    qwen_dist, qwen_n_fail = recovery_distribution(qwen_rows)
    grit_dist, grit_n_fail = recovery_distribution(grit_rows)

    x = np.arange(1, 8)
    qwen_vals = [qwen_dist.get(k, 0) for k in x]
    grit_vals  = [grit_dist.get(k, 0)  for k in x]
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_q = ax.bar(x - w/2, qwen_vals, w, label=f"Qwen3B — direct  (failed on {qwen_n_fail})",
                    color=COL_QWEN, edgecolor="black", linewidth=0.6)
    bars_g = ax.bar(x + w/2, grit_vals, w, label=f"GRIT — visual CoT  (failed on {grit_n_fail})",
                    color=COL_GRIT, edgecolor="black", linewidth=0.6)

    for b, v in zip(bars_q, qwen_vals):
        if v > 0:
            ax.text(b.get_x() + b.get_width()/2, v + 0.3, str(v),
                    ha="center", va="bottom", fontsize=10, color=COL_QWEN, fontweight="bold")
    for b, v in zip(bars_g, grit_vals):
        if v > 0:
            ax.text(b.get_x() + b.get_width()/2, v + 0.3, str(v),
                    ha="center", va="bottom", fontsize=10, color=COL_GRIT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}" for k in x])
    ax.set_xlabel("Number of augmented candidates that recover the question\n"
                  "(1 = only one augmentation works   →   7 = every augmentation works)")
    ax.set_ylabel("Number of questions  (original image failed)")
    ax.set_title("Augmentation diversity: complementary rescues vs redundant rescues\n"
                 "(T=0 ablation — pure image augmentation effect, no temperature noise)")
    ax.legend(loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(qwen_vals), max(grit_vals)) * 1.3)

    # Annotations
    ax.annotate("Redundant\n(any aug works)", xy=(7 - w/2, qwen_dist.get(7, 0)),
                xytext=(5.5, qwen_dist.get(7, 0) + 3),
                arrowprops=dict(arrowstyle="->", color=COL_QWEN), color=COL_QWEN, fontsize=9)
    ax.annotate("Complementary\n(unique rescue)", xy=(1 + w/2, grit_dist.get(1, 0)),
                xytext=(2.3, grit_dist.get(1, 0) + 3),
                arrowprops=dict(arrowstyle="->", color=COL_GRIT), color=COL_GRIT, fontsize=9)

    fig.tight_layout()
    path = FIG_DIR / "E_augmentation_diversity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure F — Oracle coverage breakdown: original vs augmented candidates
# ─────────────────────────────────────────────────────────────────────────────
def fig_oracle_breakdown():
    configs = {
        "Qwen3B\n(T=0)": load(REPO / "results/tts_scale_t0/qwen3b_results.jsonl"),
        "GRIT\n(T=0)":   load(REPO / "results/tts_scale_t0/grit_results.jsonl"),
        "Qwen3B\n(T=0.7)": load(REPO / "results/tts_scale/qwen3b_results.jsonl"),
        "GRIT\n(T=0.7)":   load(REPO / "results/tts_scale/grit_results.jsonl"),
    }
    orig_idxs = {0, 2}
    aug_idxs  = {1, 3, 4, 5, 6, 7, 8}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(configs))
    w = 0.55
    colors_both  = ["#7fa8d6", "#e8937a", "#7fa8d6", "#e8937a"]
    colors_aug   = ["#2a4d8f", "#b82a0c", "#2a4d8f", "#b82a0c"]
    colors_orig  = ["#c9d8ee", "#f5c4b8", "#c9d8ee", "#f5c4b8"]

    for i, (label, rows) in enumerate(configs.items()):
        n = len(rows)
        orig_only = aug_only = both = 0
        for r in rows:
            cands = {c["candidate_idx"]: c for c in r["candidates"]}
            orig_ok = any(is_correct(cands[j]["answer"], r) for j in orig_idxs if j in cands)
            aug_ok  = any(is_correct(cands[j]["answer"], r) for j in aug_idxs  if j in cands)
            if orig_ok and aug_ok:   both += 1
            elif orig_ok:            orig_only += 1
            elif aug_ok:             aug_only += 1

        orig_pct = (orig_only + both) / n * 100
        aug_add  = aug_only / n * 100
        base     = (orig_only + both) / n * 100

        # stacked: bottom = covered by original, top addition = aug-only
        bar_orig = ax.bar(i, base, w, color=colors_orig[i], edgecolor="black", linewidth=0.6)
        bar_aug  = ax.bar(i, aug_add, w, bottom=base, color=colors_aug[i],
                          edgecolor="black", linewidth=0.6)

        ax.text(i, base / 2, f"{base:.0f}%", ha="center", va="center",
                fontsize=10, color="black")
        if aug_add > 0.5:
            ax.text(i, base + aug_add / 2, f"+{aug_add:.0f}%", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(list(configs.keys()))
    ax.set_ylabel("% of questions with ≥1 correct candidate")
    ax.set_title("Oracle coverage: what original candidates provide vs what augmentation adds")
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.3)

    orig_patch = mpatches.Patch(facecolor="#c9d8ee", edgecolor="black", label="Covered by original image alone")
    aug_patch  = mpatches.Patch(facecolor="#555577", edgecolor="black", label="Uniquely recovered by augmentation")
    ax.legend(handles=[orig_patch, aug_patch], loc="upper left")

    fig.tight_layout()
    path = FIG_DIR / "F_oracle_breakdown.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Writing to {FIG_DIR}/")
    fig_slot_marginal()
    fig_oracle_breakdown()
    print("Done.")
