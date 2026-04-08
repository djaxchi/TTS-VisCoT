"""
Comprehensive results visualization for TTS-VisCoT experiments.

Generates figures comparing Qwen2.5-VL (3B) and GRIT (3B) on:
  - VQAv2 benchmark (vqa / counting / ocr tasks, n=20 each)
  - TreeBench spatial-reasoning benchmark (n=20)

Run from repo root:
    python scripts/plot_results.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
CMP_FILE        = ROOT / "results/comparison/run_qwen_grit_tts9_20_tokenstore.json"
CMP_JUDGED_FILE = ROOT / "results/comparison/run_qwen_grit_tts9_20_tokenstore_judged.json"
QWEN_TB_PREDS = ROOT / "results/tts_eval/qwen_run1/predictions.jsonl"
GRIT_TB_PREDS = ROOT / "results/tts_eval/grit_run2/predictions.jsonl"
QWEN_TB_ARTS  = ROOT / "results/tts_eval/qwen_run1/candidate_artifacts.jsonl"
GRIT_TB_ARTS  = ROOT / "results/tts_eval/grit_run2/candidate_artifacts.jsonl"
OUT_DIR = ROOT / "results/comparison"

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
QWEN_COLOR = "#4C72B0"
GRIT_COLOR = "#DD8452"
CORRECT_COLOR = "#55A868"
WRONG_COLOR   = "#C44E52"


# ── loaders ──────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── helpers ──────────────────────────────────────────────────────────────────
def accuracy(items, key_fn):
    return sum(1 for i in items if key_fn(i)) / len(items) if items else 0.0


def bar_label(ax, rects, fmt="{:.0%}"):
    for r in rects:
        h = r.get_height()
        ax.text(
            r.get_x() + r.get_width() / 2.0,
            h + 0.01,
            fmt.format(h),
            ha="center", va="bottom", fontsize=8.5,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — VQAv2 benchmark: baseline vs TTS scaling (majority-3/5/9)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_vqa_scaling():
    cmp = load_json(CMP_FILE)

    # ── collect data ──────────────────────────────────────────────────────────
    tasks = ["vqa", "counting", "ocr"]
    task_labels = ["VQA", "Counting", "OCR"]
    models = {
        "Qwen2.5-VL (3B, no CoT)": ("Qwen", QWEN_COLOR),
        "GRIT (3B)": ("GRIT", GRIT_COLOR),
    }
    voting_keys = ["baseline", "majority_3", "majority_5", "majority_9"]
    voting_labels = ["Baseline\n(×1)", "Majority-3\n(×3)", "Majority-5\n(×5)", "Majority-9\n(×9)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle(
        "VQAv2 Benchmark — Accuracy vs. Test-Time Scaling\n"
        "Qwen2.5-VL 3B vs. GRIT 3B  |  n=20 per task",
        fontsize=13, fontweight="bold", y=1.02,
    )

    x = np.arange(len(voting_labels))
    width = 0.35

    for ax, task, task_label in zip(axes, tasks, task_labels):
        for i, (model_key, (short_name, color)) in enumerate(models.items()):
            items = cmp[model_key][task]
            accs = []
            for vk in voting_keys:
                if vk == "baseline":
                    acc = accuracy(items, lambda it: it["correct"])
                else:
                    acc = accuracy(
                        items,
                        lambda it, vk=vk: it["voting"][vk]["answer"] in it["references"],
                    )
                accs.append(acc)
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, accs, width, label=short_name, color=color, alpha=0.88)
            bar_label(ax, bars)

        ax.set_title(task_label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(voting_labels, fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy" if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Chance (25%)")

    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "fig1_vqa_tts_scaling.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — TreeBench: baseline vs TTS for both models
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_treebench_overview():
    qwen_preds = load_jsonl(QWEN_TB_PREDS)
    grit_preds = load_jsonl(GRIT_TB_PREDS)

    # ── accuracy ──────────────────────────────────────────────────────────────
    q_base = accuracy(qwen_preds, lambda p: p["baseline"]["is_correct"])
    q_tts  = accuracy(qwen_preds, lambda p: p["tts"]["is_correct"])
    g_base = accuracy(grit_preds, lambda p: p["baseline"]["is_correct"])
    g_tts  = accuracy(grit_preds, lambda p: p["tts"]["is_correct"])

    # ── agreement & entropy ───────────────────────────────────────────────────
    q_agree = [p["tts"]["agreement_rate"] for p in qwen_preds]
    g_agree = [p["tts"]["agreement_rate"] for p in grit_preds]
    q_entropy = [p["tts"]["answer_entropy"] for p in qwen_preds]
    g_entropy = [p["tts"]["answer_entropy"] for p in grit_preds]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "TreeBench Spatial Reasoning — Test-Time Scaling  |  n=20",
        fontsize=13, fontweight="bold",
    )

    # ─── panel A: baseline vs TTS accuracy ────────────────────────────────────
    ax = axes[0]
    categories = ["Baseline (×1)", "TTS Majority-9 (×9)"]
    qwen_vals = [q_base, q_tts]
    grit_vals  = [g_base, g_tts]
    x = np.arange(2)
    w = 0.3
    b1 = ax.bar(x - w/2, qwen_vals, w, label="Qwen", color=QWEN_COLOR, alpha=0.88)
    b2 = ax.bar(x + w/2, grit_vals, w, label="GRIT", color=GRIT_COLOR, alpha=0.88)
    bar_label(ax, b1)
    bar_label(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_title("Accuracy: Baseline vs. TTS", fontweight="bold")
    ax.legend(fontsize=9)

    # ─── panel B: agreement rate distribution ─────────────────────────────────
    ax = axes[1]
    ax.violinplot([q_agree, g_agree], positions=[1, 2], showmedians=True,
                  showextrema=True)
    ax.scatter([1]*len(q_agree), q_agree, alpha=0.5, color=QWEN_COLOR, s=20, zorder=3)
    ax.scatter([2]*len(g_agree), g_agree, alpha=0.5, color=GRIT_COLOR, s=20, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Qwen", "GRIT"])
    ax.set_ylabel("Vote Agreement Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Candidate Agreement Rate\n(among 9 candidates)", fontweight="bold")

    # ─── panel C: entropy distribution ────────────────────────────────────────
    ax = axes[2]
    ax.violinplot([q_entropy, g_entropy], positions=[1, 2], showmedians=True,
                  showextrema=True)
    ax.scatter([1]*len(q_entropy), q_entropy, alpha=0.5, color=QWEN_COLOR, s=20, zorder=3)
    ax.scatter([2]*len(g_entropy), g_entropy, alpha=0.5, color=GRIT_COLOR, s=20, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Qwen", "GRIT"])
    ax.set_ylabel("Vote Entropy")
    ax.set_title("Answer Entropy\n(higher = more uncertain)", fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / "fig2_treebench_overview.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Image transform ablation (TreeBench)
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_image_transform():
    def transform_acc(arts):
        stats = defaultdict(lambda: {"c": 0, "t": 0})
        for a in arts:
            tid = a["image_transform_id"]
            stats[tid]["t"] += 1
            if a["is_correct"]:
                stats[tid]["c"] += 1
        return {k: v["c"] / v["t"] for k, v in stats.items()}

    qwen_arts = load_jsonl(QWEN_TB_ARTS)
    grit_arts  = load_jsonl(GRIT_TB_ARTS)
    q_acc = transform_acc(qwen_arts)
    g_acc = transform_acc(grit_arts)

    transforms = sorted(q_acc.keys())
    labels = [t.replace("_", "\n") for t in transforms]
    x = np.arange(len(transforms))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - w/2, [q_acc[t] for t in transforms], w, label="Qwen",
                color=QWEN_COLOR, alpha=0.88)
    b2 = ax.bar(x + w/2, [g_acc[t] for t in transforms], w, label="GRIT",
                color=GRIT_COLOR, alpha=0.88)
    bar_label(ax, b1)
    bar_label(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 0.60)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_title(
        "TreeBench: Per-Image-Transform Accuracy\n(candidates across all text variants & stages)",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "fig3_image_transform.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Text variant ablation (TreeBench)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_text_variant():
    def variant_acc(arts):
        stats = defaultdict(lambda: {"c": 0, "t": 0})
        for a in arts:
            tv = a["text_variant_id"]
            stats[tv]["t"] += 1
            if a["is_correct"]:
                stats[tv]["c"] += 1
        return {k: v["c"] / v["t"] for k, v in stats.items()}

    qwen_arts = load_jsonl(QWEN_TB_ARTS)
    grit_arts  = load_jsonl(GRIT_TB_ARTS)
    q_acc = variant_acc(qwen_arts)
    g_acc = variant_acc(grit_arts)

    variants  = sorted(q_acc.keys())
    labels    = ["Hardcoded\nParaphrase", "Model\nParaphrase", "Original\nQuestion"]
    x = np.arange(len(variants))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, [q_acc[v] for v in variants], w, label="Qwen",
                color=QWEN_COLOR, alpha=0.88)
    b2 = ax.bar(x + w/2, [g_acc[v] for v in variants], w, label="GRIT",
                color=GRIT_COLOR, alpha=0.88)
    bar_label(ax, b1)
    bar_label(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_title(
        "TreeBench: Per-Text-Variant Accuracy\n"
        "(across all image transforms & stages)",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "fig4_text_variant.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Agreement rate vs correctness (scatter + bars)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_agreement_vs_correctness():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "TreeBench: Vote Agreement Rate vs. Answer Correctness",
        fontsize=13, fontweight="bold",
    )

    for ax, preds_path, model_name, color in [
        (axes[0], QWEN_TB_PREDS, "Qwen2.5-VL (3B)", QWEN_COLOR),
        (axes[1], GRIT_TB_PREDS, "GRIT (3B)", GRIT_COLOR),
    ]:
        preds = load_jsonl(preds_path)
        correct_agree = [p["tts"]["agreement_rate"] for p in preds if p["tts"]["is_correct"]]
        wrong_agree   = [p["tts"]["agreement_rate"] for p in preds if not p["tts"]["is_correct"]]

        # bin into 5 buckets and show bar
        bins = np.linspace(0, 1, 6)
        bin_labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

        def hist(data):
            counts, _ = np.histogram(data, bins=bins)
            return counts / max(len(data), 1)

        x = np.arange(5)
        w = 0.35
        b1 = ax.bar(x - w/2, hist(correct_agree), w, label="Correct", color=CORRECT_COLOR, alpha=0.88)
        b2 = ax.bar(x + w/2, hist(wrong_agree),   w, label="Wrong",   color=WRONG_COLOR,   alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=8.5)
        ax.set_xlabel("Vote Agreement Rate")
        ax.set_ylabel("Fraction of Predictions")
        ax.set_title(f"{model_name}\n(correct={len(correct_agree)}, wrong={len(wrong_agree)})",
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.0)

    fig.tight_layout()
    out = OUT_DIR / "fig5_agreement_vs_correct.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Token-level logprob signal (Qwen, TreeBench)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_token_logprobs():
    def collect_logprobs(arts):
        correct, wrong = [], []
        for a in arts:
            tm  = a.get("token_metadata", {})
            lps = tm.get("option_logprobs", [])
            if not lps:
                continue
            ans = a["normalized_answer"]
            lp_dict = lps[0]["logprobs"]
            if ans not in lp_dict:
                continue
            lp = lp_dict[ans]
            (correct if a["is_correct"] else wrong).append(lp)
        return correct, wrong

    qwen_arts = load_jsonl(QWEN_TB_ARTS)
    grit_arts  = load_jsonl(GRIT_TB_ARTS)
    q_correct, q_wrong = collect_logprobs(qwen_arts)
    g_correct, g_wrong = collect_logprobs(grit_arts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Token-Level Logprob of Predicted Answer Token\n"
        "Correct vs. Incorrect Candidates — TreeBench",
        fontsize=13, fontweight="bold",
    )

    for ax, correct, wrong, model_name in [
        (axes[0], q_correct, q_wrong, "Qwen2.5-VL (3B)"),
        (axes[1], g_correct, g_wrong, "GRIT (3B)"),
    ]:
        # violin
        data = [d for d in [correct, wrong] if d]
        positions = [i+1 for i, d in enumerate([correct, wrong]) if d]
        if len(data) >= 2:
            parts = ax.violinplot(data, positions=positions, showmedians=True)
            for pc, col in zip(parts["bodies"], [CORRECT_COLOR, WRONG_COLOR]):
                pc.set_facecolor(col)
                pc.set_alpha(0.7)
        # scatter points
        if correct:
            ax.scatter([1]*len(correct), correct, alpha=0.4, color=CORRECT_COLOR, s=18, zorder=3)
            ax.axhline(np.mean(correct), color=CORRECT_COLOR, linestyle="--", linewidth=1.2,
                       label=f"Correct mean: {np.mean(correct):.2f}")
        if wrong:
            ax.scatter([2]*len(wrong), wrong, alpha=0.4, color=WRONG_COLOR, s=18, zorder=3)
            ax.axhline(np.mean(wrong), color=WRONG_COLOR, linestyle="--", linewidth=1.2,
                       label=f"Wrong mean: {np.mean(wrong):.2f}")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Correct", "Wrong"])
        ax.set_ylabel("Log-Prob of Predicted Answer Token")
        ax.set_title(f"{model_name}\n(n_correct={len(correct)}, n_wrong={len(wrong)})",
                     fontweight="bold")
        ax.legend(fontsize=8.5)

    fig.tight_layout()
    out = OUT_DIR / "fig6_token_logprobs.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — Combined summary heatmap: Accuracy across models × benchmarks
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_summary_heatmap():
    cmp = load_json(CMP_FILE)
    qwen_preds = load_jsonl(QWEN_TB_PREDS)
    grit_preds = load_jsonl(GRIT_TB_PREDS)

    # rows = conditions, cols = datasets
    rows = ["Baseline (×1)", "TTS Majority-3 (×3)", "TTS Majority-5 (×5)", "TTS Majority-9 (×9)"]
    cols = ["VQA", "Counting", "OCR", "TreeBench"]
    task_map = {"VQA": "vqa", "Counting": "counting", "OCR": "ocr"}

    def vqa_acc(model_key, task, voting):
        items = cmp[model_key][task]
        if voting == "baseline":
            return accuracy(items, lambda i: i["correct"])
        return accuracy(items, lambda i, v=voting: i["voting"][v]["answer"] in i["references"])

    def tb_acc(preds, voting):
        if voting == "baseline":
            return accuracy(preds, lambda p: p["baseline"]["is_correct"])
        return accuracy(preds, lambda p: p["tts"]["is_correct"])  # TTS uses majority-9

    voting_keys = ["baseline", "majority_3", "majority_5", "majority_9"]

    for model_key, (model_name, preds) in [
        ("Qwen2.5-VL (3B, no CoT)", ("Qwen2.5-VL (3B, no CoT)", qwen_preds)),
        ("GRIT (3B)", ("GRIT (3B)", grit_preds)),
    ]:
        data = np.zeros((len(rows), len(cols)))
        for r, vk in enumerate(voting_keys):
            for c, col in enumerate(cols):
                if col == "TreeBench":
                    data[r, c] = tb_acc(preds, vk)
                else:
                    data[r, c] = vqa_acc(model_key, task_map[col], vk)

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=10)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=9)
        for r in range(len(rows)):
            for c in range(len(cols)):
                ax.text(c, r, f"{data[r,c]:.0%}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if data[r,c] < 0.45 or data[r,c] > 0.75 else "black")
        plt.colorbar(im, ax=ax, format="{x:.0%}")
        ax.set_title(
            f"{model_name}\nAccuracy Heatmap: Voting Method × Benchmark",
            fontweight="bold", pad=10,
        )
        fig.tight_layout()
        safe = model_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        out = OUT_DIR / f"fig7_heatmap_{safe}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — Vote margin vs correctness (TreeBench)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_vote_margin():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "TreeBench: Vote Margin Distribution (Correct vs. Wrong)",
        fontsize=13, fontweight="bold",
    )

    for ax, preds_path, model_name, color in [
        (axes[0], QWEN_TB_PREDS, "Qwen2.5-VL (3B)", QWEN_COLOR),
        (axes[1], GRIT_TB_PREDS, "GRIT (3B)", GRIT_COLOR),
    ]:
        preds = load_jsonl(preds_path)
        correct_margin = [p["tts"]["vote_margin"] for p in preds if p["tts"]["is_correct"]]
        wrong_margin   = [p["tts"]["vote_margin"] for p in preds if not p["tts"]["is_correct"]]

        max_margin = max(
            max(correct_margin, default=0),
            max(wrong_margin, default=0),
        ) + 1
        bins = np.arange(0, max_margin + 1) - 0.5

        ax.hist(correct_margin, bins=bins, alpha=0.7, color=CORRECT_COLOR, label=f"Correct (n={len(correct_margin)})", density=True)
        ax.hist(wrong_margin,   bins=bins, alpha=0.7, color=WRONG_COLOR,   label=f"Wrong (n={len(wrong_margin)})",   density=True)
        ax.set_xlabel("Vote Margin (winning votes − runner-up votes)")
        ax.set_ylabel("Density")
        ax.set_title(f"{model_name}", fontweight="bold")
        ax.legend(fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "fig8_vote_margin.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 9 — Stage-1 vs Stage-2 accuracy contribution (TreeBench)
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_stage_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "TreeBench: Accuracy by Candidate Stage (Stage 1 = original, Stage 2 = augmented)",
        fontsize=13, fontweight="bold",
    )

    for ax, arts_path, model_name in [
        (axes[0], QWEN_TB_ARTS, "Qwen2.5-VL (3B)"),
        (axes[1], GRIT_TB_ARTS, "GRIT (3B)"),
    ]:
        arts = load_jsonl(arts_path)
        stages = {1: {"c": 0, "t": 0}, 2: {"c": 0, "t": 0}}
        for a in arts:
            s = a["stage"]
            stages[s]["t"] += 1
            if a["is_correct"]:
                stages[s]["c"] += 1

        labels = [f"Stage 1\n(n={stages[1]['t']})", f"Stage 2\n(n={stages[2]['t']})"]
        accs   = [stages[s]["c"] / stages[s]["t"] for s in [1, 2]]
        colors = [QWEN_COLOR if "Qwen" in model_name else GRIT_COLOR] * 2
        bars = ax.bar(labels, accs, color=colors, alpha=0.88, width=0.4)
        bar_label(ax, bars)
        ax.set_ylim(0, 0.55)
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Chance")
        ax.set_title(f"{model_name}", fontweight="bold")
        ax.legend(fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "fig9_stage_breakdown.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 10 — Combined transform × text-variant heatmap (TreeBench, GRIT)
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_transform_variant_heatmap():
    transforms = ["brightness_contrast", "edge_enhance", "grayscale",
                  "jpeg_recompress", "original", "rotation"]
    variants   = ["hardcoded_paraphrase", "model_paraphrase", "original"]
    t_labels   = ["bright\ncontrast", "edge\nenhance", "gray\nscale",
                  "jpeg\ncompress", "original", "rotation"]
    v_labels   = ["Hardcoded\nParaphrase", "Model\nParaphrase", "Original"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(
        "TreeBench: Per-Candidate Accuracy  ×  Image Transform × Text Variant",
        fontsize=13, fontweight="bold",
    )

    for ax, arts_path, model_name in [
        (axes[0], QWEN_TB_ARTS, "Qwen2.5-VL (3B)"),
        (axes[1], GRIT_TB_ARTS, "GRIT (3B)"),
    ]:
        arts = load_jsonl(arts_path)
        grid = defaultdict(lambda: {"c": 0, "t": 0})
        for a in arts:
            key = (a["image_transform_id"], a["text_variant_id"])
            grid[key]["t"] += 1
            if a["is_correct"]:
                grid[key]["c"] += 1

        data = np.zeros((len(transforms), len(variants)))
        for r, t in enumerate(transforms):
            for c, v in enumerate(variants):
                d = grid[(t, v)]
                data[r, c] = d["c"] / d["t"] if d["t"] else 0.0

        im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(v_labels, fontsize=8.5)
        ax.set_yticks(range(len(transforms)))
        ax.set_yticklabels(t_labels, fontsize=8.5)
        for r in range(len(transforms)):
            for c in range(len(variants)):
                ax.text(c, r, f"{data[r,c]:.0%}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if data[r,c] < 0.25 or data[r,c] > 0.48 else "black")
        plt.colorbar(im, ax=ax, format="{x:.0%}")
        ax.set_title(f"{model_name}", fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / "fig10_transform_variant_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ── LLM-judge accuracy helpers ───────────────────────────────────────────────
def _vqa_acc_judged(items: list[dict], voting_key: str) -> float:
    """Accuracy using LLM judge: checks if the majority-vote winner is judged correct."""
    correct = 0
    for it in items:
        if voting_key == "baseline":
            if it.get("llm_judge", False):
                correct += 1
        else:
            winner = it["voting"][voting_key]["answer"]
            norms  = it["candidate_answers_normalized"]
            judged = it["candidate_answers_judged"]
            verdict = next((ok for ans, ok in zip(norms, judged) if ans == winner), False)
            if verdict:
                correct += 1
    return correct / len(items) if items else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1b — VQAv2: string-match (top) vs LLM-judge (bottom) TTS scaling
# ═══════════════════════════════════════════════════════════════════════════════
def fig1b_vqa_scaling_judged():
    cmp = load_json(CMP_JUDGED_FILE)

    tasks        = ["vqa", "counting", "ocr"]
    task_labels  = ["VQA", "Counting", "OCR"]
    models = {
        "Qwen2.5-VL (3B, no CoT)": ("Qwen", QWEN_COLOR),
        "GRIT (3B)":               ("GRIT", GRIT_COLOR),
    }
    voting_keys   = ["baseline", "majority_3", "majority_5", "majority_9"]
    voting_labels = ["Baseline\n(×1)", "Majority-3\n(×3)", "Majority-5\n(×5)", "Majority-9\n(×9)"]

    fig, axes = plt.subplots(
        2, 3, figsize=(14, 8), sharey=True,
        gridspec_kw={"hspace": 0.50},
    )
    row_titles = ["String-Match Accuracy", "LLM-Judge Accuracy"]
    fig.suptitle(
        "VQAv2 Benchmark — TTS Scaling: String-Match vs. LLM-Judge Evaluation\n"
        "Qwen2.5-VL 3B vs. GRIT 3B  |  n=20 per task",
        fontsize=13, fontweight="bold", y=1.02,
    )

    x = np.arange(len(voting_labels))
    width = 0.35

    for row, eval_label in enumerate(row_titles):
        for col, (task, task_label) in enumerate(zip(tasks, task_labels)):
            ax = axes[row][col]
            for i, (model_key, (short_name, color)) in enumerate(models.items()):
                items = cmp[model_key][task]
                accs  = []
                for vk in voting_keys:
                    if row == 0:  # string match
                        if vk == "baseline":
                            acc = accuracy(items, lambda it: it["correct"])
                        else:
                            acc = accuracy(
                                items,
                                lambda it, v=vk: it["voting"][v]["answer"] in it["references"],
                            )
                    else:  # LLM judge
                        acc = _vqa_acc_judged(items, vk)
                    accs.append(acc)

                offset = (i - 0.5) * width
                hatch  = None if row == 0 else "//"
                bars   = ax.bar(x + offset, accs, width, label=short_name,
                                color=color, alpha=0.88, hatch=hatch,
                                edgecolor="white" if hatch is None else color)
                bar_label(ax, bars)

            ax.set_xticks(x)
            ax.set_xticklabels(voting_labels, fontsize=8.5)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel(eval_label if col == 0 else "")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8)
            if row == 0:
                ax.set_title(task_label, fontweight="bold")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=9)

    # shared legend for hatch meaning
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="gray", alpha=0.6, label="String-Match (solid)"),
        Patch(facecolor="gray", alpha=0.6, hatch="//", label="LLM-Judge (hatched)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    out = OUT_DIR / "fig1b_vqa_tts_scaling_judged.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 14 — LLM-judge impact: accuracy gain over string-match
# ═══════════════════════════════════════════════════════════════════════════════
def fig14_judge_impact():
    cmp = load_json(CMP_JUDGED_FILE)
    tasks       = ["vqa", "counting", "ocr"]
    task_labels = ["VQA", "Counting", "OCR"]
    voting_keys   = ["baseline", "majority_3", "majority_5", "majority_9"]
    voting_labels = ["Baseline", "Maj-3", "Maj-5", "Maj-9"]
    model_specs = [
        ("Qwen2.5-VL (3B, no CoT)", "Qwen", QWEN_COLOR),
        ("GRIT (3B)",               "GRIT", GRIT_COLOR),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle(
        "LLM-Judge Accuracy Gain over String-Match\n"
        "(positive = judge recovers answers missed by exact string matching)",
        fontsize=13, fontweight="bold",
    )

    x = np.arange(len(voting_labels))
    width = 0.35

    for ax, task, task_label in zip(axes, tasks, task_labels):
        for i, (model_key, short_name, color) in enumerate(model_specs):
            items = cmp[model_key][task]
            deltas = []
            for vk in voting_keys:
                if vk == "baseline":
                    sm  = accuracy(items, lambda it: it["correct"])
                else:
                    sm  = accuracy(items, lambda it, v=vk: it["voting"][v]["answer"] in it["references"])
                llm = _vqa_acc_judged(items, vk)
                deltas.append(llm - sm)

            offset = (i - 0.5) * width
            bars   = ax.bar(x + offset, deltas, width, label=short_name,
                            color=color, alpha=0.88)
            bar_label(ax, bars, fmt="{:+.0%}")

        ax.set_xticks(x)
        ax.set_xticklabels(voting_labels)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Accuracy gain (LLM − String)" if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
        ax.set_title(task_label, fontweight="bold")
        ax.set_ylim(-0.05, 0.25)

    axes[0].legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "fig14_judge_impact.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 15 — Updated summary heatmaps with LLM-judge rows
# ═══════════════════════════════════════════════════════════════════════════════
def fig15_heatmap_with_judge():
    cmp_judged = load_json(CMP_JUDGED_FILE)
    qwen_preds = load_jsonl(QWEN_TB_PREDS)
    grit_preds = load_jsonl(GRIT_TB_PREDS)

    voting_keys = ["baseline", "majority_3", "majority_5", "majority_9"]
    row_labels  = [
        "Baseline (×1) — SM",  "Maj-3 (×3) — SM",  "Maj-5 (×5) — SM",  "Maj-9 (×9) — SM",
        "Baseline (×1) — LLM", "Maj-3 (×3) — LLM", "Maj-5 (×5) — LLM", "Maj-9 (×9) — LLM",
    ]
    cols      = ["VQA", "Counting", "OCR", "TreeBench"]
    task_map  = {"VQA": "vqa", "Counting": "counting", "OCR": "ocr"}

    def tb_acc(preds, vk):
        if vk == "baseline":
            return accuracy(preds, lambda p: p["baseline"]["is_correct"])
        return accuracy(preds, lambda p: p["tts"]["is_correct"])

    for model_key, (model_name, preds) in [
        ("Qwen2.5-VL (3B, no CoT)", ("Qwen2.5-VL (3B, no CoT)", qwen_preds)),
        ("GRIT (3B)",               ("GRIT (3B)",               grit_preds)),
    ]:
        data = np.zeros((8, 4))
        items_map = {t: cmp_judged[model_key][task_map[t]] for t in ["VQA", "Counting", "OCR"]}

        for r, vk in enumerate(voting_keys):
            for c, col in enumerate(cols):
                if col == "TreeBench":
                    data[r, c] = tb_acc(preds, vk)
                elif vk == "baseline":
                    data[r, c] = accuracy(items_map[col], lambda it: it["correct"])
                else:
                    data[r, c] = accuracy(
                        items_map[col],
                        lambda it, v=vk: it["voting"][v]["answer"] in it["references"],
                    )

        for r, vk in enumerate(voting_keys):
            for c, col in enumerate(cols):
                if col == "TreeBench":
                    data[r + 4, c] = tb_acc(preds, vk)  # TreeBench is MC, same value
                else:
                    data[r + 4, c] = _vqa_acc_judged(items_map[col], vk)

        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

        # separator line between SM and LLM rows
        ax.axhline(3.5, color="white", linewidth=3)

        ax.set_xticks(range(4))
        ax.set_xticklabels(cols, fontsize=10)
        ax.set_yticks(range(8))
        ax.set_yticklabels(row_labels, fontsize=8.5)

        for r in range(8):
            for c in range(4):
                ax.text(c, r, f"{data[r,c]:.0%}", ha="center", va="center",
                        fontsize=9.5, fontweight="bold",
                        color="white" if data[r,c] < 0.45 or data[r,c] > 0.75 else "black")

        plt.colorbar(im, ax=ax, format="{x:.0%}")

        # row-group labels on the right
        ax.annotate("String-Match", xy=(1.12, 0.75), xycoords="axes fraction",
                    fontsize=9, rotation=90, va="center", color="#555")
        ax.annotate("LLM-Judge",    xy=(1.12, 0.25), xycoords="axes fraction",
                    fontsize=9, rotation=90, va="center", color="#555")

        safe = model_name.replace("/","_").replace(" ","_").replace("(","").replace(")","").replace(",","")
        ax.set_title(
            f"{model_name}\nAccuracy Heatmap — String-Match vs. LLM-Judge × Benchmark",
            fontweight="bold", pad=10,
        )
        fig.tight_layout()
        out = OUT_DIR / f"fig15_heatmap_judge_{safe}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close(fig)


# ── artifact paths for all tasks ─────────────────────────────────────────────
TASK_ARTS = {
    "Qwen2.5-VL (3B)": {
        "VQA":       ROOT / "results/tts_eval/qwen_vqa/candidate_artifacts.jsonl",
        "Counting":  ROOT / "results/tts_eval/qwen_counting/candidate_artifacts.jsonl",
        "OCR":       ROOT / "results/tts_eval/qwen_ocr/candidate_artifacts.jsonl",
        "TreeBench": ROOT / "results/tts_eval/qwen_run1/candidate_artifacts.jsonl",
    },
    "GRIT (3B)": {
        "VQA":       ROOT / "results/tts_eval/grit_vqa/candidate_artifacts.jsonl",
        "Counting":  ROOT / "results/tts_eval/grit_counting/candidate_artifacts.jsonl",
        "OCR":       ROOT / "results/tts_eval/grit_ocr/candidate_artifacts.jsonl",
        "TreeBench": ROOT / "results/tts_eval/grit_run2/candidate_artifacts.jsonl",
    },
}
TASKS_ORDER  = ["VQA", "Counting", "OCR", "TreeBench"]
TRANSFORMS   = ["brightness_contrast", "edge_enhance", "grayscale",
                 "jpeg_recompress", "original", "rotation"]
T_LABELS     = ["Bright/\nContrast", "Edge\nEnhance", "Gray\nScale",
                 "JPEG\nCompress", "Original", "Rotation"]
VARIANTS     = ["hardcoded_paraphrase", "model_paraphrase", "original"]
V_LABELS     = ["Hardcoded\nParaphrase", "Model\nParaphrase", "Original"]


def _arts_acc_by(arts: list[dict], key: str, order: list[str]) -> dict[str, float]:
    """Accuracy grouped by arts[key], returning a dict keyed by order values."""
    stats: dict[str, dict] = defaultdict(lambda: {"c": 0, "t": 0})
    for a in arts:
        v = a[key]
        stats[v]["t"] += 1
        if a["is_correct"]:
            stats[v]["c"] += 1
    return {v: stats[v]["c"] / stats[v]["t"] if stats[v]["t"] else 0.0 for v in order}


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 11 — Image transform accuracy: all tasks × both models
# ═══════════════════════════════════════════════════════════════════════════════
def fig11_transform_all_tasks():
    fig, axes = plt.subplots(
        2, 4, figsize=(18, 8), sharey="row",
        gridspec_kw={"hspace": 0.45, "wspace": 0.25},
    )
    fig.suptitle(
        "Per-Image-Transform Candidate Accuracy — All Tasks",
        fontsize=14, fontweight="bold",
    )

    x = np.arange(len(TRANSFORMS))
    w = 0.55

    for row, (model_name, color) in enumerate([
        ("Qwen2.5-VL (3B)", QWEN_COLOR),
        ("GRIT (3B)",        GRIT_COLOR),
    ]):
        for col, task in enumerate(TASKS_ORDER):
            ax = axes[row][col]
            arts = load_jsonl(TASK_ARTS[model_name][task])
            acc  = _arts_acc_by(arts, "image_transform_id", TRANSFORMS)
            vals = [acc[t] for t in TRANSFORMS]

            bars = ax.bar(x, vals, w, color=color, alpha=0.85)
            bar_label(ax, bars)
            ax.set_xticks(x)
            ax.set_xticklabels(T_LABELS, fontsize=7.5)
            ax.set_ylim(0, 1.15)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8)

            if col == 0:
                ax.set_ylabel(f"{model_name}\nAccuracy", fontsize=9)
            if row == 0:
                ax.set_title(task, fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / "fig11_transform_all_tasks.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 12 — Text variant accuracy: all tasks × both models
# ═══════════════════════════════════════════════════════════════════════════════
def fig12_variant_all_tasks():
    fig, axes = plt.subplots(
        2, 4, figsize=(15, 8), sharey="row",
        gridspec_kw={"hspace": 0.45, "wspace": 0.25},
    )
    fig.suptitle(
        "Per-Text-Variant Candidate Accuracy — All Tasks",
        fontsize=14, fontweight="bold",
    )

    x = np.arange(len(VARIANTS))
    w = 0.5

    for row, (model_name, color) in enumerate([
        ("Qwen2.5-VL (3B)", QWEN_COLOR),
        ("GRIT (3B)",        GRIT_COLOR),
    ]):
        for col, task in enumerate(TASKS_ORDER):
            ax = axes[row][col]
            arts = load_jsonl(TASK_ARTS[model_name][task])
            acc  = _arts_acc_by(arts, "text_variant_id", VARIANTS)
            vals = [acc[v] for v in VARIANTS]

            bars = ax.bar(x, vals, w, color=color, alpha=0.85)
            bar_label(ax, bars)
            ax.set_xticks(x)
            ax.set_xticklabels(V_LABELS, fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8)

            if col == 0:
                ax.set_ylabel(f"{model_name}\nAccuracy", fontsize=9)
            if row == 0:
                ax.set_title(task, fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / "fig12_variant_all_tasks.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 13 — Transform × variant heatmap: all tasks × both models
# ═══════════════════════════════════════════════════════════════════════════════
def fig13_heatmap_all_tasks():
    for model_name in ["Qwen2.5-VL (3B)", "GRIT (3B)"]:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
        safe = model_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        fig.suptitle(
            f"{model_name} — Image Transform × Text Variant Accuracy (All Tasks)",
            fontsize=13, fontweight="bold",
        )

        for ax, task in zip(axes, TASKS_ORDER):
            arts = load_jsonl(TASK_ARTS[model_name][task])
            grid: dict = defaultdict(lambda: {"c": 0, "t": 0})
            for a in arts:
                key = (a["image_transform_id"], a["text_variant_id"])
                grid[key]["t"] += 1
                if a["is_correct"]:
                    grid[key]["c"] += 1

            data = np.zeros((len(TRANSFORMS), len(VARIANTS)))
            for r, t in enumerate(TRANSFORMS):
                for c, v in enumerate(VARIANTS):
                    d = grid[(t, v)]
                    data[r, c] = d["c"] / d["t"] if d["t"] else 0.0

            im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_xticks(range(len(VARIANTS)))
            ax.set_xticklabels(V_LABELS, fontsize=8)
            ax.set_yticks(range(len(TRANSFORMS)))
            ax.set_yticklabels(T_LABELS if task == TASKS_ORDER[0] else [], fontsize=8)
            for r in range(len(TRANSFORMS)):
                for c in range(len(VARIANTS)):
                    ax.text(c, r, f"{data[r,c]:.0%}", ha="center", va="center",
                            fontsize=8.5, fontweight="bold",
                            color="white" if data[r,c] < 0.3 or data[r,c] > 0.7 else "black")
            plt.colorbar(im, ax=ax, format="{x:.0%}", shrink=0.85)
            ax.set_title(task, fontweight="bold")

        fig.tight_layout()
        out = OUT_DIR / f"fig13_heatmap_all_tasks_{safe}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
        plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures…")
    fig1_vqa_scaling()
    fig2_treebench_overview()
    fig3_image_transform()
    fig4_text_variant()
    fig5_agreement_vs_correctness()
    fig6_token_logprobs()
    fig7_summary_heatmap()
    fig8_vote_margin()
    fig9_stage_breakdown()
    fig10_transform_variant_heatmap()
    fig11_transform_all_tasks()
    fig12_variant_all_tasks()
    fig13_heatmap_all_tasks()
    fig1b_vqa_scaling_judged()
    fig14_judge_impact()
    fig15_heatmap_with_judge()
    print("\nAll figures saved to results/comparison/")
