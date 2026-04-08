"""
Two presentation-ready figures for TTS-VisCoT results.

Figure A — TTS accuracy per task, Qwen vs GRIT (bar chart)
Figure B — Per-variation accuracy heatmap with split Qwen / GRIT cells

Run from repo root:
    python scripts/plot_presentation.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/comparison"

# ── colours ───────────────────────────────────────────────────────────────────
QWEN_COLOR = "#4C72B0"
GRIT_COLOR = "#DD8452"

plt.rcParams.update({
    "figure.dpi": 180,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── data sources ──────────────────────────────────────────────────────────────
# VQAv2 tasks: merge *1 (q0-14, no-quant) + *2 (q15-29, no-quant) = 30 questions each
# TreeBench: original runs (20 questions each, quantized) — new runs still pending

TASK_ARTS = {
    "Qwen": {
        "VQA":       [ROOT / "results/tts_eval/qwen_vqa1/candidate_artifacts.jsonl",
                      ROOT / "results/tts_eval/qwen_vqa2/candidate_artifacts.jsonl"],
        "Counting":  [ROOT / "results/tts_eval/qwen_counting1/candidate_artifacts.jsonl",
                      ROOT / "results/tts_eval/qwen_counting2/candidate_artifacts.jsonl"],
        "OCR":       [ROOT / "results/tts_eval/qwen_ocr1/candidate_artifacts.jsonl",
                      ROOT / "results/tts_eval/qwen_ocr2/candidate_artifacts.jsonl"],
        "TreeBench": [ROOT / "results/tts_eval/qwen_run1/candidate_artifacts.jsonl",
                      ROOT / "results/tts_eval/qwen_run2/candidate_artifacts.jsonl"],
    },
    "GRIT": {
        "VQA":       [ROOT / "results/tts_eval/grit_vqa1/candidate_artifacts.jsonl",
                      ROOT / "results/tts_eval/grit_vqa2/candidate_artifacts.jsonl"],
        "Counting":  [ROOT / "results/tts_eval/grit_counting1/candidate_artifacts.jsonl",
                      ROOT / "results/tts_eval/grit_counting2/candidate_artifacts.jsonl"],
        "OCR":       [ROOT / "results/tts_eval/grit_ocr1/candidate_artifacts.jsonl",
                      ROOT / "results/tts_eval/grit_ocr2/candidate_artifacts.jsonl"],
        "TreeBench": [ROOT / "results/tts_eval/grit_run2/candidate_artifacts.jsonl",
                      ROOT / "results/tts_eval/grit_run3/candidate_artifacts.jsonl"],
    },
}

TASK_PREDS = {
    "Qwen": {
        "VQA":       [ROOT / "results/tts_eval/qwen_vqa1/predictions.jsonl",
                      ROOT / "results/tts_eval/qwen_vqa2/predictions.jsonl"],
        "Counting":  [ROOT / "results/tts_eval/qwen_counting1/predictions.jsonl",
                      ROOT / "results/tts_eval/qwen_counting2/predictions.jsonl"],
        "OCR":       [ROOT / "results/tts_eval/qwen_ocr1/predictions.jsonl",
                      ROOT / "results/tts_eval/qwen_ocr2/predictions.jsonl"],
        "TreeBench": [ROOT / "results/tts_eval/qwen_run1/predictions.jsonl",
                      ROOT / "results/tts_eval/qwen_run2/predictions.jsonl"],
    },
    "GRIT": {
        "VQA":       [ROOT / "results/tts_eval/grit_vqa1/predictions.jsonl",
                      ROOT / "results/tts_eval/grit_vqa2/predictions.jsonl"],
        "Counting":  [ROOT / "results/tts_eval/grit_counting1/predictions.jsonl",
                      ROOT / "results/tts_eval/grit_counting2/predictions.jsonl"],
        "OCR":       [ROOT / "results/tts_eval/grit_ocr1/predictions.jsonl",
                      ROOT / "results/tts_eval/grit_ocr2/predictions.jsonl"],
        "TreeBench": [ROOT / "results/tts_eval/grit_run2/predictions.jsonl",
                      ROOT / "results/tts_eval/grit_run3/predictions.jsonl"],
    },
}

TASKS = ["VQA", "Counting", "OCR", "TreeBench"]

# The 9 exact candidate slots in fixed order (image_transform, text_variant)
SLOTS: list[tuple[str, str]] = [
    ("original",            "original"),
    ("original",            "hardcoded_paraphrase"),
    ("original",            "model_paraphrase"),
    ("edge_enhance",        "original"),
    ("grayscale",           "original"),
    ("jpeg_recompress",     "original"),
    ("brightness_contrast", "original"),
    ("rotation",            "original"),
    ("edge_enhance",        "model_paraphrase"),
]
N_COLS = len(SLOTS)   # 9

# Short two-line labels shown under each column
COL_LABELS = [
    "Orig\nOrig",
    "Orig\nHard↑",
    "Orig\nModel↑",
    "Edge\nOrig",
    "Gray\nOrig",
    "JPEG\nOrig",
    "Bright\nOrig",
    "Rotat.\nOrig",
    "Edge\nModel↑",
]

# Group spans for the secondary header
# (label, first_col_index, last_col_index_inclusive)
COL_GROUPS = [
    ("Original image\n+ text variants", 0, 2),
    ("Image transforms\n+ original text", 3, 7),
    ("Mixed", 8, 8),
]


# ── loaders ───────────────────────────────────────────────────────────────────
def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_combined(paths: list[Path]) -> list[dict]:
    """Merge multiple JSONL files into one list."""
    rows = []
    for p in paths:
        if p.exists():
            rows.extend(_load_jsonl(p))
    return rows


# ── accuracy helpers ──────────────────────────────────────────────────────────
def _tts_acc_judged(items: list[dict]) -> float:
    """Majority-9 accuracy using LLM judge."""
    correct = 0
    for it in items:
        winner = it["voting"]["majority_9"]["answer"]
        verdict = next(
            (ok for a, ok in zip(it["candidate_answers_normalized"],
                                  it["candidate_answers_judged"]) if a == winner),
            False,
        )
        if verdict:
            correct += 1
    return correct / len(items)


def _tts_acc_treebench(preds: list[dict]) -> float:
    return sum(1 for p in preds if p["tts"]["is_correct"]) / len(preds)


def _top3_slot_vote_acc(arts: list[dict]) -> tuple[float, list[tuple[str, str]]]:
    """
    Among the 9 fixed (transform × variant) slots, pick the 3 with highest
    per-slot accuracy, then re-run majority vote using only those 3 candidates
    per question.  Returns (accuracy, top3_slots).
    """
    # per-slot accuracy across all questions
    slot_stats: dict[tuple, dict] = defaultdict(lambda: {"c": 0, "t": 0})
    for a in arts:
        slot = (a["image_transform_id"], a["text_variant_id"])
        slot_stats[slot]["t"] += 1
        if a["is_correct"]:
            slot_stats[slot]["c"] += 1
    slot_acc = {s: v["c"] / v["t"] for s, v in slot_stats.items()}
    top3 = sorted(slot_acc, key=lambda s: -slot_acc[s])[:3]
    top3_set = set(top3)

    # filter to top-3 slots and majority-vote per question
    by_q: dict[str, list[dict]] = defaultdict(list)
    for a in arts:
        if (a["image_transform_id"], a["text_variant_id"]) in top3_set:
            by_q[a["sample_id"]].append(a)

    correct = 0
    for cands in by_q.values():
        winner = Counter(a["normalized_answer"] for a in cands).most_common(1)[0][0]
        if any(a["normalized_answer"] == winner and a["is_correct"] for a in cands):
            correct += 1
    return correct / len(by_q) if by_q else 0.0, top3


def _variation_acc(arts: list[dict], key: str, values: list[str]) -> dict[str, float]:
    stats: dict[str, dict] = defaultdict(lambda: {"c": 0, "t": 0})
    for a in arts:
        v = a[key]
        stats[v]["t"] += 1
        if a["is_correct"]:
            stats[v]["c"] += 1
    return {v: stats[v]["c"] / stats[v]["t"] if stats[v]["t"] else 0.0 for v in values}


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE A — TTS accuracy per task: Qwen vs GRIT
# ═══════════════════════════════════════════════════════════════════════════════
def figure_a_tts_accuracy():
    qwen_acc, grit_acc = [], []
    for task in TASKS:
        for model, acc_list in [("Qwen", qwen_acc), ("GRIT", grit_acc)]:
            preds = _load_combined(TASK_PREDS[model][task])
            acc = sum(1 for p in preds if p["tts"]["is_correct"]) / len(preds)
            acc_list.append(acc)

    x     = np.arange(len(TASKS))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_q = ax.bar(x - width / 2, qwen_acc, width,
                    label="Qwen2.5-VL 3B (TTS)", color=QWEN_COLOR, alpha=0.90,
                    edgecolor="white", linewidth=0.5)
    bars_g = ax.bar(x + width / 2, grit_acc, width,
                    label="GRIT 3B (TTS)", color=GRIT_COLOR, alpha=0.90,
                    edgecolor="white", linewidth=0.5)

    # value labels
    for bars in (bars_q, bars_g):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # chance line for TreeBench (4-way MC)
    ax.annotate("Chance (25%)", xy=(2.82, 0.255), fontsize=8, color="gray",
                va="bottom", ha="left")
    ax.axhline(0.25, xmin=0.70, xmax=1.0, color="gray",
               linestyle="--", linewidth=1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=12)
    ax.set_ylabel("TTS Accuracy (Majority-9)", fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(
        "Test-Time Scaling Accuracy by Task (Majority-9)\n"
        "String-match evaluation  ·  VQAv2 = 30 questions  ·  TreeBench = 40 questions (4-way MC)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10, framealpha=0.3)

    # light vertical separator before TreeBench
    ax.axvline(2.5, color="#cccccc", linewidth=1.2, linestyle=":")
    ax.text(2.55, 1.05, "Spatial reasoning\nbenchmark", fontsize=7.5,
            color="#888", va="top")

    fig.tight_layout()
    out = OUT_DIR / "figA_tts_accuracy_per_task.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE B — Split-cell heatmap: variation × task, Qwen (top) / GRIT (bottom)
# ═══════════════════════════════════════════════════════════════════════════════
def figure_b_variation_heatmap():
    # Build data matrices using direct per-slot accuracy (no marginals)
    qwen_data = np.zeros((len(TASKS), N_COLS))
    grit_data = np.zeros((len(TASKS), N_COLS))

    for t_idx, task in enumerate(TASKS):
        for model, mat in [("Qwen", qwen_data), ("GRIT", grit_data)]:
            arts = _load_combined(TASK_ARTS[model][task])
            # per-slot accuracy
            slot_stats: dict = defaultdict(lambda: {"c": 0, "t": 0})
            for a in arts:
                slot = (a["image_transform_id"], a["text_variant_id"])
                slot_stats[slot]["t"] += 1
                if a["is_correct"]:
                    slot_stats[slot]["c"] += 1
            for c, slot in enumerate(SLOTS):
                s = slot_stats[slot]
                mat[t_idx, c] = s["c"] / s["t"] if s["t"] else 0.0

    # ── figure setup ─────────────────────────────────────────────────────────
    n_rows = len(TASKS) * 2   # 8 visual rows (Qwen + GRIT per task)

    # y-space above row 0: enough room for col labels (-0.9) and group headers (-1.9)
    Y_TOP = -2.2

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(top=0.91)          # leave room for suptitle
    ax.set_xlim(-0.5, N_COLS - 0.5)
    ax.set_ylim(Y_TOP, n_rows - 0.5)      # expanded upward to fit headers
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # alternating light/white row-pair backgrounds for readability
    BAND_COLORS = ["#f7f7f7", "#ffffff"]

    for t_idx, task in enumerate(TASKS):
        row_q = t_idx * 2
        row_g = t_idx * 2 + 1
        band  = BAND_COLORS[t_idx % 2]

        # row-pair background band
        for row in (row_q, row_g):
            ax.add_patch(mpatches.FancyBboxPatch(
                (-0.5, row - 0.5), N_COLS, 1.0,
                boxstyle="square,pad=0",
                facecolor=band, edgecolor="none", zorder=0,
            ))

        for c in range(N_COLS):
            q_val  = qwen_data[t_idx, c]
            g_val  = grit_data[t_idx, c]
            q_wins = q_val > g_val
            g_wins = g_val > q_val

            for row, val, wins in [(row_q, q_val, q_wins), (row_g, g_val, g_wins)]:
                # thin cell border
                ax.add_patch(mpatches.FancyBboxPatch(
                    (c - 0.48, row - 0.48), 0.96, 0.96,
                    boxstyle="square,pad=0",
                    facecolor="none", edgecolor="#cccccc", linewidth=0.7, zorder=1,
                ))
                # value: bold + underline effect for winner, normal otherwise
                ax.text(c, row, f"{val:.0%}",
                        ha="center", va="center", zorder=2,
                        fontsize=8.5,
                        fontweight="bold" if wins else "normal",
                        color="#111111" if wins else "#555555")

        # thick task separator (not after last task)
        if t_idx < len(TASKS) - 1:
            sep_y = row_g + 0.5
            ax.plot([-0.5, N_COLS - 0.5], [sep_y, sep_y],
                    color="#444", linewidth=1.8, zorder=5)

    # ── column labels (just above the cells) ─────────────────────────────────
    for c, label in enumerate(COL_LABELS):
        ax.text(c, -0.65, label, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#333",
                linespacing=1.3)

    # ── group headers (above the column labels) ───────────────────────────────
    group_colors = ["#e8eef7", "#fef3e8", "#f0ece8"]
    for (glabel, c_start, c_end), gcol in zip(COL_GROUPS, group_colors):
        mid = (c_start + c_end) / 2
        ax.text(mid, -1.55, glabel, ha="center", va="center",
                fontsize=8, fontweight="bold", color="#333",
                linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=gcol, edgecolor="none"))
        # bracket line underneath the group label
        ax.plot([c_start - 0.45, c_end + 0.45], [-1.18, -1.18],
                color="#aaa", linewidth=1.0, solid_capstyle="round")

    # vertical separators between column groups
    for sep_x in [2.5, 7.5]:
        ax.plot([sep_x, sep_x], [Y_TOP, n_rows - 0.5],
                color="#999", linewidth=1.2, linestyle=":", zorder=5)

    # ── row labels ────────────────────────────────────────────────────────────
    for t_idx, task in enumerate(TASKS):
        row_q = t_idx * 2
        row_g = t_idx * 2 + 1
        mid   = (row_q + row_g) / 2
        ax.text(-1.0, mid, task, ha="right", va="center",
                fontsize=10, fontweight="bold", color="#222")
        for row, lbl, col in [(row_q, "Qwen", QWEN_COLOR), (row_g, "GRIT", GRIT_COLOR)]:
            ax.text(-0.54, row, lbl, ha="right", va="center",
                    fontsize=7.5, color=col, fontstyle="italic")

    # ── legend ────────────────────────────────────────────────────────────────
    bold_patch = mpatches.Patch(facecolor="#111111",
                                label="Bold = better score (per task × variation)")
    dim_patch  = mpatches.Patch(facecolor="#555555",
                                label="Normal = lower score")
    ax.legend(handles=[bold_patch, dim_patch], loc="lower right",
              bbox_to_anchor=(1.02, -0.04), fontsize=8.5, framealpha=0.35)

    # ── title at figure level so it never overlaps axes content ──────────────
    fig.suptitle(
        "Per-Slot Candidate Accuracy — All Tasks\n"
        "Top = Qwen2.5-VL 3B  ·  Bottom = GRIT 3B  ·  Bold = better score  ·  ↑ = text variation",
        fontsize=11, fontweight="bold", y=0.97,
    )
    out = OUT_DIR / "figB_variation_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE C — TTS accuracy using only top-3 variations per task × model
# ═══════════════════════════════════════════════════════════════════════════════

# Short readable labels for each of the 9 slots
SLOT_SHORT = {
    ("original",            "original"):             "Orig × Orig",
    ("original",            "hardcoded_paraphrase"):  "Orig × HardPara",
    ("original",            "model_paraphrase"):       "Orig × ModelPara",
    ("edge_enhance",        "original"):               "Edge × Orig",
    ("grayscale",           "original"):               "Gray × Orig",
    ("jpeg_recompress",     "original"):               "JPEG × Orig",
    ("brightness_contrast", "original"):               "Bright × Orig",
    ("rotation",            "original"):               "Rot × Orig",
    ("edge_enhance",        "model_paraphrase"):        "Edge × ModelPara",
}


def figure_c_top3_accuracy():
    qwen_acc, grit_acc = [], []
    qwen_top3_labels, grit_top3_labels = [], []

    for task in TASKS:
        for model, acc_list, label_list in [
            ("Qwen", qwen_acc, qwen_top3_labels),
            ("GRIT", grit_acc, grit_top3_labels),
        ]:
            arts = _load_combined(TASK_ARTS[model][task])
            acc, top3 = _top3_slot_vote_acc(arts)
            acc_list.append(acc)
            label_list.append([SLOT_SHORT.get(s, str(s)) for s in top3])

    x     = np.arange(len(TASKS))
    width = 0.32

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bars_q = ax.bar(x - width / 2, qwen_acc, width,
                    label="Qwen2.5-VL 3B", color=QWEN_COLOR, alpha=0.90,
                    edgecolor="white", linewidth=0.5)
    bars_g = ax.bar(x + width / 2, grit_acc, width,
                    label="GRIT 3B", color=GRIT_COLOR, alpha=0.90,
                    edgecolor="white", linewidth=0.5)

    # value labels above bars
    for bars in (bars_q, bars_g):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.0%}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    # top-3 slot annotations below each bar group
    for i, task in enumerate(TASKS):
        for bars, labels in [(bars_q, qwen_top3_labels), (bars_g, grit_top3_labels)]:
            bar   = bars[i]
            cx    = bar.get_x() + bar.get_width() / 2
            slots = labels[i]
            annotation = "\n".join(f"• {s}" for s in slots)
            ax.text(cx, -0.07, annotation,
                    ha="center", va="top", fontsize=5.8,
                    color="#555", transform=ax.get_xaxis_transform(),
                    linespacing=1.4)

    # chance line for TreeBench
    ax.axhline(0.25, xmin=0.72, xmax=1.0, color="gray",
               linestyle="--", linewidth=1.1)
    ax.annotate("Chance (25%)", xy=(2.84, 0.255), fontsize=7.5, color="gray",
                va="bottom", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=12)
    ax.set_ylabel("TTS Accuracy — Top-3 Variations (Majority-3)", fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(
        "TTS Accuracy Using Top-3 Performing Variations (per Model × Task)\n"
        "Top-3 selected by per-slot candidate accuracy  ·  voted by majority among 3 candidates",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=10, framealpha=0.3)
    ax.axvline(2.5, color="#cccccc", linewidth=1.2, linestyle=":")
    ax.text(2.55, 1.05, "Spatial reasoning\nbenchmark",
            fontsize=7.5, color="#888", va="top")

    fig.tight_layout(rect=[0, 0.12, 1, 1])   # leave space at bottom for slot labels
    out = OUT_DIR / "figC_top3_variation_accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating presentation figures…")
    figure_a_tts_accuracy()
    figure_b_variation_heatmap()
    figure_c_top3_accuracy()
    print("Done.")
