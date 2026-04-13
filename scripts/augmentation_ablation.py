"""Augmentation ablation analysis.

Analyzes per-augmentation diversity from TTS candidate data. For each image
augmentation, computes how often it flips the answer away from greedy, and
whether that flip is toward or away from the correct answer.

Usage:
    python -m scripts.augmentation_ablation \
        --grit results/tts_hard_bench_t0/grit_results.jsonl \
        --qwen results/tts_hard_bench_t0/qwen3b_results.jsonl \
        --out results/augmentation_ablation/
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

IMAGE_AUGS = [
    "edge_enhance",
    "grayscale",
    "jpeg_recompress",
    "brightness_contrast",
    "rotation_90",
]


def extract_greedy_and_aug_answers(
    row: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract the greedy answer and image-augmentation-only candidates at T=0.

    Returns:
        (greedy_answer, list of aug candidate dicts) where each aug dict has
        keys: image_aug, answer, text_variant, temperature.
    """
    greedy = None
    augs = []

    for c in row["candidates"]:
        # Greedy = candidate 0: original/original/T=0.0
        if (c["image_aug"] == "original"
                and c["text_variant"] == "original"
                and c["temperature"] == 0.0):
            greedy = c["answer"]
            continue

        # Image-augmentation candidates: non-original image, original text, T=0.0
        if (c["image_aug"] != "original"
                and c["text_variant"] == "original"
                and c["temperature"] == 0.0):
            augs.append(c)

    return greedy, augs


def compute_flip_stats(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Compute per-augmentation flip statistics across all questions.

    Returns:
        Dict mapping augmentation name to stats dict with keys:
        total, flips, flip_to_correct, flip_from_correct.
    """
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "flips": 0, "flip_to_correct": 0, "flip_from_correct": 0}
    )

    for row in rows:
        gt = row["gt_answer"]
        greedy, augs = extract_greedy_and_aug_answers(row)

        if greedy is None:
            continue

        for c in augs:
            aug_name = c["image_aug"]
            aug_answer = c["answer"]

            if aug_answer is None:
                continue

            stats[aug_name]["total"] += 1

            if aug_answer != greedy:
                stats[aug_name]["flips"] += 1

                if aug_answer == gt:
                    stats[aug_name]["flip_to_correct"] += 1

                if greedy == gt:
                    stats[aug_name]["flip_from_correct"] += 1

    # Ensure all known augs appear even if no data
    for aug in IMAGE_AUGS:
        _ = stats[aug]

    return dict(stats)


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load JSONL results file."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def print_stats_table(stats: dict[str, dict[str, int]], model: str) -> None:
    """Print formatted stats table."""
    print(f"\n{'='*70}")
    print(f"  {model} — Per-augmentation flip analysis")
    print(f"{'='*70}")
    print(f"  {'Augmentation':<22} {'Total':>6} {'Flips':>6} {'Rate':>7}"
          f" {'→correct':>9} {'→wrong':>8} {'←correct':>9}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*8} {'-'*9}")

    for aug in IMAGE_AUGS:
        s = stats.get(aug, {"total": 0, "flips": 0, "flip_to_correct": 0, "flip_from_correct": 0})
        total = s["total"]
        flips = s["flips"]
        rate = f"{flips/total*100:.1f}%" if total > 0 else "n/a"
        to_correct = s["flip_to_correct"]
        to_wrong = flips - to_correct
        from_correct = s["flip_from_correct"]
        print(f"  {aug:<22} {total:>6} {flips:>6} {rate:>7}"
              f" {to_correct:>9} {to_wrong:>8} {from_correct:>9}")

    total_all = sum(s["total"] for s in stats.values())
    flips_all = sum(s["flips"] for s in stats.values())
    to_correct_all = sum(s["flip_to_correct"] for s in stats.values())
    from_correct_all = sum(s["flip_from_correct"] for s in stats.values())
    rate_all = f"{flips_all/total_all*100:.1f}%" if total_all > 0 else "n/a"
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*8} {'-'*9}")
    print(f"  {'TOTAL':<22} {total_all:>6} {flips_all:>6} {rate_all:>7}"
          f" {to_correct_all:>9} {flips_all - to_correct_all:>8} {from_correct_all:>9}")


def print_per_task_stats(rows: list[dict[str, Any]], model: str) -> None:
    """Print flip stats broken down by task."""
    tasks = sorted(set(r["task"] for r in rows))
    for task in tasks:
        task_rows = [r for r in rows if r["task"] == task]
        stats = compute_flip_stats(task_rows)
        print_stats_table(stats, f"{model} — {task} (n={len(task_rows)})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Augmentation ablation analysis")
    parser.add_argument("--grit", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/augmentation_ablation"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for label, path in [("GRIT", args.grit), ("Qwen3B", args.qwen)]:
        rows = load_results(path)
        print(f"\nLoaded {len(rows)} rows from {path}")

        # Overall
        stats = compute_flip_stats(rows)
        print_stats_table(stats, label)

        # Per task
        print_per_task_stats(rows, label)

        # Save JSON
        out_path = args.out / f"{label.lower()}_flip_stats.json"
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
