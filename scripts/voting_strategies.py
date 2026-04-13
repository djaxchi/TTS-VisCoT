"""Alternative voting strategies for TTS candidate selection.

Replays existing candidate data under different voting rules to measure
how much of the oracle gap can be closed without additional inference.

Usage:
    python -m scripts.voting_strategies \
        --data results/tts_hard_bench/grit_results.jsonl \
              results/tts_hard_bench/qwen3b_results.jsonl \
        --out results/voting_strategies/
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def vote_plurality(
    candidates: list[str | None], *, greedy: str | None
) -> str | None:
    """Standard plurality vote. Most common answer wins, ties broken by first seen."""
    counts: Counter[str] = Counter()
    order: list[str] = []
    for a in candidates:
        if a is None:
            continue
        if a not in counts:
            order.append(a)
        counts[a] += 1

    if not counts:
        return None

    max_count = max(counts.values())
    for a in order:
        if counts[a] == max_count:
            return a
    return None


def vote_greedy_tiebreak(
    candidates: list[str | None], *, greedy: str | None
) -> str | None:
    """Plurality vote, but ties are broken in favor of the greedy answer."""
    counts: Counter[str] = Counter()
    order: list[str] = []
    for a in candidates:
        if a is None:
            continue
        if a not in counts:
            order.append(a)
        counts[a] += 1

    if not counts:
        return None

    max_count = max(counts.values())
    tied = [a for a in order if counts[a] == max_count]

    if greedy in tied:
        return greedy
    return tied[0]


def vote_greedy_unless_supermajority(
    candidates: list[str | None],
    *,
    greedy: str | None,
    threshold: int,
) -> str | None:
    """Use greedy answer unless an alternative has >= threshold votes.

    Conservative strategy: trust the greedy (T=0) answer unless a strong
    consensus forms around a different answer.
    """
    counts: Counter[str] = Counter()
    order: list[str] = []
    for a in candidates:
        if a is None:
            continue
        if a not in counts:
            order.append(a)
        counts[a] += 1

    if not counts:
        return None

    if greedy is None:
        return vote_plurality(candidates, greedy=None)

    # Check if any non-greedy answer meets the threshold
    for a in order:
        if a != greedy and counts[a] >= threshold:
            return a

    return greedy


def vote_consistency_filter(
    candidates: list[str | None],
    *,
    greedy: str | None,
    min_count: int,
) -> str | None:
    """Filter out singleton answers, then vote over the rest.

    If no answer survives the filter, fall back to greedy (or plurality if
    greedy is None).
    """
    counts: Counter[str] = Counter()
    order: list[str] = []
    for a in candidates:
        if a is None:
            continue
        if a not in counts:
            order.append(a)
        counts[a] += 1

    survivors = [a for a in order if counts[a] >= min_count]

    if not survivors:
        if greedy is not None:
            return greedy
        return vote_plurality(candidates, greedy=None)

    max_count = max(counts[a] for a in survivors)
    for a in survivors:
        if counts[a] == max_count:
            return a
    return None


def evaluate_strategies(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int | float]]:
    """Evaluate all voting strategies on a set of result rows.

    Returns dict mapping strategy name to {correct, total, accuracy}.
    """
    strategy_defs = {
        "greedy": lambda cands, g: g,
        "plurality": lambda cands, g: vote_plurality(cands, greedy=g),
        "greedy_tiebreak": lambda cands, g: vote_greedy_tiebreak(cands, greedy=g),
        "supermajority_3": lambda cands, g: vote_greedy_unless_supermajority(
            cands, greedy=g, threshold=3),
        "supermajority_4": lambda cands, g: vote_greedy_unless_supermajority(
            cands, greedy=g, threshold=4),
        "supermajority_5": lambda cands, g: vote_greedy_unless_supermajority(
            cands, greedy=g, threshold=5),
        "consistency_2": lambda cands, g: vote_consistency_filter(
            cands, greedy=g, min_count=2),
        "consistency_3": lambda cands, g: vote_consistency_filter(
            cands, greedy=g, min_count=3),
    }

    results: dict[str, dict[str, int]] = {
        name: {"correct": 0, "total": 0} for name in strategy_defs
    }
    results["oracle"] = {"correct": 0, "total": 0}

    for row in rows:
        gt = row["gt_answer"]
        all_answers = [c["answer"] for c in row["candidates"]]
        greedy_ans = row.get("greedy")

        # If greedy not stored, extract from candidate 0
        if greedy_ans is None:
            for c in row["candidates"]:
                if c["candidate_idx"] == 0:
                    greedy_ans = c["answer"]
                    break

        for name, fn in strategy_defs.items():
            selected = fn(all_answers, greedy_ans)
            results[name]["total"] += 1
            if selected == gt:
                results[name]["correct"] += 1

        # Oracle
        results["oracle"]["total"] += 1
        non_null = [a for a in all_answers if a is not None]
        if gt in non_null:
            results["oracle"]["correct"] += 1

    # Compute accuracy
    for name in results:
        r = results[name]
        r["accuracy"] = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0.0

    return results


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load JSONL results file."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def print_results_table(
    results: dict[str, dict[str, int | float]],
    label: str,
) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Strategy':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*9}")

    # Fixed display order
    order = [
        "greedy", "plurality", "greedy_tiebreak",
        "consistency_2", "consistency_3",
        "supermajority_3", "supermajority_4", "supermajority_5",
        "oracle",
    ]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        print(f"  {name:<25} {r['correct']:>8} {r['total']:>6} {r['accuracy']:>8.1f}%")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Voting strategy comparison")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/voting_strategies"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for path in args.data:
        rows = load_results(path)
        model = path.stem.replace("_results", "")
        print(f"\nLoaded {len(rows)} rows from {path}")

        # Overall
        results = evaluate_strategies(rows)
        print_results_table(results, f"{model} — all tasks (n={len(rows)})")

        # Per task
        tasks = sorted(set(r["task"] for r in rows))
        per_task = {}
        for task in tasks:
            task_rows = [r for r in rows if r["task"] == task]
            task_results = evaluate_strategies(task_rows)
            print_results_table(task_results, f"{model} — {task} (n={len(task_rows)})")
            per_task[task] = task_results

        # Save
        out_path = args.out / f"{model}_voting.json"
        with open(out_path, "w") as f:
            json.dump({"all": results, "per_task": per_task}, f, indent=2)
        print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
