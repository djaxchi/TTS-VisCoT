#!/usr/bin/env python3
"""Summarize re-ranking results from $SCRATCH/results/rerank/."""

import json
import os
import pathlib
import collections

RERANK_DIR = pathlib.Path(os.environ.get("SCRATCH", ".")) / "results" / "rerank"


METRICS = [
    ("correct_greedy", "Greedy"),
    ("correct_rerank",  "Rerank"),
    ("correct_vote9",   "Vote@9"),
    ("correct_oracle",  "Oracle"),
]


def summarize(path: pathlib.Path) -> dict[str, dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    tasks: dict[str, dict] = collections.defaultdict(
        lambda: {key: 0 for key, _ in METRICS} | {"total": 0}
    )
    for r in rows:
        task = r.get("task", "unknown")
        tasks[task]["total"] += 1
        for key, _ in METRICS:
            if r.get(key, False):
                tasks[task][key] += 1
    return dict(tasks)


def acc(v: dict, key: str) -> float:
    return v[key] / v["total"] * 100 if v["total"] else 0.0


def main() -> None:
    files = sorted(RERANK_DIR.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {RERANK_DIR}")
        return

    for path in files:
        tasks = summarize(path)
        # compute totals
        totals: dict = {key: sum(t[key] for t in tasks.values()) for key, _ in METRICS}
        totals["total"] = sum(t["total"] for t in tasks.values())

        header = f"{'Task':12s}" + "".join(f"  {label:>8s}" for _, label in METRICS)
        sep = "-" * len(header)
        print(f"\n=== {path.name} ===")
        print(header)
        print(sep)
        for task in sorted(tasks):
            v = tasks[task]
            row = f"{task:12s}" + "".join(f"  {acc(v, k):7.1f}%" for k, _ in METRICS)
            print(row)
        print(sep)
        row = f"{'OVERALL':12s}" + "".join(f"  {acc(totals, k):7.1f}%" for k, _ in METRICS)
        print(row)


if __name__ == "__main__":
    main()
