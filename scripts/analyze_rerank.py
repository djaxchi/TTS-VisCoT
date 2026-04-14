#!/usr/bin/env python3
"""Summarize re-ranking results from $SCRATCH/results/rerank/."""

import json
import os
import pathlib
import collections

RERANK_DIR = pathlib.Path(os.environ.get("SCRATCH", ".")) / "results" / "rerank"


def summarize(path: pathlib.Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    tasks: dict[str, dict] = collections.defaultdict(lambda: {"correct": 0, "total": 0})
    for r in rows:
        task = r.get("task", "unknown")
        tasks[task]["total"] += 1
        if r.get("rerank_correct", False):
            tasks[task]["correct"] += 1
    return dict(tasks)


def main() -> None:
    files = sorted(RERANK_DIR.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {RERANK_DIR}")
        return

    for path in files:
        tasks = summarize(path)
        overall_c = sum(v["correct"] for v in tasks.values())
        overall_t = sum(v["total"] for v in tasks.values())

        print(f"\n=== {path.name} ===")
        for task, v in sorted(tasks.items()):
            acc = v["correct"] / v["total"] * 100 if v["total"] else 0
            print(f"  {task:12s}: {v['correct']:3d}/{v['total']:3d}  =  {acc:5.1f}%")

        if overall_t:
            print(f"  {'OVERALL':12s}: {overall_c:3d}/{overall_t:3d}  =  {overall_c/overall_t*100:5.1f}%")


if __name__ == "__main__":
    main()
