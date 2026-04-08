#!/usr/bin/env python3
"""Export TreeBench questions to a JSONL ready for external paraphrasing.

Each exported row has the question text and answer options but leaves
``model_paraphrase`` blank.  Fill that field with your external LLM, then
pass the file to run_tts_eval.py via --paraphrase-source static.

Usage
-----
Export questions (no images needed):
    python scripts/export_treebench_questions.py \\
        --n 100 \\
        --output results/treebench_paraphrase_template.jsonl

After filling in ``model_paraphrase`` in each row, run inference with:
    python experiments/run_tts_eval.py \\
        --paraphrase-source static \\
        --static-paraphrase-path results/treebench_paraphrase_template.jsonl \\
        --model-type grit \\
        --save-dir results/tts_eval/grit_run1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TreeBench questions for external paraphrase generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of questions to export (default: 100).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="TreeBench split to load (default: train).",
    )
    parser.add_argument(
        "--output",
        default="results/treebench_paraphrase_template.jsonl",
        help="Destination JSONL file (default: results/treebench_paraphrase_template.jsonl).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        print(f"ERROR: {out_path} already exists. Use --overwrite to replace it.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading TreeBench (split={args.split}, n={args.n})…")
    from src.data.datasets.treebench import TreeBenchDataset

    ds = TreeBenchDataset(split=args.split, max_samples=args.n)
    try:
        ds.load()
    except ValueError as exc:
        if "Unknown split" in str(exc) and args.split != "train":
            print(f"  Split '{args.split}' not found, falling back to 'train'.")
            ds = TreeBenchDataset(split="train", max_samples=args.n)
            ds.load()
        else:
            raise

    n_total = min(args.n, len(ds))
    print(f"  Loaded {n_total} samples.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n_total):
            ex = ds.get_example(i)
            if ex is None:
                continue
            row = {
                "sample_id": str(ex.image_id),
                "question": ex.question,
                "options": dict(ex.options),
                "correct_answer": ex.correct_answer,
                # ↓ Fill this in with your LLM before running inference
                "model_paraphrase": "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            exported += 1

    print(f"  Exported {exported} rows -> {out_path}")
    print()
    print("Next steps:")
    print(f"  1. Open {out_path}")
    print("  2. Fill in the 'model_paraphrase' field for each row using your LLM")
    print("     Prompt suggestion: 'Rewrite this question with different wording,")
    print("     same meaning, same answer options. Output only the rewritten question.'")
    print("  3. Run inference:")
    print("       python experiments/run_tts_eval.py \\")
    print("           --paraphrase-source static \\")
    print(f"           --static-paraphrase-path {out_path} \\")
    print("           --model-type grit \\")
    print("           --save-dir results/tts_eval/grit_run1")


if __name__ == "__main__":
    main()
