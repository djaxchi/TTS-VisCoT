#!/usr/bin/env python3
"""Build a static paraphrase cache from the first N benchmark questions.

This avoids runtime model paraphrase calls by precomputing one rewritten
question per sample.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _rewrite_question(question: str, idx: int) -> str:
    q = " ".join(question.strip().split())

    # A few deterministic structure-level rewrites, rotated by index.
    patterns = [
        (r"^From the perspective of (.+?), in which direction is (.+?) to (.+?)\?$", r"From \1's point of view, where is \2 relative to \3?"),
        (r"^Which part of (.+?) is (.+?)\?$", r"What part of \1 corresponds to \2?"),
        (r"^What is (.+?)\?$", r"Which is \1?"),
    ]
    for pattern, repl in patterns:
        if re.match(pattern, q, flags=re.IGNORECASE):
            out = re.sub(pattern, repl, q, flags=re.IGNORECASE)
            return out

    transforms = [
        lambda s: s.replace(" in which direction ", " where "),
        lambda s: s.replace(" to the ", " relative to the "),
        lambda s: s.replace("Which", "What", 1) if s.startswith("Which") else s,
        lambda s: s.replace("What", "Which", 1) if s.startswith("What") else s,
    ]
    t = transforms[idx % len(transforms)]
    out = t(q)
    if out == q:
        out = f"In the image, {q[0].lower() + q[1:] if q else q}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create static paraphrase cache JSONL from benchmark questions.")
    parser.add_argument("--input", default="data/benchmark/vqa_100.jsonl", help="Input JSONL with question rows.")
    parser.add_argument("--output", default="results/tts_eval/static_paraphrase_cache.jsonl", help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=100, help="Number of rows to rewrite.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = _load_jsonl(in_path)[: max(args.limit, 0)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            sample_id = str(row.get("question_id") or row.get("image_id") or f"row_{i}")
            question = str(row.get("question") or "").strip()
            paraphrase = _rewrite_question(question, i)
            out_row = {
                "sample_id": sample_id,
                "question": question,
                "model_paraphrase": paraphrase,
                "source": "static_offline_rewrite_v1",
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} static paraphrases to {out_path}")


if __name__ == "__main__":
    main()
