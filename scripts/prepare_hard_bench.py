#!/usr/bin/env python3
"""Prepare the hard_bench dataset JSONL files.

Generates:
  data/hard_bench/vqa_100.jsonl       — MMMU-Pro (standard 10-option, test split)
  data/hard_bench/ocr_100.jsonl       — OCRBench v1 (echo840/OCRBench, test split)
  data/hard_bench/counting_100.jsonl  — GQA hard counting (val_balanced_instructions)

Images are NOT stored here; they are fetched lazily by the dataset loader.
Run this script once to regenerate the JSONL files.
"""

from __future__ import annotations

import ast
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "hard_bench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# ---------------------------------------------------------------------------
# VQA — MMMU-Pro (standard 10-option)
# ---------------------------------------------------------------------------

def _format_mmmu_question(question: str, options_raw: str) -> str:
    """Append lettered options to the question string."""
    try:
        options = ast.literal_eval(options_raw)
    except Exception:
        options = [o.strip() for o in options_raw.strip("[]").split(",")]
    letters = "ABCDEFGHIJ"
    opts = "\n".join(f"{letters[i]}) {opt}" for i, opt in enumerate(options))
    return f"{question}\n{opts}"


def build_vqa() -> None:
    from datasets import load_dataset  # type: ignore

    print("Building VQA (MMMU-Pro)…")
    ds = load_dataset(
        "MMMU/MMMU_Pro",
        "standard (10 options)",
        split="test",
        streaming=True,
    )

    # Sample evenly across subjects for diversity
    by_subject: dict[str, list[dict]] = {}
    for row in ds:
        subj = row.get("subject", "unknown")
        by_subject.setdefault(subj, []).append(row)

    rng = random.Random(SEED)
    subjects = sorted(by_subject.keys())
    selected: list[dict] = []
    # Round-robin across subjects until we have 100
    idx = {s: 0 for s in subjects}
    rounds = 0
    while len(selected) < 100 and rounds < 1000:
        for s in subjects:
            pool = by_subject[s]
            i = idx[s]
            if i < len(pool):
                selected.append(pool[i])
                idx[s] += 1
                if len(selected) == 100:
                    break
        rounds += 1

    rows = []
    for r in selected:
        qid = str(r["id"])
        question = _format_mmmu_question(r["question"], str(r.get("options", "[]")))
        rows.append({
            "question_id": qid,
            "question": question,
            "answer": str(r["answer"]).strip(),
            "image_id": qid,
            "image_source": "mmmu_pro",
        })

    _write(OUT_DIR / "vqa_100.jsonl", rows)
    print(f"  Wrote {len(rows)} VQA rows ({len(by_subject)} subjects).")


# ---------------------------------------------------------------------------
# OCR — OCRBench v1
# ---------------------------------------------------------------------------

def build_ocr() -> None:
    from datasets import load_dataset  # type: ignore

    print("Building OCR (OCRBench)…")
    ds = load_dataset("echo840/OCRBench", split="test", streaming=True)

    # Collect by question_type, favour harder types
    by_type: dict[str, list[dict]] = {}
    for i, row in enumerate(ds):
        qt = row.get("question_type", "unknown")
        by_type.setdefault(qt, []).append((i, row))

    print(f"  Available types: { {k: len(v) for k, v in by_type.items()} }")

    # Priority order: hardest first
    priority = [
        "Handwriting Recognition",
        "Artistic Text Recognition",
        "Irregular Text Recognition",
        "Regular Text Recognition",
        "Digit String Recognition",
    ]
    other_types = [t for t in by_type if t not in priority]
    ordered = priority + other_types

    rng = random.Random(SEED)
    selected: list[tuple[int, dict]] = []
    per_type = max(1, 100 // max(1, len([t for t in ordered if t in by_type])))

    for qt in ordered:
        if qt not in by_type:
            continue
        pool = by_type[qt]
        rng.shuffle(pool)
        selected.extend(pool[:per_type])
        if len(selected) >= 100:
            break

    # Top up if needed
    all_remaining = [
        item for qt in ordered for item in by_type.get(qt, [])
        if item not in selected
    ]
    rng.shuffle(all_remaining)
    selected.extend(all_remaining[: max(0, 100 - len(selected))])
    selected = selected[:100]

    def _parse_ocr_answer(raw: Any) -> str:
        """OCRBench stores answers as Python list strings — extract first element."""
        s = str(raw).strip()
        try:
            val = ast.literal_eval(s)
            if isinstance(val, list) and val:
                return str(val[0]).strip()
        except Exception:
            pass
        return s

    rows = []
    for idx, (ds_idx, r) in enumerate(selected):
        rows.append({
            "question_id": str(ds_idx),
            "question": r["question"],
            "answer": _parse_ocr_answer(r["answer"]),
            "image_id": str(ds_idx),
            "image_source": "ocrbench",
        })

    _write(OUT_DIR / "ocr_100.jsonl", rows)
    print(f"  Wrote {len(rows)} OCR rows.")


# ---------------------------------------------------------------------------
# Counting — ChartQA "how many / count" questions (chart element counting)
# ---------------------------------------------------------------------------
# Charts require precise visual counting of bars, segments, and data points —
# genuinely hard for 7B VLMs (~85% overall on ChartQA, lower on counting subset).
# We prefer human-authored questions and non-trivial counts (answer not a single
# small integer) to filter out the easiest cases.

_COUNT_KEYWORDS = re.compile(
    r"\b(how many|number of|count|in total|altogether)\b", re.IGNORECASE
)


def _is_trivial_count(answer: str) -> bool:
    """True if the answer is a small integer ≤ 3 (subitizable, too easy)."""
    try:
        return float(answer) <= 3
    except ValueError:
        return False


def build_counting() -> None:
    from datasets import load_dataset  # type: ignore

    print("Building Counting (ChartQA — chart element counting)…")
    ds = load_dataset("lmms-lab/ChartQA", split="test", streaming=True)

    # Collect with sequential index (no stable ID in this dataset)
    human_candidates: list[dict] = []
    augmented_candidates: list[dict] = []
    for idx, row in enumerate(ds):
        q = str(row.get("question", ""))
        a = str(row.get("answer", "")).strip()
        qtype = str(row.get("type", ""))
        if not _COUNT_KEYWORDS.search(q):
            continue
        if _is_trivial_count(a):
            continue
        entry = {
            "question_id": str(idx),
            "question": q,
            "answer": a,
            "image_id": str(idx),
            "image_source": "chartqa",
        }
        if "human" in qtype:
            human_candidates.append(entry)
        else:
            augmented_candidates.append(entry)

    print(
        f"  Found {len(human_candidates)} human + {len(augmented_candidates)} augmented candidates."
    )

    # Prefer human-authored; fill with augmented if needed
    rng = random.Random(SEED)
    rng.shuffle(human_candidates)
    rng.shuffle(augmented_candidates)
    rows = (human_candidates + augmented_candidates)[:100]

    _write(OUT_DIR / "counting_100.jsonl", rows)
    print(f"  Wrote {len(rows)} counting rows.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tasks = sys.argv[1:] or ["vqa", "ocr", "counting"]
    if "vqa" in tasks:
        build_vqa()
    if "ocr" in tasks:
        build_ocr()
    if "counting" in tasks:
        build_counting()
    print("Done. Files written to", OUT_DIR)
