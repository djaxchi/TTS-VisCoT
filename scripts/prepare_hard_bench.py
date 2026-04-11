#!/usr/bin/env python3
"""Prepare the hard_bench dataset JSONL files.

Generates:
  data/hard_bench/vqa_100.jsonl       — MMMU-Pro (standard 10-option, test split)
  data/hard_bench/ocr_100.jsonl       — OCRBench v2 (lmms-lab/OCRBench-v2, test split)
  data/hard_bench/counting_100.jsonl  — MMStar instance-counting subset (val split)

Images are NOT stored here; they are fetched lazily by the dataset loader.
Run this script once to regenerate the JSONL files.
"""

from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path
from typing import Any

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
# OCR — OCRBench v2 (lmms-lab/OCRBench-v2)
# ---------------------------------------------------------------------------

def build_ocr() -> None:
    from datasets import load_dataset  # type: ignore

    print("Building OCR (OCRBench v2)…")
    ds = load_dataset("lmms-lab/OCRBench-v2", split="test", streaming=True)

    # Collect by question type for diverse sampling
    by_type: dict[str, list[dict]] = {}
    for row in ds:
        qt = str(row.get("type", "unknown"))
        by_type.setdefault(qt, []).append(row)

    print(f"  Available types ({len(by_type)}): { {k: len(v) for k, v in by_type.items()} }")

    # Round-robin across types for maximum type diversity
    rng = random.Random(SEED)
    ordered = sorted(by_type.keys())
    for t in ordered:
        rng.shuffle(by_type[t])

    selected: list[dict] = []
    idx_per_type = {t: 0 for t in ordered}

    while len(selected) < 100:
        made_progress = False
        for t in ordered:
            i = idx_per_type[t]
            if i < len(by_type[t]):
                selected.append(by_type[t][i])
                idx_per_type[t] += 1
                made_progress = True
                if len(selected) == 100:
                    break
        if not made_progress:
            break

    rows = []
    for r in selected:
        answers_raw = r.get("answers", [])
        if isinstance(answers_raw, list):
            answers_list = [str(a).strip() for a in answers_raw if str(a).strip()]
        else:
            answers_list = [str(answers_raw).strip()]

        primary_answer = answers_list[0] if answers_list else ""
        iid = str(r["id"])
        rows.append({
            "question_id": iid,
            "question": r["question"],
            "answer": primary_answer,
            "answers_all": answers_list,
            "image_id": iid,
            "image_source": "ocrbench_v2",
        })

    _write(OUT_DIR / "ocr_100.jsonl", rows)
    print(f"  Wrote {len(rows)} OCR rows across {len({r['image_source'] for r in rows})} source(s).")


# ---------------------------------------------------------------------------
# Counting — MMStar instance-counting subset
# ---------------------------------------------------------------------------
# MMStar (NeurIPS 2024) is curated explicitly to be vision-indispensable:
# each sample was verified to require visual content (no text-only shortcuts).
# The "instance counting" category tests object enumeration in real images
# with MCQ format (A/B/C/D), making it resistant to training contamination.

def build_counting() -> None:
    from datasets import load_dataset  # type: ignore

    print("Building Counting (MMStar — instance counting)…")
    ds = load_dataset("Lin-Chen/MMStar", split="val", streaming=True)

    candidates: list[dict] = []
    for row in ds:
        cat = str(row.get("category", "")).lower()
        if "count" in cat or "instance" in cat:
            candidates.append(row)

    print(f"  Found {len(candidates)} instance-counting candidates.")

    rng = random.Random(SEED)
    rng.shuffle(candidates)
    selected = candidates[:100]

    rows = []
    for r in selected:
        iid = str(r["index"])
        rows.append({
            "question_id": iid,
            "question": r["question"],
            "answer": str(r["answer"]).strip(),
            "image_id": iid,
            "image_source": "mmstar",
        })

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
