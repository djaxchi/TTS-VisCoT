#!/usr/bin/env python3
"""Generate counting_100.jsonl from GQA with real object-counting questions.

Streams lmms-lab/GQA (val_balanced_questions) and keeps only questions where:
  - The question contains "how many"
  - The answer is a plausible object count (small integer 0-20 or a number word)
  - The question is NOT about a measurement or textual number (ml, minutes, years…)

Usage:
    python scripts/prepare_counting_data.py
    python scripts/prepare_counting_data.py --n 100 --out data/benchmark/counting_100.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Filter logic (public so tests can import it)
# ---------------------------------------------------------------------------

# Unit / measurement keywords that indicate OCR number-reading, not counting.
_MEASUREMENT_KEYWORDS = {
    "ml", "mg", "kg", "km", "miles", "mile", "mph", "calories", "cal",
    "minutes", "minute", "hours", "hour", "days", "day", "weeks", "week",
    "months", "month", "years", "year", "seconds", "second",
    "dollars", "cents", "pounds", "pence", "euros",
    "ounces", "liters", "litres", "gallons", "spf", "mbps",
    "percent", "%", "ratings", "copies", "blogs", "maps", "phones",
    "repayments", "blogs",
}

# English number words accepted as object counts (1–20 range).
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
}

# Maximum plausible object count.  Answers larger than this (as an integer)
# are almost certainly textual numbers, not visual counts.
_MAX_COUNT = 20


def is_object_counting_question(question: str, answer: str) -> bool:
    """Return True if (question, answer) represents a genuine object-count task.

    Genuine counting questions ask "how many <object> are …?" and have a
    small-integer answer derived by visually counting objects in the image,
    NOT by reading a number printed in the image.

    Args:
        question: The question string (any casing).
        answer: The reference answer string.

    Returns:
        True if the pair is a valid object-counting question.
    """
    q_lower = question.lower()
    a_lower = answer.strip().lower()

    # Must ask "how many".
    if "how many" not in q_lower:
        return False

    # Reject measurement / OCR keywords anywhere in the question.
    # Strip punctuation from each token so "miles?" matches "miles".
    q_words = {w.strip("?.,!:;'\"") for w in q_lower.split()}
    if q_words & _MEASUREMENT_KEYWORDS:
        return False

    # Accept word-number answers (one, two, three … twenty).
    if a_lower in _NUMBER_WORDS:
        return True

    # Accept small integer answers (0–MAX_COUNT).
    try:
        count = int(a_lower)
        return 0 <= count <= _MAX_COUNT
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# GQA streaming
# ---------------------------------------------------------------------------


def _stream_vqa2_counting(n: int) -> list[dict]:
    """Stream VQAv2 validation and return up to *n* object-counting QA rows.

    VQAv2 has a ``question_type`` field that is literally ``"how many"`` for
    counting questions and an ``answer_type`` of ``"number"`` for numeric
    answers — making it the right source for genuine object-counting questions.
    """
    from datasets import load_dataset  # type: ignore

    print(f"Streaming lmms-lab/VQAv2 (validation) for {n} counting samples…")
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)

    rows: list[dict] = []
    seen_images: set[str] = set()

    for row in ds:
        if len(rows) >= n:
            break

        question_type: str = str(row.get("question_type", "")).lower()
        answer_type: str = str(row.get("answer_type", "")).lower()

        # Only keep genuine counting questions.
        if question_type != "how many" or answer_type != "number":
            continue

        question: str = str(row.get("question", ""))
        answer: str = str(row.get("multiple_choice_answer", ""))
        question_id: str = str(row.get("question_id", ""))
        image_id: str = str(row.get("image_id", ""))

        if not is_object_counting_question(question, answer):
            continue

        # Skip duplicate images for visual diversity.
        if image_id in seen_images:
            continue
        seen_images.add(image_id)

        rows.append({
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "image_id": image_id,
            "image_source": "vqa2",
        })
        print(f"  [{len(rows):>3}/{n}]  {question[:70]}  -> {answer}")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100,
                        help="Number of counting samples to collect (default 100).")
    parser.add_argument("--out", default="data/benchmark/counting_100.jsonl",
                        metavar="PATH", help="Output JSONL path.")
    args = parser.parse_args()

    rows = _stream_vqa2_counting(args.n)

    if not rows:
        print("ERROR: no counting questions found — check dataset availability.", file=sys.stderr)
        sys.exit(1)

    out_path = _PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} counting questions -> {out_path}")


if __name__ == "__main__":
    main()
