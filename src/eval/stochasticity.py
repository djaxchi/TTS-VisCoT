"""Stochasticity metrics for the TTS entropy pilot experiment.

Provides :func:`compute_entropy` (per-question Shannon entropy over answer
distributions) and :func:`entropy_summary` (mean entropy per task).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional


def compute_entropy(answers: List[Optional[str]]) -> float:
    """Compute Shannon entropy (bits) over a list of normalized answer labels.

    ``None`` entries (model failed to produce an answer) are ignored.
    Labels are uppercased before counting so ``'a'`` and ``'A'`` are the same.

    Args:
        answers: List of MCQ letter strings or ``None``.

    Returns:
        Entropy in bits in ``[0, log2(len(answers))]``.  Returns ``0.0`` for
        empty input or when all valid answers are identical.
    """
    valid = [a.upper() for a in answers if a is not None]
    if not valid:
        return 0.0
    counts = Counter(valid)
    total = len(valid)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def entropy_summary(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean entropy per task from a list of per-question result rows.

    Args:
        rows: List of dicts, each with at least ``"task"`` (str) and
              ``"entropy"`` (float) keys.

    Returns:
        Dict mapping task name to mean entropy over all rows for that task.
    """
    if not rows:
        return {}
    from collections import defaultdict
    buckets: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        buckets[row["task"]].append(row["entropy"])
    return {task: sum(vals) / len(vals) for task, vals in buckets.items()}
