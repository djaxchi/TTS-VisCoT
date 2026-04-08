"""Replay and compare voting strategies using saved candidate traces."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from src.eval.vqa_eval import evaluate_vqa


def _argmax_stable(score_by_answer: Dict[str, float], candidates: List[str]) -> str:
    """Pick max score answer, tie-breaking by first appearance in candidates."""
    if not score_by_answer:
        return ""
    best = max(score_by_answer.values())
    tied = {a for a, s in score_by_answer.items() if s == best}
    for a in candidates:
        if a in tied:
            return a
    return next(iter(tied))


def _weighted_slot_vote(candidates: List[str], weights: List[float]) -> str:
    score: Dict[str, float] = {}
    for i, ans in enumerate(candidates[: len(weights)]):
        if not ans:
            continue
        score[ans] = score.get(ans, 0.0) + float(weights[i])
    return _argmax_stable(score, candidates)


def _token_majority_vote(candidates: List[str]) -> str:
    """Build an answer by majority vote at each token position.

    This replays a token-like aggregation from saved text candidates only
    (no model logits required).
    """
    valid = [c for c in candidates if c]
    if not valid:
        return ""

    token_rows = [c.split() for c in valid]
    max_len = max((len(r) for r in token_rows), default=0)
    if max_len == 0:
        return ""

    out_tokens: List[str] = []
    for pos in range(max_len):
        toks = [row[pos] for row in token_rows if pos < len(row)]
        if not toks:
            break
        counts = Counter(toks)
        top = max(counts.values())
        tied = {t for t, c in counts.items() if c == top}
        chosen = next(t for t in toks if t in tied)
        out_tokens.append(chosen)

    return " ".join(out_tokens).strip()


def replay_method_answer(
    entry: Dict[str, Any],
    method: str,
    *,
    weights: List[float] | None = None,
    threshold: float = 0.8,
) -> str:
    """Reconstruct an answer from a saved trace entry under a chosen method."""
    candidates = list(entry.get("candidate_answers_normalized") or [])
    voting = entry.get("voting", {})

    zero_shot = candidates[0] if candidates else ""
    m3 = str(voting.get("majority_3", {}).get("answer", "") or "")
    m5 = str(voting.get("majority_5", {}).get("answer", "") or "")
    agree5 = float(voting.get("majority_5", {}).get("agreement_rate", 0.0) or 0.0)

    if method == "zero_shot":
        return zero_shot
    if method == "majority_3":
        return m3
    if method == "majority_5":
        return m5
    if method == "token_majority":
        return _token_majority_vote(candidates)
    if method == "weighted_slot":
        w = weights if weights is not None else [0.30, 0.25, 0.20, 0.15, 0.10]
        return _weighted_slot_vote(candidates, w)
    if method == "gated_majority_5":
        return m5 if agree5 >= threshold else zero_shot
    if method == "gated_weighted":
        w = weights if weights is not None else [0.30, 0.25, 0.20, 0.15, 0.10]
        weighted = _weighted_slot_vote(candidates, w)
        return weighted if agree5 >= threshold else zero_shot

    raise ValueError(f"Unknown method '{method}'")


def compute_reliability_weights(entries: List[Dict[str, Any]], k: int = 5) -> List[float]:
    """Estimate per-slot reliability weights from correctness frequency.

    Note: this uses the same entries for estimating and evaluating (exploratory).
    """
    if not entries:
        return [1.0 / k] * k

    correct_counts = [0.0] * k
    totals = [0.0] * k

    for e in entries:
        refs = e.get("references", [])
        cands = list(e.get("candidate_answers_normalized") or [])
        for i in range(min(k, len(cands))):
            totals[i] += 1.0
            if evaluate_vqa(cands[i], refs):
                correct_counts[i] += 1.0

    raw = []
    for i in range(k):
        if totals[i] == 0:
            raw.append(0.0)
        else:
            raw.append(correct_counts[i] / totals[i])

    s = sum(raw)
    if s == 0:
        return [1.0 / k] * k
    return [v / s for v in raw]


def evaluate_methods_on_entries(
    entries: List[Dict[str, Any]],
    methods: List[str],
    *,
    learned_weights: List[float] | None = None,
    threshold: float = 0.8,
) -> Dict[str, Dict[str, float]]:
    """Return per-method metrics on a list of trace entries."""
    out: Dict[str, Dict[str, float]] = {}
    n = len(entries)
    for method in methods:
        correct = 0
        for e in entries:
            ans = replay_method_answer(
                e,
                method,
                weights=learned_weights,
                threshold=threshold,
            )
            refs = e.get("references", [])
            correct += int(evaluate_vqa(ans, refs))
        out[method] = {
            "n": n,
            "correct": correct,
            "accuracy": (correct / n) if n else 0.0,
        }
    return out
