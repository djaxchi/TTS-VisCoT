#!/usr/bin/env python3
"""Replay candidate votes using confidence (logprob) weighting.

The diversity analysis showed that voting is the bottleneck: the correct answer
appears in the candidates but loses the majority vote because wrong answers
cluster.  This script re-runs voting on EXISTING results (no new inference)
using each candidate's first-answer-token log-probability as its vote weight.

Three strategies are compared side-by-side:
  majority      — standard plurality vote (current approach)
  conf_prob     — weighted by first-token probability
  conf_logprob  — weighted by exp(-|logprob|), i.e. higher weight when model
                  is more certain regardless of sign
  oracle        — upper bound: correct if ref appears in any candidate

Usage:
    python scripts/replay_confidence_voting.py
    python scripts/replay_confidence_voting.py results/tts/TTS_Hard.json
    python scripts/replay_confidence_voting.py results/tts/TTS_Temperature.json -v
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

RESULTS_DIR = _PROJECT_ROOT / "results" / "tts"

W = 72
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
GREEN = "\033[32m"; RED = "\033[31m"; YELLOW = "\033[33m"; CYAN = "\033[36m"


# ---------------------------------------------------------------------------
# Voting strategies
# ---------------------------------------------------------------------------

def _majority(answers: List[Optional[str]]) -> Optional[str]:
    valid = [a for a in answers if a]
    if not valid:
        return None
    counts = Counter(valid)
    return max(counts, key=lambda k: (counts[k], k))


def _weighted_vote(
    answers: List[Optional[str]],
    weights: List[float],
) -> Optional[str]:
    """Return the answer with the highest total weight."""
    scores: Dict[str, float] = defaultdict(float)
    for ans, w in zip(answers, weights):
        if ans:
            scores[ans] += w
    if not scores:
        return None
    return max(scores, key=lambda k: (scores[k], k))


def _conf_weights_prob(confidences: List[Optional[Dict[str, Any]]]) -> List[float]:
    """First-token probability as weight; missing → uniform 1.0."""
    weights = []
    for c in confidences:
        if c and isinstance(c.get("prob"), float):
            weights.append(max(c["prob"], 1e-9))
        else:
            weights.append(1.0)
    return weights


def _conf_weights_logprob(confidences: List[Optional[Dict[str, Any]]]) -> List[float]:
    """Certainty = exp(-|logprob|); higher when model is more certain."""
    weights = []
    for c in confidences:
        if c and isinstance(c.get("logprob"), float):
            weights.append(math.exp(-abs(c["logprob"])))
        else:
            weights.append(1.0)
    return weights


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

def _oracle(answers: List[Optional[str]], references: List[str]) -> bool:
    ref_lower = {r.strip().lower() for r in references}
    return any(a and a.strip().lower() in ref_lower for a in answers)


def _is_correct(answer: Optional[str], references: List[str]) -> bool:
    if not answer:
        return False
    a_lower = answer.strip().lower()
    return any(a_lower == r.strip().lower() for r in references)


# ---------------------------------------------------------------------------
# Per-question replay
# ---------------------------------------------------------------------------

def replay_question(entry: Dict[str, Any]) -> Dict[str, Any]:
    answers: List[Optional[str]] = entry.get("candidate_answers_normalized", [])
    confidences: List[Optional[Dict]] = entry.get("candidate_confidences", [None] * len(answers))
    references: List[str] = entry.get("references", [])

    # Pad confidences to match answers length
    if len(confidences) < len(answers):
        confidences = confidences + [None] * (len(answers) - len(confidences))

    weights_prob    = _conf_weights_prob(confidences)
    weights_logprob = _conf_weights_logprob(confidences)

    winner_majority  = _majority(answers)
    winner_prob      = _weighted_vote(answers, weights_prob)
    winner_logprob   = _weighted_vote(answers, weights_logprob)

    has_conf = any(c is not None for c in confidences)

    return {
        "question_id":      entry.get("question_id", "?"),
        "references":       references,
        "majority_correct": _is_correct(winner_majority,  references),
        "prob_correct":     _is_correct(winner_prob,      references),
        "logprob_correct":  _is_correct(winner_logprob,   references),
        "oracle":           _oracle(answers, references),
        "has_confidence":   has_conf,
        "majority_winner":  winner_majority,
        "prob_winner":      winner_prob,
        "logprob_winner":   winner_logprob,
        # Did confidence weighting CHANGE the winner?
        "prob_changed":     winner_prob != winner_majority,
        "logprob_changed":  winner_logprob != winner_majority,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(replays: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(replays)
    if n == 0:
        return {}
    has_conf = sum(1 for r in replays if r["has_confidence"])
    return {
        "n":                  n,
        "has_confidence_pct": has_conf / n,
        "oracle_accuracy":    sum(r["oracle"] for r in replays) / n,
        "majority_accuracy":  sum(r["majority_correct"] for r in replays) / n,
        "prob_accuracy":      sum(r["prob_correct"] for r in replays) / n,
        "logprob_accuracy":   sum(r["logprob_correct"] for r in replays) / n,
        # How often did confidence weighting change the vote?
        "prob_change_rate":   sum(r["prob_changed"] for r in replays) / n,
        "logprob_change_rate":sum(r["logprob_changed"] for r in replays) / n,
        # Among changed votes, how often was the change correct?
        "prob_change_helped": (
            sum(r["prob_correct"] and r["prob_changed"] for r in replays) /
            max(1, sum(r["prob_changed"] for r in replays))
        ),
        "logprob_change_helped": (
            sum(r["logprob_correct"] and r["logprob_changed"] for r in replays) /
            max(1, sum(r["logprob_changed"] for r in replays))
        ),
    }


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _delta(base: float, new: float) -> str:
    d = new - base
    color = GREEN if d > 0.005 else (RED if d < -0.005 else DIM)
    return f"{color}({d:+.1%}){RESET}"


def print_summary(model: str, task: str, s: Dict[str, Any], verbose_replays: Optional[List] = None) -> None:
    print(f"\n  {BOLD}{model}{RESET}  ·  {task}  ·  {s['n']} questions")
    print(f"  {'─' * (W - 2)}")
    print(f"  {'Confidence data available':<38} {_pct(s['has_confidence_pct'])}")
    print(f"  {'─' * (W - 2)}")
    maj = s["majority_accuracy"]
    print(f"  {'Oracle accuracy (ceiling)':<38} {_pct(s['oracle_accuracy'])}")
    print(f"  {'Majority vote (baseline)':<38} {_pct(maj)}")
    print(f"  {'Conf-weighted (prob)':<38} {_pct(s['prob_accuracy'])}  {_delta(maj, s['prob_accuracy'])}")
    print(f"  {'Conf-weighted (logprob)':<38} {_pct(s['logprob_accuracy'])}  {_delta(maj, s['logprob_accuracy'])}")
    print(f"  {'─' * (W - 2)}")
    print(f"  {'Prob-weighted changed vote':<38} {_pct(s['prob_change_rate'])}  →  helped {_pct(s['prob_change_helped'])} of changes")
    print(f"  {'Logprob-weighted changed vote':<38} {_pct(s['logprob_change_rate'])}  →  helped {_pct(s['logprob_change_helped'])} of changes")

    if verbose_replays:
        # Show questions where confidence voting changed the outcome
        changed = [r for r in verbose_replays if r["prob_changed"] or r["logprob_changed"]]
        if changed:
            print(f"\n  {DIM}Questions where confidence weighting changed winner:{RESET}")
            for r in changed[:8]:
                refs = r["references"][:1]
                print(
                    f"    qid={r['question_id'][:12]}  "
                    f"maj={r['majority_winner']!r}  "
                    f"prob={r['prob_winner']!r}  "
                    f"ref={refs!r}  "
                    f"{'✓' if r['prob_correct'] else '✗'}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(path: Path, verbose: bool) -> None:
    if not path.exists():
        print(f"{RED}File not found: {path}{RESET}")
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))

    print(f"\n{BOLD}{'═' * W}{RESET}")
    print(f"{BOLD}  Confidence-Weighted Voting Replay{RESET}")
    print(f"{BOLD}  File: {path.name}{RESET}")
    print(f"{BOLD}{'═' * W}{RESET}")

    for model_label, tasks in data.items():
        for task_name, questions in tasks.items():
            if not questions:
                continue
            replays = [replay_question(e) for e in questions]
            s = summarize(replays)
            if not s:
                continue
            print_summary(model_label, task_name, s, replays if verbose else None)

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay confidence-weighted voting on TTS results")
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=RESULTS_DIR / "TTS_Hard.json",
        help="TTS result JSON file (default: results/tts/TTS_Hard.json)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-question changes")
    args = parser.parse_args()

    main(args.file, verbose=args.verbose)
