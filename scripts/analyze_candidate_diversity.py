#!/usr/bin/env python3
"""Analyze candidate diversity in TTS result JSON files.

Reads any file in the TTS.json / TTS_Hard.json / TTS_Temperature.json format
and prints per-model × per-task diversity statistics.

Usage:
    python scripts/analyze_candidate_diversity.py
    python scripts/analyze_candidate_diversity.py results/tts/TTS_Hard.json
    python scripts/analyze_candidate_diversity.py results/tts/TTS_Temperature.json
    python scripts/analyze_candidate_diversity.py results/tts/TTS.json results/tts/TTS_Temperature.json

Metrics per question:
    entropy      — Shannon entropy (bits) of normalized candidate answers
    unique_ratio — unique answers / total candidates  (0 = all same, 1 = all different)
    oracle       — 1 if the reference appears among ANY candidate
    majority     — 1 if the majority-vote winner matches the reference

Summary interpretation:
    diversity_gap = oracle_accuracy - majority_accuracy
        > 0  → voting is the bottleneck (candidates are good but vote wastes them)
        ≈ 0  → candidates are the bottleneck (oracle ≈ majority)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
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
# Core metrics
# ---------------------------------------------------------------------------

def _entropy(answers: List[Optional[str]]) -> float:
    """Shannon entropy (bits) over non-None normalized answers."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0.0
    counts = Counter(valid)
    total = len(valid)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _unique_ratio(answers: List[Optional[str]]) -> float:
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0.0
    return len(set(valid)) / len(valid)


def _oracle_hit(candidate_answers: List[Optional[str]], references: List[str]) -> bool:
    """True if any reference appears among the candidates (case-insensitive)."""
    ref_lower = {r.strip().lower() for r in references}
    for a in candidate_answers:
        if a is not None and a.strip().lower() in ref_lower:
            return True
    return False


def _majority_correct(entry: Dict[str, Any]) -> bool:
    """True if the stored majority-vote answer matches any reference."""
    return bool(entry.get("correct", False))


# ---------------------------------------------------------------------------
# Per-question analysis
# ---------------------------------------------------------------------------

def analyze_question(entry: Dict[str, Any]) -> Dict[str, Any]:
    candidates: List[Optional[str]] = entry.get("candidate_answers_normalized", [])
    references: List[str] = entry.get("references", [])
    n = len(candidates)

    return {
        "question_id":    entry.get("question_id", "?"),
        "n_candidates":   n,
        "entropy":        _entropy(candidates),
        "unique_ratio":   _unique_ratio(candidates),
        "oracle":         _oracle_hit(candidates, references),
        "majority":       _majority_correct(entry),
        "candidate_answers": candidates,
        "references":     references,
        "image_transforms": entry.get("candidate_image_transforms", []),
        "text_variants":    entry.get("candidate_text_variants", []),
    }


# ---------------------------------------------------------------------------
# Per-task summary
# ---------------------------------------------------------------------------

def summarize_task(questions: List[Dict[str, Any]]) -> Dict[str, float]:
    if not questions:
        return {}
    qs = [analyze_question(e) for e in questions]
    n_qs = len(qs)
    return {
        "n_questions":       n_qs,
        "n_candidates":      qs[0]["n_candidates"] if qs else 0,
        "mean_entropy":      sum(q["entropy"] for q in qs) / n_qs,
        "max_entropy":       math.log2(qs[0]["n_candidates"]) if qs and qs[0]["n_candidates"] > 1 else 0.0,
        "mean_unique_ratio": sum(q["unique_ratio"] for q in qs) / n_qs,
        "oracle_accuracy":   sum(q["oracle"] for q in qs) / n_qs,
        "majority_accuracy": sum(q["majority"] for q in qs) / n_qs,
        "diversity_gap":     sum(q["oracle"] - q["majority"] for q in qs) / n_qs,
        # Distribution of how often candidates agree
        "pct_all_same":      sum(1 for q in qs if q["unique_ratio"] == 0 or q["n_candidates"] <= 1 or q["entropy"] < 0.01) / n_qs,
        "pct_all_diff":      sum(1 for q in qs if q["unique_ratio"] > 0.99) / n_qs,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _bar(c: str = "─", w: int = W) -> str:
    return c * w


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _bits(v: float, max_v: float) -> str:
    frac = f" ({v / max_v * 100:.0f}% of max)" if max_v > 0 else ""
    return f"{v:.3f} bits{frac}"


def print_summary(
    label: str,
    task: str,
    s: Dict[str, float],
    questions: List[Dict[str, Any]],
    verbose: bool = False,
) -> None:
    print(f"\n  {BOLD}{label}{RESET}  ·  task={task}  ·  {int(s['n_questions'])} questions  ·  {int(s['n_candidates'])} candidates each")
    print(f"  {_bar()}")
    print(f"  {'Mean candidate entropy':<35} {_bits(s['mean_entropy'], s['max_entropy'])}")
    print(f"  {'Mean unique-answer ratio':<35} {_pct(s['mean_unique_ratio'])}")
    print(f"  {'% questions where ALL answers same':<35} {_pct(s['pct_all_same'])}")
    print(f"  {'% questions where ALL answers differ':<35} {_pct(s['pct_all_diff'])}")
    print(f"  {_bar('·')}")
    print(f"  {'Oracle accuracy':<35} {_pct(s['oracle_accuracy'])}  (ref in any candidate)")
    print(f"  {'Majority-vote accuracy':<35} {_pct(s['majority_accuracy'])}")

    gap = s["diversity_gap"]
    gap_color = GREEN if gap > 0.05 else (YELLOW if gap > 0 else RED)
    gap_interp = (
        "→ voting is bottleneck" if gap > 0.05
        else "→ candidates are bottleneck" if gap <= 0.01
        else "→ small headroom"
    )
    print(f"  {'Diversity gap (oracle − majority)':<35} {gap_color}{_pct(gap)}  {gap_interp}{RESET}")

    if verbose:
        # Show bottom 5 most diverse (entropy = 0) and top 5
        qs = sorted([analyze_question(e) for e in questions], key=lambda x: x["entropy"])
        print(f"\n  {DIM}5 least diverse questions (entropy ≈ 0):{RESET}")
        for q in qs[:5]:
            cnts = Counter(a for a in q["candidate_answers"] if a)
            print(f"    qid={q['question_id'][:12]}  entropy={q['entropy']:.2f}  dist={dict(cnts)}")
        print(f"\n  {DIM}5 most diverse questions (high entropy):{RESET}")
        for q in qs[-5:][::-1]:
            cnts = Counter(a for a in q["candidate_answers"] if a)
            print(f"    qid={q['question_id'][:12]}  entropy={q['entropy']:.2f}  dist={dict(cnts)}  oracle={q['oracle']}")


# ---------------------------------------------------------------------------
# Comparison across two files (perturbation vs temperature)
# ---------------------------------------------------------------------------

def compare_two(
    data_a: Dict[str, Any],
    label_a: str,
    data_b: Dict[str, Any],
    label_b: str,
) -> None:
    """Print a compact comparison table between two result files."""
    all_models = sorted(set(data_a) | set(data_b))
    print(f"\n{BOLD}{'═' * W}{RESET}")
    print(f"{BOLD}  Comparison: {label_a}  vs  {label_b}{RESET}")
    print(f"{BOLD}{'═' * W}{RESET}")

    header = f"  {'Model · Task':<28} {'Entropy':<12} {'Oracle':>8} {'Majority':>10} {'Gap':>8}"
    print(f"\n  {BOLD}{' ' * 28} {label_a[:12]:<12} {'→ ' + label_b[:12]:<16}{RESET}")
    print(f"  {_bar('·')}")

    for model in all_models:
        tasks_a = data_a.get(model, {})
        tasks_b = data_b.get(model, {})
        all_tasks = sorted(set(tasks_a) | set(tasks_b))
        for task in all_tasks:
            qa = tasks_a.get(task, [])
            qb = tasks_b.get(task, [])
            if not qa or not qb:
                continue
            sa = summarize_task(qa)
            sb = summarize_task(qb)

            def delta(a: float, b: float, pct: bool = True) -> str:
                d = b - a
                s = ("+" if d >= 0 else "") + (f"{d * 100:.1f}pp" if pct else f"{d:.3f}")
                c = GREEN if d > 0 else (RED if d < 0 else DIM)
                return f"{c}{s}{RESET}"

            ent_delta  = delta(sa["mean_entropy"], sb["mean_entropy"], pct=False)
            ora_delta  = delta(sa["oracle_accuracy"], sb["oracle_accuracy"])
            maj_delta  = delta(sa["majority_accuracy"], sb["majority_accuracy"])
            gap_delta  = delta(sa["diversity_gap"], sb["diversity_gap"])

            tag = f"{model[:14]} · {task}"
            print(f"  {tag:<28} ent {ent_delta}  oracle {ora_delta}  maj {maj_delta}  gap {gap_delta}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(paths: List[Path], verbose: bool = False) -> None:
    loaded: List[Tuple[Path, Dict[str, Any]]] = []
    for p in paths:
        if not p.exists():
            print(f"{RED}File not found: {p}{RESET}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        loaded.append((p, data))

    for p, data in loaded:
        print(f"\n{BOLD}{'═' * W}{RESET}")
        print(f"{BOLD}  File: {p.name}{RESET}")
        print(f"{BOLD}{'═' * W}{RESET}")

        for model_label, tasks in data.items():
            for task_name, questions in tasks.items():
                if not questions:
                    continue
                s = summarize_task(questions)
                if not s:
                    continue
                print_summary(model_label, task_name, s, questions, verbose=verbose)

    if len(loaded) == 2:
        (pa, da), (pb, db) = loaded
        compare_two(da, pa.stem, db, pb.stem)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze TTS candidate diversity")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=[RESULTS_DIR / "TTS_Hard.json"],
        help="Path(s) to TTS result JSON file(s). Pass two to get a side-by-side comparison.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-question breakdown")
    args = parser.parse_args()

    main(args.files, verbose=args.verbose)
