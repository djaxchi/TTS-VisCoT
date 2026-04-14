"""Analyze 292-question TTS scale results.

Produces:
  1. Per-task accuracy table (greedy / vote@9 / oracle@9) for all 4 configs.
  2. Voting-strategy replay including token-level (logprob) aggregation.
  3. All outputs printed as markdown tables.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]

CONFIGS = {
    "qwen3b_standard": REPO / "results/tts_scale/qwen3b_results.jsonl",
    "qwen3b_t0":       REPO / "results/tts_scale_t0/qwen3b_results.jsonl",
    "grit_standard":   REPO / "results/tts_scale/grit_results.jsonl",
    "grit_t0":         REPO / "results/tts_scale_t0/grit_results.jsonl",
}


def load(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def normalize(s: str) -> str:
    """Normalize an answer string for comparison."""
    if s is None:
        return ""
    return s.strip().lower()


def is_correct(predicted: str, row: Dict[str, Any]) -> bool:
    """Check if prediction matches any acceptable answer."""
    pred = normalize(predicted)
    if not pred:
        return False
    for acc in row.get("answers_all", [row["gt_answer"]]):
        if pred == normalize(acc):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Voting strategies
# ─────────────────────────────────────────────────────────────────────────────

def vote_plurality(candidates: List[Dict[str, Any]]) -> str:
    counts = Counter(normalize(c["answer"]) for c in candidates if c["answer"])
    counts.pop("", None)
    if not counts:
        return ""
    # Iterate in original order to break ties by first-seen
    seen = {}
    for c in candidates:
        a = normalize(c["answer"])
        if a and a not in seen:
            seen[a] = c
    return max(counts.items(), key=lambda kv: (kv[1], -list(seen.keys()).index(kv[0])))[0]


def vote_greedy_tiebreak(candidates: List[Dict[str, Any]], greedy: str) -> str:
    counts = Counter(normalize(c["answer"]) for c in candidates if c["answer"])
    counts.pop("", None)
    if not counts:
        return normalize(greedy)
    top = max(counts.values())
    tied = [a for a, c in counts.items() if c == top]
    g = normalize(greedy)
    if g in tied:
        return g
    return tied[0]


def vote_greedy_unless_supermajority(
    candidates: List[Dict[str, Any]], greedy: str, threshold: int = 6
) -> str:
    counts = Counter(normalize(c["answer"]) for c in candidates if c["answer"])
    counts.pop("", None)
    g = normalize(greedy)
    if not counts:
        return g
    top_ans, top_cnt = max(counts.items(), key=lambda kv: kv[1])
    if top_ans != g and top_cnt >= threshold:
        return top_ans
    return g


def vote_consistency_filter(
    candidates: List[Dict[str, Any]], greedy: str, min_count: int = 2
) -> str:
    counts = Counter(normalize(c["answer"]) for c in candidates if c["answer"])
    counts.pop("", None)
    filtered = {a: c for a, c in counts.items() if c >= min_count}
    if not filtered:
        return normalize(greedy)
    return max(filtered.items(), key=lambda kv: kv[1])[0]


def vote_logprob_sum(candidates: List[Dict[str, Any]]) -> str:
    """Sum log-probs across candidates per option, argmax.

    Only works when at least one candidate has option_logprobs.
    Returns "" if no candidate has logprobs.
    """
    agg: Dict[str, float] = defaultdict(float)
    has_any = False
    for c in candidates:
        lp = c.get("option_logprobs") or {}
        if not lp:
            continue
        has_any = True
        for k, v in lp.items():
            agg[k] += v
    if not has_any:
        return ""
    best = max(agg.items(), key=lambda kv: kv[1])[0]
    return normalize(best)


def vote_logprob_mean(candidates: List[Dict[str, Any]]) -> str:
    """Average log-probs across candidates that reported them per option."""
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for c in candidates:
        lp = c.get("option_logprobs") or {}
        for k, v in lp.items():
            sums[k] += v
            counts[k] += 1
    if not sums:
        return ""
    means = {k: sums[k] / counts[k] for k in sums}
    best = max(means.items(), key=lambda kv: kv[1])[0]
    return normalize(best)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def basic_accuracy(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Per-task + overall accuracy: greedy / vote@9 / oracle@9."""
    by_task = defaultdict(lambda: {"n": 0, "greedy": 0, "vote": 0, "oracle": 0})
    for r in rows:
        t = r["task"]
        by_task[t]["n"] += 1
        if r["correct_greedy"]:
            by_task[t]["greedy"] += 1
        if r["correct_9"]:
            by_task[t]["vote"] += 1
        if r["correct_any"]:
            by_task[t]["oracle"] += 1

    out = {}
    total = {"n": 0, "greedy": 0, "vote": 0, "oracle": 0}
    for t, d in by_task.items():
        out[t] = {k: d[k] / d["n"] for k in ("greedy", "vote", "oracle")}
        out[t]["n"] = d["n"]
        for k in ("n", "greedy", "vote", "oracle"):
            total[k] += d[k]
    out["overall"] = {k: total[k] / total["n"] for k in ("greedy", "vote", "oracle")}
    out["overall"]["n"] = total["n"]
    return out


def voting_strategies_replay(rows: List[Dict[str, Any]], counting_only: bool = False) -> Dict[str, float]:
    """Run all voting strategies; return accuracy per strategy."""
    if counting_only:
        rows = [r for r in rows if r["task"] == "counting"]
    strategies = {
        "plurality":                 lambda r: vote_plurality(r["candidates"]),
        "greedy_tiebreak":           lambda r: vote_greedy_tiebreak(r["candidates"], r["greedy"]),
        "greedy_unless_supermaj":    lambda r: vote_greedy_unless_supermajority(r["candidates"], r["greedy"]),
        "consistency_filter":        lambda r: vote_consistency_filter(r["candidates"], r["greedy"]),
        "logprob_sum":               lambda r: vote_logprob_sum(r["candidates"]),
        "logprob_mean":              lambda r: vote_logprob_mean(r["candidates"]),
    }
    greedy_correct = sum(1 for r in rows if r["correct_greedy"])
    oracle_correct = sum(1 for r in rows if r["correct_any"])
    out = {
        "_n": len(rows),
        "greedy": greedy_correct / len(rows) if rows else 0.0,
        "oracle": oracle_correct / len(rows) if rows else 0.0,
    }
    for name, fn in strategies.items():
        correct = 0
        skipped = 0
        for r in rows:
            pred = fn(r)
            if not pred:
                skipped += 1
                # fall back to greedy when strategy abstains
                if r["correct_greedy"]:
                    correct += 1
            elif is_correct(pred, r):
                correct += 1
        out[name] = correct / len(rows) if rows else 0.0
        out[f"{name}_skipped"] = skipped
    return out


def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def print_basic_table() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Print the headline greedy/vote/oracle table and return the data."""
    all_results = {}
    for config_name, path in CONFIGS.items():
        all_results[config_name] = basic_accuracy(load(path))

    print("\n## Headline accuracy — 292 questions\n")
    print("| Config | Task | n | Greedy | Vote@9 | Oracle@9 | Δ(vote-greedy) |")
    print("|---|---|---|---|---|---|---|")
    for config_name, accs in all_results.items():
        for task in ["vqa", "ocr", "counting", "overall"]:
            if task not in accs:
                continue
            a = accs[task]
            delta = a["vote"] - a["greedy"]
            sign = "+" if delta >= 0 else ""
            print(f"| {config_name} | {task} | {a['n']} | {fmt_pct(a['greedy'])} | "
                  f"{fmt_pct(a['vote'])} | {fmt_pct(a['oracle'])} | {sign}{delta*100:.1f}pp |")
    return all_results


def print_voting_strategies_table() -> Dict[str, Dict[str, float]]:
    out = {}
    print("\n## Voting strategies — standard recipe, 292 questions\n")
    print("Strategies tested on Qwen3B and GRIT standard recipe. `logprob_*` strategies")
    print("use only VQA (gt in A-D subset) + counting, since OCR has no logprobs and")
    print("VQA only captured A-D.\n")
    print("| Config | n | Greedy | Plurality | Greedy+tiebreak | GreedyUnlessSupermaj | Consistency | Logprob-sum | Logprob-mean | Oracle |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for config_name in ["qwen3b_standard", "grit_standard"]:
        rows = load(CONFIGS[config_name])
        r = voting_strategies_replay(rows)
        out[config_name] = r
        print(f"| {config_name} | {r['_n']} | {fmt_pct(r['greedy'])} | {fmt_pct(r['plurality'])} | "
              f"{fmt_pct(r['greedy_tiebreak'])} | {fmt_pct(r['greedy_unless_supermaj'])} | "
              f"{fmt_pct(r['consistency_filter'])} | {fmt_pct(r['logprob_sum'])} | "
              f"{fmt_pct(r['logprob_mean'])} | {fmt_pct(r['oracle'])} |")

    print("\n### Counting-only (clean 4-option case, valid logprob comparison)\n")
    print("| Config | n | Greedy | Plurality | Logprob-sum | Logprob-mean | Oracle |")
    print("|---|---|---|---|---|---|---|")
    for config_name in ["qwen3b_standard", "grit_standard"]:
        rows = load(CONFIGS[config_name])
        r = voting_strategies_replay(rows, counting_only=True)
        out[f"{config_name}_counting"] = r
        print(f"| {config_name} | {r['_n']} | {fmt_pct(r['greedy'])} | {fmt_pct(r['plurality'])} | "
              f"{fmt_pct(r['logprob_sum'])} | {fmt_pct(r['logprob_mean'])} | {fmt_pct(r['oracle'])} |")
    return out


if __name__ == "__main__":
    basic = print_basic_table()
    voting = print_voting_strategies_table()

    # Persist as JSON for figure generation
    out_path = REPO / "results/analysis/scale_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"basic": basic, "voting": voting}, f, indent=2)
    print(f"\n\nSaved analysis to {out_path}")
