"""Analyze how many of the 9 candidates are correct per question.

Answers the diagnostic question: why do voting strategies fail?

For each config × task we report:
  - Distribution of #correct candidates in {0..9}
  - Mean #correct given oracle==True
  - Fraction of oracle-correct questions where the correct answer is the MODE
  - Fraction of oracle-correct questions where the correct answer is a MINORITY
    (< plurality winner count)
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = {
    "qwen3b_standard": REPO / "results/tts_scale/qwen3b_results.jsonl",
    "qwen3b_t0":       REPO / "results/tts_scale_t0/qwen3b_results.jsonl",
    "grit_standard":   REPO / "results/tts_scale/grit_results.jsonl",
    "grit_t0":         REPO / "results/tts_scale_t0/grit_results.jsonl",
}


def norm(s): return (s or "").strip().lower()


def load(p): return [json.loads(l) for l in open(p)]


def is_correct(ans, row):
    a = norm(ans)
    if not a:
        return False
    return any(a == norm(x) for x in (row.get("answers_all") or [row["gt_answer"]]))


def analyze(rows):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    out = {}
    for task, rs in list(by_task.items()) + [("overall", rows)]:
        dist = Counter()
        correct_is_mode = 0
        correct_is_minority = 0
        oracle_rows = 0
        mean_correct_given_oracle = 0
        total_correct_tokens = 0
        for r in rs:
            cands = r["candidates"]
            n_correct = sum(1 for c in cands if is_correct(c["answer"], r))
            dist[n_correct] += 1
            total_correct_tokens += n_correct
            if n_correct > 0:
                oracle_rows += 1
                mean_correct_given_oracle += n_correct
                counts = Counter(norm(c["answer"]) for c in cands if c["answer"])
                counts.pop("", None)
                if not counts:
                    continue
                top_ans, top_cnt = counts.most_common(1)[0]
                # find the max count among CORRECT answers
                correct_counts = [c for a, c in counts.items()
                                  if any(norm(a) == norm(x) for x in (r.get("answers_all") or [r["gt_answer"]]))]
                max_correct = max(correct_counts) if correct_counts else 0
                if max_correct >= top_cnt:
                    correct_is_mode += 1
                else:
                    correct_is_minority += 1
        n = len(rs)
        out[task] = {
            "n": n,
            "dist": dict(sorted(dist.items())),
            "mean_correct_per_q": total_correct_tokens / n if n else 0,
            "oracle_n": oracle_rows,
            "oracle_rate": oracle_rows / n if n else 0,
            "mean_correct_given_oracle": mean_correct_given_oracle / oracle_rows if oracle_rows else 0,
            "correct_is_mode_frac": correct_is_mode / oracle_rows if oracle_rows else 0,
            "correct_is_minority_frac": correct_is_minority / oracle_rows if oracle_rows else 0,
        }
    return out


def fmt_dist(d):
    total = sum(d.values())
    bins = []
    for k in range(10):
        v = d.get(k, 0)
        bins.append(f"{k}:{v}({100*v/total:.0f}%)")
    return " ".join(bins)


if __name__ == "__main__":
    all_out = {}
    for cfg, p in CONFIGS.items():
        print(f"\n{'='*80}\n{cfg}  ({p.name})\n{'='*80}")
        out = analyze(load(p))
        all_out[cfg] = out
        print(f"{'task':<10} {'n':>4} {'mean#corr':>10} {'oracle%':>8} {'mean|oracle':>12} {'mode%':>8} {'minority%':>10}")
        for task in ("vqa", "ocr", "counting", "overall"):
            if task not in out:
                continue
            d = out[task]
            print(f"{task:<10} {d['n']:>4} {d['mean_correct_per_q']:>10.2f} "
                  f"{100*d['oracle_rate']:>7.1f}% {d['mean_correct_given_oracle']:>12.2f} "
                  f"{100*d['correct_is_mode_frac']:>7.1f}% {100*d['correct_is_minority_frac']:>9.1f}%")
        print("\ndistribution of #correct-in-9 (overall):")
        print("  " + fmt_dist(out["overall"]["dist"]))
        for task in ("vqa", "ocr", "counting"):
            if task in out:
                print(f"  {task:<9} " + fmt_dist(out[task]["dist"]))

    out_path = REPO / "results/analysis/candidate_correctness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved to {out_path}")
