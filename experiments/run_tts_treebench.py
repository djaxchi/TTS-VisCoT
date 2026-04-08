#!/usr/bin/env python3
"""Test-Time Scaling results for TreeBench — built from pregenerated predictions.

Reads pregenerated candidate answers from results/tts_eval/ (no model inference),
converts them to the same JSON structure as TTS.json, and writes
results/tts/TTS_TreeBench.json.

─── Dataset ──────────────────────────────────────────────────────────────────
  TreeBench — 40 spatial-reasoning multiple-choice questions (A/B/C/D).
  One task only (no sub-categories in local data).

─── Pregenerated predictions ─────────────────────────────────────────────────
  Qwen2.5-VL (3B):  qwen_run1 (example_0–19) + qwen_run2 (example_20–39)
  GRIT (3B):        grit_run1 (example_0–19) + grit_run3 (example_20–39)

  Each prediction contains 1 baseline call + 9 TTS candidates with
  raw_output, normalized_answer, elapsed_s, and option_scores (raw logits
  for A/B/C/D), from which we derive logprob/prob confidence values.

─── Output format ────────────────────────────────────────────────────────────
  Identical structure to TTS.json:
    { "<model_label>": { "treebench": [ <result>, ... ] } }

  Each result has the same keys as TTS.json entries plus:
    choices        — {"A": ..., "B": ..., "C": ..., "D": ...}
    correct_answer — ground-truth letter

Usage:
    python experiments/run_tts_treebench.py
    python experiments/run_tts_treebench.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
except Exception:
    pass

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN = "\033[36m"; GREEN = "\033[32m"; RED = "\033[31m"
W = 80

TTS_EVAL_DIR = _PROJECT_ROOT / "results" / "tts_eval"
OUTPUT_PATH  = _PROJECT_ROOT / "results" / "tts" / "TTS_TreeBench.json"
TASK_LABEL   = "treebench"

# Pregenerated prediction files per model, in order (they will be merged by image_id)
PREGENERATED: Dict[str, List[str]] = {
    "Qwen2.5-VL (3B)": ["qwen_run1", "qwen_run2"],
    "GRIT (3B)":        ["grit_run1", "grit_run3"],
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _bar(c: str = "─", w: int = W) -> str:
    return c * w


def _header(title: str) -> str:
    pad = (W - len(title) - 2) // 2
    return f"{BOLD}{'═' * pad} {title} {'═' * (W - pad - len(title) - 2)}{RESET}"


# ---------------------------------------------------------------------------
# Confidence from option_scores (raw logits → softmax → logprob)
# ---------------------------------------------------------------------------

def _option_scores_to_confidence(
    token_metadata: Dict[str, Any],
    answer: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Convert raw option logits to the same confidence dict as TTS.json.

    option_scores is a list of step dicts; we use step_1's scores dict
    (keys A/B/C/D, values are raw logits).
    """
    option_scores = token_metadata.get("option_scores", [])
    if not option_scores:
        return None
    scores_dict: Dict[str, float] = option_scores[0].get("scores", {})
    if not scores_dict:
        return None

    keys = list(scores_dict.keys())
    vals = [scores_dict[k] for k in keys]
    max_v = max(vals)
    exp_vals = [math.exp(v - max_v) for v in vals]
    total = sum(exp_vals)
    probs    = {k: exp_vals[i] / total for i, k in enumerate(keys)}
    log_probs = {k: math.log(probs[k]) for k in keys}

    target = (answer or "").strip().upper()
    target_prob   = probs.get(target, 0.0)
    target_logprob = log_probs.get(target, -float("inf"))

    top5 = [
        {
            "token":    k,
            "token_id": None,
            "prob":     probs[k],
            "logprob":  log_probs[k],
        }
        for k in sorted(probs, key=lambda k: probs[k], reverse=True)[:5]
    ]

    return {
        "answer_first_token":    target,
        "answer_first_token_id": None,
        "logprob": target_logprob,
        "prob":    target_prob,
        "top5_distribution": top5,
    }


# ---------------------------------------------------------------------------
# Result dataclass (same fields as run_test_time_scaling.py TTSResult
# plus choices / correct_answer)
# ---------------------------------------------------------------------------

@dataclass
class TTSResult:
    question_id:    str
    question:       str
    choices:        Dict[str, str]   # {"A": ..., "B": ..., "C": ..., "D": ...}
    correct_answer: str              # ground-truth letter

    baseline_answer:            str
    baseline_answer_normalized: Optional[str]
    baseline_correct:           bool
    baseline_confidence:        Optional[Dict[str, Any]]

    answer:    str    # majority_9 winner (normalised letter)
    correct:   bool
    tokens:    int
    elapsed_s: float

    candidate_image_transforms:   List[str]                       = field(default_factory=list)
    candidate_text_variants:      List[str]                       = field(default_factory=list)
    candidate_prompts:            List[str]                       = field(default_factory=list)
    candidate_answers:            List[str]                       = field(default_factory=list)
    candidate_answers_normalized: List[Optional[str]]             = field(default_factory=list)
    candidate_confidences:        List[Optional[Dict[str, Any]]]  = field(default_factory=list)

    voting: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    winning_answer_mean_logprob: Optional[float] = None
    winning_answer_mean_prob:    Optional[float] = None


def _result_to_dict(r: TTSResult) -> Dict[str, Any]:
    return {
        "question_id":    r.question_id,
        "question":       r.question,
        "choices":        r.choices,
        "correct_answer": r.correct_answer,
        "baseline_answer":             r.baseline_answer,
        "baseline_answer_normalized":  r.baseline_answer_normalized,
        "baseline_correct":            r.baseline_correct,
        "baseline_confidence":         r.baseline_confidence,
        "answer":    r.answer,
        "correct":   r.correct,
        "tokens":    r.tokens,
        "elapsed_s": r.elapsed_s,
        "candidate_image_transforms":    r.candidate_image_transforms,
        "candidate_text_variants":       r.candidate_text_variants,
        "candidate_prompts":             r.candidate_prompts,
        "candidate_answers":             r.candidate_answers,
        "candidate_answers_normalized":  r.candidate_answers_normalized,
        "candidate_confidences":         r.candidate_confidences,
        "voting":                        r.voting,
        "winning_answer_mean_logprob":   r.winning_answer_mean_logprob,
        "winning_answer_mean_prob":      r.winning_answer_mean_prob,
    }


def _dict_to_result(d: Dict[str, Any]) -> TTSResult:
    return TTSResult(
        question_id=d["question_id"],
        question=d["question"],
        choices=d.get("choices", {}),
        correct_answer=d["correct_answer"],
        baseline_answer=d.get("baseline_answer", ""),
        baseline_answer_normalized=d.get("baseline_answer_normalized"),
        baseline_correct=d.get("baseline_correct", False),
        baseline_confidence=d.get("baseline_confidence"),
        answer=d["answer"],
        correct=d["correct"],
        tokens=d["tokens"],
        elapsed_s=d["elapsed_s"],
        candidate_image_transforms=d.get("candidate_image_transforms", []),
        candidate_text_variants=d.get("candidate_text_variants", []),
        candidate_prompts=d.get("candidate_prompts", []),
        candidate_answers=d.get("candidate_answers", []),
        candidate_answers_normalized=d.get("candidate_answers_normalized", []),
        candidate_confidences=d.get("candidate_confidences", []),
        voting=d.get("voting", {}),
        winning_answer_mean_logprob=d.get("winning_answer_mean_logprob"),
        winning_answer_mean_prob=d.get("winning_answer_mean_prob"),
    )


# ---------------------------------------------------------------------------
# Majority voting (A/B/C/D)
# ---------------------------------------------------------------------------

def _majority_vote(normalized_answers: List[Optional[str]]) -> Dict[str, Any]:
    valid = [a for a in normalized_answers if a]
    if not valid:
        return {"answer": "", "vote_counts": {}, "agreement_rate": 0.0, "valid_votes": 0}
    counts = Counter(valid)
    top = max(counts.values())
    tied = {a for a, c in counts.items() if c == top}
    winner = next(a for a in valid if a in tied)
    return {
        "answer":         winner,
        "vote_counts":    dict(counts),
        "agreement_rate": top / len(valid),
        "valid_votes":    len(valid),
    }


# ---------------------------------------------------------------------------
# Convert one pregenerated prediction row → TTSResult
# ---------------------------------------------------------------------------

def _row_to_result(row: Dict[str, Any]) -> TTSResult:
    candidates: List[Dict[str, Any]] = row["tts"]["candidates"]

    # ── Per-candidate fields ──────────────────────────────────────────────────
    img_transforms  = [c["image_transform_id"]  for c in candidates]
    txt_variants    = [c["text_variant_id"]      for c in candidates]
    prompts         = [c.get("prompt", "")       for c in candidates]
    raw_answers     = [c.get("raw_output", "")   for c in candidates]
    norm_answers    = [c.get("normalized_answer") for c in candidates]
    elapsed_total   = sum(c.get("elapsed_s", 0.0) for c in candidates)
    token_total     = sum(
        len(c.get("token_metadata", {}).get("generated_token_ids", [])) or
        len((c.get("raw_output") or "").split())
        for c in candidates
    )

    confidences: List[Optional[Dict[str, Any]]] = [
        _option_scores_to_confidence(c.get("token_metadata", {}), c.get("normalized_answer"))
        for c in candidates
    ]

    # ── Baseline (separate call stored in row["baseline"]) ────────────────────
    base        = row["baseline"]
    base_raw    = base.get("raw_output", "")
    base_norm   = base.get("normalized_answer")
    base_ok     = bool(base.get("is_correct", False))
    # Derive baseline confidence from candidate 0 (original+original), same call type
    base_conf   = _option_scores_to_confidence(
        candidates[0].get("token_metadata", {}),
        base_norm,
    )

    # ── Majority vote (@9) ────────────────────────────────────────────────────
    voting = {"majority_9": _majority_vote(norm_answers)}
    final_answer  = voting["majority_9"]["answer"]
    correct_ltr   = row["correct_answer"].upper()
    final_correct = final_answer == correct_ltr if final_answer else False

    # ── Aggregate confidence for the winning answer ───────────────────────────
    winning_lps = [
        c["logprob"] for c, n in zip(confidences, norm_answers)
        if c is not None and n == final_answer
    ]
    winning_ps = [
        c["prob"] for c, n in zip(confidences, norm_answers)
        if c is not None and n == final_answer
    ]
    mean_lp = sum(winning_lps) / len(winning_lps) if winning_lps else None
    mean_p  = sum(winning_ps)  / len(winning_ps)  if winning_ps  else None

    return TTSResult(
        question_id=row["image_id"],
        question=row["question"],
        choices=row["choices"],
        correct_answer=correct_ltr,
        baseline_answer=base_raw,
        baseline_answer_normalized=base_norm,
        baseline_correct=base_ok,
        baseline_confidence=base_conf,
        answer=final_answer,
        correct=final_correct,
        tokens=token_total,
        elapsed_s=elapsed_total,
        candidate_image_transforms=img_transforms,
        candidate_text_variants=txt_variants,
        candidate_prompts=prompts,
        candidate_answers=raw_answers,
        candidate_answers_normalized=norm_answers,
        candidate_confidences=confidences,
        voting=voting,
        winning_answer_mean_logprob=mean_lp,
        winning_answer_mean_prob=mean_p,
    )


# ---------------------------------------------------------------------------
# Load pregenerated predictions for one model
# ---------------------------------------------------------------------------

def _load_model_results(run_names: List[str]) -> List[TTSResult]:
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    for run_name in run_names:
        path = TTS_EVAL_DIR / run_name / "predictions.jsonl"
        if not path.exists():
            print(f"  {RED}MISS{RESET}  {path} not found — skipping")
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                iid = row.get("image_id", "")
                rows_by_id[iid] = row
        print(f"  {DIM}Loaded {path.name} ({path.parent.name}){RESET}")

    # Sort by numeric suffix of image_id (example_0, example_1, …)
    def _sort_key(iid: str) -> int:
        try:
            return int(iid.split("_")[-1])
        except ValueError:
            return 0

    results: List[TTSResult] = []
    for iid in sorted(rows_by_id, key=_sort_key):
        try:
            results.append(_row_to_result(rows_by_id[iid]))
        except Exception as exc:
            print(f"  {RED}ERR{RESET}  {iid}: {exc}")

    return results


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_table(
    all_results: Dict[str, List[TTSResult]],
    model_labels: List[str],
) -> None:
    print(f"\n{_header('TTS TreeBench Results')}")
    col_w = 45
    print(f"  {BOLD}{'Model':<25}{'TREEBENCH':>{col_w}}{RESET}")
    print(f"  {_bar('─', W - 2)}")
    for ml in model_labels:
        res = all_results.get(ml, [])
        if not res:
            print(f"  {ml:<25}{'N/A':>{col_w}}")
            continue
        n    = len(res)
        base = sum(r.baseline_correct for r in res)
        c9   = sum(r.correct for r in res)
        cell = f"base={base/n:.0%}  @9={c9/n:.0%}  ({n} questions)"
        print(f"  {ml:<25}{cell:>{col_w}}")
    print(f"  {_bar('─', W - 2)}")
    print(f"\n  base=original-pass  @9=majority vote over 9 TTS candidates\n")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_tts_comparison(
    all_results: Dict[str, List[TTSResult]],
    model_labels: List[str],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"  {DIM}matplotlib not available — skipping plot.{RESET}")
        return

    COLORS   = ["#4C72B0", "#DD8452"]
    X_KEYS   = ["baseline", "majority_9"]
    X_LABELS = ["Baseline\n(×1)", "Majority-9\n(×9)"]

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.suptitle(
        "Test-Time Scaling — TreeBench\n"
        "Qwen2.5-VL 3B vs. GRIT 3B",
        fontsize=13, fontweight="bold", y=1.02,
    )

    x     = np.arange(len(X_LABELS))
    width = 0.35

    for model_idx, (ml, color) in enumerate(zip(model_labels, COLORS)):
        items = all_results.get(ml, [])
        if not items:
            continue
        accs = []
        for xk in X_KEYS:
            if xk == "baseline":
                acc = sum(r.baseline_correct for r in items) / len(items)
            else:
                acc = sum(r.correct for r in items) / len(items)
            accs.append(acc)

        short  = ml.split(" ")[0]
        offset = (model_idx - 0.5) * width
        bars   = ax.bar(x + offset, accs, width, label=short, color=color, alpha=0.88)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.01,
                f"{h:.0%}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_title("TreeBench (spatial reasoning, A/B/C/D)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  {GREEN}Plot saved → {out_path}{RESET}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pregenerated TreeBench TTS predictions to TTS_TreeBench.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", default=str(OUTPUT_PATH),
                        help="Destination JSON (default: results/tts/TTS_TreeBench.json).")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip conversion — regenerate plot from existing TTS_TreeBench.json.")
    args = parser.parse_args()

    print(_header("TTS TreeBench — pregenerated predictions"))

    model_labels = list(PREGENERATED.keys())
    output_path  = Path(args.output)

    # ── Plot-only ──────────────────────────────────────────────────────────────
    if args.plot_only:
        if not output_path.exists():
            print(f"{RED}No checkpoint at {output_path}{RESET}")
            sys.exit(1)
        data = json.loads(output_path.read_text(encoding="utf-8"))
        plot_results = {
            ml: [_dict_to_result(r) for r in data.get(ml, {}).get(TASK_LABEL, [])]
            for ml in model_labels
        }
        _plot_tts_comparison(plot_results, model_labels,
                             output_path.parent / "tts_scaling_treebench.png")
        _print_table(plot_results, model_labels)
        return

    # ── Load and convert ───────────────────────────────────────────────────────
    all_results: Dict[str, List[TTSResult]] = {}

    for label, run_names in PREGENERATED.items():
        print(f"\n{_header(label)}")
        results = _load_model_results(run_names)
        all_results[label] = results
        n    = len(results)
        base = sum(r.baseline_correct for r in results)
        c9   = sum(r.correct for r in results)
        print(
            f"  {GREEN}{n} questions{RESET}  "
            f"base={base/n:.0%}  @9={c9/n:.0%}" if n else f"  {RED}0 questions loaded{RESET}"
        )

    # ── Table + save + plot ────────────────────────────────────────────────────
    _print_table(all_results, model_labels)

    payload = {
        ml: {TASK_LABEL: [_result_to_dict(r) for r in all_results.get(ml, [])]}
        for ml in model_labels
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOutput saved → {output_path}")

    _plot_tts_comparison(
        all_results, model_labels,
        output_path.parent / "tts_scaling_treebench.png",
    )


if __name__ == "__main__":
    main()
