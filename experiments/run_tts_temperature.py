#!/usr/bin/env python3
"""Test-Time Scaling for GRIT using temperature sampling.

Motivation
──────────
Input-perturbation TTS (image augmentations + text paraphrases) may not move
the model's distribution enough to produce diverse candidates.  This script
tests the alternative: generate N candidates by sampling at temperature T > 0
from the SAME input (original image + original prompt).  Diversity is driven
entirely by decoding stochasticity, not input variation.

Experiment design
─────────────────
For each question in hard_bench (vqa / counting / ocr):

  GRIT@1  greedy  (T=0, 1 call)   ← internal baseline
  GRIT@N  temp    (T=0.7, N calls) ← temperature TTS

Both conditions run on the same questions so results are directly comparable.
Output mirrors the TTS_Hard.json schema so analyze_candidate_diversity.py
can process it without modification.

Output
──────
  results/tts/TTS_Temperature.json

  Each entry carries:
    baseline_correct   — greedy single-pass accuracy
    correct            — majority-vote accuracy over N temperature samples
    candidate_answers_normalized — all N sampled answers
    candidate_image_transforms   — all "original" (no perturbation)
    candidate_text_variants      — all "original" (no perturbation)

Usage
─────
    python experiments/run_tts_temperature.py
    python experiments/run_tts_temperature.py --n-samples 9 --temperature 0.7
    python experiments/run_tts_temperature.py --resume
    python experiments/run_tts_temperature.py --tasks vqa ocr
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
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

OUTPUT_PATH = _PROJECT_ROOT / "results" / "tts" / "TTS_Temperature.json"

MODEL_LABEL  = "GRIT (3B)"
MODEL_ID     = "yfan1997/GRIT-20-Qwen2.5-VL-3B"
LOAD_IN_8BIT = False
MAX_NEW_TOKENS = 512

TASK_LABELS  = ["vqa", "counting", "ocr"]

W = 80
BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
GREEN = "\033[32m"; RED = "\033[31m"; YELLOW = "\033[33m"; CYAN = "\033[36m"

# MCQ answer labels (MMMU-Pro uses A-J, MMStar uses A-D)
_MCQ_PATTERN = re.compile(r"(?<![A-Z0-9])([A-J])(?![A-Z0-9])", re.IGNORECASE)
_MCQ_EXPLICIT = [
    re.compile(r"^\s*[\(\[]?\s*([A-J])\s*[\)\]]?\s*[.!?]?\s*$", re.IGNORECASE),
    re.compile(r"\bOPTION\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"\b(?:ANSWER|FINAL ANSWER|CHOICE)\s*(?:IS|:)?\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"[\(\[]\s*([A-J])\s*[\)\]]", re.IGNORECASE),
]


def _normalize_mcq(text: str) -> Optional[str]:
    """Extract a single MCQ letter (A-J) from model output, or None."""
    if not text:
        return None
    upper = text.strip().upper()
    for pat in _MCQ_EXPLICIT:
        m = pat.search(upper)
        if m:
            return m.group(1).upper()
    tokens = [m.group(1).upper() for m in _MCQ_PATTERN.finditer(upper)]
    unique = set(tokens)
    if len(unique) == 1:
        return next(iter(unique))
    return None


def _normalize_open_ended(text: str) -> Optional[str]:
    """Lowercase + strip articles and trailing punctuation."""
    if not text:
        return None
    t = text.strip().lower()
    t = re.sub(r"[.!?,;:]+$", "", t).strip()
    t = re.sub(r"^(a|an|the)\s+", "", t).strip()
    return t if t else None


def _normalize(answer: str, task: str) -> Optional[str]:
    """Task-aware normalization: MCQ for vqa/counting, open-ended for ocr."""
    if task == "ocr":
        return _normalize_open_ended(answer)
    return _normalize_mcq(answer)


def _is_correct(pred: Optional[str], references: List[str], task: str) -> bool:
    """Check prediction against all reference answers (case-insensitive)."""
    if pred is None:
        return False
    pred_l = pred.strip().lower()
    # For OCR, compare against all acceptable answers
    for ref in references:
        ref_norm = _normalize(ref, task)
        if ref_norm is not None and pred_l == ref_norm:
            return True
        # Fallback: direct lowercase match
        if pred_l == ref.strip().lower():
            return True
    return False


def _majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return ranked[0][0]


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _save(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _run_single(
    model: Any,
    image: Any,
    prompt: str,
    temperature: float,
) -> tuple[Dict[str, Any], float]:
    """Run GRIT once; return (result_dict, elapsed_seconds)."""
    t0 = time.perf_counter()
    results = model.generate(
        image,
        prompt,
        n=1,
        temperature=temperature,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return results[0], time.perf_counter() - t0


def _build_prompt(question: str, task: str) -> str:
    """Build the bare inference prompt (no choices dict — options embedded in question)."""
    from src.augmentation.text import generate_question_variants
    variants = generate_question_variants(question, choices={}, add_constraint=True)
    return variants[0]  # "original" variant


def _run_question(
    model: Any,
    sample: Dict[str, Any],
    task: str,
    n_samples: int,
    temperature: float,
    q_index: int,
    q_total: int,
) -> Dict[str, Any]:
    """Run greedy baseline + N temperature samples for one question, printing per-candidate timing."""
    image    = sample["image"]
    question = sample["question"]
    refs     = sample.get("references", [sample.get("answer", "")])
    if isinstance(refs, str):
        refs = [refs]
    qid = str(sample["question_id"])

    prompt = _build_prompt(question, task)

    # ── Greedy baseline (T=0) ──────────────────────────────────────────
    baseline_out, base_t = _run_single(model, image, prompt, temperature=0.0)
    baseline_ans  = baseline_out["answer"]
    baseline_norm = _normalize(baseline_ans, task)
    baseline_ok   = _is_correct(baseline_norm, refs, task)

    b_sym = f"{GREEN}✓{RESET}" if baseline_ok else f"{RED}✗{RESET}"
    print(f"\n  [{q_index:3d}/{q_total}]  qid={qid[:14]}  ref={refs[0]!r:.20}")
    print(f"    base  {b_sym}  ans={baseline_norm!r:.12}  ({base_t:.1f}s)")

    # ── Temperature samples (T > 0) ───────────────────────────────────
    cand_raws:   List[str]           = []
    cand_norms:  List[Optional[str]] = []
    cand_times:  List[float]         = []

    for c_idx in range(n_samples):
        out, t = _run_single(model, image, prompt, temperature=temperature)
        raw    = out["answer"]
        norm   = _normalize(raw, task)
        cand_raws.append(raw)
        cand_norms.append(norm)
        cand_times.append(t)
        sym = f"{GREEN}✓{RESET}" if _is_correct(norm, refs, task) else f"{RED}✗{RESET}"
        print(f"    cand {c_idx+1:d}  {sym}  ans={norm!r:.12}  ({t:.1f}s)")

    elapsed = base_t + sum(cand_times)

    majority = _majority_vote(cand_norms)
    tts_ok   = _is_correct(majority, refs, task)

    return {
        "question_id": qid,
        "question":    question,
        "references":  refs,
        # Greedy baseline
        "baseline_answer":            baseline_ans,
        "baseline_answer_normalized": baseline_norm,
        "baseline_correct":           baseline_ok,
        "baseline_confidence":        None,
        # TTS result
        "answer":   majority or "",
        "correct":  tts_ok,
        "tokens":   0,
        "elapsed_s": elapsed,
        # Candidate detail
        "candidate_image_transforms":   ["original"] * n_samples,
        "candidate_text_variants":      ["original"] * n_samples,
        "candidate_prompts":            [prompt] * n_samples,
        "candidate_answers":            cand_raws,
        "candidate_answers_normalized": [a or "" for a in cand_norms],
        "candidate_confidences":        [None] * n_samples,
        # Metadata
        "method":      "temperature_sampling",
        "temperature": temperature,
        "voting": {
            "majority_9": {
                "answer":         majority or "",
                "vote_counts":    dict(Counter(a for a in cand_norms if a)),
                "agreement_rate": (max(Counter(a for a in cand_norms if a).values()) / max(1, len([a for a in cand_norms if a])))
                                  if any(a for a in cand_norms) else 0.0,
                "valid_votes":    len([a for a in cand_norms if a]),
            }
        },
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _check_gpu() -> None:
    """Print GPU status and warn loudly if CUDA is unavailable."""
    import torch
    if torch.cuda.is_available():
        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        print(f"{GREEN}  GPU  {name}  ({mem:.1f} GB){RESET}")
    else:
        print(f"{YELLOW}  WARNING: CUDA not available — running on CPU (will be very slow){RESET}")


def run(
    tasks: List[str],
    n_samples: int,
    temperature: float,
    resume: bool,
    n_questions: int,
) -> None:
    from src.models.grit import GRITModel
    from src.data.datasets.viscot_benchmark import load_task

    # Load checkpoint
    checkpoint = _load(OUTPUT_PATH) if resume else {}
    results: Dict[str, Any] = {MODEL_LABEL: checkpoint.get(MODEL_LABEL, {})}

    print(f"\n{'═' * W}")
    print(f"{BOLD}  GRIT Temperature-Sampling TTS{RESET}")
    print(f"  model       = {MODEL_ID}")
    print(f"  N samples   = {n_samples}  temperature = {temperature}")
    print(f"  n_questions = {n_questions} per task")
    print(f"  tasks       = {tasks}")
    print(f"  output      = {OUTPUT_PATH}")
    _check_gpu()
    print(f"{'═' * W}\n")

    print(f"{CYAN}Loading GRIT model…{RESET}")
    model = GRITModel(model_id=MODEL_ID, load_in_8bit=LOAD_IN_8BIT)
    model._load()

    # Confirm model device after loading
    import torch
    if torch.cuda.is_available() and hasattr(model, "_model") and model._model is not None:
        devices = {str(p.device) for p in model._model.parameters()}
        print(f"{GREEN}Model loaded.  Parameter devices: {devices}{RESET}\n")
    else:
        print(f"{GREEN}Model loaded.{RESET}\n")

    for task in tasks:
        already_done = results[MODEL_LABEL].get(task, [])
        done_ids = {e["question_id"] for e in already_done}

        print(f"\n{BOLD}{'─' * W}{RESET}")
        print(f"{BOLD}  Task: {task.upper()}{RESET}  ({len(already_done)} already done, target {n_questions})\n")

        try:
            samples = load_task(task, n=None)
        except FileNotFoundError as exc:
            print(f"{RED}  SKIP — {exc}{RESET}")
            continue

        # Attach references field
        for s in samples:
            if "references" not in s:
                ans_all = s.get("answers_all")
                s["references"] = ans_all if ans_all else [s["answer"]]

        task_results: List[Dict[str, Any]] = list(already_done)
        n_baseline_ok = sum(1 for e in already_done if e.get("baseline_correct"))
        n_tts_ok      = sum(1 for e in already_done if e.get("correct"))

        new_this_task = 0
        for i, samp in enumerate(samples):
            qid = str(samp["question_id"])
            if qid in done_ids:
                continue
            if new_this_task >= n_questions:
                break

            try:
                entry = _run_question(
                    model, samp, task, n_samples, temperature,
                    q_index=len(task_results) + 1,
                    q_total=n_questions,
                )
            except Exception as exc:
                import traceback
                print(f"  {RED}ERR{RESET}  qid={qid}: {exc}")
                traceback.print_exc()
                continue

            task_results.append(entry)
            new_this_task += 1
            n_baseline_ok += entry["baseline_correct"]
            n_tts_ok      += entry["correct"]

            v_sym = f"{GREEN}✓{RESET}" if entry["correct"] else f"{RED}✗{RESET}"
            n_done = len(task_results)
            vote_counts = entry.get("voting", {}).get("majority_9", {}).get("vote_counts", {})
            print(
                f"    → vote: {entry['answer']!r:.12}  tts={v_sym}  "
                f"dist={vote_counts}  total={entry['elapsed_s']:.1f}s  "
                f"[run acc  base={n_baseline_ok/n_done:.0%}  tts={n_tts_ok/n_done:.0%}]"
            )

            # Save incrementally
            results[MODEL_LABEL][task] = task_results
            _save(OUTPUT_PATH, results)

        n_done = len(task_results)
        if n_done:
            print(f"\n  {BOLD}Task summary  ({task}){RESET}")
            print(f"    GRIT@1 greedy  accuracy : {n_baseline_ok/n_done:.1%}")
            print(f"    GRIT@{n_samples} temp T={temperature} : {n_tts_ok/n_done:.1%}")
            delta = (n_tts_ok - n_baseline_ok) / n_done
            color = GREEN if delta > 0 else (RED if delta < 0 else DIM)
            print(f"    TTS delta                : {color}{delta:+.1%}{RESET}")

    print(f"\n{BOLD}{'═' * W}{RESET}")
    print(f"{GREEN}Done. Results saved to {OUTPUT_PATH}{RESET}")
    print(f"\nRun diversity analysis with:")
    print(f"  python scripts/analyze_candidate_diversity.py {OUTPUT_PATH}")
    print(f"  python scripts/analyze_candidate_diversity.py results/tts/TTS_Hard.json {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRIT temperature-sampling TTS experiment")
    parser.add_argument(
        "--tasks", nargs="+", default=TASK_LABELS,
        choices=TASK_LABELS, metavar="TASK",
        help="Tasks to run (default: all three)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=9,
        help="Number of temperature samples per question (default: 9)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--n-questions", type=int, default=15,
        help="Number of questions to run per task (default: 15)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint at OUTPUT_PATH",
    )
    args = parser.parse_args()

    run(
        tasks=args.tasks,
        n_samples=args.n_samples,
        temperature=args.temperature,
        resume=args.resume,
        n_questions=args.n_questions,
    )
