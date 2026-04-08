#!/usr/bin/env python3
"""Test-Time Scaling on baseline-failure questions.

─── Motivation ───────────────────────────────────────────────────────────────
  TTS is most valuable when the baseline already fails.  This script selects
  exactly the questions that expose that weakness: for each model × task we
  run a fast single-call baseline sweep on all 100 questions, collect the ones
  answered incorrectly, then run the full 9-candidate TTS on the first 30.

─── Pipeline ─────────────────────────────────────────────────────────────────
  Phase 1 — Baseline sweep (1 call / question × 100 questions × 2 models)
    • original image, original prompt, greedy decoding
    • answered correctly  → skipped (not interesting for TTS)
    • answered wrongly    → queued for TTS

  Phase 2 — TTS on failures (9 calls / question × ≤30 questions × 2 models)
    • DEFAULT_CANDIDATE_RECIPE from pipeline_tts.py (same as TTS.json)
    • majority_9 vote over all 9 candidates

─── Output ───────────────────────────────────────────────────────────────────
  results/tts/TTS_Hard.json  (same JSON structure as TTS.json)

  Each result entry carries:
    baseline_correct  = False  (always, by construction)
    correct           = whether majority_9 recovered the right answer

  The recovery rate (correct / 30) is the key metric.

─── The 9 candidates (DEFAULT_CANDIDATE_RECIPE) ──────────────────────────────
  Stage 1 — original image, three text variants:
    1. original image   + original prompt
    2. original image   + hardcoded paraphrase
    3. original image   + model/static paraphrase
  Stage 2 — image augmentations:
    4. edge_enhance        + original
    5. grayscale           + original
    6. jpeg_recompress     + original
    7. brightness_contrast + original
    8. rotation_90         + original
    9. edge_enhance        + model/static paraphrase

Usage:
    python experiments/run_tts_hard.py
    python experiments/run_tts_hard.py --resume
    python experiments/run_tts_hard.py --plot-only
    python experiments/run_tts_hard.py --n-hard 30 --n-pool 100
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
CYAN = "\033[36m"; GREEN = "\033[32m"; RED = "\033[31m"; YELLOW = "\033[33m"
W = 80

OUTPUT_PATH     = _PROJECT_ROOT / "results" / "tts" / "TTS_Hard.json"
PARAPHRASE_PATH = _PROJECT_ROOT / "results" / "questions_to_rephrase.json"

N_HARD_DEFAULT = 30   # number of baseline-failure questions to run TTS on
N_POOL_DEFAULT = 100  # how many questions to sweep for baseline failures

TEMPERATURE = 0.0  # greedy decoding

MODEL_CONFIGS: List[Dict[str, Any]] = [
    {
        "label": "Qwen2.5-VL (3B)",
        "type": "direct_vlm",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "load_in_8bit": False,
        "max_new_tokens": 256,
    },
    {
        "label": "GRIT (3B)",
        "type": "grit",
        "model_id": "yfan1997/GRIT-20-Qwen2.5-VL-3B",
        "load_in_8bit": False,
        "max_new_tokens": 512,
    },
]

TASK_LABELS = ["vqa", "counting", "ocr"]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _bar(c: str = "─", w: int = W) -> str:
    return c * w


def _header(title: str) -> str:
    pad = (W - len(title) - 2) // 2
    return f"{BOLD}{'═' * pad} {title} {'═' * (W - pad - len(title) - 2)}{RESET}"


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

def _count_tokens(model_obj: Any, text: str) -> int:
    proc = getattr(model_obj, "_processor", None)
    if proc is not None:
        tok = getattr(proc, "tokenizer", None)
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=False))
    tok = getattr(model_obj, "_tokenizer", None)
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return len(text.split())


# ---------------------------------------------------------------------------
# Single-pass inference: answer + first-answer-token confidence
# ---------------------------------------------------------------------------

def _generate_with_confidence(
    model_obj: Any,
    image: Any,
    prompt: str,
    max_new_tokens: int,
    model_type: str,
) -> Dict[str, Any]:
    """Generate answer + first-token confidence in one forward pass.

    Returns dict with keys: answer, raw_output, confidence (dict or None).
    """
    import base64
    import io as _io
    import torch
    from qwen_vl_utils import process_vision_info

    proc = model_obj._processor
    core = model_obj._model
    tokenizer = proc.tokenizer

    if model_type == "grit":
        from src.models.grit import _SYSTEM_PROMPT as _SYS, _parse_grit_answer
    else:
        from src.models.direct_vlm import _SYSTEM_PROMPT as _SYS
        _parse_grit_answer = None  # type: ignore[assignment]

    buf = _io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    uri = f"data:image;base64,{base64.b64encode(buf.getvalue()).decode()}"

    messages = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": [
            {"type": "image", "image": uri},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = proc(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
    inputs = {k: v.to(core.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        out = core.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = out.sequences[0, prompt_len:]
    scores = out.scores

    if model_type == "grit":
        raw_output = proc.batch_decode(
            [generated_ids], skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )[0].strip()
        for tok in ("<|im_end|>", "<|endoftext|>", "<|end|>", "<pad>", "<eos>"):
            raw_output = raw_output.replace(tok, "")
        answer = (_parse_grit_answer(raw_output) or raw_output).strip()
    else:
        raw_output = proc.batch_decode(
            [generated_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        answer = raw_output

    confidence: Optional[Dict[str, Any]] = None
    if answer and scores:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if answer_ids:
            target_id = answer_ids[0]
            gen_list  = generated_ids.tolist()

            if model_type == "direct_vlm":
                answer_pos: Optional[int] = 0
            else:
                answer_pos = None
                tag_ids = tokenizer.encode("<answer>", add_special_tokens=False)
                for i in range(len(gen_list) - len(tag_ids) + 1):
                    if gen_list[i: i + len(tag_ids)] == tag_ids:
                        answer_pos = i + len(tag_ids)
                        break
                if answer_pos is None:
                    for i, tid in enumerate(gen_list):
                        if tid == target_id:
                            answer_pos = i
                            break

            if answer_pos is not None and answer_pos < len(scores):
                logits_at = scores[answer_pos].squeeze(0)
                log_probs = torch.log_softmax(logits_at, dim=-1)
                logprob   = float(log_probs[target_id].item())
                prob      = float(torch.exp(log_probs[target_id]).item())
                top_vals, top_ids = torch.topk(logits_at, k=min(5, int(logits_at.shape[-1])))
                top5 = [
                    {
                        "token":    tokenizer.decode([int(tid)], skip_special_tokens=False),
                        "token_id": int(tid),
                        "prob":     float(torch.exp(log_probs[int(tid)]).item()),
                        "logprob":  float(log_probs[int(tid)].item()),
                    }
                    for _, tid in zip(top_vals.tolist(), top_ids.tolist())
                ]
                confidence = {
                    "answer_first_token":    tokenizer.decode([target_id], skip_special_tokens=False),
                    "answer_first_token_id": int(target_id),
                    "logprob": logprob,
                    "prob":    prob,
                    "top5_distribution": top5,
                }

    del inputs, out
    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.empty_cache()

    return {"answer": answer, "raw_output": raw_output, "confidence": confidence}


# ---------------------------------------------------------------------------
# Majority voting
# ---------------------------------------------------------------------------

def _majority_vote(normalized_answers: List[str]) -> Dict[str, Any]:
    valid = [a for a in normalized_answers if a]
    if not valid:
        return {"answer": "", "vote_counts": {}, "agreement_rate": 0.0, "valid_votes": 0}
    counts = Counter(valid)
    top    = max(counts.values())
    tied   = {a for a, c in counts.items() if c == top}
    winner = next(a for a in valid if a in tied)
    return {
        "answer":         winner,
        "vote_counts":    dict(counts),
        "agreement_rate": top / len(valid),
        "valid_votes":    len(valid),
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TTSResult:
    question_id: str
    question:    str
    references:  List[str]

    baseline_answer:            str
    baseline_answer_normalized: str
    baseline_correct:           bool   # always False (by construction)
    baseline_confidence:        Optional[Dict[str, Any]]

    answer:    str    # majority_9 winner
    correct:   bool   # whether TTS recovered the right answer
    tokens:    int
    elapsed_s: float

    candidate_image_transforms:   List[str]                       = field(default_factory=list)
    candidate_text_variants:      List[str]                       = field(default_factory=list)
    candidate_prompts:            List[str]                       = field(default_factory=list)
    candidate_answers:            List[str]                       = field(default_factory=list)
    candidate_answers_normalized: List[str]                       = field(default_factory=list)
    candidate_confidences:        List[Optional[Dict[str, Any]]] = field(default_factory=list)

    voting: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    winning_answer_mean_logprob: Optional[float] = None
    winning_answer_mean_prob:    Optional[float] = None


def _result_to_dict(r: TTSResult) -> Dict[str, Any]:
    return {
        "question_id": r.question_id,
        "question":    r.question,
        "references":  r.references,
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
        references=d["references"],
        baseline_answer=d.get("baseline_answer", ""),
        baseline_answer_normalized=d.get("baseline_answer_normalized", ""),
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
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    all_results: Dict[str, Dict[str, List[TTSResult]]],
    model_labels: List[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        ml: {tl: [_result_to_dict(r) for r in all_results.get(ml, {}).get(tl, [])]
             for tl in TASK_LABELS}
        for ml in model_labels
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Phase 1: baseline sweep — find questions the model fails on
# ---------------------------------------------------------------------------

def _baseline_sweep(
    model_obj: Any,
    samples: List[Dict[str, Any]],
    max_new_tokens: int,
    model_type: str,
    model_label: str,
    task: str,
    n_hard: int,
) -> List[Dict[str, Any]]:
    """Run baseline on all samples; return up to n_hard where model is wrong.

    Each returned entry is the original sample dict augmented with:
        baseline_answer, baseline_answer_normalized, baseline_confidence.
    """
    from src.eval.vqa_eval import evaluate_vqa, vqa_normalize
    from src.augment_text import generate_question_variants

    failures: List[Dict[str, Any]] = []
    n_correct = 0

    print(f"\n  {BOLD}Phase 1 — baseline sweep ({len(samples)} questions){RESET}")

    for i, samp in enumerate(samples):
        if len(failures) >= n_hard:
            print(
                f"  {GREEN}Found {n_hard} failures after {i} questions "
                f"({n_correct} correct so far) — stopping sweep.{RESET}"
            )
            break

        q    = samp["question"]
        refs = samp["references"]
        qid  = str(samp["question_id"])

        # Build the bare original prompt (same as TTS candidate 0)
        prompts = generate_question_variants(q, choices={}, add_constraint=True)
        prompt  = prompts[0]  # "original"

        try:
            out = _generate_with_confidence(
                model_obj, samp["image"], prompt, max_new_tokens, model_type,
            )
        except Exception as exc:
            print(f"    {RED}ERR{RESET}  {qid}: {exc} — skipping")
            continue

        norm    = vqa_normalize(out["answer"])
        correct = evaluate_vqa(norm, refs)

        status = f"{GREEN}✓{RESET}" if correct else f"{RED}✗{RESET}"
        print(
            f"  [{i+1:3d}/{len(samples)}] {status}  "
            f"pred={norm[:30]!r}  ref={refs[0]!r}"
        )

        if correct:
            n_correct += 1
            continue

        failures.append({
            **samp,
            "baseline_answer":            out["answer"],
            "baseline_answer_normalized": norm,
            "baseline_confidence":        out["confidence"],
        })

    n_scanned = min(i + 1, len(samples))
    print(
        f"\n  Scanned {n_scanned} questions: "
        f"{YELLOW}{len(failures)} failures{RESET}, {n_correct} correct"
    )
    return failures


# ---------------------------------------------------------------------------
# Phase 2: full TTS on failure questions
# ---------------------------------------------------------------------------

def _run_tts_on_failures(
    model_obj: Any,
    failure_samples: List[Dict[str, Any]],
    max_new_tokens: int,
    model_type: str,
    model_label: str,
    task: str,
    paraphrase_lookup: Optional[Dict[tuple, str]] = None,
) -> List[TTSResult]:
    from src.pipeline_tts import DEFAULT_CANDIDATE_RECIPE, build_candidate_inputs
    from src.augment_image import ImageVariationConfig
    from src.eval.vqa_eval import evaluate_vqa, vqa_normalize

    image_config = ImageVariationConfig(preset="strong")
    results: List[TTSResult] = []

    print(f"\n  {BOLD}Phase 2 — TTS on {len(failure_samples)} failure questions{RESET}")

    for i, samp in enumerate(failure_samples):
        q    = samp["question"]
        refs = samp["references"]
        qid  = str(samp["question_id"])

        base_raw  = samp["baseline_answer"]
        base_norm = samp["baseline_answer_normalized"]
        base_conf = samp["baseline_confidence"]

        print(
            f"\n  [{model_label}] {task.upper()} {i+1}/{len(failure_samples)}: "
            f"{q[:60]}{'…' if len(q) > 60 else ''}"
        )
        print(f"    baseline: {RED}✗{RESET}  pred={base_norm!r}  ref={refs[0]!r}")

        # Paraphrase
        static_para = (paraphrase_lookup or {}).get((task, qid))

        def _para_fn(
            _q: str,
            _choices: Dict[str, str],
            _idx: int,
            _p: str = static_para or "",
        ) -> str:
            return _p if _p else _q

        model_paraphrase_fn: Optional[Callable] = _para_fn if static_para else None

        # Build 9 candidate inputs
        t0 = time.perf_counter()
        try:
            candidates_input = build_candidate_inputs(
                image=samp["image"],
                question=q,
                choices={},
                max_candidates=len(DEFAULT_CANDIDATE_RECIPE),
                candidate_recipe=DEFAULT_CANDIDATE_RECIPE,
                model_paraphrase_fn=model_paraphrase_fn,
                image_config=image_config,
            )
        except Exception as exc:
            print(f"    {RED}SKIP{RESET}  build_candidate_inputs error: {exc}")
            continue

        img_transforms: List[str]                       = []
        txt_variants:   List[str]                       = []
        prompts:        List[str]                       = []
        raw_answers:    List[str]                       = []
        confidences:    List[Optional[Dict[str, Any]]] = []
        total_tokens = 0
        skip = False

        for c in candidates_input:
            try:
                out = _generate_with_confidence(
                    model_obj, c["image"], c["prompt"], max_new_tokens, model_type,
                )
            except Exception as exc:
                print(f"    {RED}SKIP{RESET}  candidate {c['candidate_id']} error: {exc}")
                skip = True
                break

            img_transforms.append(c.get("image_transform_id", "original"))
            txt_variants.append(c.get("text_variant_id", "original"))
            prompts.append(c["prompt"])
            raw_answers.append(out["answer"])
            total_tokens += _count_tokens(model_obj, out.get("raw_output", out["answer"]))
            confidences.append(out["confidence"])

        if skip:
            continue

        elapsed = time.perf_counter() - t0

        # Majority vote @9
        normalized = [vqa_normalize(a) for a in raw_answers]
        voting: Dict[str, Dict[str, Any]] = {"majority_9": _majority_vote(normalized)}
        final_answer  = voting["majority_9"]["answer"]
        final_correct = evaluate_vqa(final_answer, refs)

        # Aggregate confidence for the winning answer
        winning_lps = [
            c["logprob"] for c, n in zip(confidences, normalized)
            if c is not None and n == final_answer
        ]
        winning_ps = [
            c["prob"] for c, n in zip(confidences, normalized)
            if c is not None and n == final_answer
        ]
        mean_lp = sum(winning_lps) / len(winning_lps) if winning_lps else None
        mean_p  = sum(winning_ps)  / len(winning_ps)  if winning_ps  else None

        recovery_tag = f"{GREEN}RECOVERED ✓{RESET}" if final_correct else f"{RED}still wrong ✗{RESET}"
        conf_str = f"  conf={mean_p:.2f}" if mean_p is not None else ""
        print(
            f"    TTS @9={final_answer[:25]!r}  {recovery_tag}"
            f"  tok={total_tokens}  t={elapsed:.1f}s{conf_str}"
        )

        results.append(TTSResult(
            question_id=qid,
            question=q,
            references=refs,
            baseline_answer=base_raw,
            baseline_answer_normalized=base_norm,
            baseline_correct=False,
            baseline_confidence=base_conf,
            answer=final_answer,
            correct=final_correct,
            tokens=total_tokens,
            elapsed_s=elapsed,
            candidate_image_transforms=img_transforms,
            candidate_text_variants=txt_variants,
            candidate_prompts=prompts,
            candidate_answers=raw_answers,
            candidate_answers_normalized=normalized,
            candidate_confidences=confidences,
            voting=voting,
            winning_answer_mean_logprob=mean_lp,
            winning_answer_mean_prob=mean_p,
        ))

    return results


# ---------------------------------------------------------------------------
# Load all 100 questions per task (with images)
# ---------------------------------------------------------------------------

def _load_all_samples(task: str, n_pool: int) -> List[Dict[str, Any]]:
    from src.data.datasets.viscot_benchmark import load_task

    print(f"  {CYAN}{task.upper()}: loading {n_pool} questions + images…{RESET}")
    samples = load_task(task, n=n_pool)

    # Normalise field names: JSONL uses "answer" (str), TTS eval needs "references" (list)
    result: List[Dict[str, Any]] = []
    for s in samples:
        refs = s.get("references") or s.get("answers")
        if refs is None:
            raw = s.get("answer", "")
            refs = [raw] if raw else []
        if isinstance(refs, str):
            refs = [refs]
        result.append({
            "question_id": str(s["question_id"]),
            "question":    s["question"],
            "references":  refs,
            "image":       s["image"],
        })

    print(f"    {len(result)}/{n_pool} images ready")
    return result


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_table(
    all_results: Dict[str, Dict[str, List[TTSResult]]],
    model_labels: List[str],
) -> None:
    print(f"\n{_header('TTS Hard — Recovery Results')}")
    col_w = 38
    header = f"{'Model':<22}" + "".join(f"{t.upper():>{col_w}}" for t in TASK_LABELS)
    print(f"  {BOLD}{header}{RESET}")
    print(f"  {_bar('─', W - 2)}")

    for ml in model_labels:
        row_parts = [f"{ml:<22}"]
        for tl in TASK_LABELS:
            res = all_results.get(ml, {}).get(tl, [])
            if not res:
                row_parts.append(f"{'N/A':>{col_w}}")
                continue
            n        = len(res)
            recovered = sum(r.correct for r in res)
            cell = f"recovered={recovered}/{n} ({recovered/n:.0%})"
            row_parts.append(f"{cell:>{col_w}}")
        print("  " + "".join(row_parts))

    print(f"  {_bar('─', W - 2)}")
    print(f"\n  All {n} questions had baseline_correct=False (model initially wrong).")
    print(f"  recovered = TTS majority_9 got the right answer.\n")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_recovery(
    all_results: Dict[str, Dict[str, List[TTSResult]]],
    model_labels: List[str],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"  {DIM}matplotlib not available — skipping plot.{RESET}")
        return

    COLORS       = ["#4C72B0", "#DD8452"]
    TASK_DISPLAY = {"vqa": "VQA", "counting": "Counting", "ocr": "OCR"}
    X_KEYS       = ["baseline", "majority_9"]
    X_LABELS     = ["Baseline\n(×1, wrong)", "Majority-9\n(×9)"]

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size":  11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    n_tasks = len(TASK_LABELS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4.5), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    fig.suptitle(
        "TTS Recovery on Baseline Failures\n"
        "Qwen2.5-VL 3B vs. GRIT 3B  |  questions where baseline = wrong",
        fontsize=13, fontweight="bold", y=1.02,
    )

    x     = np.arange(len(X_LABELS))
    width = 0.35

    for ax, task in zip(axes, TASK_LABELS):
        for model_idx, (ml, color) in enumerate(zip(model_labels, COLORS)):
            items = all_results.get(ml, {}).get(task, [])
            if not items:
                continue

            # Baseline acc = 0% by construction; TTS recovery rate
            accs = [0.0, sum(r.correct for r in items) / len(items)]

            short  = ml.split(" ")[0]
            offset = (model_idx - 0.5) * width
            bars   = ax.bar(x + offset, accs, width, label=short, color=color, alpha=0.88)
            for bar, val in zip(bars, accs):
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.01,
                    f"{val:.0%}",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_title(TASK_DISPLAY.get(task, task.upper()), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy" if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    axes[0].legend(loc="upper left", fontsize=9)
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
        description=(
            "TTS Hard: baseline sweep → select failures → run TTS on failures. "
            "Proves that TTS recovers accuracy the baseline cannot reach."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", default=str(OUTPUT_PATH),
                        help="Destination JSON (default: results/tts/TTS_Hard.json).")
    parser.add_argument("--paraphrase-path", default=str(PARAPHRASE_PATH))
    parser.add_argument("--n-hard", type=int, default=N_HARD_DEFAULT,
                        help="Number of baseline-failure questions to run TTS on (default: 30).")
    parser.add_argument("--n-pool", type=int, default=N_POOL_DEFAULT,
                        help="Size of baseline sweep pool per task (default: 100).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume TTS phase from existing checkpoint (skips completed tasks).")
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate plot from existing TTS_Hard.json without inference.")
    args = parser.parse_args()

    print(_header("TTS Hard — Baseline-Failure Recovery"))

    model_labels = [mc["label"] for mc in MODEL_CONFIGS]
    output_path  = Path(args.output)

    # ── Plot-only ──────────────────────────────────────────────────────────────
    if args.plot_only:
        ckpt = load_checkpoint(output_path)
        if not ckpt:
            print(f"{RED}No checkpoint at {output_path}{RESET}")
            sys.exit(1)
        plot_results = {
            ml: {tl: [_dict_to_result(r) for r in ckpt.get(ml, {}).get(tl, [])]
                 for tl in TASK_LABELS}
            for ml in model_labels
        }
        _plot_recovery(plot_results, model_labels, output_path.parent / "tts_hard_recovery.png")
        _print_table(plot_results, model_labels)
        return

    # ── Paraphrases ────────────────────────────────────────────────────────────
    paraphrase_lookup: Dict[tuple, str] = {}
    pp = Path(args.paraphrase_path)
    if pp.exists():
        for e in json.loads(pp.read_text(encoding="utf-8")):
            if e.get("paraphrase"):
                paraphrase_lookup[(e["task"], str(e["question_id"]))] = e["paraphrase"]
        print(f"\n{CYAN}Loaded {len(paraphrase_lookup)} paraphrases.{RESET}")
    else:
        print(f"\n{DIM}No paraphrase file — model_paraphrase slot uses original question.{RESET}")

    # ── Checkpoint ─────────────────────────────────────────────────────────────
    checkpoint: Dict[str, Any] = {}
    if args.resume and output_path.exists():
        checkpoint = load_checkpoint(output_path)
        if checkpoint:
            print(f"\n{CYAN}Resuming from checkpoint: {output_path}{RESET}")

    # ── Load images (once, shared across models) ───────────────────────────────
    print(f"\n{CYAN}Loading questions and images (n_pool={args.n_pool} per task)…{RESET}")
    samples_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for task in TASK_LABELS:
        samples_by_task[task] = _load_all_samples(task, args.n_pool)

    # ── Inference ──────────────────────────────────────────────────────────────
    import torch

    all_results: Dict[str, Dict[str, List[TTSResult]]] = {}

    for mc in MODEL_CONFIGS:
        label = mc["label"]
        all_results[label] = {}

        tasks_done   = [
            t for t in TASK_LABELS
            if len(checkpoint.get(label, {}).get(t, [])) >= args.n_hard
        ]
        tasks_needed = [t for t in TASK_LABELS if t not in tasks_done]

        for tl in tasks_done:
            all_results[label][tl] = [
                _dict_to_result(r) for r in checkpoint[label][tl]
            ]

        if not tasks_needed:
            print(f"\n{DIM}Skipping {label} — all tasks complete in checkpoint.{RESET}")
            continue

        print(f"\n{_header(label)}")
        if tasks_done:
            print(f"  {DIM}Re-using from checkpoint: {tasks_done}{RESET}")
        print(f"  {BOLD}Running: {tasks_needed}{RESET}")

        mid = mc["model_id"]
        print(f"  Loading {CYAN}{mid}{RESET}…")
        t_load = time.perf_counter()
        if mc["type"] == "direct_vlm":
            from src.models.direct_vlm import DirectVLMModel
            model_obj = DirectVLMModel(model_id=mid, load_in_8bit=False)
        else:
            from src.models.grit import GRITModel
            model_obj = GRITModel(model_id=mid, load_in_8bit=False)
        model_obj._load()
        print(f"  {GREEN}Loaded in {time.perf_counter() - t_load:.1f}s{RESET}")

        for task in tasks_needed:
            samples = samples_by_task[task]
            print(f"\n  {BOLD}{task.upper()}{RESET}")

            # Phase 1 — sweep to find failures
            failures = _baseline_sweep(
                model_obj, samples,
                max_new_tokens=mc["max_new_tokens"],
                model_type=mc["type"],
                model_label=label,
                task=task,
                n_hard=args.n_hard,
            )

            if not failures:
                print(f"  {YELLOW}No failures found for {label} / {task} — skipping TTS phase.{RESET}")
                all_results[label][task] = []
                save_checkpoint(output_path, all_results, model_labels)
                continue

            # Phase 2 — TTS on failures
            all_results[label][task] = _run_tts_on_failures(
                model_obj, failures,
                max_new_tokens=mc["max_new_tokens"],
                model_type=mc["type"],
                model_label=label,
                task=task,
                paraphrase_lookup=paraphrase_lookup or None,
            )
            save_checkpoint(output_path, all_results, model_labels)
            print(f"  {DIM}Checkpoint saved → {output_path}{RESET}")

        print(f"\n  {DIM}Unloading {label}…{RESET}")
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Table + save + plot ────────────────────────────────────────────────────
    _print_table(all_results, model_labels)

    save_checkpoint(output_path, all_results, model_labels)
    print(f"\nOutput saved → {output_path}")

    _plot_recovery(
        all_results, model_labels,
        output_path.parent / "tts_hard_recovery.png",
    )


if __name__ == "__main__":
    main()
