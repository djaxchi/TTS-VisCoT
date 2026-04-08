#!/usr/bin/env python3
"""Test-Time Scaling (TTS) experiment on the ModelBenchmark question set.

Reads the 20 questions per task from results/comparison/ModelBenchmark.json,
runs Qwen2.5-VL-3B and GRIT-3B (no quantisation) with 9 TTS candidates each,
and writes cumulative results to results/tts/TTS.json.

─── The 9 candidates (DEFAULT_CANDIDATE_RECIPE from pipeline_tts.py) ─────────

  Stage 1 — original image, three text variants:
    1. original image   + original prompt           ("Answer briefly.")
    2. original image   + hardcoded paraphrase       ("From the image, determine …")
    3. original image   + model/static paraphrase

  Stage 2 — image augmentations, original or paraphrase prompt:
    4. edge_enhance     + original   (UnsharpMask r=2.2 p=340 + EDGE_ENHANCE_MORE)
    5. grayscale        + original   (L→RGB)
    6. jpeg_recompress  + original   (quality=28)
    7. brightness_contrast + original (×1.80 brightness, ×1.85 contrast)
    8. rotation_90      + original   (90° CCW, canvas expands, no crop)
    9. edge_enhance     + model/static paraphrase

  All image parameters use the "strong" preset (ImageVariationConfig default).

─── Baseline ─────────────────────────────────────────────────────────────────
  Candidate 1 in DEFAULT_CANDIDATE_RECIPE is (original image, original text),
  which is the bare-question baseline.  No separate baseline call is made.

─── Majority voting ──────────────────────────────────────────────────────────
  All 9 TTS candidates are always run.  The majority vote is tallied at @9.

─── Token-level answer confidence ────────────────────────────────────────────
  After each candidate call, a single no-generation forward pass extracts
  P(first_answer_token | image, prompt) from the last-position logits.
  Stored per candidate: logprob, prob, top-5 competing tokens at that position.
  Aggregated on the @9 majority-voted answer: mean_logprob, mean_prob.

Usage:
    python experiments/run_test_time_scaling.py
    python experiments/run_test_time_scaling.py --resume
    python experiments/run_test_time_scaling.py --plot-only
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
CYAN = "\033[36m"; GREEN = "\033[32m"; RED = "\033[31m"
W = 80

BACKBONE_PATH   = _PROJECT_ROOT / "results" / "comparison" / "ModelBenchmark.json"
OUTPUT_PATH     = _PROJECT_ROOT / "results" / "tts" / "TTS.json"
PARAPHRASE_PATH = _PROJECT_ROOT / "results" / "questions_to_rephrase.json"

TEMPERATURE = 0.0  # greedy decoding — not a per-model hyperparameter

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
    image: Any,         # PIL.Image
    prompt: str,
    max_new_tokens: int,
    model_type: str,    # "direct_vlm" or "grit"
) -> Dict[str, Any]:
    """Run model.generate with output_scores=True → answer + confidence in one pass.

    Uses return_dict_in_generate=True so that scores[i] contains the logits
    that produced generated token i.  For direct_vlm the first generated token
    is the answer token; for GRIT we search for the position right after the
    <answer> tag in the generated sequence.

    Returns:
        Dict with keys: answer, raw_output, confidence (dict or None).
        confidence contains: answer_first_token, answer_first_token_id,
        logprob, prob, top5_distribution.
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

    generated_ids = out.sequences[0, prompt_len:]   # (gen_len,)
    scores = out.scores                              # tuple of (1, vocab) tensors

    # Decode answer
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

    # Find position in generated sequence where first answer token lives.
    # direct_vlm: position 0 (first generated token IS the answer).
    # grit: scan for <answer> tag, then the next position is the answer content.
    confidence: Optional[Dict[str, Any]] = None
    if answer and scores:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if answer_ids:
            target_id   = answer_ids[0]
            gen_list    = generated_ids.tolist()

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
                    # fallback: first occurrence of the target token id
                    for i, tid in enumerate(gen_list):
                        if tid == target_id:
                            answer_pos = i
                            break

            if answer_pos is not None and answer_pos < len(scores):
                logits_at   = scores[answer_pos].squeeze(0)        # (vocab,)
                log_probs   = torch.log_softmax(logits_at, dim=-1)
                logprob     = float(log_probs[target_id].item())
                prob        = float(torch.exp(log_probs[target_id]).item())
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"answer": answer, "raw_output": raw_output, "confidence": confidence}


# ---------------------------------------------------------------------------
# Majority voting
# ---------------------------------------------------------------------------

def _majority_vote(normalized_answers: List[str]) -> Dict[str, Any]:
    """First-seen tie-break majority vote over normalised open-ended strings."""
    valid = [a for a in normalized_answers if a]
    if not valid:
        return {"answer": "", "vote_counts": {}, "agreement_rate": 0.0, "valid_votes": 0}
    counts = Counter(valid)
    top = max(counts.values())
    tied = {a for a, c in counts.items() if c == top}
    winner = next(a for a in valid if a in tied)
    return {
        "answer": winner,
        "vote_counts": dict(counts),
        "agreement_rate": top / len(valid),
        "valid_votes": len(valid),
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TTSResult:
    question_id: str
    question:    str
    references:  List[str]

    # Bare-question baseline (original image, bare question, temp=0, 1 call)
    baseline_answer:            str
    baseline_answer_normalized: str
    baseline_correct:           bool
    baseline_confidence:        Optional[Dict[str, Any]]

    # TTS final answer (majority_9)
    answer:    str
    correct:   bool
    tokens:    int      # total tokens across all 9 TTS candidates
    elapsed_s: float    # wall time for all 9 TTS candidates

    # Per-candidate TTS data (length = 9, order = DEFAULT_CANDIDATE_RECIPE)
    candidate_image_transforms: List[str]                        = field(default_factory=list)
    candidate_text_variants:    List[str]                        = field(default_factory=list)
    candidate_prompts:          List[str]                        = field(default_factory=list)
    candidate_answers:          List[str]                        = field(default_factory=list)
    candidate_answers_normalized: List[str]                      = field(default_factory=list)
    candidate_confidences:      List[Optional[Dict[str, Any]]]  = field(default_factory=list)

    # Majority votes at @3, @5, @9
    voting: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Aggregated confidence for the majority-voted answer
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
    task_labels:  List[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        ml: {tl: [_result_to_dict(r) for r in all_results.get(ml, {}).get(tl, [])]
             for tl in task_labels}
        for ml in model_labels
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_checkpoint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _task_is_complete(label: str, task: str, checkpoint: Dict[str, Any], n: int) -> bool:
    return n == 0 or len(checkpoint.get(label, {}).get(task, [])) >= n


# ---------------------------------------------------------------------------
# Core TTS inference
# ---------------------------------------------------------------------------

def _run_tts_on_samples(
    model_obj: Any,
    samples: List[Dict[str, Any]],
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

    for i, samp in enumerate(samples):
        q     = samp["question"]
        refs  = samp["references"]
        qid   = str(samp["question_id"])
        image = samp["image"]

        print(
            f"  [{model_label}] {task.upper()} {i + 1}/{len(samples)}: "
            f"{q[:60]}{'…' if len(q) > 60 else ''}"
        )

        # ── Paraphrase function for this sample ───────────────────────────────
        static_para = (paraphrase_lookup or {}).get((task, qid))

        def _para_fn(
            _q: str,
            _choices: Dict[str, str],
            _idx: int,
            _p: str = static_para or "",
        ) -> str:
            return _p if _p else _q

        model_paraphrase_fn: Optional[Callable] = _para_fn if static_para else None

        # ── Build all 9 candidate inputs (images + prompts) ───────────────────
        t0 = time.perf_counter()
        try:
            candidates_input = build_candidate_inputs(
                image=image,
                question=q,
                choices={},
                max_candidates=len(DEFAULT_CANDIDATE_RECIPE),
                candidate_recipe=DEFAULT_CANDIDATE_RECIPE,
                model_paraphrase_fn=model_paraphrase_fn,
                image_config=image_config,
            )
        except Exception as exc:
            print(f"    {RED}SKIP{RESET}  build_candidate_inputs error on {qid}: {exc}")
            continue

        img_transforms:  List[str]                        = []
        txt_variants:    List[str]                        = []
        prompts:         List[str]                        = []
        raw_answers:     List[str]                        = []
        confidences:     List[Optional[Dict[str, Any]]]  = []
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

        # ── Candidate 0 is the baseline (original image, original text) ───────
        baseline_answer     = raw_answers[0]
        baseline_norm       = vqa_normalize(baseline_answer)
        baseline_correct    = evaluate_vqa(baseline_norm, refs)
        baseline_confidence = confidences[0]

        # ── Majority vote (@9) ────────────────────────────────────────────────
        normalized = [vqa_normalize(a) for a in raw_answers]
        voting: Dict[str, Dict[str, Any]] = {
            "majority_9": _majority_vote(normalized),
        }
        final_answer  = voting["majority_9"]["answer"]
        final_correct = evaluate_vqa(final_answer, refs)

        # ── Aggregate confidence for the winning answer ───────────────────────
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

        # ── Logging ───────────────────────────────────────────────────────────
        base_tag  = f"{GREEN}b✓{RESET}" if baseline_correct else f"{RED}b✗{RESET}"
        final_tag = f"{GREEN}✓{RESET}"  if final_correct    else f"{RED}✗{RESET}"
        conf_str  = f"  conf={mean_p:.2f}" if mean_p is not None else ""
        print(
            f"    {base_tag} base={baseline_norm[:25]!r}  "
            f"{final_tag} @9={final_answer[:25]!r}  ref={refs[0]!r}  "
            f"tok={total_tokens}  t={elapsed:.1f}s{conf_str}"
        )

        results.append(TTSResult(
            question_id=qid,
            question=q,
            references=refs,
            baseline_answer=baseline_answer,
            baseline_answer_normalized=baseline_norm,
            baseline_correct=baseline_correct,
            baseline_confidence=baseline_confidence,
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
# Image loading from backbone
# ---------------------------------------------------------------------------

def _load_backbone_samples(
    backbone: Dict[str, Any],
    task_labels: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    from src.data.datasets.viscot_benchmark import load_task

    first_model_data = next(iter(backbone.values()))
    result: Dict[str, List[Dict[str, Any]]] = {}

    for task in task_labels:
        backbone_entries = first_model_data.get(task, [])
        if not backbone_entries:
            result[task] = []
            continue

        print(f"  {CYAN}{task.upper()}: fetching {len(backbone_entries)} images…{RESET}")
        all_samples = load_task(task, n=100)
        lookup = {str(s["question_id"]): s for s in all_samples}

        samples: List[Dict[str, Any]] = []
        for entry in backbone_entries:
            qid = str(entry["question_id"])
            s   = lookup.get(qid)
            if s is None:
                print(f"    {RED}MISS{RESET}  question_id {qid} not in cache")
                continue
            samples.append({
                "question_id": qid,
                "question":    entry["question"],
                "references":  entry["references"],
                "image":       s["image"],
            })

        print(f"    {len(samples)}/{len(backbone_entries)} images ready")
        result[task] = samples

    return result


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_table(
    all_results: Dict[str, Dict[str, List[TTSResult]]],
    model_labels: List[str],
    task_labels:  List[str],
) -> None:
    from src.eval.vqa_eval import evaluate_vqa

    print(f"\n{_header('TTS Results')}")
    col_w = 30
    header = f"{'Model':<22}" + "".join(f"{t.upper():>{col_w}}" for t in task_labels) + f"{'OVERALL':>{col_w}}"
    print(f"  {BOLD}{header}{RESET}")
    print(f"  {_bar('─', W - 2)}")

    for ml in model_labels:
        row_parts = [f"{ml:<22}"]
        tot_n = tot_base = tot_9 = 0
        for tl in task_labels:
            res = all_results.get(ml, {}).get(tl, [])
            if not res:
                row_parts.append(f"{'N/A':>{col_w}}")
                continue
            n    = len(res)
            base = sum(r.baseline_correct for r in res)
            c9   = sum(r.correct for r in res)
            tot_n += n; tot_base += base; tot_9 += c9
            cell = f"base={base/n:.0%} @9={c9/n:.0%}"
            row_parts.append(f"{cell:>{col_w}}")
        if tot_n:
            cell = f"base={tot_base/tot_n:.0%} @9={tot_9/tot_n:.0%}"
        else:
            cell = "N/A"
        row_parts.append(f"{cell:>{col_w}}")
        print("  " + "".join(row_parts))

    print(f"  {_bar('─', W - 2)}")
    print(f"\n  Column: base=bare-question  @9=majority vote over 9 TTS candidates\n")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_tts_comparison(
    all_results: Dict[str, Dict[str, List[TTSResult]]],
    model_labels: List[str],
    task_labels:  List[str],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(f"  {DIM}matplotlib not available — skipping plot.{RESET}")
        return

    from src.eval.vqa_eval import evaluate_vqa

    COLORS       = ["#4C72B0", "#DD8452"]
    TASK_DISPLAY = {"vqa": "VQA", "counting": "Counting", "ocr": "OCR"}
    X_KEYS   = ["baseline", "majority_9"]
    X_LABELS = ["Baseline\n(×1)", "Majority-9\n(×9)"]

    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    n_tasks = len(task_labels)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4.5), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    fig.suptitle(
        "Test-Time Scaling — Baseline vs. Majority-9\n"
        "Qwen2.5-VL 3B vs. GRIT 3B  (no quantisation)  |  n=20 per task",
        fontsize=13, fontweight="bold", y=1.02,
    )

    x     = np.arange(len(X_LABELS))
    width = 0.35

    for ax, task in zip(axes, task_labels):
        for model_idx, (ml, color) in enumerate(zip(model_labels, COLORS)):
            items = all_results.get(ml, {}).get(task, [])
            if not items:
                continue

            accs: List[float] = []
            for xk in X_KEYS:
                if xk == "baseline":
                    acc = sum(r.baseline_correct for r in items) / len(items)
                else:
                    acc = sum(
                        evaluate_vqa(r.voting[xk]["answer"], r.references)
                        for r in items
                    ) / len(items)
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
                    ha="center", va="bottom", fontsize=8,
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
            "TTS experiment: Qwen3B + GRIT3B, 1 baseline + 9 TTS candidates "
            "(DEFAULT_CANDIDATE_RECIPE), on the ModelBenchmark question set."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backbone", default=str(BACKBONE_PATH),
                        help="Source benchmark JSON (default: ModelBenchmark.json).")
    parser.add_argument("--output", default=str(OUTPUT_PATH),
                        help="Destination TTS JSON (default: results/tts/TTS.json).")
    parser.add_argument("--paraphrase-path", default=str(PARAPHRASE_PATH),
                        help="JSON file with human paraphrases (task+question_id keyed).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing --output checkpoint.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip inference — regenerate plot from existing TTS.json.")
    args = parser.parse_args()

    print(_header("Test-Time Scaling Experiment"))

    backbone_path = Path(args.backbone)
    if not backbone_path.exists():
        print(f"\n{RED}Backbone not found: {backbone_path}{RESET}")
        sys.exit(1)

    backbone: Dict[str, Any] = json.loads(backbone_path.read_text(encoding="utf-8"))
    first_model_data = next(iter(backbone.values()))
    task_labels  = [t for t, entries in first_model_data.items() if entries]
    model_labels = [mc["label"] for mc in MODEL_CONFIGS]
    output_path  = Path(args.output)

    print(f"\n{CYAN}Backbone: {backbone_path.name}{RESET}")
    for t in task_labels:
        n = len(first_model_data[t])
        print(f"  {t.upper()}: {n} questions  ({n * 9} generate calls per model)")

    # ── Plot-only ──────────────────────────────────────────────────────────────
    if args.plot_only:
        ckpt = load_checkpoint(output_path)
        if not ckpt:
            print(f"{RED}No TTS.json at {output_path}{RESET}")
            sys.exit(1)
        plot_results: Dict[str, Dict[str, List[TTSResult]]] = {
            ml: {tl: [_dict_to_result(r) for r in ckpt.get(ml, {}).get(tl, [])]
                 for tl in task_labels}
            for ml in model_labels
        }
        _plot_tts_comparison(plot_results, model_labels, task_labels,
                             output_path.parent / "tts_scaling.png")
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
        print(f"\n{DIM}No paraphrase file found — model_paraphrase slot will use original question.{RESET}")

    # ── Checkpoint ─────────────────────────────────────────────────────────────
    checkpoint: Dict[str, Any] = {}
    if args.resume and output_path.exists():
        checkpoint = load_checkpoint(output_path)
        if checkpoint:
            print(f"\n{CYAN}Resuming from checkpoint: {output_path}{RESET}")

    # ── Fetch images ───────────────────────────────────────────────────────────
    print(f"\n{CYAN}Fetching images for backbone questions…{RESET}")
    samples_by_task = _load_backbone_samples(backbone, task_labels)

    # ── Inference ──────────────────────────────────────────────────────────────
    import torch

    all_results: Dict[str, Dict[str, List[TTSResult]]] = {}

    for mc in MODEL_CONFIGS:
        label = mc["label"]
        all_results[label] = {}

        n_for_task = {t: len(samples_by_task.get(t, [])) for t in task_labels}

        tasks_needed  = [t for t in task_labels if n_for_task[t] > 0
                         and not _task_is_complete(label, t, checkpoint, n_for_task[t])]
        tasks_from_cp = [t for t in task_labels if n_for_task[t] > 0
                         and _task_is_complete(label, t, checkpoint, n_for_task[t])]

        for tl in tasks_from_cp:
            all_results[label][tl] = [
                _dict_to_result(r) for r in checkpoint.get(label, {}).get(tl, [])
            ]

        if not tasks_needed:
            print(f"\n{DIM}Skipping {label} (all tasks complete in checkpoint).{RESET}")
            continue

        print(f"\n{_header(label)}")
        if tasks_from_cp:
            print(f"  {DIM}Re-using from checkpoint: {tasks_from_cp}{RESET}")
        print(f"  {BOLD}Running: {tasks_needed}{RESET}")

        mid = mc["model_id"]
        print(f"  Loading {CYAN}{mid}{RESET} (no quantisation)…")
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
            print(f"\n  {BOLD}{task.upper()} ({len(samples)} questions × 9 candidates){RESET}")
            all_results[label][task] = _run_tts_on_samples(
                model_obj,
                samples,
                max_new_tokens=mc["max_new_tokens"],
                model_type=mc["type"],
                model_label=label,
                task=task,
                paraphrase_lookup=paraphrase_lookup or None,
            )
            save_checkpoint(output_path, all_results, model_labels, task_labels)
            print(f"  {DIM}Checkpoint saved → {output_path}{RESET}")

        print(f"\n  {DIM}Unloading {label}…{RESET}")
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Table + save + plot ────────────────────────────────────────────────────
    _print_table(all_results, model_labels, task_labels)

    save_checkpoint(output_path, all_results, model_labels, task_labels)
    print(f"\nOutput saved → {output_path}")

    _plot_tts_comparison(
        all_results, model_labels, task_labels,
        output_path.parent / "tts_scaling.png",
    )


if __name__ == "__main__":
    main()
