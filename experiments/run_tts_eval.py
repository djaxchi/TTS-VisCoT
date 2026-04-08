#!/usr/bin/env python3
"""Evaluate baseline vs adaptive TTS on a set of local TreeBench examples.

Usage — real model (requires GPU + loaded weights):
    python experiments/run_tts_eval.py \\
        --data-dir results/treebench_samples \\
        --n-questions 3 \\
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
        --save-dir results/tts_eval/run1

Usage — dry-run (no model, no GPU, deterministic mock answers):
    python experiments/run_tts_eval.py --dry-run --n-questions 3

The script prints per-question details (question, correct answer, baseline vs TTS
prediction, stopping info) and a final accuracy / stopping-stats summary.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.eval.tts_eval import compute_summary, evaluate_one, make_predict_fn
from src.augment_image import ImageVariationConfig
from src.pipeline_tts import DEFAULT_CANDIDATE_RECIPE, build_candidate_inputs
from src.utils_normalize import normalize_answer


DEFAULT_PARAPHRASE_CACHE_PATH = _PROJECT_ROOT / "results" / "tts_eval" / "model_paraphrase_cache.jsonl"
DEFAULT_STATIC_PARAPHRASE_PATH = _PROJECT_ROOT / "results" / "tts_eval" / "static_paraphrase_cache.jsonl"
PARAPHRASE_CACHE_VERSION = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_samples(data_dir: str, n: int) -> List[Dict[str, Any]]:
    """Read up to *n* samples from a local exported TreeBench folder.

    Expects a ``metadata.jsonl`` file (one JSON dict per line) with keys:
    ``image_id``, ``question``, ``options``, ``correct_answer``, ``image_path``.
    """
    meta_path = Path(data_dir) / "metadata.jsonl"
    if not meta_path.exists():
        return []

    try:
        lines = meta_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    samples: List[Dict[str, Any]] = []
    for line in lines[:n]:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            image_path = Path(data_dir) / row["image_path"]
            if not image_path.exists():
                print(f"  [WARN] image not found: {image_path}")
                continue
            image = Image.open(image_path).convert("RGB")
            samples.append(
                {
                    "image_id": str(row["image_id"]),
                    "image": image,
                    "question": str(row["question"]),
                    "choices": dict(row["options"]),
                    "correct_answer": str(row.get("correct_answer", "")),
                }
            )
        except (KeyError, OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"  [WARN] skipping malformed sample: {exc}")

    return samples


def _load_benchmark_samples(task: str, n: int, offset: int = 0) -> List[Dict[str, Any]]:
    """Load up to *n* open-ended samples from the VisCoT benchmark."""
    from src.data.datasets.viscot_benchmark import load_task
    rows = load_task(task, n=n, offset=offset)
    return [
        {
            "image_id": r["question_id"],
            "question": r["question"],
            "choices": {},  # open-ended — no MC options
            "correct_answer": r["answer"],
            "image": r["image"],
        }
        for r in rows
    ]


def _format_choices(choices: Dict[str, str]) -> str:
    ordered = [k for k in ("A", "B", "C", "D") if k in choices]
    return "\n".join(f"{k}. {choices[k]}" for k in ordered)


def _build_paraphrase_prompt(question: str, choices: Dict[str, str]) -> str:
    return (
        "Rewrite the following visual multiple-choice question into one semantically equivalent "
        "question for the same image. Use noticeably different wording, but preserve all entities, "
        "attributes, and spatial relations exactly. Do not answer the question. Do not mention the "
        "correct option. Output only the rewritten question sentence.\n\n"
        "Example 1\n"
        "Original: What color is the car next to the tree?\n"
        "Rewrite: Which color does the car beside the tree have?\n\n"
        "Example 2\n"
        "Original: From the man's point of view, where is the red ball relative to the chair?\n"
        "Rewrite: Relative to the chair, where is the red ball from the man's perspective?\n\n"
        f"Question: {question.strip()}\n"
        "Choices:\n"
        f"{_format_choices(choices)}"
    )


def _build_retry_paraphrase_prompt(question: str, choices: Dict[str, str]) -> str:
    return (
        "Rewrite the question below with noticeably different wording while preserving the exact meaning. "
        "Keep the same entities, attributes, and spatial relations. Do not add instructions like 'inspect the image' or 'choose the best option'. "
        "Change the sentence structure, not just one word. Do not answer. Output only one rewritten question sentence.\n\n"
        "Bad rewrite example: Inspect the image carefully and choose the best option. Where is the sign?\n"
        "Good rewrite example: Where is the sign positioned relative to the viewer?\n\n"
        f"Original question: {question.strip()}\n"
        "Choices:\n"
        f"{_format_choices(choices)}"
    )


def _normalize_paraphrase_text(raw_text: str, original_question: str) -> str:
    text = " ".join((raw_text or "").strip().split())
    text = re.sub(r'^(?:rephrased question|rewritten question|paraphrase|question)\s*:\s*', "", text, flags=re.IGNORECASE)
    text = text.strip().strip('"').strip("'").strip()
    if not text:
        return original_question.strip()
    if normalize_answer(text) is not None:
        return original_question.strip()
    return text


def _paraphrase_is_usable(paraphrase: str, original_question: str) -> bool:
    candidate = " ".join((paraphrase or "").strip().split()).lower()
    original = " ".join((original_question or "").strip().split()).lower()
    if not candidate:
        return False
    if candidate == original:
        return False
    banned_prefixes = (
        "inspect the image carefully and choose the best option",
        "from the image determine which option is correct",
        "from the image, determine which option is correct",
        "determine the best answer from the image",
        "using the image, answer this question",
    )
    if candidate.startswith(banned_prefixes):
        return False
    similarity = difflib.SequenceMatcher(None, original, candidate).ratio()
    token_overlap = len(set(original.split()) ^ set(candidate.split()))
    if similarity > 0.95:
        return False
    if similarity > 0.88 and token_overlap < 3:
        return False
    return True


def _paraphrase_cache_key(sample_id: str, question: str) -> str:
    sample_key = str(sample_id).strip()
    return sample_key or question.strip()


def load_paraphrase_cache(cache_path: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    cache: Dict[str, Dict[str, Any]] = {}
    for line in rows:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = str(row.get("sample_id") or row.get("question") or "").strip()
        if key:
            cache[key] = row
    return cache


def load_static_paraphrase_cache(
    cache_path: str | Path,
    task_filter: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Load static paraphrases keyed by sample_id and question text.

    Handles two file formats:
    - JSONL  (one JSON object per line) with fields ``sample_id``, ``model_paraphrase``.
    - JSON array  (``questions_to_rephrase.json`` style) with fields
      ``question_id``, ``question``, ``paraphrase``, ``task``.

    Args:
        cache_path: Path to the JSONL or JSON paraphrase file.
        task_filter: When loading JSON-array format, keep only rows whose
            ``task`` field equals this value (e.g. ``"vqa"``).
    """
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}

    # Detect JSON array format (starts with '[')
    if raw.startswith("["):
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        cache: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            if task_filter and entry.get("task") != task_filter:
                continue
            sid = str(entry.get("question_id") or "").strip()
            q = str(entry.get("question") or "").strip()
            # Normalise to the standard cache row shape
            row = {
                "sample_id": sid,
                "question": q,
                "model_paraphrase": str(entry.get("paraphrase") or "").strip(),
            }
            if sid:
                cache[sid] = row
            if q:
                cache[f"q::{q}"] = row
        return cache

    # JSONL format
    cache = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = str(row.get("sample_id") or "").strip()
        q = str(row.get("question") or "").strip()
        if sid:
            cache[sid] = row
        if q:
            cache[f"q::{q}"] = row
    return cache


def resolve_static_paraphrase(
    static_cache: Dict[str, Dict[str, Any]],
    *,
    sample_id: str,
    question: str,
) -> str | None:
    sid = str(sample_id).strip()
    q = str(question).strip()
    by_id = static_cache.get(sid)
    if by_id and by_id.get("model_paraphrase"):
        return str(by_id["model_paraphrase"])
    by_q = static_cache.get(f"q::{q}")
    if by_q and by_q.get("model_paraphrase"):
        return str(by_q["model_paraphrase"])
    return None


def _save_paraphrase_cache(cache_path: str | Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [cache[k] for k in sorted(cache)]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_or_generate_model_paraphrase(
    *,
    cache: Dict[str, Dict[str, Any]],
    cache_path: str | Path,
    sample_id: str,
    image: Image.Image,
    question: str,
    choices: Dict[str, str],
    model: Any,
    model_label: str,
    max_new_tokens: int = 96,
) -> str:
    key = _paraphrase_cache_key(sample_id, question)
    cached = cache.get(key)
    if (
        cached
        and int(cached.get("cache_version", 0)) == PARAPHRASE_CACHE_VERSION
        and cached.get("model_paraphrase")
        and _paraphrase_is_usable(str(cached["model_paraphrase"]), question)
    ):
        return str(cached["model_paraphrase"])

    raw_paraphrase = ""
    paraphrase = question.strip()
    attempt_rows: List[Dict[str, Any]] = []
    for attempt_idx, prompt in enumerate((
        _build_paraphrase_prompt(question, choices),
        _build_retry_paraphrase_prompt(question, choices),
    ), start=1):
        chain = model.predict(
            image,
            prompt,
            temperature=0.2,
            max_new_tokens=max_new_tokens,
        )
        raw_paraphrase = str(chain.get("answer", "") or "")
        paraphrase = _normalize_paraphrase_text(raw_paraphrase, question)
        attempt_rows.append(
            {
                "attempt": attempt_idx,
                "prompt": prompt,
                "raw_model_output": raw_paraphrase,
                "normalized_paraphrase": paraphrase,
                "accepted": _paraphrase_is_usable(paraphrase, question),
            }
        )
        if _paraphrase_is_usable(paraphrase, question):
            break

    cache[key] = {
        "sample_id": sample_id,
        "question": question,
        "choices": dict(choices),
        "model_paraphrase": paraphrase,
        "raw_model_output": raw_paraphrase,
        "source_model": model_label,
        "cache_version": PARAPHRASE_CACHE_VERSION,
        "attempts": attempt_rows,
    }
    _save_paraphrase_cache(cache_path, cache)
    return paraphrase


def make_cached_model_paraphrase_fn(
    *,
    cache: Dict[str, Dict[str, Any]],
    cache_path: str | Path,
    sample_id: str,
    image: Image.Image,
    question: str,
    choices: Dict[str, str],
    model: Any,
    model_label: str,
    max_new_tokens: int = 96,
) -> Callable[[str, Dict[str, str], int], str]:
    def _paraphrase(_question: str, _choices: Dict[str, str], _idx: int) -> str:
        return get_or_generate_model_paraphrase(
            cache=cache,
            cache_path=cache_path,
            sample_id=sample_id,
            image=image,
            question=question,
            choices=choices,
            model=model,
            model_label=model_label,
            max_new_tokens=max_new_tokens,
        )

    return _paraphrase


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _sep(width: int = 70) -> None:
    print("\n" + "-" * width)


def _header(title: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def _print_result(idx: int, sample: Dict[str, Any], result: Dict[str, Any], verbose: bool = False) -> None:
    _sep()
    print(f"Q{idx + 1} [{sample['image_id']}]")
    q = sample["question"]
    print(f"  question: {q[:90]}{'...' if len(q) > 90 else ''}")
    print(f"  correct:  {sample['correct_answer']}")

    if result["baseline"] is not None:
        b = result["baseline"]
        tick = "[OK]" if b["is_correct"] else "[X]"
        raw_preview = repr(b["raw_output"][:60]) if b["raw_output"] else "''"
        print(f"  baseline: {b['normalized_answer']} {tick}  (raw: {raw_preview})")

    if result["tts"] is not None:
        t = result["tts"]
        tick = "[OK]" if t["is_correct"] else "[X]"
        stop_label = "early-stop" if t["stopped_early"] else f"stage-2 ({t['used_candidates']} cands)"
        s1_rate = t.get("stage_1_agreement_rate")
        s2_flip = t.get("stage2_changed_answer")
        entropy = t.get("answer_entropy", 0.0)
        s1_str = f"  s1_agree={s1_rate:.0%}" if s1_rate is not None else ""
        s2_str = f"  s2_flip={s2_flip}" if s2_flip is not None else ""
        print(f"  tts:      {t['winning_answer']} {tick}  [{stop_label}]")
        print(f"            candidates: {t['candidate_answers']}  agreement={t['agreement_rate']:.0%}  H={entropy:.2f}bits{s1_str}{s2_str}")

        if verbose and t.get("candidates"):
            import math
            print("  --- per-candidate breakdown ---")
            for c in t["candidates"]:
                cid = c.get("candidate_id", "?")
                tf = c.get("image_transform_id", c.get("image_key", "?"))
                tv = c.get("text_variant_id", c.get("prompt_key", "?"))
                ans = c.get("normalized_answer", None)
                raw = c.get("raw_output", "")
                elapsed = c.get("elapsed_s")
                correct_flag = c.get("is_correct")
                tm = c.get("token_metadata", {})
                olp = tm.get("option_logprobs", [])
                probs_str = ""
                best_logit = None
                if olp:
                    lp = olp[0]["logprobs"]
                    best_logit = max(lp, key=lp.get)
                    probs = {k: round(math.exp(v), 3) for k, v in lp.items()}
                    probs_str = f"  probs={probs}"
                fallback = f" [logit->{best_logit}]" if best_logit and ans is None else ""
                truncated = " [TRUNCATED]" if raw and not raw.endswith((".", "?", "!", ">", "\n")) else ""
                correct_mark = " [OK]" if correct_flag is True else (" [X]" if correct_flag is False else "")
                time_str = f"  {elapsed:.1f}s" if elapsed is not None else ""
                print(
                    f"    cand={cid} img={tf!r:22s} txt={tv!r:18s} "
                    f"ans={str(ans):4s}{correct_mark}{fallback}{truncated}{time_str}"
                    f"{probs_str}"
                )
                print(f"      raw: {repr(raw[:120])}")


def _print_summary(summary: Dict[str, Any]) -> None:
    _header("SUMMARY")
    print(f"  n_questions: {summary['n_questions']}")

    if "baseline_accuracy" in summary:
        print(
            f"  baseline accuracy : {summary['baseline_correct']}/{summary['baseline_n']}"
            f"  ({summary['baseline_accuracy']:.0%})"
        )

    if "tts_accuracy" in summary:
        print(
            f"  tts accuracy      : {summary['tts_correct']}/{summary['tts_n']}"
            f"  ({summary['tts_accuracy']:.0%})"
        )
        print(f"  tts early-stop    : {summary['tts_early_stop_rate']:.0%}"
              f"  ({sum(1 for _ in range(summary['tts_n'])) * summary['tts_early_stop_rate']:.0f}/{summary['tts_n']})")
        print(f"  tts avg candidates: {summary['tts_avg_candidates']:.1f}")

        # Highlight if TTS helped
        if "baseline_accuracy" in summary:
            delta = summary["tts_accuracy"] - summary["baseline_accuracy"]
            direction = "+" if delta >= 0 else ""
            print(f"\n  tts vs baseline   : {direction}{delta:.0%}")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serializable(result: Dict[str, Any]) -> Dict[str, Any]:
    """Strip non-JSON-serializable objects (PIL Images) from a result dict."""
    out = {k: v for k, v in result.items() if k != "image"}
    if out.get("tts") is not None:
        tts_copy = dict(out["tts"])
        # CandidateResult dicts inside tts["candidates"] contain no Images
        # (run_tts_pipeline returns only primitive fields), so no stripping needed.
        out["tts"] = tts_copy
    return out


def _candidate_artifact_rows(
    *,
    result: Dict[str, Any],
    model_name: str,
    model_variant: str,
    decoding_settings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    tts = result.get("tts")
    if tts is None:
        return []
    rows: List[Dict[str, Any]] = []
    for c in tts.get("candidates", []):
        rows.append(
            {
                "sample_id": result.get("image_id", ""),
                "model_name": model_name,
                "model_variant": model_variant,
                "candidate_id": c.get("candidate_id"),
                "stage": c.get("stage"),
                "image_transform_id": c.get("image_transform_id", c.get("image_key")),
                "image_transform_parameters": c.get("image_transform_parameters", {}),
                "image_transform_preset": c.get("image_transform_preset", ""),
                "text_variant_id": c.get("text_variant_id", c.get("prompt_key")),
                "prompt": c.get("prompt", ""),
                "decoding_settings": c.get("decoding_settings", decoding_settings),
                "raw_output": c.get("raw_output", ""),
                "normalized_answer": c.get("normalized_answer"),
                "parse_status": c.get("parse_status", "invalid"),
                "is_valid": c.get("is_valid", False),
                "is_correct": c.get("is_correct"),       # True/False/None (None = parse failed)
                "elapsed_s": c.get("elapsed_s"),          # wall-clock time for this candidate
                "token_metadata": c.get("token_metadata", {}),
                # sample-level stage diagnostics repeated per candidate for easy grouping
                "stage_1_agreement_rate": tts.get("stage_1_agreement_rate"),
                "stage2_changed_answer": tts.get("stage2_changed_answer"),
                "answer_entropy": tts.get("answer_entropy"),
            }
        )
    return rows


def save_candidate_views(
    *,
    save_dir: str | Path,
    image_id: str,
    image: Image.Image,
    question: str,
    choices: Dict[str, str],
    image_preset: str,
    text_mode: str = "rule",
    model_paraphrase_fn: Callable[[str, Dict[str, str], int], str] | None = None,
) -> List[Dict[str, Any]]:
    """Save candidate view images/prompts and return manifest rows."""
    out = Path(save_dir) / "candidate_views" / image_id
    out.mkdir(parents=True, exist_ok=True)

    candidates = build_candidate_inputs(
        image=image,
        question=question,
        choices=choices,
        max_candidates=len(DEFAULT_CANDIDATE_RECIPE),
        candidate_recipe=DEFAULT_CANDIDATE_RECIPE,
        text_mode=text_mode,
        model_paraphrase_fn=model_paraphrase_fn,
        image_config=ImageVariationConfig(preset=image_preset),
    )

    rows: List[Dict[str, Any]] = []
    for c in candidates:
        cid = int(c["candidate_id"])
        img_path = out / f"candidate_{cid}.png"
        prompt_path = out / f"candidate_{cid}_prompt.txt"
        c["image"].convert("RGB").save(img_path)
        prompt_path.write_text(c["prompt"], encoding="utf-8")

        rows.append(
            {
                "sample_id": image_id,
                "candidate_id": cid,
                "stage": int(c["stage"]),
                "image_transform_id": c.get("image_transform_id", c.get("image_key", "")),
                "image_transform_parameters": c.get("image_transform_parameters", {}),
                "image_transform_preset": c.get("image_transform_preset", image_preset),
                "text_variant_id": c.get("text_variant_id", c.get("prompt_key", "")),
                "prompt": c.get("prompt", ""),
                "image_path": str(img_path).replace("\\", "/"),
                "prompt_path": str(prompt_path).replace("\\", "/"),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs adaptive TTS on local TreeBench examples."
    )
    parser.add_argument(
        "--data-dir",
        default="results/treebench_samples",
        help="Folder with exported TreeBench samples (metadata.jsonl + images/).",
    )
    parser.add_argument(
        "--benchmark-task",
        choices=["vqa", "counting", "ocr"],
        default=None,
        help="Load VisCoT benchmark task (open-ended) instead of TreeBench MC samples.",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=3,
        help="Number of questions to evaluate (limited by available samples).",
    )
    parser.add_argument(
        "--question-offset",
        type=int,
        default=0,
        help="Skip the first N questions (to run a different slice of the benchmark).",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["baseline", "tts", "both"],
        help="Which evaluation mode(s) to run.",
    )
    parser.add_argument(
        "--model-type",
        default="direct_vlm",
        choices=["direct_vlm", "grit"],
        help="Model wrapper to evaluate.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="HuggingFace model ID (defaults to Qwen 3B for direct_vlm, GRIT 3B for grit).",
    )
    parser.add_argument(
        "--no-8bit",
        action="store_true",
        help="Disable 8-bit quantization (needs more VRAM).",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable stage-1 early stopping — always run all 9 candidates.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens per model call.",
    )
    parser.add_argument(
        "--image-preset",
        choices=["conservative", "moderate", "strong"],
        default="strong",
        help="Strength preset for image transforms in the adaptive candidate pool.",
    )
    parser.add_argument(
        "--token-storage-mode",
        choices=["none", "options_only", "topk", "full"],
        default="options_only",
        help="Per-candidate token metadata storage mode.",
    )
    parser.add_argument(
        "--token-topk",
        type=int,
        default=5,
        help="Top-k width when --token-storage-mode=topk.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="If set, write predictions.jsonl and metrics.json here.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a deterministic mock predict_fn instead of loading the real model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-candidate breakdown (image transform, text variant, answer, logit probs, raw output preview).",
    )
    parser.add_argument(
        "--paraphrase-cache-path",
        default=str(DEFAULT_PARAPHRASE_CACHE_PATH),
        help="JSONL cache used to store one model-generated paraphrase per sample for reuse across later model runs.",
    )
    parser.add_argument(
        "--paraphrase-max-new-tokens",
        type=int,
        default=96,
        help="Token budget for the one-time model paraphrase generation call.",
    )
    parser.add_argument(
        "--paraphrase-source",
        choices=["model", "static"],
        default="model",
        help="Use model-generated paraphrases or a precomputed static paraphrase file.",
    )
    parser.add_argument(
        "--static-paraphrase-path",
        default=str(DEFAULT_STATIC_PARAPHRASE_PATH),
        help="JSONL file with precomputed paraphrases (sample_id/question/model_paraphrase).",
    )
    parser.add_argument(
        "--task-paraphrase-path",
        default=str(_PROJECT_ROOT / "results" / "questions_to_rephrase.json"),
        help="JSON array file with LLM paraphrases for benchmark tasks "
             "(question_id/question/paraphrase/task). Used when --benchmark-task is set.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load samples
    # ------------------------------------------------------------------
    if args.benchmark_task:
        samples = _load_benchmark_samples(args.benchmark_task, args.n_questions, offset=args.question_offset)
        if not samples:
            print(f"ERROR: no samples loaded for benchmark task '{args.benchmark_task}'", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(samples)} sample(s) from benchmark task '{args.benchmark_task}'")
    else:
        samples = _load_samples(args.data_dir, args.n_questions)
        if not samples:
            print(f"ERROR: no samples loaded from '{args.data_dir}'", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(samples)} sample(s) from '{args.data_dir}'")

    model_type = args.model_type
    model_id = args.model_id
    if model_id is None:
        if model_type == "grit":
            model_id = "yfan1997/GRIT-20-Qwen2.5-VL-3B"
        else:
            model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    # ------------------------------------------------------------------
    # Build predict_fn
    # ------------------------------------------------------------------
    if args.dry_run:
        print("DRY RUN -- no model loaded; using cyclic mock answers [A, B, C, D, ...]")
        _mock_answers = cycle(["A", "B", "C", "D"])

        def predict_fn(image: Image.Image, prompt: str) -> str:  # noqa: ARG001
            return next(_mock_answers)
        model = None

    else:
        if model_type == "grit":
            from src.models.grit import GRITModel
            model = GRITModel(model_id=model_id, load_in_8bit=not args.no_8bit)
        else:
            from src.models.direct_vlm import DirectVLMModel
            model = DirectVLMModel(model_id=model_id, load_in_8bit=not args.no_8bit)

        print(f"Loading model '{model_id}' [{model_type}] (8-bit={not args.no_8bit})...")
        model._load()
        predict_fn = make_predict_fn(
            model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            return_details=args.token_storage_mode != "none",
            token_storage_mode=args.token_storage_mode,
            token_topk=args.token_topk,
            open_ended=bool(args.benchmark_task),
        )
        print("Model loaded.")

    paraphrase_cache = load_paraphrase_cache(args.paraphrase_cache_path)
    if args.benchmark_task:
        static_paraphrases = load_static_paraphrase_cache(
            args.task_paraphrase_path, task_filter=args.benchmark_task
        )
    else:
        static_paraphrases = load_static_paraphrase_cache(args.static_paraphrase_path)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    _header(f"Running {args.mode.upper()} evaluation on {len(samples)} question(s)")

    results: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    candidate_view_rows: List[Dict[str, Any]] = []
    decoding_settings = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "token_storage_mode": args.token_storage_mode,
        "token_topk": args.token_topk,
    }
    for idx, sample in enumerate(samples):
        print(f"\n[{idx + 1}/{len(samples)}] {sample['image_id']} ...", end=" ", flush=True)
        model_paraphrase_fn = None
        if args.paraphrase_source == "static":
            static_text = resolve_static_paraphrase(
                static_paraphrases,
                sample_id=sample["image_id"],
                question=sample["question"],
            )
            if static_text is None:
                static_text = sample["question"].strip()

            def _static_paraphrase(_question: str, _choices: Dict[str, str], _idx: int, text: str = static_text) -> str:
                return text

            model_paraphrase_fn = _static_paraphrase
        elif model is not None:
            model_paraphrase_fn = make_cached_model_paraphrase_fn(
                cache=paraphrase_cache,
                cache_path=args.paraphrase_cache_path,
                sample_id=sample["image_id"],
                image=sample["image"],
                question=sample["question"],
                choices=sample["choices"],
                model=model,
                model_label=f"{model_type}:{model_id}",
                max_new_tokens=args.paraphrase_max_new_tokens,
            )
        result = evaluate_one(
            image=sample["image"],
            question=sample["question"],
            choices=sample["choices"],
            correct_answer=sample["correct_answer"],
            predict_fn=predict_fn,
            mode=args.mode,
            tts_kwargs={
                "image_config": ImageVariationConfig(preset=args.image_preset),
                "decoding_settings": decoding_settings,
                "model_paraphrase_fn": model_paraphrase_fn,
                "allow_early_stop": not args.no_early_stop,
            },
        )
        result["image_id"] = sample["image_id"]
        results.append(result)
        if args.save_dir:
            candidate_view_rows.extend(
                save_candidate_views(
                    save_dir=args.save_dir,
                    image_id=sample["image_id"],
                    image=sample["image"],
                    question=sample["question"],
                    choices=sample["choices"],
                    image_preset=args.image_preset,
                    model_paraphrase_fn=model_paraphrase_fn,
                )
            )
        candidate_rows.extend(
            _candidate_artifact_rows(
                result=result,
                model_name=model_type,
                model_variant=model_id,
                decoding_settings=decoding_settings,
            )
        )
        print("done")
        _print_result(idx, sample, result, verbose=args.verbose)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = compute_summary(results)
    _print_summary(summary)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        predictions_path = save_path / "predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(_serializable(r), ensure_ascii=False) + "\n")

        metrics_path = save_path / "metrics.json"
        metrics_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        candidate_artifacts_path = save_path / "candidate_artifacts.jsonl"
        with candidate_artifacts_path.open("w", encoding="utf-8") as f:
            for row in candidate_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        candidate_views_path = save_path / "candidate_views.jsonl"
        with candidate_views_path.open("w", encoding="utf-8") as f:
            for row in candidate_view_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        _header("Saved")
        print(f"  predictions : {predictions_path}")
        print(f"  metrics     : {metrics_path}")
        print(f"  artifacts   : {candidate_artifacts_path}")
        print(f"  views       : {candidate_views_path}")


if __name__ == "__main__":
    main()
