"""Adaptive test-time scaling pipeline for TreeBench multiple-choice VQA."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from PIL import Image

from src.augmentation.image import ImageVariationConfig, generate_image_variant_specs
from src.augmentation.text import generate_prompt_variants
from src.utils_normalize import normalize_answer, normalize_open_ended_answer
from src.voting_tts import VoteStats, compute_vote_stats, weighted_vote


PredictFn = Callable[[Image.Image, str], str]


@dataclass
class CandidateResult:
    candidate_id: int
    stage: int
    image_transform_id: str
    image_transform_parameters: Dict[str, Any]
    image_transform_preset: str
    text_variant_id: str
    prompt: str
    raw_output: str
    normalized_answer: str | None
    is_valid: bool
    parse_status: str
    decoding_settings: Dict[str, Any]
    token_metadata: Dict[str, Any]
    elapsed_s: float


DEFAULT_CANDIDATE_RECIPE: List[tuple[int, str, str]] = [
    (1, "original",            "original"),
    (1, "original",            "hardcoded_paraphrase"),
    (1, "original",            "model_paraphrase"),
    (2, "edge_enhance",        "original"),
    (2, "grayscale",           "original"),
    (2, "jpeg_recompress",     "original"),
    (2, "brightness_contrast", "original"),
    (2, "rotation_90",         "original"),
    (2, "edge_enhance",        "model_paraphrase"),
]


def make_temperature_recipe(n: int) -> List[tuple[int, str, str]]:
    """Build a candidate recipe for pure temperature sampling (no input perturbation).

    All *n* candidates use the original image and original prompt.  Diversity
    comes entirely from stochastic decoding (temperature > 0 in the predict_fn).

    Args:
        n: Number of candidates to generate.

    Returns:
        Recipe list of length *n*, all entries ``(1, "original", "original")``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return [(1, "original", "original")] * n


def export_debug_artifacts(
    output_dir: str | Path,
    original_image: Image.Image,
    original_question: str,
    choices: Dict[str, str],
    candidates: List[Dict[str, Any]],
) -> Path:
    """Export debug artifacts for one TTS wrapper run.

    Writes the original and varied images, text prompts, and metadata for
    a single experimental example so candidate construction is inspectable.
    """
    out = Path(output_dir)
    images_dir = out / "images"
    text_dir = out / "text"
    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    original_image.convert("RGB").save(images_dir / "original.png")

    saved_image_keys = {"original"}
    for c in candidates:
        key = str(c["image_key"])
        if key in saved_image_keys:
            continue
        c["image"].convert("RGB").save(images_dir / f"{key}.png")
        saved_image_keys.add(key)

    (text_dir / "original_question.txt").write_text(original_question.strip() + "\n", encoding="utf-8")
    (text_dir / "choices.json").write_text(json.dumps(choices, indent=2, ensure_ascii=False), encoding="utf-8")

    for c in candidates:
        cid = c["candidate_id"]
        (text_dir / f"candidate_{cid}_prompt.txt").write_text(c["prompt"], encoding="utf-8")

    metadata = {
        "original_question": original_question,
        "choices": choices,
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "stage": c["stage"],
                "image_key": c["image_key"],
                "prompt_key": c["prompt_key"],
                "image_transform_id": c.get("image_transform_id", c["image_key"]),
                "image_transform_parameters": c.get("image_transform_parameters", {}),
                "image_transform_preset": c.get("image_transform_preset", ""),
                "text_variant_id": c.get("text_variant_id", c["prompt_key"]),
            }
            for c in candidates
        ],
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def build_candidate_inputs(
    image: Image.Image,
    question: str,
    choices: Dict[str, str],
    max_candidates: int | None = None,
    text_mode: str = "rule",
    model_paraphrase_fn: Callable[[str, Dict[str, str], int], str] | None = None,
    candidate_recipe: Sequence[tuple[int, str, str]] | None = None,
    image_config: ImageVariationConfig | None = None,
) -> List[Dict[str, Any]]:
    """Build candidate inputs for the TTS wrapper.

    Args:
        image: Input image.
        question: Original question.
        choices: Multiple-choice options (A/B/C/D).
        max_candidates: Number of candidates to build. Defaults to the full recipe length.
        text_mode: Text variation mode ("rule" or "model").
        model_paraphrase_fn: Optional callback for model-based paraphrases.

    Returns:
        Ordered list of candidate dicts with ``candidate_id``, ``stage``, ``image``, and ``prompt``.
    """
    prompt_variants = generate_prompt_variants(
        question,
        choices,
        mode=text_mode,
        model_paraphrase_fn=model_paraphrase_fn,
    )
    image_specs = generate_image_variant_specs(image, config=image_config)

    recipe = list(candidate_recipe or DEFAULT_CANDIDATE_RECIPE)
    n = max_candidates if max_candidates is not None else len(recipe)
    if len(recipe) < n:
        raise ValueError("candidate_recipe does not contain enough entries for requested max_candidates")

    candidates: List[Dict[str, Any]] = []
    for idx, (stage, image_id, text_id) in enumerate(recipe[:n], start=1):
        if image_id not in image_specs:
            raise ValueError(f"Unknown image transform in candidate recipe: {image_id}")
        if text_id not in prompt_variants:
            raise ValueError(f"Unknown text variant in candidate recipe: {text_id}")

        image_spec = image_specs[image_id]
        prompt_spec = prompt_variants[text_id]
        candidates.append(
            {
                "candidate_id": idx,
                "stage": int(stage),
                "image_key": image_id,
                "prompt_key": text_id,
                "image": image_spec["image"],
                "prompt": prompt_spec["prompt"],
                "image_transform_id": image_spec["transform_id"],
                "image_transform_parameters": dict(image_spec["parameters"]),
                "image_transform_preset": image_spec["preset"],
                "text_variant_id": prompt_spec["text_variant_id"],
            }
        )

    return candidates


def _run_candidate(
    candidate_id: int,
    stage: int,
    image: Image.Image,
    prompt: str,
    predict_fn: PredictFn,
    image_transform_id: str = "original",
    image_transform_parameters: Dict[str, Any] | None = None,
    image_transform_preset: str = "",
    text_variant_id: str = "original",
    decoding_settings: Dict[str, Any] | None = None,
    open_ended: bool = False,
) -> CandidateResult:
    t0 = time.perf_counter()
    model_out = predict_fn(image, prompt)
    elapsed_s = time.perf_counter() - t0

    token_metadata: Dict[str, Any] = {}
    if isinstance(model_out, dict):
        raw = str(model_out.get("answer", model_out.get("raw_output", "")))
        token_metadata = dict(model_out.get("token_metadata") or {})
    else:
        raw = str(model_out)

    norm = normalize_open_ended_answer(raw) if open_ended else normalize_answer(raw)
    parse_status = "valid" if norm is not None else "invalid"
    return CandidateResult(
        candidate_id=candidate_id,
        stage=stage,
        image_transform_id=image_transform_id,
        image_transform_parameters=dict(image_transform_parameters or {}),
        image_transform_preset=image_transform_preset,
        text_variant_id=text_variant_id,
        prompt=prompt,
        raw_output=raw,
        normalized_answer=norm,
        is_valid=norm is not None,
        parse_status=parse_status,
        decoding_settings=dict(decoding_settings or {}),
        token_metadata=token_metadata,
        elapsed_s=elapsed_s,
    )


def run_baseline(
    image: Image.Image,
    question: str,
    choices: Dict[str, str],
    predict_fn: PredictFn,
    text_mode: str = "rule",
    image_config: ImageVariationConfig | None = None,
    decoding_settings: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run single-input baseline (original image + original question)."""
    candidate = build_candidate_inputs(
        image=image,
        question=question,
        choices=choices,
        max_candidates=1,
        candidate_recipe=[(1, "original", "original")],
        text_mode=text_mode,
        image_config=image_config,
    )[0]
    result = _run_candidate(
        1,
        1,
        candidate["image"],
        candidate["prompt"],
        predict_fn,
        image_transform_id=candidate.get("image_transform_id", "original"),
        image_transform_parameters=candidate.get("image_transform_parameters", {}),
        image_transform_preset=candidate.get("image_transform_preset", ""),
        text_variant_id=candidate.get("text_variant_id", "original"),
        decoding_settings=decoding_settings,
        open_ended=not choices,
    )
    return {
        "raw_output": result.raw_output,
        "normalized_answer": result.normalized_answer,
        "is_valid": result.is_valid,
    }


def run_tts_pipeline(
    image: Image.Image,
    question: str,
    choices: Dict[str, str],
    predict_fn: PredictFn,
    text_mode: str = "rule",
    model_paraphrase_fn: Callable[[str, Dict[str, str], int], str] | None = None,
    candidate_recipe: Sequence[tuple[int, str, str]] | None = None,
    image_config: ImageVariationConfig | None = None,
    decoding_settings: Dict[str, Any] | None = None,
    candidate_weights: Sequence[float] | None = None,
    allow_early_stop: bool = True,
) -> Dict[str, Any]:
    """Run adaptive 3->5 candidate test-time scaling wrapper.

    Stage 1: candidates 1..3
      1) original image + original question
      2) original image + text paraphrase 1
      3) image variation 1 + original question

    Stop early when agreement reaches >= 2/3 over valid normalized votes.

    Stage 2 (if needed): candidates 4..5
      4) original image + text paraphrase 2
      5) image variation 2 + original question
    """
    recipe = list(candidate_recipe or DEFAULT_CANDIDATE_RECIPE)
    planned = build_candidate_inputs(
        image=image,
        question=question,
        choices=choices,
        max_candidates=len(recipe),
        text_mode=text_mode,
        model_paraphrase_fn=model_paraphrase_fn,
        candidate_recipe=recipe,
        image_config=image_config,
    )

    open_ended = not choices
    stage1_candidates = [c for c in planned if c["stage"] == 1]
    results: List[CandidateResult] = [
        _run_candidate(
            c["candidate_id"],
            c["stage"],
            c["image"],
            c["prompt"],
            predict_fn,
            image_transform_id=c.get("image_transform_id", c.get("image_key", "original")),
            image_transform_parameters=c.get("image_transform_parameters", {}),
            image_transform_preset=c.get("image_transform_preset", ""),
            text_variant_id=c.get("text_variant_id", c.get("prompt_key", "original")),
            decoding_settings=decoding_settings,
            open_ended=open_ended,
        )
        for c in stage1_candidates
    ]

    stage1_answers = [r.normalized_answer for r in results]
    stage1_stats = compute_vote_stats(stage1_answers)

    early_stop = allow_early_stop and stage1_stats.valid_votes >= 2 and stage1_stats.agreement_rate >= (2 / 3)
    if not early_stop:
        stage2_candidates = [c for c in planned if c["stage"] == 2]
        for c in stage2_candidates:
            results.append(
                _run_candidate(
                    c["candidate_id"],
                    c["stage"],
                    c["image"],
                    c["prompt"],
                    predict_fn,
                    image_transform_id=c.get("image_transform_id", c.get("image_key", "original")),
                    image_transform_parameters=c.get("image_transform_parameters", {}),
                    image_transform_preset=c.get("image_transform_preset", ""),
                    text_variant_id=c.get("text_variant_id", c.get("prompt_key", "original")),
                    decoding_settings=decoding_settings,
                    open_ended=open_ended,
                )
            )

    all_answers = [r.normalized_answer for r in results]
    final_stats: VoteStats = compute_vote_stats(all_answers)

    # Stage 2 utility: did adding stage-2 candidates change the winning answer?
    if early_stop:
        stage2_changed_answer = False
    else:
        stage2_changed_answer = stage1_stats.winning_answer != final_stats.winning_answer
    weighted_winner = None
    weighted_scores: Dict[str, float] = {}
    if candidate_weights is not None:
        if len(candidate_weights) < len(all_answers):
            raise ValueError("candidate_weights must have at least used_candidates entries")
        weighted_winner, weighted_scores = weighted_vote(
            all_answers,
            list(candidate_weights[: len(all_answers)]),
        )

    return {
        "stopped_early": early_stop,
        "used_candidates": len(results),
        "winning_answer": final_stats.winning_answer,
        "weighted_winning_answer": weighted_winner,
        "weighted_vote_scores": weighted_scores,
        "agreement_rate": final_stats.agreement_rate,
        "answer_entropy": final_stats.answer_entropy,
        "vote_margin": final_stats.vote_margin,
        "vote_counts": final_stats.vote_counts,
        "stage_1_agreement_rate": stage1_stats.agreement_rate,
        "stage_1_valid_votes": stage1_stats.valid_votes,
        "stage2_changed_answer": stage2_changed_answer,
        "candidate_answers": all_answers,
        "candidates": [
            {
                "candidate_id": r.candidate_id,
                "stage": r.stage,
                "image_transform_id": r.image_transform_id,
                "image_transform_parameters": r.image_transform_parameters,
                "image_transform_preset": r.image_transform_preset,
                "text_variant_id": r.text_variant_id,
                "prompt": r.prompt,
                "parse_status": r.parse_status,
                "decoding_settings": r.decoding_settings,
                "token_metadata": r.token_metadata,
                "raw_output": r.raw_output,
                "normalized_answer": r.normalized_answer,
                "is_valid": r.is_valid,
                "elapsed_s": r.elapsed_s,
            }
            for r in results
        ],
    }
