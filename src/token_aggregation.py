"""Token-level and answer-level aggregation helpers for TTS experiments."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image

from src.utils_normalize import normalize_answer


def top_k_token_distribution(
    logits: np.ndarray,
    k: int,
    token_id_to_text: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Return the top-k tokens from a logit vector with their probabilities.

    Args:
        logits: 1-D array of shape [vocab_size] containing raw (pre-softmax) logits.
        k: Number of top tokens to return.  Clipped to ``vocab_size`` if larger.
        token_id_to_text: Mapping from token ID to its string representation.

    Returns:
        List of up to *k* dicts sorted by descending probability, each containing:

        * ``token_id``   – integer token index
        * ``token_text`` – decoded string (empty string if ID not in mapping)
        * ``logit``      – raw logit value (float)
        * ``prob``       – softmax probability in [0, 1] (float)
        * ``log_prob``   – natural log of softmax probability (float)
    """
    arr = np.asarray(logits, dtype=float).ravel()
    vocab_size = arr.shape[0]
    k = min(k, vocab_size)

    # Numerically stable log-softmax: log p_i = x_i - max(x) - log Σ exp(x_j - max(x))
    shifted = arr - arr.max()
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    log_probs = shifted - log_sum_exp
    probs = np.exp(log_probs)

    # top-k indices sorted by descending logit (equivalent to descending prob)
    top_ids = np.argpartition(arr, -k)[-k:]
    top_ids = top_ids[np.argsort(arr[top_ids])[::-1]]

    return [
        {
            "token_id": int(tid),
            "token_text": token_id_to_text.get(int(tid), ""),
            "logit": float(arr[tid]),
            "prob": float(probs[tid]),
            "log_prob": float(log_probs[tid]),
        }
        for tid in top_ids
    ]


def aggregate_answer_level(
    candidates_or_answers: Sequence[Any],
    predict_fn: Any | None = None,
) -> Dict[str, Any]:
    """Aggregate candidates at answer level.

    Supports two modes:
    - ``aggregate_answer_level(["A", "B", ...])``
    - ``aggregate_answer_level(candidates, predict_fn=callable)`` where each
      candidate has ``image`` and ``prompt`` keys.
    """
    raw_answers: List[str] = []
    normalized: List[str | None] = []

    if predict_fn is None:
        raw_answers = [str(x) for x in candidates_or_answers]
    else:
        for c in candidates_or_answers:
            raw_answers.append(str(predict_fn(c["image"], c["prompt"])))

    normalized = [normalize_answer(a) for a in raw_answers]
    valid = [a for a in normalized if a is not None]

    if not valid:
        return {
            "winning_answer": None,
            "raw_answers": raw_answers,
            "normalized_answers": normalized,
            "vote_counts": {},
            "agreement_rate": 0.0,
        }

    counts = Counter(valid)
    top = max(counts.values())
    ties = {k for k, v in counts.items() if v == top}
    winner = next(a for a in valid if a in ties)

    return {
        "winning_answer": winner,
        "raw_answers": raw_answers,
        "normalized_answers": normalized,
        "vote_counts": dict(counts),
        "agreement_rate": top / len(valid),
    }


def aggregate_token_level_from_logits_steps(
    step_logits: Sequence[np.ndarray],
    token_id_to_text: Dict[int, str],
    *,
    choice_token_ids: Sequence[int] | None = None,
    max_steps: int = 8,
    top_k: int = 0,
) -> Dict[str, Any]:
    """Reference token-level aggregation from prepared per-step logits.

    This pure function is intended for testing and algorithm sanity checks.

    Args:
        step_logits: Per-step arrays of shape [n_candidates, vocab_size].
        token_id_to_text: Token ID → string mapping (e.g. from the tokenizer).
        choice_token_ids: If set, restrict token selection to these IDs (A/B/C/D).
        max_steps: Upper bound on decoding steps.
        top_k: If > 0, each step dict includes a ``"top_k"`` key with the top-k
            tokens from the aggregated logits, each as
            ``{"token_id", "token_text", "logit", "prob", "log_prob"}``.
            Defaults to 0 (disabled).
    """
    chosen: List[int] = []
    generated_text = ""
    step_debug: List[Dict[str, Any]] = []

    for step_idx, logits in enumerate(step_logits[:max_steps]):
        arr = np.asarray(logits, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Each step logits array must have shape [n_candidates, vocab_size].")
        agg = arr.mean(axis=0)

        if choice_token_ids:
            allowed = np.asarray(choice_token_ids, dtype=int)
            best_local = int(np.argmax(agg[allowed]))
            token_id = int(allowed[best_local])
        else:
            token_id = int(np.argmax(agg))

        chosen.append(token_id)
        token_text = token_id_to_text.get(token_id, "")
        generated_text += token_text

        cand_pref = [int(np.argmax(arr[i])) for i in range(arr.shape[0])]
        step_row: Dict[str, Any] = {
            "step": step_idx + 1,
            "candidate_top_token_ids": cand_pref,
            "chosen_token_id": token_id,
            "chosen_token_text": token_text,
        }
        if top_k > 0:
            step_row["top_k"] = top_k_token_distribution(agg, top_k, token_id_to_text)
        step_debug.append(step_row)

        norm = normalize_answer(generated_text.strip())
        if norm is not None:
            break

    return {
        "generated_text": generated_text,
        "normalized_answer": normalize_answer(generated_text.strip()),
        "chosen_token_ids": chosen,
        "steps": step_debug,
    }


def _qwen_choice_token_ids(tokenizer: Any) -> List[int]:
    ids: set[int] = set()
    for txt in ("A", "B", "C", "D", " A", " B", " C", " D"):
        encoded = tokenizer.encode(txt, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    return sorted(ids)


def _prepare_qwen_candidate_inputs(model: Any, image: Image.Image, prompt: str) -> Dict[str, torch.Tensor]:
    from qwen_vl_utils import process_vision_info
    import base64
    import io as _io

    buf = _io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    image_uri = f"data:image;base64,{b64}"

    system_prompt = "You are a helpful visual question answering assistant. Answer concisely."
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = model._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    out = model._processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(model._model.device) if hasattr(v, "to") else v for k, v in out.items()}


def _next_logits_for_candidate(
    model: Any,
    base_inputs: Dict[str, torch.Tensor],
    shared_generated_ids: torch.Tensor | None,
) -> torch.Tensor:
    input_ids = base_inputs["input_ids"]
    attention_mask = base_inputs.get("attention_mask")

    if shared_generated_ids is not None and shared_generated_ids.numel() > 0:
        input_ids = torch.cat([input_ids, shared_generated_ids], dim=1)
        if attention_mask is not None:
            ones = torch.ones(
                (attention_mask.shape[0], shared_generated_ids.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, ones], dim=1)

    kwargs: Dict[str, Any] = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    for key in ("pixel_values", "image_grid_thw", "video_grid_thw"):
        if key in base_inputs:
            kwargs[key] = base_inputs[key]

    with torch.inference_mode():
        outputs = model._model(**kwargs, use_cache=False, return_dict=True)
    return outputs.logits[:, -1, :].squeeze(0)


def aggregate_token_level(
    model: Any,
    candidates: Sequence[Dict[str, Any]],
    *,
    max_steps: int = 8,
    restrict_to_choices: bool = True,
    top_k: int = 0,
) -> Dict[str, Any]:
    """Token-level aggregation prototype for Qwen-based VLM backends.

    Args:
        model: Loaded ``DirectVLMModel`` or ``GRITModel`` instance.
        candidates: Candidate inputs with ``image`` and ``prompt``.
        max_steps: Maximum synchronized decoding steps.
        restrict_to_choices: Restrict each chosen token to A/B/C/D-like token IDs.
        top_k: If > 0, each step dict includes a ``"top_k"`` key with the top-k
            tokens from the *aggregated* logits, each as
            ``{"token_id", "token_text", "logit", "prob", "log_prob"}``.
            Defaults to 0 (disabled).

    Returns:
        Dict containing generated text, normalized answer, and per-step diagnostics.
    """
    model._load()

    prepared = [_prepare_qwen_candidate_inputs(model, c["image"], c["prompt"]) for c in candidates]
    tokenizer = model._processor.tokenizer
    choice_ids = _qwen_choice_token_ids(tokenizer) if restrict_to_choices else None

    chosen_ids: List[int] = []
    shared_ids: torch.Tensor | None = None
    generated_text = ""
    steps: List[Dict[str, Any]] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for step in range(max_steps):
        cand_logits: List[torch.Tensor] = []
        cand_top_ids: List[int] = []
        for inp in prepared:
            l = _next_logits_for_candidate(model, inp, shared_ids)
            cand_logits.append(l)
            cand_top_ids.append(int(torch.argmax(l).item()))

        stacked = torch.stack(cand_logits, dim=0)
        agg = stacked.mean(dim=0)

        if choice_ids:
            allowed = torch.tensor(choice_ids, dtype=torch.long, device=agg.device)
            pick_local = int(torch.argmax(agg[allowed]).item())
            token_id = int(allowed[pick_local].item())
        else:
            token_id = int(torch.argmax(agg).item())

        token_piece = tokenizer.decode([token_id], skip_special_tokens=False)
        chosen_ids.append(token_id)
        generated_text += token_piece

        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=prepared[0]["input_ids"].device)
        shared_ids = token_tensor if shared_ids is None else torch.cat([shared_ids, token_tensor], dim=1)

        step_row: Dict[str, Any] = {
            "step": step + 1,
            "candidate_top_token_ids": cand_top_ids,
            "candidate_top_tokens": [tokenizer.decode([t], skip_special_tokens=False) for t in cand_top_ids],
            "chosen_token_id": token_id,
            "chosen_token": token_piece,
        }
        if top_k > 0:
            agg_np = agg.cpu().numpy() if hasattr(agg, "cpu") else np.asarray(agg)
            vocab = {i: tokenizer.decode([i], skip_special_tokens=False) for i in range(len(agg_np))}
            step_row["top_k"] = top_k_token_distribution(agg_np, top_k, vocab)
        steps.append(step_row)

        norm = normalize_answer(generated_text.strip())
        if norm is not None:
            break
        if eos_id is not None and token_id == int(eos_id):
            break

    return {
        "generated_text": generated_text,
        "normalized_answer": normalize_answer(generated_text.strip()),
        "chosen_token_ids": chosen_ids,
        "steps": steps,
        "choice_token_ids": choice_ids or [],
    }
