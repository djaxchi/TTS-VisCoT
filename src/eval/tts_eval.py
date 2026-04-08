"""Evaluation helpers for TTS pipeline experiments on TreeBench."""

from __future__ import annotations

import base64
import io
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from src.pipeline_tts import PredictFn, run_baseline, run_tts_pipeline
from src.utils_normalize import normalize_open_ended_answer


def _find_answer_tag_end(raw_output: str) -> Optional[str]:
    """Return the raw_output prefix up to and including the opening ``<answer>`` tag.

    Used to locate the exact token position where a CoT model (e.g. GRIT)
    predicts its answer letter, so that option logprobs are extracted at the
    right decoding step rather than at step 1.

    Args:
        raw_output: Full generated text including CoT tags.

    Returns:
        Prefix string ending with ``<answer>``, or ``None`` if tag not found.
    """
    tag = "<answer>"
    idx = raw_output.find(tag)
    if idx == -1:
        return None
    return raw_output[: idx + len(tag)]


def _qwen_letter_token_ids(tokenizer: Any) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for letter in ("A", "B", "C", "D"):
        ids: set[int] = set()
        for text in (letter, f" {letter}"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
        out[letter] = sorted(ids)
    return out


def _extract_option_stats_at_prefix(
    model: Any,
    image: Image.Image,
    prompt: str,
    *,
    topk: int,
    assistant_prefix: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract option-letter logits and top-k at a specific decoding position.

    For direct VLMs (Qwen baseline): extracts at step 1 (first generated token).
    For CoT models (GRIT): pass ``assistant_prefix`` = everything generated up to
    and including the ``<answer>`` tag so that logits are extracted at the exact
    position where the model predicts the answer letter — after its full reasoning.

    Args:
        model: Model with ``_processor`` and ``_model`` attributes (Qwen-based).
        image: Input image.
        prompt: Question/instruction text.
        topk: Number of top-k tokens to record.
        assistant_prefix: If provided, build an assistant-prefill context so
            logits are extracted at the token AFTER this prefix.  Pass the
            raw output text up to and including ``<answer>`` for GRIT.
        system_prompt: Optional system message prepended before the user turn.

    Returns:
        Dict with ``option_scores``, ``option_logprobs``, ``option_token_ids``,
        and ``topk`` lists, or empty dict when the backend is unsupported.
    """
    processor = getattr(model, "_processor", None)
    core_model = getattr(model, "_model", None)
    if processor is None or core_model is None:
        return {}

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return {}

    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except Exception:
        return {}

    try:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        image_uri = f"data:image;base64,{base64.b64encode(buf.getvalue()).decode()}"

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        if assistant_prefix is not None:
            # Prefill the assistant turn up to the answer position.
            # add_generation_prompt=False because the assistant turn is already started.
            messages.append({"role": "assistant", "content": assistant_prefix})
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(core_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = core_model(**inputs, use_cache=False, return_dict=True)
        logits = outputs.logits[:, -1, :].squeeze(0)
        log_probs = torch.log_softmax(logits, dim=-1)

        letter_ids = _qwen_letter_token_ids(tokenizer)
        option_scores: Dict[str, float] = {}
        option_logprobs: Dict[str, float] = {}
        option_token_ids: Dict[str, List[int]] = {}

        for letter, ids in letter_ids.items():
            option_token_ids[letter] = list(ids)
            if not ids:
                option_scores[letter] = float("-inf")
                option_logprobs[letter] = float("-inf")
                continue
            idx = torch.tensor(ids, dtype=torch.long, device=logits.device)
            option_scores[letter] = float(torch.max(logits[idx]).item())
            option_logprobs[letter] = float(torch.max(log_probs[idx]).item())

        tk = max(int(topk), 1)
        top_vals, top_ids = torch.topk(logits, k=min(tk, int(logits.shape[-1])))
        topk_rows = []
        for val, tid in zip(top_vals.tolist(), top_ids.tolist()):
            tid_int = int(tid)
            lp = float(log_probs[tid_int].item())
            topk_rows.append(
                {
                    "token_id": tid_int,
                    "token": tokenizer.decode([tid_int], skip_special_tokens=False),
                    "logit": float(val),
                    "log_prob": lp,
                    "prob": float(torch.exp(torch.tensor(lp)).item()),
                }
            )

        step_label = "answer_step" if assistant_prefix is not None else "step_1"
        return {
            "option_scores": [{"step": step_label, "scores": option_scores}],
            "option_logprobs": [{"step": step_label, "logprobs": option_logprobs}],
            "option_token_ids": option_token_ids,
            "topk": [{"step": step_label, "tokens": topk_rows}],
        }
    except Exception:
        return {}


def make_predict_fn(
    model: Any,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    return_details: bool = False,
    token_storage_mode: str = "none",
    token_topk: int = 5,
    open_ended: bool = False,
) -> PredictFn:
    """Wrap a BaseVisualCoTModel into the PredictFn interface.

    Calls ``model.predict(image, prompt, ...)`` and extracts the ``"answer"``
    string so the pipeline does not need to know about chain-dict structure.

    Args:
        model: Any model with a ``predict(image, prompt, **kwargs) -> dict`` method.
        temperature: Passed through to ``model.predict``.
        max_new_tokens: Passed through to ``model.predict``.

    Returns:
        A callable conforming to ``PredictFn = Callable[[Image, str], str]``.
    """

    def _predict(image: Image.Image, prompt: str) -> Any:
        chain = model.predict(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        answer = chain.get("answer", "")
        if not return_details:
            return answer

        token_metadata: Dict[str, Any] = {
            "storage_mode": token_storage_mode,
            "generated_token_ids": [],
            "generated_tokens": [],
        }
        if token_storage_mode != "none":
            proc = getattr(model, "_processor", None)
            tok = getattr(proc, "tokenizer", None) if proc is not None else None
            if tok is not None:
                ids = tok.encode(answer or "", add_special_tokens=False)
                token_metadata["generated_token_ids"] = [int(i) for i in ids]
                if token_storage_mode in {"full", "topk", "options_only"}:
                    token_metadata["generated_tokens"] = list(tok.convert_ids_to_tokens(ids))
                if token_storage_mode == "topk":
                    token_metadata["topk_k"] = int(token_topk)

            # For CoT models (e.g. GRIT) that store raw_output in the chain dict,
            # extract option logprobs at the answer token position (after <answer>)
            # rather than at step 1, which would be inside the <think> block.
            chain_raw_output = chain.get("raw_output", "")
            answer_prefix = _find_answer_tag_end(chain_raw_output) if chain_raw_output else None
            sys_prompt = getattr(model, "system_prompt", None)

            if not open_ended:
                stats = _extract_option_stats_at_prefix(
                    model,
                    image,
                    prompt,
                    topk=int(token_topk),
                    assistant_prefix=answer_prefix,
                    system_prompt=sys_prompt,
                )
            else:
                stats = {}
            if token_storage_mode in {"options_only", "full"}:
                token_metadata["option_scores"] = stats.get("option_scores", [])
                token_metadata["option_logprobs"] = stats.get("option_logprobs", [])
                token_metadata["option_token_ids"] = stats.get("option_token_ids", {})
            if token_storage_mode in {"topk", "full"}:
                token_metadata["topk"] = stats.get("topk", [])

        return {
            "answer": answer,
            "token_metadata": token_metadata,
        }

    return _predict


def evaluate_one(
    image: Image.Image,
    question: str,
    choices: Dict[str, str],
    correct_answer: str,
    predict_fn: PredictFn,
    mode: str = "both",
    tts_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run baseline and/or TTS on one example, returning a structured result dict.

    Args:
        image: Input image.
        question: Question text.
        choices: Multiple-choice options (A/B/C/D).
        correct_answer: Ground-truth label (A/B/C/D).
        predict_fn: Callable ``(image, prompt) -> raw_str`` adapter.
        mode: ``"baseline"``, ``"tts"``, or ``"both"``.

    Returns:
        Dict with ``question``, ``correct_answer``, and ``baseline`` / ``tts``
        sub-dicts (``None`` when not run).
    """
    result: Dict[str, Any] = {
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "baseline": None,
        "tts": None,
    }

    open_ended = not choices
    norm_gt = normalize_open_ended_answer(correct_answer) if open_ended else correct_answer

    if mode in ("baseline", "both"):
        bl = run_baseline(image, question, choices, predict_fn)
        result["baseline"] = {
            "raw_output": bl["raw_output"],
            "normalized_answer": bl["normalized_answer"],
            "is_valid": bl["is_valid"],
            "is_correct": bl["normalized_answer"] == norm_gt,
        }

    if mode in ("tts", "both"):
        tts = run_tts_pipeline(image, question, choices, predict_fn, **(tts_kwargs or {}))
        # Annotate each candidate with whether its individual answer was correct.
        # is_correct is None when the candidate produced an invalid (un-parseable) answer.
        annotated_candidates = []
        for c in tts["candidates"]:
            c_copy = dict(c)
            norm = c_copy.get("normalized_answer")
            if norm is None:
                c_copy["is_correct"] = None
            else:
                c_copy["is_correct"] = norm == norm_gt
            annotated_candidates.append(c_copy)
        result["tts"] = {
            "winning_answer": tts["winning_answer"],
            "weighted_winning_answer": tts.get("weighted_winning_answer"),
            "weighted_vote_scores": tts.get("weighted_vote_scores", {}),
            "is_correct": tts["winning_answer"] == norm_gt,
            "stopped_early": tts["stopped_early"],
            "used_candidates": tts["used_candidates"],
            "candidate_answers": tts["candidate_answers"],
            "agreement_rate": tts["agreement_rate"],
            "answer_entropy": tts.get("answer_entropy", 0.0),
            "vote_margin": tts["vote_margin"],
            "vote_counts": tts["vote_counts"],
            "stage_1_agreement_rate": tts.get("stage_1_agreement_rate"),
            "stage2_changed_answer": tts.get("stage2_changed_answer"),
            "candidates": annotated_candidates,
        }

    return result


def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics from a list of ``evaluate_one`` results.

    Handles mixed modes: computes only the metrics available across all results.

    Args:
        results: List of result dicts returned by ``evaluate_one``.

    Returns:
        Summary dict with accuracy, TTS stopping rate, and average candidates used.
    """
    n = len(results)
    summary: Dict[str, Any] = {"n_questions": n}
    if n == 0:
        return summary

    baseline_rows = [r["baseline"] for r in results if r.get("baseline") is not None]
    if baseline_rows:
        correct = sum(1 for b in baseline_rows if b["is_correct"])
        summary["baseline_accuracy"] = correct / len(baseline_rows)
        summary["baseline_correct"] = correct
        summary["baseline_n"] = len(baseline_rows)

    tts_rows = [r["tts"] for r in results if r.get("tts") is not None]
    if tts_rows:
        correct = sum(1 for t in tts_rows if t["is_correct"])
        stopped = sum(1 for t in tts_rows if t["stopped_early"])
        summary["tts_accuracy"] = correct / len(tts_rows)
        summary["tts_correct"] = correct
        summary["tts_n"] = len(tts_rows)
        summary["tts_early_stop_rate"] = stopped / len(tts_rows)
        summary["tts_avg_candidates"] = (
            sum(t["used_candidates"] for t in tts_rows) / len(tts_rows)
        )

    return summary
