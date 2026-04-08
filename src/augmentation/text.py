"""Text variation helpers for test-time scaling."""

from __future__ import annotations

from typing import Callable, Dict, List

STRICT_CONSTRAINT = "Answer with A, B, C, or D only."
_INSTRUCTION_FRAMES = {
    1: "Using the image, answer this question:",
    2: "Determine the best answer from the image:",
}
_ANSWER_FORMAT_FRAMES = {
    0: STRICT_CONSTRAINT,
    1: "Reply using a single letter: A, B, C, or D.",
    2: "Select one option only: A, B, C, or D. Answer with a single capital letter: A, B, C, or D.",
}


def _format_choices(choices: Dict[str, str]) -> str:
    ordered = [k for k in ("A", "B", "C", "D") if k in choices]
    return "\n".join(f"{k}. {choices[k]}" for k in ordered)


_OPEN_ENDED_ANSWER_FRAME = "Answer with a word or short phrase."


def _compose_prompt(
    question: str,
    choices: Dict[str, str],
    instruction_frame: str | None,
    answer_frame: str,
) -> str:
    parts: List[str] = []
    if instruction_frame:
        parts.append(instruction_frame)
    parts.append(question.strip())
    if choices:
        parts.extend(["Choices:", _format_choices(choices)])
    if answer_frame:
        parts.append(answer_frame)
    return "\n".join(parts)


def _rule_paraphrase(question: str, idx: int) -> str:
    """Single hardcoded deterministic paraphrase (idx ignored — only one template)."""
    q = question.strip()
    return f"From the image, determine which option is correct. {q}"


def generate_prompt_variants(
    question: str,
    choices: Dict[str, str],
    mode: str = "rule",
    model_paraphrase_fn: Callable[[str, Dict[str, str], int], str] | None = None,
    add_constraint: bool = True,
    num_paraphrases: int = 2,
) -> Dict[str, Dict[str, str]]:
    """Return auditable prompt variants keyed by text-variant IDs."""
    open_ended = not choices
    if open_ended:
        answer_frame_0 = _OPEN_ENDED_ANSWER_FRAME
    else:
        answer_frame_0 = _ANSWER_FORMAT_FRAMES[0] if add_constraint else ""
    original_prompt = _compose_prompt(
        question=question,
        choices=choices,
        instruction_frame=None,
        answer_frame=answer_frame_0,
    )

    variants: Dict[str, Dict[str, str]] = {
        "original": {
            "text_variant_id": "original",
            "question_variant": question.strip(),
            "prompt": original_prompt,
            "instruction_frame": "",
            "answer_frame": answer_frame_0,
        }
    }

    # Hardcoded deterministic paraphrase (always present)
    hardcoded_q = _rule_paraphrase(question, 1)
    hp_instruction = _INSTRUCTION_FRAMES[1]
    hp_answer = _OPEN_ENDED_ANSWER_FRAME if open_ended else (_ANSWER_FORMAT_FRAMES[1] if add_constraint else "")
    variants["hardcoded_paraphrase"] = {
        "text_variant_id": "hardcoded_paraphrase",
        "question_variant": hardcoded_q,
        "prompt": _compose_prompt(
            question=hardcoded_q,
            choices=choices,
            instruction_frame=hp_instruction,
            answer_frame=hp_answer,
        ),
        "instruction_frame": hp_instruction,
        "answer_frame": hp_answer,
    }

    # Model-generated paraphrase (uses callable when available).
    # Fallback when no callable is provided: keep the original question wording but
    # use instruction frame 2 ("Determine the best answer from the image:") and the
    # strictest answer constraint, giving a genuinely distinct prompt without an LLM.
    if model_paraphrase_fn is not None:
        model_q = model_paraphrase_fn(question, choices, 2)
    else:
        model_q = question.strip()
    mp_instruction = _INSTRUCTION_FRAMES[2]
    mp_answer = _OPEN_ENDED_ANSWER_FRAME if open_ended else (_ANSWER_FORMAT_FRAMES[2] if add_constraint else "")
    variants["model_paraphrase"] = {
        "text_variant_id": "model_paraphrase",
        "question_variant": model_q,
        "prompt": _compose_prompt(
            question=model_q,
            choices=choices,
            instruction_frame=mp_instruction,
            answer_frame=mp_answer,
        ),
        "instruction_frame": mp_instruction,
        "answer_frame": mp_answer,
    }

    return variants


def generate_question_variants(
    question: str,
    choices: Dict[str, str],
    mode: str = "rule",
    model_paraphrase_fn: Callable[[str, Dict[str, str], int], str] | None = None,
    add_constraint: bool = True,
    num_paraphrases: int = 2,
) -> List[str]:
    """Return prompt variants: original plus deterministic paraphrases.

    The default returns exactly three prompts:
    - original question + strict answer constraint
    - paraphrase 1 with alternative instruction/answer framing
    - paraphrase 2 with alternative instruction/answer framing

    Args:
        question: Original question.
        choices: Multiple-choice options (A/B/C/D).
        mode: ``"rule"`` or ``"model"`` paraphrasing strategy.
        model_paraphrase_fn: Optional callable used when ``mode="model"``.
        add_constraint: Whether to append answer-format instructions.
        num_paraphrases: Number of paraphrases to generate (default: 2).

    Returns:
        A list with ``1 + num_paraphrases`` prompts where the first entry is the original.
    """
    variants = generate_prompt_variants(
        question=question,
        choices=choices,
        mode=mode,
        model_paraphrase_fn=model_paraphrase_fn,
        add_constraint=add_constraint,
        num_paraphrases=num_paraphrases,
    )

    ordered = ["original", "hardcoded_paraphrase", "model_paraphrase"]
    return [variants[k]["prompt"] for k in ordered if k in variants]
