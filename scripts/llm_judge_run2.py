"""LLM-as-a-judge: annotate each answer in results/comparison/run2.json.

Uses the same Qwen2.5-VL-7B-Instruct model that is already used in the
comparison experiment.  For every VQA entry the model is asked (text-only,
no image) whether the predicted answer is semantically correct given the
question and reference answers.  The result is stored as a new key
``"llm_judge"`` (bool) on each entry.  Already-judged entries are skipped so
the script is safe to re-run.

Usage:
    python scripts/llm_judge_run2.py
    python scripts/llm_judge_run2.py --input  results/comparison/run2.json
    python scripts/llm_judge_run2.py --output results/comparison/run2_judged.json
    python scripts/llm_judge_run2.py --no-8bit
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Model ID — same checkpoint as the comparison experiment
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

JUDGE_SYSTEM_PROMPT = (
    "You are a strict VQA answer evaluator.  "
    "Decide whether a model answer is semantically correct given the question "
    "and a list of reference answers.  "
    "Ignore capitalisation, punctuation, and minor wording differences "
    "(e.g. 'yes.' == 'yes', 'light blue' == 'light-blue', 'river' may equal 'shore' "
    "only if they describe the same concept in context).  "
    "Reply with exactly one word: YES or NO."
)


def _build_prompt(question: str, references: list[str], answer: str) -> str:
    ref_str = ", ".join(f'"{r}"' for r in references)
    return (
        f"Question: {question}\n"
        f"Reference answer(s): {ref_str}\n"
        f"Model answer: \"{answer}\"\n\n"
        "Is the model answer correct? Reply YES or NO."
    )


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_judge(model_id: str, load_in_8bit: bool) -> tuple[Any, Any]:
    """Load Qwen2.5-VL processor and model weights.

    Returns:
        (processor, model) tuple ready for text-only inference.
    """
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    print(f"Loading judge model '{model_id}' (8-bit={load_in_8bit}) …")
    quant_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quant_config,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Judge model loaded.\n")
    return processor, model


# ---------------------------------------------------------------------------
# Single judgment
# ---------------------------------------------------------------------------

def judge(
    processor: Any,
    model: Any,
    question: str,
    references: list[str],
    answer: str,
) -> bool:
    """Return True if the model judges the answer as correct."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(question, references, answer)},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    reply = processor.batch_decode(
        out_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip().upper()

    del inputs, out_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return reply.startswith("YES")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate run2.json with LLM-as-a-judge verdicts using Qwen2.5-VL."
    )
    parser.add_argument(
        "--input",
        default="results/comparison/run2.json",
        help="Path to the input JSON file (default: results/comparison/run2.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (defaults to --input, i.e. in-place update)",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID to use as judge (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--no-8bit",
        action="store_true",
        help="Disable 8-bit quantisation (requires >=40 GB VRAM)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    with open(input_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    # Count work to do
    all_entries = [
        (model_name, task_name, entry)
        for model_name, model_data in data.items()
        for task_name, entries in model_data.items()
        for entry in entries
    ]
    todo = [e for e in all_entries if "llm_judge" not in e[2]]
    already_done = len(all_entries) - len(todo)

    print(f"Total entries : {len(all_entries)}")
    print(f"Already judged: {already_done}  (will be skipped)")
    print(f"To judge now  : {len(todo)}\n")

    if not todo:
        print("Nothing to do — all entries already have 'llm_judge'.")
        return

    processor, model = load_judge(args.model_id, load_in_8bit=not args.no_8bit)

    judged = 0
    for model_name, task_name, entry in todo:
        result = judge(
            processor,
            model,
            entry["question"],
            entry["references"],
            entry["answer"],
        )
        entry["llm_judge"] = result
        judged += 1

        verdict = "YES" if result else "NO "
        print(
            f"[{judged}/{len(todo)}] {model_name} | {task_name} | "
            f"q={entry['question_id']} | judge={verdict} | "
            f"correct={entry.get('correct')} | answer={entry['answer']!r}"
        )

    # Unload model
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Judged {judged} entries. Saved to: {output_path}")


if __name__ == "__main__":
    main()
