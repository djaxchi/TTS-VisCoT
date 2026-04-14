#!/usr/bin/env python3
"""Candidate re-ranking experiment.

Given existing 9-candidate results, this script:
  1. Loads a JSONL file produced by run_tts_scale.py.
  2. For each question, collects the unique answers proposed by the 9 candidates.
  3. Re-queries the model with the original image + question and the unique candidate
     answers as labelled options, asking it to pick the best one.
  4. Evaluates whether re-ranking beats greedy, majority vote, and oracle@9.

The re-ranker is always DirectVLM (Qwen2.5-VL-3B-Instruct) regardless of which model
generated the candidates — this tests whether a fast, direct model can act as a
verifier over a richer candidate pool.

Usage
-----
  # Re-rank GRIT standard results
  python experiments/run_rerank.py --source results/tts_scale/grit_results.jsonl

  # Re-rank Qwen3B T=0 results
  python experiments/run_rerank.py --source results/tts_scale_t0/qwen3b_results.jsonl

  # Re-rank a single task only
  python experiments/run_rerank.py --source results/tts_scale/grit_results.jsonl --task vqa

Output
------
  results/rerank/<stem>_reranked.jsonl   — one row per question with rerank result
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.utils.logging import get_logger

logger = get_logger(__name__)

OUT_DIR = _PROJECT_ROOT / "results" / "rerank"

# ── Model registry (mirrors run_tts_scale.py) ─────────────────────────────────

RERANKER_CONFIGS = {
    "qwen3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "module":   "src.models.direct_vlm",
        "class":    "DirectVLMModel",
        "kwargs":   {"model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "load_in_8bit": False},
        "system_prompt": "You are a helpful visual question answering assistant.",
        "answer_parser": lambda raw: raw.strip(),
    },
    "grit": {
        "model_id": "yfan1997/GRIT-20-Qwen2.5-VL-3B",
        "module":   "src.models.grit",
        "class":    "GRITModel",
        "kwargs":   {"model_id": "yfan1997/GRIT-20-Qwen2.5-VL-3B", "load_in_8bit": False},
        "system_prompt": (
            "First, think between <think> and </think> while output necessary coordinates "
            "needed to answer the question in JSON with key 'bbox_2d'. "
            "Then, based on the thinking contents and coordinates, rethink between "
            "<rethink></rethink> and then answer the question after <answer>."
        ),
        "answer_parser": lambda raw: (
            __import__("re").search(r"<answer>(.*?)(?:</answer>|$)", raw, __import__("re").DOTALL)
            or type("_", (), {"group": lambda s, i: raw})()
        ).group(1).strip(),
    },
}


def _detect_model_key(source: Path) -> str:
    """Infer which model to use as re-ranker from the source filename."""
    stem = source.stem.lower()
    if stem.startswith("grit"):
        return "grit"
    if "qwen" in stem:
        return "qwen3b"
    raise ValueError(
        f"Cannot infer model from source filename '{source.name}'. "
        "Pass --rerank_model grit|qwen3b explicitly."
    )

# ── Answer normalisation ──────────────────────────────────────────────────────

from src.utils_normalize import normalize_answer, normalize_open_ended_answer


def _norm(raw: str, task: str) -> str:
    if not raw:
        return ""
    if task == "ocr":
        return normalize_open_ended_answer(raw) or ""
    return normalize_answer(raw) or ""


def _is_correct(pred: str, row: Dict[str, Any]) -> bool:
    p = _norm(pred, row["task"])
    if not p:
        return False
    task = row["task"]
    for a in (row.get("answers_all") or [row["gt_answer"]]):
        if p == _norm(a, task):
            return True
    return False


# ── Build unique proposals from candidates ────────────────────────────────────

def _unique_proposals(candidates: List[Dict[str, Any]], task: str) -> List[str]:
    """Return deduplicated, normalised answers from the candidate pool."""
    seen: Dict[str, str] = {}  # norm -> raw (keep first occurrence)
    for c in candidates:
        raw = c.get("answer") or ""
        if not raw:
            continue
        n = _norm(raw, task)
        if n and n not in seen:
            seen[n] = raw
    return list(seen.values())  # raw answers, deduped by normalised form


# ── Re-ranking prompt ─────────────────────────────────────────────────────────

def _build_rerank_prompt(original_question: str, proposals: List[str], task: str) -> str:
    """Construct the re-ranking prompt shown to the verifier model.

    For MCQ tasks (vqa, counting) the candidates are already option letters
    (e.g. "A", "C").  We must NOT re-label them with new A/B/C prefixes —
    that creates a confusing double layer ("A. a", "B. c").  Instead we show
    the original question (which already contains the option text) and list
    only the proposed letters so the model can cross-reference.

    For OCR (free-form) we assign fresh A/B/C labels since the proposals are
    plain text strings with no existing letter identity.
    """
    if task in ("vqa", "counting"):
        # proposals are letters like ["A", "C", "D"] — list them, keep original Q
        proposed_str = "  " + ",  ".join(p.upper() for p in proposals)
        return (
            f"{original_question}\n\n"
            f"Multiple candidates were generated. They proposed these answers: "
            f"{proposed_str.strip()}.\n"
            f"Based on the image, which answer is correct? "
            f"Reply with only the single letter."
        )
    else:  # ocr — free-form text, assign fresh labels
        labelled = "\n".join(f"  {chr(65+i)}. {p}" for i, p in enumerate(proposals))
        return (
            f"{original_question}\n\n"
            f"The following candidate answers were proposed. "
            f"Based on the image, select the most accurate one. "
            f"Reply with only the letter (A, B, C, …).\n\n"
            f"Candidates:\n{labelled}"
        )


# ── Model load / unload ───────────────────────────────────────────────────────

def _load_reranker(model_key: str):
    """Load the reranker model using its native class (GRIT or DirectVLM)."""
    import importlib
    from transformers import AutoProcessor

    cfg = RERANKER_CONFIGS[model_key]
    logger.info("Loading re-ranker '{}' ({}) …", model_key, cfg["model_id"])
    mod = importlib.import_module(cfg["module"])
    cls = getattr(mod, cfg["class"])
    model_obj = cls(**cfg["kwargs"])
    model_obj._load()
    # Widen pixel budget for GRIT so it sees fine-grained details during rerank
    if model_key == "grit" and hasattr(model_obj, "_processor"):
        model_obj._processor = AutoProcessor.from_pretrained(
            cfg["model_id"],
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
    logger.info("Re-ranker loaded.")
    return model_obj


def _unload(model_obj) -> None:
    import torch
    try:
        model_obj._model.cpu()
    except Exception:
        pass
    del model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Single-question re-rank ───────────────────────────────────────────────────

def _rerank_one(model_obj, image, prompt: str, model_key: str) -> str:
    """Run one reranking inference using the model's native generate().

    Returns the raw answer string (after answer-tag parsing for GRIT).
    """
    # Use n=1, T=0 (greedy) — we want the model's best single answer
    outs = model_obj.generate(image, prompt, n=1, temperature=0.0, max_new_tokens=512)
    raw = outs[0].get("answer", "") if outs else ""

    # Apply model-specific answer parser (strips <answer> tags for GRIT)
    parser = RERANKER_CONFIGS[model_key]["answer_parser"]
    return parser(raw)


def _parse_letter(raw: str, proposals: List[str], task: str) -> Optional[str]:
    """Map the model reply back to a proposal answer.

    MCQ (vqa/counting): proposals ARE letters (e.g. ["A","C"]).  The model
    replies with a letter — return it directly if it's in the proposals.

    OCR: proposals are free-form text labelled A, B, C…  Map the reply
    letter to the proposal at that index.
    """
    raw = raw.strip()
    if not raw:
        return None
    letter = raw[0].upper()
    if task in ("vqa", "counting"):
        # model replies with the original option letter — return it directly
        if letter in [p.upper() for p in proposals]:
            return letter
        # fallback: first letter of output if it's a plausible option
        return letter if letter.isalpha() else None
    else:
        # OCR: map new label A/B/C back to proposal index
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(proposals):
            return proposals[idx]
        return raw


# ── Checkpoint / resume ───────────────────────────────────────────────────────

def _out_path(source: Path, model_key: str) -> Path:
    # Include parent dir name so tts_scale vs tts_scale_t0 don't collide
    recipe = source.parent.name  # e.g. "tts_scale" or "tts_scale_t0"
    return OUT_DIR / f"{recipe}_{source.stem}_reranked_by_{model_key}.jsonl"


def _load_done(source: Path, model_key: str) -> set:
    p = _out_path(source, model_key)
    if not p.exists():
        return set()
    done = set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                done.add(str(json.loads(line)["question_id"]))
            except Exception:
                pass
    return done


def _append(row: Dict[str, Any], source: Path, model_key: str) -> None:
    with _out_path(source, model_key).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_images_for_task(task: str) -> Dict[str, Any]:
    from src.data.datasets.viscot_benchmark import load_task
    examples = load_task(task)
    return {str(ex["question_id"]): ex["image"] for ex in examples}


# ── Main runner ───────────────────────────────────────────────────────────────

def run(source: Path, tasks_filter: Optional[List[str]], model_key: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load source rows
    all_rows: List[Dict[str, Any]] = []
    with source.open(encoding="utf-8") as f:
        for line in f:
            try:
                all_rows.append(json.loads(line))
            except Exception:
                pass

    if tasks_filter:
        all_rows = [r for r in all_rows if r["task"] in tasks_filter]

    done_ids = _load_done(source, model_key)
    todo = [r for r in all_rows if str(r["question_id"]) not in done_ids]
    print(f"\nSource: {source.name}  total={len(all_rows)}  done={len(done_ids)}  todo={len(todo)}", flush=True)

    print(f"Re-ranker: {model_key} ({RERANKER_CONFIGS[model_key]['model_id']})", flush=True)

    if not todo:
        print("All done.", flush=True)
        return

    # Pre-load images per task
    print("Loading images…", flush=True)
    tasks_needed = list({r["task"] for r in todo})
    images: Dict[str, Dict[str, Any]] = {}
    for task in tasks_needed:
        images[task] = _load_images_for_task(task)
        print(f"  {task}: {len(images[task])} images", flush=True)

    # Load re-ranker
    model_obj = _load_reranker(model_key)

    try:
        t_start = time.time()
        for idx, row in enumerate(todo):
            task = row["task"]
            qid = str(row["question_id"])
            image = images[task].get(qid)
            if image is None:
                logger.warning("Image not found for qid={}", qid)
                continue

            proposals = _unique_proposals(row["candidates"], task)

            # If only one unique proposal, reranking degenerates — record directly
            if len(proposals) <= 1:
                chosen_raw = proposals[0] if proposals else ""
                rerank_raw_output = "(single proposal — no choice)"
            else:
                prompt = _build_rerank_prompt(row["question"], proposals, task)
                try:
                    rerank_raw_output = _rerank_one(model_obj, image, prompt, model_key)
                except Exception as e:
                    logger.warning("Rerank failed qid={}: {}", qid, e)
                    rerank_raw_output = ""
                chosen_raw = _parse_letter(rerank_raw_output, proposals, task) or ""

            chosen_norm = _norm(chosen_raw, task)
            correct_rerank = _is_correct(chosen_raw, row)

            out_row = {
                "model": row["model"],
                "task": task,
                "question_id": qid,
                "gt_answer": row["gt_answer"],
                "answers_all": row.get("answers_all", [row["gt_answer"]]),
                "n_unique_proposals": len(proposals),
                "proposals": proposals,
                "rerank_raw_output": rerank_raw_output,
                "rerank_answer": chosen_norm,
                "correct_rerank": correct_rerank,
                "correct_greedy": row["correct_greedy"],
                "correct_vote9": row["correct_9"],
                "correct_oracle": row["correct_any"],
                "greedy_answer": row["greedy"],
            }
            _append(out_row, source, model_key)
            done_ids.add(qid)

            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (len(todo) - idx - 1)
            print(
                f"\r  [{idx+1}/{len(todo)}] {elapsed/60:.0f}min  ETA {eta/60:.0f}min  "
                f"qid={qid}  n_props={len(proposals)}  rerank={'✓' if correct_rerank else '✗'}  "
                f"greedy={'✓' if row['correct_greedy'] else '✗'}",
                end="", flush=True,
            )

    finally:
        print(flush=True)
        _unload(model_obj)
        _print_summary(source, model_key)


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(source: Path, model_key: str) -> None:
    p = _out_path(source, model_key)
    if not p.exists():
        return
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    if not rows:
        return

    from collections import defaultdict
    by_task: Dict[str, List] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    by_task["overall"] = rows

    print(f"\n{'='*70}")
    print(f"  Results: {p.name}")
    print(f"{'='*70}")
    print(f"  {'task':<10} {'n':>4}  {'greedy':>8}  {'rerank':>8}  {'delta':>8}  {'oracle':>8}")
    for task in ("vqa", "ocr", "counting", "overall"):
        rs = by_task.get(task, [])
        if not rs:
            continue
        n = len(rs)
        g = sum(r["correct_greedy"] for r in rs) / n * 100
        rr = sum(r["correct_rerank"] for r in rs) / n * 100
        o = sum(r["correct_oracle"] for r in rs) / n * 100
        delta = rr - g
        sign = "+" if delta >= 0 else ""
        print(f"  {task:<10} {n:>4}  {g:>7.1f}%  {rr:>7.1f}%  {sign}{delta:>6.1f}pp  {o:>7.1f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Re-rank TTS candidates with the same model")
    parser.add_argument(
        "--source", required=True,
        help="Path to JSONL candidate file (e.g. results/tts_scale/grit_results.jsonl)",
    )
    parser.add_argument(
        "--task", choices=["vqa", "ocr", "counting"], default=None,
        help="Run only this task (default: all tasks in source file)",
    )
    parser.add_argument(
        "--rerank_model", choices=list(RERANKER_CONFIGS), default=None,
        help="Model to use as re-ranker. Defaults to auto-detect from source filename.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = _PROJECT_ROOT / source
    if not source.exists():
        print(f"ERROR: source file not found: {source}", file=sys.stderr)
        sys.exit(1)

    model_key = args.rerank_model or _detect_model_key(source)
    print(f"Source:    {source.name}", flush=True)
    print(f"Re-ranker: {model_key}  ({RERANKER_CONFIGS[model_key]['model_id']})", flush=True)

    tasks_filter = [args.task] if args.task else None
    run(source, tasks_filter, model_key)


if __name__ == "__main__":
    main()
