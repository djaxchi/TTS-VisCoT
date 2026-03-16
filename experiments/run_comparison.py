#!/usr/bin/env python3
"""Model comparison across Visual CoT baselines on VQA, counting, and OCR.

Runs the configured models across 3 tasks × n questions and prints a
comparison table of
accuracy / tokens / time per model per task.

Questions come from curated 100-sample JSONL files in data/benchmark/.
Images are fetched on-the-fly from public HuggingFace mirrors.

Usage:
    python experiments/run_comparison.py
    python experiments/run_comparison.py --n 50         # 50 questions per task (default)
    python experiments/run_comparison.py --no-viscot    # skip VisCoT
    python experiments/run_comparison.py --no-deepeyes  # skip DeepEyesV2-RL
    python experiments/run_comparison.py --save-output results/comparison/run1.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
DEFAULT_QUESTION_COUNT = 50


def _bar(c: str = "─", w: int = W) -> str:
    return c * w


def _header(title: str) -> str:
    pad = (W - len(title) - 2) // 2
    return f"{BOLD}{'═' * pad} {title} {'═' * (W - pad - len(title) - 2)}{RESET}"


# ---------------------------------------------------------------------------
# Token counting
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
# Per-model inference
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    question_id: str
    question: str
    answer: str
    references: List[str]
    correct: bool
    tokens: int
    elapsed_s: float


def _run_model_on_samples(
    model_obj: Any,
    samples: List[Dict[str, Any]],
    temperature: float,
    max_new_tokens: int,
    model_label: str,
    task: str,
) -> List[InferenceResult]:
    from src.eval.vqa_eval import evaluate_vqa

    results: List[InferenceResult] = []
    for i, samp in enumerate(samples):
        q = samp["question"]
        refs = [samp["answer"]]
        print(
            f"  [{model_label}] {task.upper()} {i+1}/{len(samples)}: "
            f"{q[:65]}{'…' if len(q) > 65 else ''}"
        )
        t0 = time.perf_counter()
        try:
            chain = model_obj.predict(
                samp["image"], q,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    {RED}SKIP{RESET}  error on {samp['question_id']}: {exc}")
            continue
        elapsed = time.perf_counter() - t0
        answer: str = chain.get("answer", "") or ""

        cot_steps: List[str] = chain.get("cot_steps", [])
        full_text = "\n".join(cot_steps) if cot_steps else answer
        tokens = _count_tokens(model_obj, full_text)

        correct = evaluate_vqa(answer, refs)
        tag = f"{GREEN}✓{RESET}" if correct else f"{RED}✗{RESET}"
        print(
            f"    {tag}  pred={answer[:55]!r}  ref={refs[0]!r}  "
            f"tok={tokens}  t={elapsed:.1f}s"
        )
        results.append(InferenceResult(
            question_id=samp["question_id"],
            question=q, answer=answer, references=refs,
            correct=correct, tokens=tokens, elapsed_s=elapsed,
        ))
    return results


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def save_checkpoint(
    out_path: Path,
    all_results: Dict[str, Dict[str, List[InferenceResult]]],
    model_labels: List[str],
    task_labels: List[str],
) -> None:
    """Write a partial results JSON so progress survives a crash.

    Args:
        out_path: Destination file (created with parents as needed).
        all_results: Results collected so far, keyed by model then task.
        model_labels: All model labels (including ones not yet started).
        task_labels: All task labels.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    for ml in model_labels:
        model_data = all_results.get(ml, {})
        payload[ml] = {}
        for tl in task_labels:
            payload[ml][tl] = [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "references": r.references,
                    "answer": r.answer,
                    "correct": r.correct,
                    "tokens": r.tokens,
                    "elapsed_s": r.elapsed_s,
                }
                for r in model_data.get(tl, [])
            ]
    # Write models not yet in all_results as empty dicts
    for ml in model_labels:
        if ml not in all_results:
            payload[ml] = {}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a previously saved checkpoint JSON, or return an empty dict.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Parsed JSON dict, or ``{}`` if the file does not exist or is invalid.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def model_is_complete(
    label: str,
    checkpoint: Dict[str, Any],
    task_labels: List[str],
    n: int,
) -> bool:
    """Return True if the checkpoint already has ≥ n results for every task.

    Args:
        label: Model label key in the checkpoint.
        checkpoint: Loaded checkpoint dict.
        task_labels: List of task names that must all be present.
        n: Minimum number of results required per task.

    Returns:
        True if the model can be skipped.
    """
    if n == 0:
        return True
    model_data = checkpoint.get(label)
    if not model_data:
        return False
    return all(len(model_data.get(t, [])) >= n for t in task_labels)


def task_is_complete(
    label: str,
    task: str,
    checkpoint: Dict[str, Any],
    n: int,
    redo_tasks: set | frozenset = frozenset(),
) -> bool:
    """Return True if *task* for *label* can be skipped (has ≥ n results and is not a redo task).

    Args:
        label: Model label key in the checkpoint.
        task: Task name to check.
        checkpoint: Loaded checkpoint dict.
        n: Minimum number of results required.
        redo_tasks: Set of task names that must be re-run regardless of checkpoint.

    Returns:
        True if the task result can be re-used from the checkpoint.
    """
    if n == 0:
        return True
    if task in redo_tasks:
        return False
    return len(checkpoint.get(label, {}).get(task, [])) >= n


def _print_table(
    all_results: Dict[str, Dict[str, List[InferenceResult]]],
    model_labels: List[str],
    task_labels: List[str],
) -> None:
    print(f"\n{_header('Results')}")
    col_w = 24
    header = f"{'Model':<22}" + "".join(f"{t.upper():>{col_w}}" for t in task_labels) + f"{'OVERALL':>{col_w}}"
    print(f"  {BOLD}{header}{RESET}")
    print(f"  {_bar('─', W - 2)}")

    for ml in model_labels:
        row_parts = [f"{ml:<22}"]
        all_correct = all_tokens = all_n = 0
        all_time = 0.0
        for tl in task_labels:
            res_list = all_results.get(ml, {}).get(tl, [])
            if not res_list:
                row_parts.append(f"{'N/A':>{col_w}}")
                continue
            n = len(res_list)
            acc = sum(r.correct for r in res_list) / n
            tok = sum(r.tokens for r in res_list)
            t = sum(r.elapsed_s for r in res_list)
            all_correct += sum(r.correct for r in res_list)
            all_tokens += tok; all_time += t; all_n += n
            cell = f"{acc:.0%} | {tok}tok | {t:.0f}s"
            row_parts.append(f"{cell:>{col_w}}")
        cell = f"{all_correct/all_n:.0%} | {all_tokens}tok | {all_time:.0f}s" if all_n else "N/A"
        row_parts.append(f"{cell:>{col_w}}")
        print("  " + "".join(row_parts))

    print(f"  {_bar('─', W - 2)}")
    print(f"\n  Column format: accuracy | total_tokens | total_wall_time_s\n")


def build_model_configs(
    *,
    include_viscot: bool,
    include_deepeyes: bool,
    load_in_8bit: bool,
) -> List[Dict[str, Any]]:
    """Build the ordered list of model configs for the comparison run."""
    model_configs: List[Dict[str, Any]] = []
    if include_viscot:
        model_configs.append({
            "label": "VisCoT (7B)",
            "type": "viscot",
            "model_path": "deepcs233/VisCoT-7b-336",
            "temperature": 0.0,
            "max_new_tokens": 512,
        })

    model_configs.append({
        "label": "Qwen2.5-VL (7B, no CoT)",
        "type": "direct_vlm",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "load_in_8bit": load_in_8bit,
        "temperature": 0.0,
        "max_new_tokens": 256,
    })

    if include_deepeyes:
        model_configs.append({
            "label": "DeepEyesV2-RL (7B)",
            "type": "deepeyes_v2",
            "model_id": "honglyhly/DeepEyesV2_7B_1031",
            "max_turns": 5,
            "load_in_8bit": load_in_8bit,
            "temperature": 0.2,
            "max_new_tokens": 512,
        })

    model_configs.append({
        "label": "GRIT (3B)",
        "type": "grit",
        "model_id": "yfan1997/GRIT-20-Qwen2.5-VL-3B",
        "load_in_8bit": load_in_8bit,
        "temperature": 0.0,
        "max_new_tokens": 512,
    })

    return model_configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VisCoT, Qwen2.5-VL, GRIT, and optional DeepEyesV2-RL on VQA/Counting/OCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=DEFAULT_QUESTION_COUNT,
                        help=f"Number of questions per task (max 100, default {DEFAULT_QUESTION_COUNT}).")
    parser.add_argument("--no-viscot", action="store_true",
                        help="Skip VisCoT (if llava not installed).")
    parser.add_argument("--no-deepeyes", action="store_true",
                        help="Skip DeepEyesV2-RL for faster comparison runs.")
    parser.add_argument("--no-8bit", action="store_true",
                        help="Disable 8-bit quantisation (needs >=40 GB VRAM).")
    parser.add_argument("--save-output", metavar="PATH", default=None)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing --save-output checkpoint, skipping completed models.",
    )
    parser.add_argument(
        "--redo-task", metavar="TASK", action="append", dest="redo_tasks", default=[],
        help="Force re-running TASK for every model even if checkpoint has results. "
             "Can be specified multiple times (e.g. --redo-task counting --redo-task vqa).",
    )
    args = parser.parse_args()

    print(_header("Model Comparison Experiment"))

    # ── 1. Load data ────────────────────────────────────────────────────────
    from src.data.datasets.viscot_benchmark import load_task

    n = min(args.n, 100)
    print(f"\n{CYAN}Loading {n} samples per task…{RESET}")
    all_samples: Dict[str, List[Dict[str, Any]]] = {
        "vqa":      load_task("vqa",      n),
        "counting": load_task("counting", n),
        "ocr":      load_task("ocr",      n),
    }
    task_labels = [t for t, s in all_samples.items() if s]

    for task, samples in all_samples.items():
        print(f"  {task.upper()}: {len(samples)} samples ready")

    if not task_labels:
        print(f"\n{RED}No samples loaded for any task — aborting.{RESET}")
        sys.exit(1)

    # ── 2. Define models ─────────────────────────────────────────────────────
    model_configs = build_model_configs(
        include_viscot=not args.no_viscot,
        include_deepeyes=not args.no_deepeyes,
        load_in_8bit=not args.no_8bit,
    )

    # ── 3. Run inference — one model at a time ───────────────────────────────
    import torch

    # Load existing checkpoint when resuming.
    checkpoint: Dict[str, Any] = {}
    if args.resume and args.save_output:
        checkpoint = load_checkpoint(Path(args.save_output))
        if checkpoint:
            print(f"\n{CYAN}Resuming from checkpoint: {args.save_output}{RESET}")
        else:
            print(f"\n{DIM}No checkpoint found at {args.save_output} — starting fresh.{RESET}")

    redo_tasks: set = set(args.redo_tasks)
    if redo_tasks:
        print(f"  {CYAN}Redo tasks: {sorted(redo_tasks)}{RESET}")

    all_results: Dict[str, Dict[str, List[InferenceResult]]] = {}
    model_labels: List[str] = []

    for mcfg in model_configs:
        label = mcfg["label"]
        model_labels.append(label)
        all_results[label] = {}

        # Determine which tasks still need inference for this model.
        tasks_needed = [
            t for t in task_labels
            if all_samples.get(t) and not task_is_complete(label, t, checkpoint, n, redo_tasks)
        ]
        tasks_from_cp = [
            t for t in task_labels
            if all_samples.get(t) and task_is_complete(label, t, checkpoint, n, redo_tasks)
        ]

        # Re-hydrate completed tasks from checkpoint.
        for tl in tasks_from_cp:
            all_results[label][tl] = [
                InferenceResult(
                    question_id=r["question_id"],
                    question=r["question"],
                    answer=r["answer"],
                    references=r["references"],
                    correct=r["correct"],
                    tokens=r["tokens"],
                    elapsed_s=r["elapsed_s"],
                )
                for r in checkpoint.get(label, {}).get(tl, [])
            ]

        if not tasks_needed:
            print(f"\n{DIM}Skipping {label} (all tasks complete in checkpoint).{RESET}")
            continue

        print(f"\n{_header(label)}")
        if tasks_from_cp:
            print(f"  {DIM}Re-using from checkpoint: {tasks_from_cp}{RESET}")
        print(f"  {BOLD}Running: {tasks_needed}{RESET}")

        mid = mcfg.get("model_id") or mcfg.get("model_path")
        print(f"  Loading {CYAN}{mid}{RESET} …")

        t_load = time.perf_counter()
        if mcfg["type"] == "viscot":
            from src.models.viscot import VisualCoTModel
            model_obj = VisualCoTModel(model_path=mcfg["model_path"])
            model_obj._load()
        elif mcfg["type"] == "direct_vlm":
            from src.models.direct_vlm import DirectVLMModel
            model_obj = DirectVLMModel(
                model_id=mcfg["model_id"],
                load_in_8bit=mcfg["load_in_8bit"],
            )
            model_obj._load()
        elif mcfg["type"] == "grit":
            from src.models.grit import GRITModel
            model_obj = GRITModel(
                model_id=mcfg["model_id"],
                load_in_8bit=mcfg["load_in_8bit"],
            )
            model_obj._load()
        else:
            from src.models.deepeyes_v2 import DeepEyesV2Model
            model_obj = DeepEyesV2Model(
                model_id=mcfg["model_id"],
                max_turns=mcfg["max_turns"],
                load_in_8bit=mcfg["load_in_8bit"],
            )
            model_obj._load()
        print(f"  {GREEN}Loaded in {time.perf_counter() - t_load:.1f}s{RESET}")

        for task in tasks_needed:
            samples = all_samples[task]
            print(f"\n  {BOLD}{task.upper()} ({len(samples)} samples){RESET}")
            all_results[label][task] = _run_model_on_samples(
                model_obj, samples,
                temperature=mcfg["temperature"],
                max_new_tokens=mcfg["max_new_tokens"],
                model_label=label,
                task=task,
            )

        print(f"\n  {DIM}Unloading {label}…{RESET}")
        del model_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.save_output:
            save_checkpoint(
                Path(args.save_output),
                all_results,
                model_labels,
                task_labels,
            )
            print(f"  {DIM}Checkpoint saved -> {args.save_output}{RESET}")

    # ── 4. Print comparison table ────────────────────────────────────────────
    _print_table(all_results, model_labels, task_labels)

    # ── 5. Save JSON ─────────────────────────────────────────────────────────
    if args.save_output:
        out_path = Path(args.save_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {}
        for ml in model_labels:
            payload[ml] = {}
            for tl in task_labels:
                payload[ml][tl] = [
                    {
                        "question_id": r.question_id,
                        "question": r.question,
                        "references": r.references,
                        "answer": r.answer,
                        "correct": r.correct,
                        "tokens": r.tokens,
                        "elapsed_s": r.elapsed_s,
                    }
                    for r in all_results.get(ml, {}).get(tl, [])
                ]
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"Output saved -> {out_path}")


if __name__ == "__main__":
    main()
