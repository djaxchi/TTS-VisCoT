"""Vote analysis for 5-candidate TTS runs with 3-vs-5 vote replay."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from src.eval.vqa_eval import evaluate_vqa


def get_model_tasks(payload: Dict[str, Any], model_label: str) -> Dict[str, List[Dict[str, Any]]]:
    """Return per-task entries for a model from a run JSON payload."""
    if model_label not in payload:
        raise KeyError(f"Model '{model_label}' not found. Available: {list(payload.keys())}")
    model_tasks = payload[model_label]
    return {k: v for k, v in model_tasks.items() if isinstance(v, list)}


def _compute_entry_flags(entry: Dict[str, Any]) -> Dict[str, Any]:
    refs = entry.get("references", [])
    voting = entry.get("voting", {})
    ans3 = voting.get("majority_3", {}).get("answer", "")
    ans5 = voting.get("majority_5", {}).get("answer", "")
    agree5 = float(voting.get("majority_5", {}).get("agreement_rate", 0.0) or 0.0)

    c3 = evaluate_vqa(ans3, refs)
    c5 = evaluate_vqa(ans5, refs)

    return {
        "c3": c3,
        "c5": c5,
        "changed": ans3 != ans5,
        "improved": (not c3) and c5,
        "worsened": c3 and (not c5),
        "agreement5": agree5,
    }


def _zero_shot_correct(entry: Dict[str, Any]) -> bool:
    """Return correctness of zero-shot candidate (candidate 1 answer)."""
    refs = entry.get("references", [])
    cands = entry.get("candidate_answers", []) or []
    zero = cands[0] if cands else ""
    return evaluate_vqa(zero, refs)


def _summarize_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(entries)
    if n == 0:
        return {
            "n": 0,
            "acc_m3": 0.0,
            "acc_m5": 0.0,
            "delta": 0.0,
            "changed": 0,
            "improved": 0,
            "worsened": 0,
            "mean_agreement_m5": 0.0,
        }

    flags = [_compute_entry_flags(e) for e in entries]
    acc3 = sum(int(f["c3"]) for f in flags) / n
    acc5 = sum(int(f["c5"]) for f in flags) / n
    return {
        "n": n,
        "acc_m3": acc3,
        "acc_m5": acc5,
        "delta": acc5 - acc3,
        "changed": sum(int(f["changed"]) for f in flags),
        "improved": sum(int(f["improved"]) for f in flags),
        "worsened": sum(int(f["worsened"]) for f in flags),
        "mean_agreement_m5": sum(float(f["agreement5"]) for f in flags) / n,
    }


def build_task_rows(payload: Dict[str, Any], model_label: str) -> List[Dict[str, Any]]:
    """Build summary rows for each task plus overall."""
    model_tasks = get_model_tasks(payload, model_label)
    rows: List[Dict[str, Any]] = []
    all_entries: List[Dict[str, Any]] = []

    for task in ("vqa", "counting", "ocr"):
        entries = model_tasks.get(task, [])
        all_entries.extend(entries)
        row = {"task": task, **_summarize_entries(entries)}
        rows.append(row)

    rows.append({"task": "overall", **_summarize_entries(all_entries)})
    return rows


def compute_transition_counts(payload: Dict[str, Any], model_label: str) -> Dict[str, Dict[str, int]]:
    """Count correctness transitions from majority_3 to majority_5."""
    model_tasks = get_model_tasks(payload, model_label)
    out: Dict[str, Dict[str, int]] = {}
    all_flags: List[Dict[str, Any]] = []

    for task in ("vqa", "counting", "ocr"):
        flags = [_compute_entry_flags(e) for e in model_tasks.get(task, [])]
        all_flags.extend(flags)
        out[task] = {
            "wrong_to_wrong": sum(int((not f["c3"]) and (not f["c5"])) for f in flags),
            "wrong_to_correct": sum(int((not f["c3"]) and f["c5"]) for f in flags),
            "correct_to_wrong": sum(int(f["c3"] and (not f["c5"])) for f in flags),
            "correct_to_correct": sum(int(f["c3"] and f["c5"]) for f in flags),
        }

    out["overall"] = {
        "wrong_to_wrong": sum(int((not f["c3"]) and (not f["c5"])) for f in all_flags),
        "wrong_to_correct": sum(int((not f["c3"]) and f["c5"]) for f in all_flags),
        "correct_to_wrong": sum(int(f["c3"] and (not f["c5"])) for f in all_flags),
        "correct_to_correct": sum(int(f["c3"] and f["c5"]) for f in all_flags),
    }
    return out


def build_agreement_bins(
    payload: Dict[str, Any],
    model_label: str,
    task: str = "overall",
) -> List[Dict[str, Any]]:
    """Return binned majority_5 agreement with per-bin accuracy."""
    model_tasks = get_model_tasks(payload, model_label)
    if task == "overall":
        entries = []
        for t in ("vqa", "counting", "ocr"):
            entries.extend(model_tasks.get(t, []))
    else:
        entries = model_tasks.get(task, [])

    bins = [
        (0.0, 0.6, "<=60%"),
        (0.6, 0.8, "60-80%"),
        (0.8, 1.0, "80-100%"),
    ]
    grouped: Dict[str, List[bool]] = defaultdict(list)

    for e in entries:
        flags = _compute_entry_flags(e)
        a = flags["agreement5"]
        for lo, hi, label in bins:
            right_closed = hi == 1.0
            if (lo < a <= hi) or (a == lo and lo == 0.0) or (right_closed and a == hi):
                grouped[label].append(bool(flags["c5"]))
                break

    rows: List[Dict[str, Any]] = []
    for _lo, _hi, label in bins:
        vals = grouped.get(label, [])
        n = len(vals)
        rows.append({
            "bin": label,
            "n": n,
            "acc_m5": (sum(int(v) for v in vals) / n) if n else 0.0,
        })
    return rows


def build_zero_shot_rows(payload: Dict[str, Any], model_label: str) -> List[Dict[str, Any]]:
    """Build per-task rows comparing zero-shot, majority_3, and majority_5."""
    model_tasks = get_model_tasks(payload, model_label)
    out: List[Dict[str, Any]] = []
    all_entries: List[Dict[str, Any]] = []

    for task in ("vqa", "counting", "ocr"):
        entries = model_tasks.get(task, [])
        all_entries.extend(entries)
        n = len(entries)
        if n == 0:
            out.append({
                "task": task,
                "n": 0,
                "acc_zero_shot": 0.0,
                "acc_m3": 0.0,
                "acc_m5": 0.0,
                "delta_m5_vs_zero": 0.0,
            })
            continue

        flags = [_compute_entry_flags(e) for e in entries]
        zs = [_zero_shot_correct(e) for e in entries]
        acc_zero = sum(int(v) for v in zs) / n
        acc_m3 = sum(int(f["c3"]) for f in flags) / n
        acc_m5 = sum(int(f["c5"]) for f in flags) / n
        out.append({
            "task": task,
            "n": n,
            "acc_zero_shot": acc_zero,
            "acc_m3": acc_m3,
            "acc_m5": acc_m5,
            "delta_m5_vs_zero": acc_m5 - acc_zero,
        })

    n_all = len(all_entries)
    if n_all == 0:
        out.append({
            "task": "overall",
            "n": 0,
            "acc_zero_shot": 0.0,
            "acc_m3": 0.0,
            "acc_m5": 0.0,
            "delta_m5_vs_zero": 0.0,
        })
        return out

    flags = [_compute_entry_flags(e) for e in all_entries]
    zs = [_zero_shot_correct(e) for e in all_entries]
    acc_zero = sum(int(v) for v in zs) / n_all
    acc_m3 = sum(int(f["c3"]) for f in flags) / n_all
    acc_m5 = sum(int(f["c5"]) for f in flags) / n_all
    out.append({
        "task": "overall",
        "n": n_all,
        "acc_zero_shot": acc_zero,
        "acc_m3": acc_m3,
        "acc_m5": acc_m5,
        "delta_m5_vs_zero": acc_m5 - acc_zero,
    })
    return out
