"""Feasibility checks for token-level aggregation on current model backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def build_feasibility_report() -> List[Dict[str, str]]:
    """Return feasibility rows for the currently tested model backends."""
    rows: List[Dict[str, str]] = [
        {
            "model": "DirectVLMModel (Qwen2.5-VL)",
            "feasible": "yes",
            "token_scores_api": "HF forward logits (model(**inputs).logits) and generate(output_scores=True)",
            "step_by_step": "yes",
            "shared_sequence": "yes",
            "limitations": "Requires custom loop outside current predict(); no cached KV optimization in first prototype.",
        },
        {
            "model": "GRITModel (Qwen2.5-VL backbone)",
            "feasible": "yes",
            "token_scores_api": "Same HF Qwen2.5-VL logits access as DirectVLMModel.",
            "step_by_step": "yes",
            "shared_sequence": "yes",
            "limitations": "Current GRIT wrapper post-processes tags; token-level prototype should target short A/B/C/D answer generation first.",
        },
        {
            "model": "VisualCoTModel (LLaVA / VisCoT wrapper)",
            "feasible": "unclear",
            "token_scores_api": "Underlying model likely has logits, but wrapper uses llava helpers and 2-turn crop pipeline.",
            "step_by_step": "unclear",
            "shared_sequence": "unclear",
            "limitations": "Two-turn bbox->crop->answer flow complicates synchronized token loop; backend-specific refactor needed.",
        },
        {
            "model": "DeepEyesV2Model (agentic tool-calling)",
            "feasible": "no",
            "token_scores_api": "Agentic loop expects tool-calls and turn-level parsing rather than pure next-token sync.",
            "step_by_step": "n/a",
            "shared_sequence": "n/a",
            "limitations": "Not in current comparison target and incompatible with first token-level MVP.",
        },
    ]
    return rows


def format_markdown(rows: List[Dict[str, str]]) -> str:
    header = (
        "# Token-Level Aggregation Feasibility Report\n\n"
        "| Model / Backend | Feasible | Next-token scores API | Step-by-step | Shared sequence | Limitations |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {r['model']} | {r['feasible']} | {r['token_scores_api']} | {r['step_by_step']} | {r['shared_sequence']} | {r['limitations']} |"
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"


def write_report(path: str | Path) -> Path:
    rows = build_feasibility_report()
    text = format_markdown(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


if __name__ == "__main__":
    output = write_report("reports/token_level_feasibility.md")
    print(f"Saved feasibility report -> {output}")
