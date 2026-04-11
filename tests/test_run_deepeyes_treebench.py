"""Tests for checkpoint/resume and accuracy logic in run_deepeyes_treebench.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.run_deepeyes_treebench import (
    compute_accuracy,
    load_predictions,
    save_prediction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(
    question_id: str = "example_0",
    correct_answer: str = "A",
    predicted_letter: str | None = "A",
    correct: bool = True,
) -> dict:
    return {
        "question_id": question_id,
        "question": "Which option is correct?",
        "correct_answer": correct_answer,
        "raw_answer": "The answer is A",
        "predicted_letter": predicted_letter,
        "correct": correct,
        "elapsed_s": 1.23,
        "cot_steps": 1,
        "tool_results": 0,
    }


# ---------------------------------------------------------------------------
# TestSavePrediction
# ---------------------------------------------------------------------------


class TestSavePrediction:
    def test_save_prediction_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "preds.jsonl"
        save_prediction(out, _make_row())
        assert out.exists()

    def test_save_prediction_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "preds.jsonl"
        save_prediction(out, _make_row())
        assert out.exists()

    def test_save_prediction_appends_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "preds.jsonl"
        save_prediction(out, _make_row(question_id="q1"))
        save_prediction(out, _make_row(question_id="q2"))
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["question_id"] == "q1"
        assert json.loads(lines[1])["question_id"] == "q2"

    def test_save_prediction_row_has_required_keys(self, tmp_path: Path) -> None:
        out = tmp_path / "preds.jsonl"
        row = _make_row()
        save_prediction(out, row)
        saved = json.loads(out.read_text(encoding="utf-8").strip())
        for key in ("question_id", "correct_answer", "predicted_letter", "correct"):
            assert key in saved


# ---------------------------------------------------------------------------
# TestLoadPredictions
# ---------------------------------------------------------------------------


class TestLoadPredictions:
    def test_load_predictions_returns_empty_when_file_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "nonexistent.jsonl"
        assert load_predictions(out) == []

    def test_load_predictions_returns_all_rows(self, tmp_path: Path) -> None:
        out = tmp_path / "preds.jsonl"
        for i in range(3):
            save_prediction(out, _make_row(question_id=f"example_{i}"))
        rows = load_predictions(out)
        assert len(rows) == 3

    def test_load_predictions_skips_blank_lines(self, tmp_path: Path) -> None:
        out = tmp_path / "preds.jsonl"
        out.write_text(
            json.dumps(_make_row(question_id="q0")) + "\n\n"
            + json.dumps(_make_row(question_id="q1")) + "\n",
            encoding="utf-8",
        )
        rows = load_predictions(out)
        assert len(rows) == 2

    def test_load_predictions_question_ids_preserved(self, tmp_path: Path) -> None:
        out = tmp_path / "preds.jsonl"
        save_prediction(out, _make_row(question_id="example_7"))
        rows = load_predictions(out)
        assert rows[0]["question_id"] == "example_7"


# ---------------------------------------------------------------------------
# TestComputeAccuracy
# ---------------------------------------------------------------------------


class TestComputeAccuracy:
    def test_compute_accuracy_all_correct(self) -> None:
        rows = [_make_row(correct=True) for _ in range(5)]
        assert compute_accuracy(rows) == pytest.approx(1.0)

    def test_compute_accuracy_none_correct(self) -> None:
        rows = [_make_row(correct=False) for _ in range(4)]
        assert compute_accuracy(rows) == pytest.approx(0.0)

    def test_compute_accuracy_partial(self) -> None:
        rows = [_make_row(correct=(i % 2 == 0)) for i in range(10)]
        assert compute_accuracy(rows) == pytest.approx(0.5)

    def test_compute_accuracy_empty_returns_zero(self) -> None:
        assert compute_accuracy([]) == pytest.approx(0.0)

    def test_compute_accuracy_single_correct(self) -> None:
        assert compute_accuracy([_make_row(correct=True)]) == pytest.approx(1.0)

    def test_compute_accuracy_single_wrong(self) -> None:
        assert compute_accuracy([_make_row(correct=False)]) == pytest.approx(0.0)
