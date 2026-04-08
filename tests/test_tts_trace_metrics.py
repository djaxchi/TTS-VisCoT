"""Tests for src.eval.tts_trace_metrics (Qwen TTS candidate-trace analytics)."""

from __future__ import annotations

from src.eval.tts_trace_metrics import (
    build_task_rows,
    build_zero_shot_rows,
    compute_transition_counts,
    get_model_tasks,
)


def _payload() -> dict:
    return {
        "Qwen2.5-VL (7B, no CoT)": {
            "vqa": [
                {
                    "references": ["cat"],
                    "candidate_answers": ["dog", "dog", "cat", "cat", "cat"],
                    "voting": {
                        "majority_3": {"answer": "dog", "agreement_rate": 2 / 3},
                        "majority_5": {"answer": "cat", "agreement_rate": 3 / 5},
                    },
                },
                {
                    "references": ["blue"],
                    "candidate_answers": ["blue", "blue", "blue", "blue", "blue"],
                    "voting": {
                        "majority_3": {"answer": "blue", "agreement_rate": 1.0},
                        "majority_5": {"answer": "blue", "agreement_rate": 1.0},
                    },
                },
                {
                    "references": ["yes"],
                    "candidate_answers": ["no", "no", "yes", "no", "no"],
                    "voting": {
                        "majority_3": {"answer": "no", "agreement_rate": 2 / 3},
                        "majority_5": {"answer": "no", "agreement_rate": 3 / 5},
                    },
                },
            ],
            "counting": [
                {
                    "references": ["2"],
                    "candidate_answers": ["2", "2", "2", "2", "2"],
                    "voting": {
                        "majority_3": {"answer": "2", "agreement_rate": 1.0},
                        "majority_5": {"answer": "2", "agreement_rate": 1.0},
                    },
                },
                {
                    "references": ["8"],
                    "candidate_answers": ["8", "8", "7", "7", "7"],
                    "voting": {
                        "majority_3": {"answer": "8", "agreement_rate": 2 / 3},
                        "majority_5": {"answer": "7", "agreement_rate": 3 / 5},
                    },
                },
            ],
            "ocr": [
                {
                    "references": ["bud light"],
                    "candidate_answers": ["bud light", "bud light", "bud light", "bud light", "bud light"],
                    "voting": {
                        "majority_3": {"answer": "bud light", "agreement_rate": 1.0},
                        "majority_5": {"answer": "bud light", "agreement_rate": 1.0},
                    },
                }
            ],
        }
    }


class TestGetModelTasks:
    def test_returns_model_tasks_dict(self) -> None:
        data = _payload()
        model_tasks = get_model_tasks(data, "Qwen2.5-VL (7B, no CoT)")
        assert set(model_tasks.keys()) == {"vqa", "counting", "ocr"}

    def test_missing_model_raises_key_error(self) -> None:
        data = _payload()
        try:
            get_model_tasks(data, "missing")
            assert False, "expected KeyError"
        except KeyError:
            assert True


class TestBuildTaskRows:
    def test_build_task_rows_contains_tasks_and_overall(self) -> None:
        rows = build_task_rows(_payload(), "Qwen2.5-VL (7B, no CoT)")
        labels = [r["task"] for r in rows]
        assert labels == ["vqa", "counting", "ocr", "overall"]

    def test_vqa_metrics_expected_values(self) -> None:
        rows = build_task_rows(_payload(), "Qwen2.5-VL (7B, no CoT)")
        vqa = next(r for r in rows if r["task"] == "vqa")
        assert vqa["n"] == 3
        assert vqa["acc_m3"] == 1 / 3
        assert vqa["acc_m5"] == 2 / 3
        assert vqa["changed"] == 1
        assert vqa["improved"] == 1
        assert vqa["worsened"] == 0

    def test_counting_metrics_expected_values(self) -> None:
        rows = build_task_rows(_payload(), "Qwen2.5-VL (7B, no CoT)")
        counting = next(r for r in rows if r["task"] == "counting")
        assert counting["n"] == 2
        assert counting["acc_m3"] == 1.0
        assert counting["acc_m5"] == 0.5
        assert counting["changed"] == 1
        assert counting["improved"] == 0
        assert counting["worsened"] == 1


class TestComputeTransitionCounts:
    def test_transition_counts_are_correct(self) -> None:
        rows = compute_transition_counts(_payload(), "Qwen2.5-VL (7B, no CoT)")
        vqa = rows["vqa"]
        # vqa rows are: wrong->correct, correct->correct, wrong->wrong
        assert vqa["wrong_to_correct"] == 1
        assert vqa["correct_to_correct"] == 1
        assert vqa["wrong_to_wrong"] == 1
        assert vqa["correct_to_wrong"] == 0

    def test_overall_transition_counts_present(self) -> None:
        rows = compute_transition_counts(_payload(), "Qwen2.5-VL (7B, no CoT)")
        overall = rows["overall"]
        assert sum(overall.values()) == 6


class TestBuildZeroShotRows:
    def test_returns_tasks_and_overall(self) -> None:
        rows = build_zero_shot_rows(_payload(), "Qwen2.5-VL (7B, no CoT)")
        assert [r["task"] for r in rows] == ["vqa", "counting", "ocr", "overall"]

    def test_vqa_zero_shot_vs_majority_values(self) -> None:
        rows = build_zero_shot_rows(_payload(), "Qwen2.5-VL (7B, no CoT)")
        vqa = next(r for r in rows if r["task"] == "vqa")
        # zero-shot from candidate 1 answers: dog, blue, no => 1/3
        assert vqa["acc_zero_shot"] == 1 / 3
        assert vqa["acc_m3"] == 1 / 3
        assert vqa["acc_m5"] == 2 / 3
        assert vqa["delta_m5_vs_zero"] == 1 / 3
