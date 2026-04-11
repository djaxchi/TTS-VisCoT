"""Tests for src/eval/stochasticity.py — entropy computation utilities."""

import math
import pytest

from src.eval.stochasticity import compute_entropy, entropy_summary


class TestComputeEntropy:
    def test_all_same_answer_returns_zero(self):
        answers = ["A", "A", "A", "A", "A"]
        assert compute_entropy(answers) == pytest.approx(0.0)

    def test_two_equally_split_answers_returns_one_bit(self):
        answers = ["A", "B", "A", "B", "A", "B", "A", "B"]
        assert compute_entropy(answers) == pytest.approx(1.0, abs=1e-6)

    def test_all_different_answers_returns_log2_n(self):
        answers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        expected = math.log2(10)
        assert compute_entropy(answers) == pytest.approx(expected, abs=1e-6)

    def test_single_answer_returns_zero(self):
        assert compute_entropy(["A"]) == pytest.approx(0.0)

    def test_empty_list_returns_zero(self):
        assert compute_entropy([]) == pytest.approx(0.0)

    def test_none_answers_are_ignored(self):
        # None = model failed to produce an answer; treat as absent, not a label
        answers = ["A", None, "A", None, "A"]
        assert compute_entropy(answers) == pytest.approx(0.0)

    def test_all_none_returns_zero(self):
        assert compute_entropy([None, None, None]) == pytest.approx(0.0)

    def test_four_way_split_correct(self):
        answers = ["A", "B", "C", "D"] * 2  # 2 each, uniform over 4 labels
        expected = math.log2(4)  # 2.0 bits
        assert compute_entropy(answers) == pytest.approx(expected, abs=1e-6)

    def test_skewed_distribution(self):
        # 8 A's, 2 B's → H = -(0.8*log2(0.8) + 0.2*log2(0.2))
        answers = ["A"] * 8 + ["B"] * 2
        expected = -(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2))
        assert compute_entropy(answers) == pytest.approx(expected, abs=1e-6)

    def test_case_insensitive(self):
        # lowercase 'a' and uppercase 'A' should count as the same label
        answers = ["a", "A", "a", "A"]
        assert compute_entropy(answers) == pytest.approx(0.0)


class TestEntropySummary:
    def test_returns_mean_per_task(self):
        rows = [
            {"task": "vqa", "entropy": 1.0},
            {"task": "vqa", "entropy": 3.0},
            {"task": "ocr", "entropy": 2.0},
        ]
        summary = entropy_summary(rows)
        assert summary["vqa"] == pytest.approx(2.0)
        assert summary["ocr"] == pytest.approx(2.0)

    def test_empty_rows_returns_empty_dict(self):
        assert entropy_summary([]) == {}

    def test_single_row(self):
        rows = [{"task": "counting", "entropy": 1.5}]
        assert entropy_summary(rows) == {"counting": pytest.approx(1.5)}
