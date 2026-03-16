"""Tests for src.eval.metrics."""

import pytest

from src.eval.metrics import (
    compute_accuracy,
    compute_bbox_metrics,
    compute_robustness_metrics,
    print_metrics_summary,
)
from src.voting.bbox_consensus import BoundingBoxPrediction


# ---------------------------------------------------------------------------
# TestComputeMetrics (accuracy)
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_accuracy_empty_input_returns_zero(self):
        result = compute_accuracy([], [])
        assert result.accuracy == 0.0
        assert result.correct == 0
        assert result.total == 0

    def test_accuracy_all_correct(self):
        result = compute_accuracy(["A", "B", "C"], ["A", "B", "C"])
        assert result.accuracy == pytest.approx(1.0)
        assert result.correct == 3

    def test_accuracy_all_wrong(self):
        result = compute_accuracy(["A", "A", "A"], ["B", "B", "B"])
        assert result.accuracy == pytest.approx(0.0)
        assert result.correct == 0

    def test_accuracy_partial(self):
        result = compute_accuracy(["A", "B", "C", "A", "D"], ["A", "B", "B", "A", "D"])
        assert result.correct == 4
        assert result.accuracy == pytest.approx(4 / 5)

    def test_accuracy_normalizes_answers(self):
        result = compute_accuracy(["(A)", "option B"], ["A", "B"], normalize=True)
        assert result.accuracy == pytest.approx(1.0)

    def test_accuracy_per_class_keys_match_ground_truth(self):
        result = compute_accuracy(["A", "B", "A"], ["A", "B", "B"])
        assert set(result.per_class_accuracy.keys()) == {"A", "B"}

    def test_accuracy_raises_on_length_mismatch(self):
        with pytest.raises(ValueError):
            compute_accuracy(["A", "B"], ["A"])

    @pytest.mark.parametrize("n_correct,n_total", [(0, 5), (3, 5), (5, 5)])
    def test_accuracy_fraction_parametrized(self, n_correct, n_total):
        preds = ["A"] * n_correct + ["B"] * (n_total - n_correct)
        gts = ["A"] * n_total
        result = compute_accuracy(preds, gts)
        assert result.accuracy == pytest.approx(n_correct / n_total)


# ---------------------------------------------------------------------------
# TestRobustnessMetrics
# ---------------------------------------------------------------------------


class TestRobustnessMetrics:
    def test_robustness_improvement_positive_when_multi_better(self):
        sv = ["A", "B", "A"]
        mv = ["A", "A", "A"]
        gt = ["A", "A", "A"]
        result = compute_robustness_metrics(sv, mv, gt)
        assert result.improvement > 0

    def test_robustness_flip_rate_zero_when_no_changes(self):
        preds = ["A", "B", "C"]
        result = compute_robustness_metrics(preds, preds, preds)
        assert result.flip_rate == 0.0

    def test_robustness_flip_rate_correct_fraction(self):
        sv = ["A", "A", "A", "A"]
        mv = ["A", "B", "A", "B"]
        gt = ["A", "A", "A", "A"]
        result = compute_robustness_metrics(sv, mv, gt)
        assert result.flip_rate == pytest.approx(0.5)

    def test_robustness_raises_on_length_mismatch(self):
        with pytest.raises(ValueError):
            compute_robustness_metrics(["A"], ["A", "B"], ["A"])

    def test_robustness_agreement_rate_from_input(self):
        preds = ["A", "B", "C"]
        result = compute_robustness_metrics(preds, preds, preds, agreement_rates=[0.5, 1.0, 0.75])
        assert result.agreement_rate == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# TestBBoxMetrics
# ---------------------------------------------------------------------------


class TestBBoxMetrics:
    _box_a = BoundingBoxPrediction(x=10, y=10, width=50, height=50)
    _box_b = BoundingBoxPrediction(x=12, y=12, width=48, height=48)  # overlaps _box_a
    _box_c = BoundingBoxPrediction(x=200, y=200, width=40, height=40)  # no overlap

    def test_bbox_metrics_no_gt_returns_zero_recall(self):
        result = compute_bbox_metrics([self._box_a], [])
        assert result.recall == 0.0
        assert result.ground_truth_boxes == 0

    def test_bbox_metrics_no_pred_returns_zero_precision(self):
        result = compute_bbox_metrics([], [self._box_a])
        assert result.precision == 0.0
        assert result.predicted_boxes == 0

    def test_bbox_metrics_perfect_match_yields_f1_one(self):
        result = compute_bbox_metrics([self._box_a], [self._box_b])
        assert result.f1_score == pytest.approx(1.0)
        assert result.matched_boxes == 1

    def test_bbox_metrics_no_match_yields_f1_zero(self):
        result = compute_bbox_metrics([self._box_c], [self._box_a])
        assert result.f1_score == pytest.approx(0.0)
        assert result.matched_boxes == 0

    def test_bbox_metrics_counts_are_consistent(self):
        preds = [self._box_a, self._box_c]
        gts = [self._box_b]
        result = compute_bbox_metrics(preds, gts)
        assert result.predicted_boxes == 2
        assert result.ground_truth_boxes == 1
        assert result.matched_boxes <= 1


# ---------------------------------------------------------------------------
# TestPrintMetricsSummary (smoke test — just ensure no crash)
# ---------------------------------------------------------------------------


class TestPrintMetricsSummary:
    def test_print_with_all_none_does_not_raise(self, capsys):
        print_metrics_summary()
        out = capsys.readouterr().out
        assert "METRICS SUMMARY" in out

    def test_print_with_accuracy_metrics(self, capsys):
        from src.eval.metrics import AccuracyMetrics

        acc = AccuracyMetrics(accuracy=0.8, correct=4, total=5)
        print_metrics_summary(accuracy_metrics=acc)
        out = capsys.readouterr().out
        assert "0.8000" in out


# ---------------------------------------------------------------------------
# TestVqaNormalize
# ---------------------------------------------------------------------------


class TestVqaNormalize:
    def test_vqa_normalize_lowercases(self):
        from src.eval.vqa_eval import vqa_normalize

        assert vqa_normalize("BANANA") == "banana"

    def test_vqa_normalize_strips_leading_trailing_whitespace(self):
        from src.eval.vqa_eval import vqa_normalize

        assert vqa_normalize("  cat  ") == "cat"

    def test_vqa_normalize_removes_articles_a(self):
        from src.eval.vqa_eval import vqa_normalize

        assert vqa_normalize("a cat") == "cat"

    def test_vqa_normalize_removes_articles_an(self):
        from src.eval.vqa_eval import vqa_normalize

        assert vqa_normalize("an apple") == "apple"

    def test_vqa_normalize_removes_articles_the(self):
        from src.eval.vqa_eval import vqa_normalize

        assert vqa_normalize("the dog") == "dog"

    def test_vqa_normalize_removes_punctuation(self):
        from src.eval.vqa_eval import vqa_normalize

        assert vqa_normalize("yes.") == "yes"
        assert vqa_normalize("no!") == "no"
        assert vqa_normalize("3,000") == "3000"

    def test_vqa_normalize_collapses_whitespace(self):
        from src.eval.vqa_eval import vqa_normalize

        assert vqa_normalize("two  cats") == "two cats"

    def test_vqa_normalize_empty_string_returns_empty(self):
        from src.eval.vqa_eval import vqa_normalize

        assert vqa_normalize("") == ""

    def test_vqa_normalize_does_not_remove_mid_word_article(self):
        from src.eval.vqa_eval import vqa_normalize

        # "theater" should not become "ter"
        result = vqa_normalize("theater")
        assert result == "theater"

    @pytest.mark.parametrize("raw,expected", [
        ("A cat", "cat"),
        ("The big dog", "big dog"),
        ("An orange", "orange"),
        ("yes", "yes"),
        ("2", "2"),
    ])
    def test_vqa_normalize_parametrized(self, raw, expected):
        from src.eval.vqa_eval import vqa_normalize

        assert vqa_normalize(raw) == expected


# ---------------------------------------------------------------------------
# TestEvaluateVqa
# ---------------------------------------------------------------------------


class TestEvaluateVqa:
    def test_evaluate_vqa_exact_match_returns_true(self):
        from src.eval.vqa_eval import evaluate_vqa

        assert evaluate_vqa("cat", ["cat"]) is True

    def test_evaluate_vqa_case_insensitive_match(self):
        from src.eval.vqa_eval import evaluate_vqa

        assert evaluate_vqa("CAT", ["cat"]) is True

    def test_evaluate_vqa_article_stripped_match(self):
        from src.eval.vqa_eval import evaluate_vqa

        assert evaluate_vqa("a cat", ["cat"]) is True

    def test_evaluate_vqa_no_match_returns_false(self):
        from src.eval.vqa_eval import evaluate_vqa

        assert evaluate_vqa("dog", ["cat"]) is False

    def test_evaluate_vqa_matches_any_reference(self):
        from src.eval.vqa_eval import evaluate_vqa

        assert evaluate_vqa("yes", ["no", "yes", "true"]) is True

    def test_evaluate_vqa_empty_references_returns_false(self):
        from src.eval.vqa_eval import evaluate_vqa

        assert evaluate_vqa("cat", []) is False

    def test_evaluate_vqa_empty_prediction_returns_false(self):
        from src.eval.vqa_eval import evaluate_vqa

        assert evaluate_vqa("", ["cat"]) is False

    def test_evaluate_vqa_punctuation_stripped_match(self):
        from src.eval.vqa_eval import evaluate_vqa

        assert evaluate_vqa("yes.", ["yes"]) is True

    def test_evaluate_vqa_numeric_string_match(self):
        from src.eval.vqa_eval import evaluate_vqa

        assert evaluate_vqa("3", ["3", "three"]) is True

    def test_evaluate_vqa_count_tokens_match(self):
        from src.eval.vqa_eval import evaluate_vqa

        # VQA counting: "two" vs reference list containing "2" — no match expected
        # (we only normalize surface form, not number words)
        assert evaluate_vqa("two", ["2"]) is False
