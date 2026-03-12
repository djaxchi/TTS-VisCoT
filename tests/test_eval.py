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
