"""Evaluation metrics: accuracy, robustness, and bounding-box quality."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils_normalize import normalize_answer as _normalize_raw


def _norm(s: str) -> str:
    """Normalize an answer string, falling back to empty string on no-match."""
    return _normalize_raw(s) or ""


# ---------------------------------------------------------------------------
# Bounding-box helpers (inlined — not exported as a separate module)
# ---------------------------------------------------------------------------


@dataclass
class BoundingBoxPrediction:
    """An axis-aligned bounding box prediction.

    Attributes:
        x: Left edge (pixels or normalised [0, 1]).
        y: Top edge (pixels or normalised [0, 1]).
        width: Box width in the same unit as ``x``.
        height: Box height in the same unit as ``y``.
        confidence: Prediction confidence score (default 1.0).
        label: Optional class label.
    """

    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    label: Optional[str] = None


def compute_iou(box1: BoundingBoxPrediction, box2: BoundingBoxPrediction) -> float:
    """Compute the Intersection-over-Union between two bounding boxes."""
    box1_x2 = box1.x + box1.width
    box1_y2 = box1.y + box1.height
    box2_x2 = box2.x + box2.width
    box2_y2 = box2.y + box2.height

    inter_x1 = max(box1.x, box2.x)
    inter_y1 = max(box1.y, box2.y)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = box1.width * box1.height + box2.width * box2.height - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def match_boxes(
    pred_boxes: List[BoundingBoxPrediction],
    gt_boxes: List[BoundingBoxPrediction],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedily match predicted boxes to ground-truth boxes by IoU."""
    if not pred_boxes or not gt_boxes:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(pb, gb)

    matches: List[Tuple[int, int]] = []
    matched_preds: set[int] = set()
    matched_gts: set[int] = set()

    while True:
        max_iou = iou_matrix.max()
        if max_iou < iou_threshold:
            break
        pred_idx, gt_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matches.append((int(pred_idx), int(gt_idx)))
        matched_preds.add(int(pred_idx))
        matched_gts.add(int(gt_idx))
        iou_matrix[pred_idx, :] = 0
        iou_matrix[:, gt_idx] = 0

    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]
    return matches, unmatched_preds, unmatched_gts


# ---------------------------------------------------------------------------
# Metric dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AccuracyMetrics:
    """Overall and per-class accuracy.

    Attributes:
        accuracy: Fraction of correct predictions.
        correct: Number of correct predictions.
        total: Total number of predictions.
        per_class_accuracy: Per-answer-letter accuracy dict.
    """

    accuracy: float
    correct: int
    total: int
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)


@dataclass
class RobustnessMetrics:
    """Comparison of single-view vs. multi-view (voting) accuracy.

    Attributes:
        single_view_accuracy: Accuracy without augmentation.
        multi_view_accuracy: Accuracy after majority voting.
        improvement: ``multi_view_accuracy − single_view_accuracy``.
        flip_rate: Fraction of examples where voting changed the answer.
        flip_correct_rate: Of the flipped answers, fraction that became correct.
        agreement_rate: Average voting agreement rate across examples.
    """

    single_view_accuracy: float
    multi_view_accuracy: float
    improvement: float
    flip_rate: float
    flip_correct_rate: float
    agreement_rate: float


@dataclass
class BBoxMetrics:
    """Bounding-box detection quality metrics.

    Attributes:
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
        f1_score: Harmonic mean of precision and recall.
        average_iou: Mean IoU over matched pairs.
        matched_boxes: Number of true-positive matches.
        predicted_boxes: Total predicted boxes.
        ground_truth_boxes: Total ground-truth boxes.
    """

    precision: float
    recall: float
    f1_score: float
    average_iou: float
    matched_boxes: int
    predicted_boxes: int
    ground_truth_boxes: int


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def compute_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    normalize: bool = True,
) -> AccuracyMetrics:
    """Compute overall and per-class accuracy.

    Args:
        predictions: Model predictions (raw or pre-normalised).
        ground_truths: Gold-standard answers.
        normalize: Whether to normalise both lists before comparison.

    Returns:
        :class:`AccuracyMetrics`.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length.")

    if not predictions:
        return AccuracyMetrics(accuracy=0.0, correct=0, total=0)

    if normalize:
        predictions = [_norm(p) for p in predictions]
        ground_truths = [_norm(g) for g in ground_truths]

    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    total = len(predictions)
    accuracy = correct / total

    per_class: Dict[str, float] = {}
    for answer in set(ground_truths):
        mask = [g == answer for g in ground_truths]
        class_correct = sum(
            p == g for p, g, m in zip(predictions, ground_truths, mask) if m
        )
        class_total = sum(mask)
        per_class[answer] = class_correct / class_total if class_total > 0 else 0.0

    return AccuracyMetrics(
        accuracy=accuracy,
        correct=correct,
        total=total,
        per_class_accuracy=per_class,
    )


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def compute_robustness_metrics(
    single_view_predictions: List[str],
    multi_view_predictions: List[str],
    ground_truths: List[str],
    agreement_rates: Optional[List[float]] = None,
) -> RobustnessMetrics:
    """Compare single-view vs. multi-view (voting) performance.

    Args:
        single_view_predictions: Predictions without augmentation.
        multi_view_predictions: Predictions after majority voting.
        ground_truths: Gold-standard answers.
        agreement_rates: Per-example voting agreement rates (optional).

    Returns:
        :class:`RobustnessMetrics`.

    Raises:
        ValueError: If the three prediction lists differ in length.
    """
    n = len(ground_truths)
    if len(single_view_predictions) != n or len(multi_view_predictions) != n:
        raise ValueError("All three lists must have the same length.")

    sv = [_norm(p) for p in single_view_predictions]
    mv = [_norm(p) for p in multi_view_predictions]
    gt = [_norm(g) for g in ground_truths]

    sv_acc = sum(p == g for p, g in zip(sv, gt)) / n
    mv_acc = sum(p == g for p, g in zip(mv, gt)) / n
    improvement = mv_acc - sv_acc

    flips = [s != m for s, m in zip(sv, mv)]
    flip_rate = sum(flips) / n

    flip_correct = sum(
        m == g
        for s, m, g, flipped in zip(sv, mv, gt, flips)
        if flipped
    )
    flip_count = sum(flips)
    flip_correct_rate = flip_correct / flip_count if flip_count > 0 else 0.0

    avg_agreement = float(np.mean(agreement_rates)) if agreement_rates else 0.0

    return RobustnessMetrics(
        single_view_accuracy=sv_acc,
        multi_view_accuracy=mv_acc,
        improvement=improvement,
        flip_rate=flip_rate,
        flip_correct_rate=flip_correct_rate,
        agreement_rate=avg_agreement,
    )


# ---------------------------------------------------------------------------
# Bounding-box metrics
# ---------------------------------------------------------------------------


def compute_bbox_metrics(
    pred_boxes: List[BoundingBoxPrediction],
    gt_boxes: List[BoundingBoxPrediction],
    iou_threshold: float = 0.5,
) -> BBoxMetrics:
    """Compute precision, recall, F1, and average IoU for bounding boxes.

    Args:
        pred_boxes: Predicted bounding boxes.
        gt_boxes: Ground-truth bounding boxes.
        iou_threshold: Minimum IoU to count a prediction as a true positive.

    Returns:
        :class:`BBoxMetrics`.
    """
    if not gt_boxes:
        return BBoxMetrics(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            average_iou=0.0,
            matched_boxes=0,
            predicted_boxes=len(pred_boxes),
            ground_truth_boxes=0,
        )
    if not pred_boxes:
        return BBoxMetrics(
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            average_iou=0.0,
            matched_boxes=0,
            predicted_boxes=0,
            ground_truth_boxes=len(gt_boxes),
        )

    matches, _, _ = match_boxes(pred_boxes, gt_boxes, iou_threshold)
    n_matches = len(matches)
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    precision = n_matches / n_pred
    recall = n_matches / n_gt
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    avg_iou = (
        float(
            np.mean(
                [compute_iou(pred_boxes[pi], gt_boxes[gi]) for pi, gi in matches]
            )
        )
        if matches
        else 0.0
    )

    return BBoxMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        average_iou=avg_iou,
        matched_boxes=n_matches,
        predicted_boxes=n_pred,
        ground_truth_boxes=n_gt,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def create_summary_table(results: List[Dict], metrics_type: str = "accuracy") -> pd.DataFrame:
    """Wrap a list of result dicts into a :class:`pandas.DataFrame`.

    Args:
        results: List of result dictionaries (e.g. per-run metric dicts).
        metrics_type: Label for the type of metrics (informational only).

    Returns:
        A :class:`pandas.DataFrame`, or an empty one if *results* is empty.
    """
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def print_metrics_summary(
    accuracy_metrics: Optional[AccuracyMetrics] = None,
    robustness_metrics: Optional[RobustnessMetrics] = None,
    bbox_metrics: Optional[BBoxMetrics] = None,
) -> None:
    """Pretty-print a metrics summary to stdout.

    Args:
        accuracy_metrics: Optional accuracy metrics to display.
        robustness_metrics: Optional robustness metrics to display.
        bbox_metrics: Optional bounding-box metrics to display.
    """
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)

    if accuracy_metrics:
        print("\nAccuracy Metrics:")
        print(f"  Overall Accuracy : {accuracy_metrics.accuracy:.4f}")
        print(f"  Correct          : {accuracy_metrics.correct}/{accuracy_metrics.total}")
        if accuracy_metrics.per_class_accuracy:
            print("  Per-class Accuracy:")
            for ans, acc in sorted(accuracy_metrics.per_class_accuracy.items()):
                print(f"    {ans}: {acc:.4f}")

    if robustness_metrics:
        print("\nRobustness Metrics:")
        print(f"  Single-view Accuracy : {robustness_metrics.single_view_accuracy:.4f}")
        print(f"  Multi-view Accuracy  : {robustness_metrics.multi_view_accuracy:.4f}")
        print(f"  Improvement          : {robustness_metrics.improvement:+.4f}")
        print(f"  Flip Rate            : {robustness_metrics.flip_rate:.4f}")
        print(f"  Flip Correct Rate    : {robustness_metrics.flip_correct_rate:.4f}")
        print(f"  Agreement Rate       : {robustness_metrics.agreement_rate:.4f}")

    if bbox_metrics:
        print("\nBounding Box Metrics:")
        print(f"  Precision   : {bbox_metrics.precision:.4f}")
        print(f"  Recall      : {bbox_metrics.recall:.4f}")
        print(f"  F1-Score    : {bbox_metrics.f1_score:.4f}")
        print(f"  Average IoU : {bbox_metrics.average_iou:.4f}")
        print(f"  Matched     : {bbox_metrics.matched_boxes}")
        print(f"  Predicted   : {bbox_metrics.predicted_boxes}")
        print(f"  Ground Truth: {bbox_metrics.ground_truth_boxes}")

    print("=" * 80 + "\n")
