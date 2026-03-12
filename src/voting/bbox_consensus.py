"""Bounding-box IoU, matching, and consensus utilities."""

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


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
    """Compute the Intersection-over-Union between two bounding boxes.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        IoU in [0, 1].
    """
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
    """Greedily match predicted boxes to ground-truth boxes by IoU.

    Args:
        pred_boxes: Predicted bounding boxes.
        gt_boxes: Ground-truth bounding boxes.
        iou_threshold: Minimum IoU to count as a match.

    Returns:
        A 3-tuple of:
        - ``matches``: list of ``(pred_idx, gt_idx)`` pairs.
        - ``unmatched_preds``: indices of unmatched predictions.
        - ``unmatched_gts``: indices of unmatched ground-truth boxes.
    """
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


def consensus_boxes(
    box_predictions: List[List[BoundingBoxPrediction]],
    iou_threshold: float = 0.5,
    min_votes: int = 2,
) -> List[BoundingBoxPrediction]:
    """Compute consensus bounding boxes from multiple per-view predictions.

    Boxes across views that overlap (IoU ≥ ``iou_threshold``) are clustered
    and averaged.  Clusters with fewer than ``min_votes`` boxes are discarded.

    Args:
        box_predictions: One list of boxes per view/model.
        iou_threshold: IoU threshold for merging into a cluster.
        min_votes: Minimum cluster size to keep.

    Returns:
        List of consensus :class:`BoundingBoxPrediction` objects.
    """
    if not box_predictions:
        return []

    all_boxes: List[Tuple[BoundingBoxPrediction, int]] = [
        (box, view_idx)
        for view_idx, boxes in enumerate(box_predictions)
        for box in boxes
    ]

    if not all_boxes:
        return []

    clusters: List[List[Tuple[BoundingBoxPrediction, int]]] = []
    used: set[int] = set()

    for i, (box_i, _) in enumerate(all_boxes):
        if i in used:
            continue
        cluster = [(box_i, i)]
        used.add(i)
        for j, (box_j, _) in enumerate(all_boxes):
            if j in used or j <= i:
                continue
            for box_in_cluster, _ in cluster:
                if compute_iou(box_in_cluster, box_j) >= iou_threshold:
                    cluster.append((box_j, j))
                    used.add(j)
                    break
        clusters.append(cluster)

    result: List[BoundingBoxPrediction] = []
    for cluster in clusters:
        if len(cluster) < min_votes:
            continue
        boxes = [b for b, _ in cluster]
        labels = [b.label for b in boxes if b.label]
        result.append(
            BoundingBoxPrediction(
                x=float(np.mean([b.x for b in boxes])),
                y=float(np.mean([b.y for b in boxes])),
                width=float(np.mean([b.width for b in boxes])),
                height=float(np.mean([b.height for b in boxes])),
                confidence=float(np.mean([b.confidence for b in boxes])),
                label=Counter(labels).most_common(1)[0][0] if labels else None,
            )
        )
    return result
