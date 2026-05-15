from __future__ import annotations

import numpy as np


def proposal_recall_at_iou(pred_label_map, gt_fiber_instance, iou_threshold: float = 0.5) -> float:
    pred = np.asarray(pred_label_map)
    gt = np.asarray(gt_fiber_instance)
    _validate_shapes(pred, gt)
    gt_ids = _nonzero_ids(gt)
    if not gt_ids:
        return 1.0
    if not _nonzero_ids(pred):
        return 0.0
    matched = sum(_best_iou(gt == gt_id, pred) >= iou_threshold for gt_id in gt_ids)
    return float(matched / len(gt_ids))


def proposal_precision_at_iou(pred_label_map, gt_fiber_instance, iou_threshold: float = 0.5) -> float:
    pred = np.asarray(pred_label_map)
    gt = np.asarray(gt_fiber_instance)
    _validate_shapes(pred, gt)
    pred_ids = _nonzero_ids(pred)
    if not pred_ids:
        return 0.0
    if not _nonzero_ids(gt):
        return 0.0
    matched = sum(_best_iou(pred == pred_id, gt) >= iou_threshold for pred_id in pred_ids)
    return float(matched / len(pred_ids))


def proposal_f1_at_iou(pred_label_map, gt_fiber_instance, iou_threshold: float = 0.5) -> float:
    recall = proposal_recall_at_iou(pred_label_map, gt_fiber_instance, iou_threshold=iou_threshold)
    precision = proposal_precision_at_iou(pred_label_map, gt_fiber_instance, iou_threshold=iou_threshold)
    if recall + precision == 0:
        return 0.0
    return float(2.0 * recall * precision / (recall + precision))


def _best_iou(mask: np.ndarray, label_map: np.ndarray) -> float:
    best = 0.0
    for instance_id in _nonzero_ids(label_map):
        candidate = label_map == instance_id
        intersection = np.logical_and(mask, candidate).sum()
        union = np.logical_or(mask, candidate).sum()
        if union:
            best = max(best, float(intersection / union))
    return best


def _nonzero_ids(label_map: np.ndarray) -> list[int]:
    return sorted(int(value) for value in np.unique(label_map) if int(value) != 0)


def _validate_shapes(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"pred_label_map and gt_fiber_instance must share shape, got {pred.shape} and {gt.shape}.")
