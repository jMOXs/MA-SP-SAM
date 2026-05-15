from __future__ import annotations

import numpy as np

from ma_sp_sam.eval.baseline import dice_for_label, match_instances
from ma_sp_sam.labels.paired_instances import PairedLabelBundle
from ma_sp_sam.refinement.pair_refinement import PairRefinementOutput


def evaluate_refined_prediction(
    pred_output: PairRefinementOutput,
    gt_bundle: PairedLabelBundle,
    *,
    fiber_iou_threshold: float = 0.5,
    class_iou_threshold: float = 0.5,
) -> dict[str, float]:
    matches = match_instances(pred_output.fiber_instance, gt_bundle.fiber_instance, iou_threshold=fiber_iou_threshold)
    pred_ids = _instance_ids(pred_output.fiber_instance)
    gt_ids = _instance_ids(gt_bundle.fiber_instance)
    recall = len(matches) / len(gt_ids) if gt_ids else 1.0
    precision = len(matches) / len(pred_ids) if pred_ids else 0.0

    pred_semantic = np.zeros_like(gt_bundle.semantic, dtype=np.uint8)
    pred_semantic[pred_output.myelin_instance > 0] = 1
    pred_semantic[pred_output.axon_instance > 0] = 2

    correct_pairs = 0
    g_errors: list[float] = []
    for match in matches:
        pred_id, gt_id = match.pred_id, match.gt_id
        axon_iou = _mask_iou(pred_output.axon_instance == pred_id, gt_bundle.axon_instance == gt_id)
        myelin_iou = _mask_iou(pred_output.myelin_instance == pred_id, gt_bundle.myelin_instance == gt_id)
        if axon_iou >= class_iou_threshold and myelin_iou >= class_iou_threshold:
            correct_pairs += 1
        pred_g = _g_ratio_from_table(pred_output.pair_table, "instance_id", pred_id)
        gt_g = _g_ratio_from_table(gt_bundle.pair_table, "fiber_id", gt_id)
        if np.isfinite(pred_g) and np.isfinite(gt_g):
            g_errors.append(abs(pred_g - gt_g))

    return {
        "fiber_iou50_recall": float(recall),
        "fiber_iou50_precision": float(precision),
        "axon_dice": dice_for_label(pred_semantic, gt_bundle.semantic, 2),
        "myelin_dice": dice_for_label(pred_semantic, gt_bundle.semantic, 1),
        "pair_accuracy_proxy": float(correct_pairs / len(matches)) if matches else 0.0,
        "g_ratio_mae": float(np.mean(g_errors)) if g_errors else float("nan"),
    }


def _instance_ids(label_map: np.ndarray) -> list[int]:
    return [int(value) for value in np.unique(label_map) if int(value) != 0]


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _g_ratio_from_table(table, id_column: str, instance_id: int) -> float:
    rows = table[table[id_column] == instance_id] if id_column in table.columns else table.iloc[0:0]
    if rows.empty or "g_ratio" not in rows.columns:
        return float("nan")
    return float(rows.iloc[0]["g_ratio"])
