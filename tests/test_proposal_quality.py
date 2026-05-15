import numpy as np

from ma_sp_sam.eval.proposal_quality import (
    proposal_f1_at_iou,
    proposal_precision_at_iou,
    proposal_recall_at_iou,
)


def test_proposal_recall_precision_f1_are_one_for_exact_match():
    gt = np.zeros((8, 8), dtype=np.uint16)
    gt[1:4, 1:4] = 1
    gt[4:7, 4:7] = 2
    pred = gt.copy()

    assert proposal_recall_at_iou(pred, gt, iou_threshold=0.5) == 1.0
    assert proposal_precision_at_iou(pred, gt, iou_threshold=0.5) == 1.0
    assert proposal_f1_at_iou(pred, gt, iou_threshold=0.5) == 1.0


def test_proposal_recall_is_zero_for_empty_prediction_with_gt():
    gt = np.zeros((6, 6), dtype=np.uint16)
    gt[2:5, 2:5] = 1
    pred = np.zeros_like(gt)

    assert proposal_recall_at_iou(pred, gt, iou_threshold=0.5) == 0.0
    assert proposal_precision_at_iou(pred, gt, iou_threshold=0.5) == 0.0
    assert proposal_f1_at_iou(pred, gt, iou_threshold=0.5) == 0.0
