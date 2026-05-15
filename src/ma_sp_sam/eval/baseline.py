from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ma_sp_sam.labels.paired_instances import PairedLabelBundle, build_paired_instances
from ma_sp_sam.utils.io import read_mask


MYELIN_VALUES = {126, 127, 128}
AXON_VALUES = {255}


@dataclass(frozen=True)
class InstanceMatch:
    pred_id: int
    gt_id: int
    iou: float


def normalize_semantic_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize either 0/1/2 semantic masks or ASTIH-style composite masks."""
    mask = np.asarray(mask)
    if set(np.unique(mask).tolist()).issubset({0, 1, 2}):
        return mask.astype(np.uint8)

    semantic = np.zeros(mask.shape, dtype=np.uint8)
    semantic[np.isin(mask, list(MYELIN_VALUES))] = 1
    semantic[np.isin(mask, list(AXON_VALUES))] = 2
    return semantic


def dice_for_label(pred_semantic: np.ndarray, gt_semantic: np.ndarray, label: int) -> float:
    pred = pred_semantic == label
    gt = gt_semantic == label
    denom = int(pred.sum() + gt.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / denom)


def _instance_ids(labels: np.ndarray) -> list[int]:
    return [int(value) for value in np.unique(labels) if value != 0]


def _mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(pred_mask, gt_mask).sum() / union)


def match_instances(pred_instance: np.ndarray, gt_instance: np.ndarray, iou_threshold: float = 0.5) -> list[InstanceMatch]:
    candidates = [
        InstanceMatch(pred_id=pred_id, gt_id=gt_id, iou=iou)
        for (pred_id, gt_id), iou in _pair_iou_lookup(pred_instance, gt_instance).items()
        if iou >= iou_threshold
    ]

    matches: list[InstanceMatch] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for candidate in sorted(candidates, key=lambda item: item.iou, reverse=True):
        if candidate.pred_id in used_pred or candidate.gt_id in used_gt:
            continue
        matches.append(candidate)
        used_pred.add(candidate.pred_id)
        used_gt.add(candidate.gt_id)
    return matches


def _pair_iou_lookup(pred_instance: np.ndarray, gt_instance: np.ndarray) -> dict[tuple[int, int], float]:
    pred_instance = np.asarray(pred_instance, dtype=np.int64)
    gt_instance = np.asarray(gt_instance, dtype=np.int64)
    if pred_instance.shape != gt_instance.shape:
        raise ValueError("pred_instance and gt_instance must have the same shape")

    max_pred = int(pred_instance.max())
    max_gt = int(gt_instance.max())
    if max_pred == 0 or max_gt == 0:
        return {}

    pred_areas = np.bincount(pred_instance.ravel(), minlength=max_pred + 1)
    gt_areas = np.bincount(gt_instance.ravel(), minlength=max_gt + 1)

    overlap_mask = (pred_instance > 0) & (gt_instance > 0)
    if not overlap_mask.any():
        return {}

    stride = max_gt + 1
    pair_codes = pred_instance[overlap_mask] * stride + gt_instance[overlap_mask]
    intersections = np.bincount(pair_codes)

    lookup: dict[tuple[int, int], float] = {}
    for code, intersection in enumerate(intersections):
        if intersection == 0:
            continue
        pred_id = code // stride
        gt_id = code % stride
        if pred_id == 0 or gt_id == 0:
            continue
        union = pred_areas[pred_id] + gt_areas[gt_id] - intersection
        iou = float(intersection / union) if union else 0.0
        lookup[(int(pred_id), int(gt_id))] = iou
    return lookup


def fiber_ap_at_iou(pred_fiber: np.ndarray, gt_fiber: np.ndarray, iou_threshold: float = 0.5) -> float:
    matches = match_instances(pred_fiber, gt_fiber, iou_threshold=iou_threshold)
    tp = len(matches)
    fp = len(_instance_ids(pred_fiber)) - tp
    fn = len(_instance_ids(gt_fiber)) - tp
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return float(tp / denom)


def _bundle_g_ratio(bundle: PairedLabelBundle, fiber_id: int) -> float:
    rows = bundle.pair_table[bundle.pair_table["fiber_id"] == fiber_id]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0]["g_ratio"])


def _valid_gt_pair_ids(bundle: PairedLabelBundle) -> list[int]:
    if bundle.pair_table.empty:
        return []
    valid = bundle.pair_table[(bundle.pair_table["axon_area"] > 0) & (bundle.pair_table["myelin_area"] > 0)]
    return [int(value) for value in valid["fiber_id"].tolist()]


def pair_accuracy_and_g_ratio_mae(
    pred_bundle: PairedLabelBundle,
    gt_bundle: PairedLabelBundle,
    *,
    fiber_iou_threshold: float = 0.5,
    class_iou_threshold: float = 0.5,
) -> tuple[float, float]:
    matches = match_instances(
        pred_bundle.fiber_instance,
        gt_bundle.fiber_instance,
        iou_threshold=fiber_iou_threshold,
    )
    match_by_gt = {match.gt_id: match.pred_id for match in matches}
    axon_iou_lookup = _pair_iou_lookup(pred_bundle.axon_instance, gt_bundle.axon_instance)
    myelin_iou_lookup = _pair_iou_lookup(pred_bundle.myelin_instance, gt_bundle.myelin_instance)
    valid_gt_ids = _valid_gt_pair_ids(gt_bundle)
    if not valid_gt_ids:
        return 1.0, 0.0

    correct_pairs = 0
    g_ratio_errors: list[float] = []
    for gt_id in valid_gt_ids:
        pred_id = match_by_gt.get(gt_id)
        if pred_id is None:
            continue
        axon_iou = axon_iou_lookup.get((pred_id, gt_id), 0.0)
        myelin_iou = myelin_iou_lookup.get((pred_id, gt_id), 0.0)
        if axon_iou >= class_iou_threshold and myelin_iou >= class_iou_threshold:
            correct_pairs += 1
            g_ratio_errors.append(abs(_bundle_g_ratio(pred_bundle, pred_id) - _bundle_g_ratio(gt_bundle, gt_id)))

    pair_accuracy = float(correct_pairs / len(valid_gt_ids))
    g_ratio_mae = float(np.mean(g_ratio_errors)) if g_ratio_errors else float("nan")
    return pair_accuracy, g_ratio_mae


def evaluate_semantic_prediction(
    pred_semantic: np.ndarray,
    gt_semantic: np.ndarray,
    *,
    sample_id: str,
    dataset: str = "",
    split: str = "",
    fiber_iou_threshold: float = 0.5,
    class_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    pred_semantic = normalize_semantic_mask(pred_semantic)
    gt_semantic = normalize_semantic_mask(gt_semantic)
    if pred_semantic.shape != gt_semantic.shape:
        raise ValueError(f"Prediction and GT shapes differ for {sample_id}: {pred_semantic.shape} vs {gt_semantic.shape}")

    pred_bundle = build_paired_instances(pred_semantic)
    gt_bundle = build_paired_instances(gt_semantic)
    pair_accuracy, g_ratio_mae = pair_accuracy_and_g_ratio_mae(
        pred_bundle,
        gt_bundle,
        fiber_iou_threshold=fiber_iou_threshold,
        class_iou_threshold=class_iou_threshold,
    )
    return {
        "dataset": dataset,
        "split": split,
        "sample_id": sample_id,
        "axon_dice": dice_for_label(pred_semantic, gt_semantic, 2),
        "myelin_dice": dice_for_label(pred_semantic, gt_semantic, 1),
        "fiber_ap50": fiber_ap_at_iou(pred_bundle.fiber_instance, gt_bundle.fiber_instance, iou_threshold=fiber_iou_threshold),
        "pair_accuracy": pair_accuracy,
        "g_ratio_mae": g_ratio_mae,
        "gt_fibers": int(gt_bundle.fiber_instance.max()),
        "pred_fibers": int(pred_bundle.fiber_instance.max()),
    }


def _iter_gt_sample_dirs(gt_root: Path, dataset: str | None, split: str | None):
    datasets = [dataset] if dataset else [path.name for path in sorted(gt_root.iterdir()) if path.is_dir()]
    for dataset_name in datasets:
        dataset_dir = gt_root / dataset_name
        if not dataset_dir.exists():
            continue
        splits = [split] if split else [path.name for path in sorted(dataset_dir.iterdir()) if path.is_dir()]
        for split_name in splits:
            split_dir = dataset_dir / split_name
            if not split_dir.exists():
                continue
            for sample_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                if (sample_dir / "semantic.png").exists():
                    yield dataset_name, split_name, sample_dir


def find_prediction_mask(pred_root: str | Path, dataset: str, split: str, sample_id: str) -> Path:
    root = Path(pred_root)
    candidates = [
        root / dataset / split / sample_id / "semantic.png",
        root / dataset / split / f"{sample_id}.png",
        root / sample_id / "semantic.png",
        root / f"{sample_id}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No prediction semantic mask found for {dataset}/{split}/{sample_id} under {root}")


def evaluate_baseline_directory(
    *,
    pred_root: str | Path,
    gt_root: str | Path,
    out_csv: str | Path,
    dataset: str | None = None,
    split: str | None = None,
    limit: int | None = None,
    fiber_iou_threshold: float = 0.5,
    class_iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    pred_root = Path(pred_root)
    gt_root = Path(gt_root)
    rows: list[dict[str, Any]] = []
    for dataset_name, split_name, sample_dir in _iter_gt_sample_dirs(gt_root, dataset, split):
        if limit is not None and len(rows) >= limit:
            break
        pred_path = find_prediction_mask(pred_root, dataset_name, split_name, sample_dir.name)
        row = evaluate_semantic_prediction(
            read_mask(pred_path),
            read_mask(sample_dir / "semantic.png"),
            sample_id=sample_dir.name,
            dataset=dataset_name,
            split=split_name,
            fiber_iou_threshold=fiber_iou_threshold,
            class_iou_threshold=class_iou_threshold,
        )
        row["prediction_path"] = str(pred_path)
        rows.append(row)

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return rows
