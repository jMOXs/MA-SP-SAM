from __future__ import annotations

import numpy as np
from skimage.segmentation import find_boundaries

from ma_sp_sam.labels.paired_instances import PairedLabelBundle


def center_heatmap_from_instances(axon_instance: np.ndarray) -> np.ndarray:
    heatmap = np.zeros(axon_instance.shape, dtype=np.float32)
    for instance_id in range(1, int(axon_instance.max()) + 1):
        ys, xs = np.where(axon_instance == instance_id)
        if len(xs) == 0:
            continue
        y, x = _instance_center_yx(ys, xs)
        heatmap[y, x] = 1.0
    return heatmap


def gaussian_center_heatmap_from_instances(axon_instance: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    axon_instance = np.asarray(axon_instance)
    heatmap = np.zeros(axon_instance.shape, dtype=np.float32)
    grid_y, grid_x = np.indices(axon_instance.shape, dtype=np.float32)
    for instance_id in range(1, int(axon_instance.max()) + 1):
        ys, xs = np.where(axon_instance == instance_id)
        if len(xs) == 0:
            continue
        center_y, center_x = _instance_center_yx(ys, xs)
        gaussian = np.exp(-((grid_y - center_y) ** 2 + (grid_x - center_x) ** 2) / (2.0 * sigma**2))
        heatmap = np.maximum(heatmap, gaussian.astype(np.float32))
    return np.clip(heatmap, 0.0, 1.0).astype(np.float32)


def hover_distance_map_from_instances(
    *,
    fiber_instance: np.ndarray,
    axon_instance: np.ndarray,
) -> np.ndarray:
    fiber_instance = np.asarray(fiber_instance)
    axon_instance = np.asarray(axon_instance)
    if fiber_instance.shape != axon_instance.shape:
        raise ValueError("fiber_instance and axon_instance must share shape.")

    distance = np.zeros((2, *fiber_instance.shape), dtype=np.float32)
    instance_ids = sorted(int(value) for value in np.unique(fiber_instance) if int(value) != 0)
    for instance_id in instance_ids:
        fiber_mask = fiber_instance == instance_id
        axon_mask = axon_instance == instance_id
        if not fiber_mask.any() or not axon_mask.any():
            continue
        center_y, center_x = _instance_center_yx(*np.where(axon_mask))
        ys, xs = np.where(fiber_mask)
        scale = max(
            float(np.max(np.abs(xs.astype(np.float32) - center_x))),
            float(np.max(np.abs(ys.astype(np.float32) - center_y))),
            1.0,
        )
        distance[0, ys, xs] = (xs.astype(np.float32) - center_x) / scale
        distance[1, ys, xs] = (ys.astype(np.float32) - center_y) / scale
    return distance


def boundary_maps_from_bundle(bundle: PairedLabelBundle) -> tuple[np.ndarray, np.ndarray]:
    axon = bundle.axon_instance > 0
    fiber = bundle.fiber_instance > 0
    inner = find_boundaries(axon, mode="inner").astype(np.uint8)
    outer = find_boundaries(fiber, mode="inner").astype(np.uint8)
    return inner, outer


def _instance_center_yx(ys: np.ndarray, xs: np.ndarray) -> tuple[int, int]:
    y = int(np.floor(float(ys.mean()) + 0.5))
    x = int(np.floor(float(xs.mean()) + 0.5))
    return y, x
