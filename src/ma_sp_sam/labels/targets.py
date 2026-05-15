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
        y = int(round(float(ys.mean())))
        x = int(round(float(xs.mean())))
        heatmap[y, x] = 1.0
    return heatmap


def boundary_maps_from_bundle(bundle: PairedLabelBundle) -> tuple[np.ndarray, np.ndarray]:
    axon = bundle.axon_instance > 0
    fiber = bundle.fiber_instance > 0
    inner = find_boundaries(axon, mode="inner").astype(np.uint8)
    outer = find_boundaries(fiber, mode="inner").astype(np.uint8)
    return inner, outer
