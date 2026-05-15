from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries

from ma_sp_sam.viz.overlays import colorize_instance_labels


SEMANTIC_PALETTE = np.array(
    [
        [0, 0, 0],
        [180, 60, 220],
        [40, 220, 80],
    ],
    dtype=np.uint8,
)


def save_heatmap_png(path: str | Path, heatmap) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    array = _normalize_unit(np.asarray(heatmap, dtype=np.float32))
    Image.fromarray((array * 255).astype(np.uint8), mode="L").save(out)


def save_semantic_prediction_png(path: str | Path, semantic_pred) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(semantic_pred, dtype=np.int64)
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in range(min(len(SEMANTIC_PALETTE), int(labels.max(initial=0)) + 1)):
        rgb[labels == label] = SEMANTIC_PALETTE[label]
    Image.fromarray(rgb).save(out)


def save_proposal_overlay(path: str | Path, image, proposal_label_map, *, alpha: float = 0.55) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    base = _to_rgb(np.asarray(image))
    labels = np.asarray(proposal_label_map)
    colorized = np.asarray(colorize_instance_labels(labels)).astype(np.float32)
    mask = labels > 0
    overlay = base.astype(np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * colorized[mask]
    boundaries = find_boundaries(labels, mode="outer")
    overlay[boundaries] = np.array([255, 255, 255], dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out)


def _normalize_unit(array: np.ndarray) -> np.ndarray:
    array = np.squeeze(array)
    if array.size == 0:
        return array.astype(np.float32)
    min_value = float(np.nanmin(array))
    max_value = float(np.nanmax(array))
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)


def _to_rgb(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3:
        image = image[..., :3]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
    else:
        raise ValueError(f"Expected 2D grayscale or RGB image, got shape {image.shape}.")
    image = image.astype(np.float32)
    if image.max(initial=0) <= 1.0:
        image = image * 255.0
    return image
