from __future__ import annotations

from pathlib import Path
import colorsys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries

from ma_sp_sam.labels.paired_instances import PairedLabelBundle


MYELIN_COLOR = np.array([180, 60, 220], dtype=np.float32)
AXON_COLOR = np.array([40, 220, 80], dtype=np.float32)
SEMANTIC_COLORS = {
    1: MYELIN_COLOR,
    2: AXON_COLOR,
}


def _id_to_rgb(instance_id: int, *, value: float = 0.95, saturation: float = 0.75) -> np.ndarray:
    hue = ((instance_id * 0.618033988749895) % 1.0)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return np.array([r * 255, g * 255, b * 255], dtype=np.uint8)


def colorize_instance_labels(labels: np.ndarray, *, value: float = 0.95, saturation: float = 0.75) -> Image.Image:
    """Render a uint instance label map as RGB, with one color per nonzero ID."""
    labels = np.asarray(labels)
    preview = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(labels):
        if instance_id == 0:
            continue
        preview[labels == instance_id] = _id_to_rgb(int(instance_id), value=value, saturation=saturation)
    return Image.fromarray(preview)


def make_paired_instance_preview(
    *,
    axon_instance: np.ndarray,
    myelin_instance: np.ndarray,
    draw_boundaries: bool = True,
) -> Image.Image:
    """Render paired axon/myelin instance labels with one hue family per pair ID.

    Axon and myelin with the same numeric ID share the same hue; axon is brighter
    and myelin is darker so their paired relationship stays visible.
    """
    axon_instance = np.asarray(axon_instance)
    myelin_instance = np.asarray(myelin_instance)
    if axon_instance.shape != myelin_instance.shape:
        raise ValueError("axon_instance and myelin_instance must have the same shape")

    preview = np.zeros((*axon_instance.shape, 3), dtype=np.uint8)
    ids = sorted(set(np.unique(axon_instance).tolist()) | set(np.unique(myelin_instance).tolist()))
    for instance_id in ids:
        if instance_id == 0:
            continue
        myelin_color = _id_to_rgb(int(instance_id), value=0.62, saturation=0.80)
        axon_color = _id_to_rgb(int(instance_id), value=1.0, saturation=0.92)
        preview[myelin_instance == instance_id] = myelin_color
        preview[axon_instance == instance_id] = axon_color

    if draw_boundaries:
        myelin_boundary = find_boundaries(myelin_instance > 0, mode="outer")
        axon_boundary = find_boundaries(axon_instance > 0, mode="outer")
        preview[myelin_boundary] = np.array([210, 210, 210], dtype=np.uint8)
        preview[axon_boundary] = np.array([255, 255, 255], dtype=np.uint8)

    return Image.fromarray(preview)


def _normalize_rgb_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        base = np.repeat(image[..., None], 3, axis=2)
    else:
        base = image[..., :3]
    base = base.astype(np.float32)
    if base.max() > 0:
        base = 255 * (base - base.min()) / max(float(base.max() - base.min()), 1.0)
    return base


def _draw_flag_boxes(out: Image.Image, fiber_instance: np.ndarray, pair_table: pd.DataFrame | None) -> None:
    if pair_table is None:
        return
    draw = ImageDraw.Draw(out)
    for _, row in pair_table.iterrows():
        flags = str(row.get("flags", ""))
        if not flags or flags == "nan":
            continue
        ys, xs = np.where(fiber_instance == int(row["fiber_id"]))
        if len(xs) == 0:
            continue
        draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(255, 0, 0), width=2)


def make_axon_myelin_overlay(
    image: np.ndarray,
    *,
    axon_instance: np.ndarray,
    myelin_instance: np.ndarray,
    fiber_instance: np.ndarray | None = None,
    pair_table: pd.DataFrame | None = None,
    alpha: float = 0.55,
    draw_boundaries: bool = True,
) -> Image.Image:
    """Overlay axon and myelin instance masks on a raw or blank image.

    Axon is green, myelin is purple. Red boxes mark flagged fibers when
    ``fiber_instance`` and ``pair_table`` are provided.
    """
    base = _normalize_rgb_image(image)
    overlay = base.copy()

    myelin_mask = myelin_instance > 0
    axon_mask = axon_instance > 0
    overlay[myelin_mask] = (1 - alpha) * overlay[myelin_mask] + alpha * MYELIN_COLOR
    overlay[axon_mask] = (1 - alpha) * overlay[axon_mask] + alpha * AXON_COLOR

    if draw_boundaries:
        boundary = find_boundaries(axon_mask | myelin_mask, mode="outer")
        overlay[boundary] = np.array([255, 255, 255], dtype=np.float32)

    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    if fiber_instance is not None:
        _draw_flag_boxes(out, fiber_instance, pair_table)
    return out


def save_axon_myelin_overlay(
    path: str | Path,
    image: np.ndarray,
    *,
    axon_instance: np.ndarray,
    myelin_instance: np.ndarray,
    fiber_instance: np.ndarray | None = None,
    pair_table: pd.DataFrame | None = None,
    alpha: float = 0.55,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_axon_myelin_overlay(
        image,
        axon_instance=axon_instance,
        myelin_instance=myelin_instance,
        fiber_instance=fiber_instance,
        pair_table=pair_table,
        alpha=alpha,
    ).save(out)


def make_qc_overlay(image: np.ndarray, bundle: PairedLabelBundle, alpha: float = 0.35) -> Image.Image:
    base = _normalize_rgb_image(image)
    overlay = base.copy()
    for label, color in SEMANTIC_COLORS.items():
        mask = bundle.semantic == label
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    _draw_flag_boxes(out, bundle.fiber_instance, bundle.pair_table)
    return out


def save_qc_overlay(path: str | Path, image: np.ndarray, bundle: PairedLabelBundle) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_qc_overlay(image, bundle).save(out)
