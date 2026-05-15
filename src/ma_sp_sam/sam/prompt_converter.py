from __future__ import annotations

from typing import Any

import numpy as np
from skimage.transform import resize

from ma_sp_sam.prompts import PromptPackage


def convert_prompt_package_to_sam_inputs(package: PromptPackage, image_shape: tuple[int, int]) -> dict[str, Any]:
    """Convert one PromptPackage into SamPredictor-compatible numpy inputs."""
    height, width = _validate_image_shape(image_shape)
    points = [*package.positive_points, *package.negative_points]
    point_coords = np.asarray([point.xy for point in points], dtype=np.float32).reshape(-1, 2)
    point_labels = np.asarray([int(point.label) for point in points], dtype=np.int64)
    box = None
    if package.box_prompt is not None:
        box = np.asarray(package.box_prompt.xyxy, dtype=np.float32)
    mask_input = None
    if package.coarse_mask_prompt is not None and package.coarse_mask_prompt.mask is not None:
        mask_input = _coarse_mask_to_sam_input(package.coarse_mask_prompt.mask, image_shape=(height, width))

    ring_points = package.ring_prompt.ring_points.astype(np.float32)
    metadata = {
        "instance_id": package.instance_id,
        "quality_prior": float(package.quality_prior),
        "source": package.source,
        "ring_points": ring_points,
        "inner_ring_points": package.ring_prompt.inner_points.astype(np.float32),
        "outer_ring_points": package.ring_prompt.outer_points.astype(np.float32),
        "ring_prompt_source": package.ring_prompt.source,
    }
    return {
        "point_coords": point_coords,
        "point_labels": point_labels,
        "box": box,
        "mask_input": mask_input,
        "metadata": metadata,
    }


def _coarse_mask_to_sam_input(mask: np.ndarray, *, image_shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask).astype(np.float32)
    if mask.shape != image_shape:
        mask = resize(mask, image_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
    return mask[None, ...].astype(np.float32)


def _validate_image_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    if len(image_shape) != 2:
        raise ValueError("image_shape must be (height, width).")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape values must be positive.")
    return height, width
