from __future__ import annotations

from typing import Any

import numpy as np
from skimage.transform import resize

from ma_sp_sam.prompts import PromptPackage


def convert_prompt_package_to_sam_inputs(
    package: PromptPackage,
    image_shape: tuple[int, int],
    *,
    use_mask_input: bool = False,
    mask_input_size: tuple[int, int] = (256, 256),
) -> dict[str, Any]:
    """Convert one PromptPackage into SamPredictor-compatible numpy inputs."""
    height, width = _validate_image_shape(image_shape)
    points = [*package.positive_points, *package.negative_points]
    point_coords = np.asarray([point.xy for point in points], dtype=np.float32).reshape(-1, 2)
    point_labels = np.asarray([int(point.label) for point in points], dtype=np.int64)
    box = None
    if package.box_prompt is not None:
        box = np.asarray(package.box_prompt.xyxy, dtype=np.float32)
    mask_input = None
    coarse_mask = None
    if package.coarse_mask_prompt is not None and package.coarse_mask_prompt.mask is not None:
        coarse_mask = np.asarray(package.coarse_mask_prompt.mask).astype(np.float32)
        if use_mask_input:
            mask_input = _coarse_mask_to_sam_input(coarse_mask, mask_input_size=mask_input_size)

    ring_points = package.ring_prompt.ring_points.astype(np.float32)
    metadata = {
        "instance_id": package.instance_id,
        "quality_prior": float(package.quality_prior),
        "source": package.source,
        "coarse_mask_shape": None if coarse_mask is None else tuple(int(value) for value in coarse_mask.shape),
        "coarse_mask_area": 0 if coarse_mask is None else int(np.count_nonzero(coarse_mask)),
        "coarse_mask_bbox": None if coarse_mask is None else _bbox_from_mask(coarse_mask > 0),
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


def _coarse_mask_to_sam_input(mask: np.ndarray, *, mask_input_size: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask).astype(np.float32)
    target_size = _validate_image_shape(mask_input_size)
    if mask.shape != target_size:
        mask = resize(mask, target_size, order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
    return mask[None, ...].astype(np.float32)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _validate_image_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    if len(image_shape) != 2:
        raise ValueError("image_shape must be (height, width).")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape values must be positive.")
    return height, width
