from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure


def compute_g_ratio(axon_area: int, fiber_area: int) -> float:
    if fiber_area <= 0 or axon_area <= 0:
        return 0.0
    return float(np.sqrt(float(axon_area) / float(fiber_area)))


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def connected_component_count(mask: np.ndarray) -> int:
    labels = measure.label(np.asarray(mask).astype(bool), connectivity=1)
    return int(labels.max())


def resolve_overlaps_by_score(candidate_masks, scores, instance_ids) -> np.ndarray:
    masks = np.asarray(candidate_masks).astype(bool)
    if masks.ndim != 3:
        raise ValueError(f"candidate_masks must be [N,H,W], got shape {masks.shape}.")
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    instance_ids = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    if not (len(masks) == len(scores) == len(instance_ids)):
        raise ValueError("candidate_masks, scores, and instance_ids must have the same length.")

    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    order = sorted(
        range(len(masks)),
        key=lambda index: (-float(scores[index]), int(areas[index]), int(instance_ids[index])),
    )
    label_map = np.zeros(masks.shape[1:], dtype=np.uint16)
    occupied = np.zeros(masks.shape[1:], dtype=bool)
    for index in order:
        assign = masks[index] & ~occupied
        if not assign.any():
            continue
        label_map[assign] = int(instance_ids[index])
        occupied[assign] = True
    return label_map


def make_pair_table(records) -> pd.DataFrame:
    rows = []
    for record in records:
        row = record.__dict__.copy() if hasattr(record, "__dict__") else dict(record)
        flags = row.get("flags", "")
        if isinstance(flags, (list, tuple, set)):
            flags = ";".join(str(flag) for flag in flags)
        row["flags"] = flags
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "instance_id",
            "source_prompt_id",
            "selected_candidate_index",
            "sam_score",
            "fiber_area",
            "axon_area",
            "myelin_area",
            "g_ratio",
            "flags",
        ],
    )
