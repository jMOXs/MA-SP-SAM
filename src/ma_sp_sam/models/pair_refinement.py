from __future__ import annotations

"""Legacy tensor-oriented pair refinement module.

The current end-to-end MA-SP-SAM V1 pipeline uses
``ma_sp_sam.refinement.pair_refinement``. This module is kept for backward
compatibility with earlier tests and experiments that refined separate axon,
myelin, and fiber candidate tensors directly.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ma_sp_sam.models.sam_adapter import CandidateMask
from ma_sp_sam.prompts import PromptPackage, PromptSynthesisResult


@dataclass(frozen=True)
class PairRefinementOutput:
    axon_i: np.ndarray
    myelin_i: np.ndarray
    fiber_i: np.ndarray
    pair_id_i: np.ndarray
    pair_table: pd.DataFrame


class PairRefinementModule:
    """Refine per-instance axon, myelin, and fiber candidates into paired labels."""

    def __init__(
        self,
        *,
        mask_threshold: float = 0.0,
        min_axon_area: int = 8,
        min_myelin_area: int = 8,
        g_ratio_min: float = 0.2,
        g_ratio_max: float = 0.95,
    ) -> None:
        if min_axon_area < 0:
            raise ValueError("min_axon_area must be non-negative.")
        if min_myelin_area < 0:
            raise ValueError("min_myelin_area must be non-negative.")
        if not 0 <= g_ratio_min <= g_ratio_max:
            raise ValueError("Expected 0 <= g_ratio_min <= g_ratio_max.")

        self.mask_threshold = mask_threshold
        self.min_axon_area = min_axon_area
        self.min_myelin_area = min_myelin_area
        self.g_ratio_min = g_ratio_min
        self.g_ratio_max = g_ratio_max

    def __call__(self, *args, **kwargs) -> PairRefinementOutput:
        return self.refine(*args, **kwargs)

    def refine(
        self,
        *,
        axon_candidate_masks,
        myelin_candidate_masks,
        fiber_candidate_masks,
        prompt_metadata=None,
    ) -> PairRefinementOutput:
        metadata = _metadata_by_instance_id(prompt_metadata)
        metadata_ids = list(metadata)
        axon_scores = _candidate_scores_by_instance(axon_candidate_masks, metadata_ids)
        myelin_scores = _candidate_scores_by_instance(myelin_candidate_masks, metadata_ids)
        fiber_scores = _candidate_scores_by_instance(fiber_candidate_masks, metadata_ids)

        instance_ids = _ordered_instance_ids(metadata_ids, axon_scores, myelin_scores, fiber_scores)
        shape = _infer_shape(axon_scores, myelin_scores, fiber_scores)
        axon_i = np.zeros(shape, dtype=np.uint16)
        myelin_i = np.zeros(shape, dtype=np.uint16)
        fiber_i = np.zeros(shape, dtype=np.uint16)
        pair_id_i = np.zeros(shape, dtype=np.uint16)

        rows: list[dict[str, object]] = []
        for instance_id in instance_ids:
            axon_score = _score_or_zeros(axon_scores, instance_id, shape)
            myelin_score = _score_or_zeros(myelin_scores, instance_id, shape)
            _validate_shape(instance_id, axon_score, shape)
            _validate_shape(instance_id, myelin_score, shape)

            axon_mask = axon_score > self.mask_threshold
            myelin_mask = myelin_score > self.mask_threshold
            axon_mask, myelin_mask = _remove_axon_myelin_overlap(
                axon_mask=axon_mask,
                myelin_mask=myelin_mask,
                axon_score=axon_score,
                myelin_score=myelin_score,
            )
            fiber_mask = axon_mask | myelin_mask

            axon_i[axon_mask] = instance_id
            myelin_i[myelin_mask] = instance_id
            fiber_i[fiber_mask] = instance_id
            pair_id_i[fiber_mask] = instance_id

            axon_area = int(axon_mask.sum())
            myelin_area = int(myelin_mask.sum())
            fiber_area = int(fiber_mask.sum())
            g_ratio = float(np.sqrt(axon_area / fiber_area)) if fiber_area else 0.0
            flags = self._flags(axon_area=axon_area, myelin_area=myelin_area, fiber_area=fiber_area, g_ratio=g_ratio)
            rows.append(
                {
                    "fiber_id": instance_id,
                    "pair_id": instance_id,
                    "axon_id": instance_id if axon_area else 0,
                    "myelin_id": instance_id if myelin_area else 0,
                    "axon_area": axon_area,
                    "myelin_area": myelin_area,
                    "fiber_area": fiber_area,
                    "g_ratio": g_ratio,
                    "quality_prior": float(metadata.get(instance_id, {}).get("quality_prior", np.nan)),
                    "flags": ";".join(flags),
                }
            )

        pair_table = pd.DataFrame(
            rows,
            columns=[
                "fiber_id",
                "pair_id",
                "axon_id",
                "myelin_id",
                "axon_area",
                "myelin_area",
                "fiber_area",
                "g_ratio",
                "quality_prior",
                "flags",
            ],
        )
        return PairRefinementOutput(
            axon_i=axon_i,
            myelin_i=myelin_i,
            fiber_i=fiber_i,
            pair_id_i=pair_id_i,
            pair_table=pair_table,
        )

    def _flags(self, *, axon_area: int, myelin_area: int, fiber_area: int, g_ratio: float) -> list[str]:
        flags: list[str] = []
        if axon_area == 0:
            flags.append("missing_axon")
        if myelin_area == 0:
            flags.append("missing_myelin")
        if axon_area > 0 and axon_area < self.min_axon_area:
            flags.append("small_axon")
        if myelin_area > 0 and myelin_area < self.min_myelin_area:
            flags.append("small_myelin")
        if axon_area > 0 and myelin_area == 0:
            flags.append("orphan_axon")
        if myelin_area > 0 and axon_area == 0:
            flags.append("orphan_myelin")
        if fiber_area > 0 and axon_area > 0 and not (self.g_ratio_min <= g_ratio <= self.g_ratio_max):
            flags.append("g_ratio_out_of_range")
        return list(dict.fromkeys(flags))


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def _candidate_scores_by_instance(candidates, metadata_ids: Sequence[int]) -> dict[int, np.ndarray]:
    if candidates is None:
        return {}
    if isinstance(candidates, Mapping):
        return {int(instance_id): _select_candidate_score(mask) for instance_id, mask in candidates.items()}
    if isinstance(candidates, CandidateMask):
        return {int(candidates.instance_id): _select_candidate_score(candidates.masks, candidates.iou_predictions)}
    if isinstance(candidates, (list, tuple)):
        return _candidate_sequence_to_dict(candidates, metadata_ids)

    array = _to_numpy(candidates)
    if array.ndim == 2:
        if _looks_like_label_map(array):
            return {int(instance_id): (array == instance_id).astype(np.float32) for instance_id in np.unique(array) if int(instance_id) != 0}
        instance_id = int(metadata_ids[0]) if len(metadata_ids) == 1 else 1
        return {instance_id: array.astype(np.float32)}
    if array.ndim == 3:
        ids = list(metadata_ids) if len(metadata_ids) == array.shape[0] else list(range(1, array.shape[0] + 1))
        return {int(instance_id): array[index].astype(np.float32) for index, instance_id in enumerate(ids)}
    if array.ndim == 4:
        ids = list(metadata_ids) if len(metadata_ids) == array.shape[0] else list(range(1, array.shape[0] + 1))
        return {int(instance_id): _select_candidate_score(array[index]) for index, instance_id in enumerate(ids)}
    raise ValueError(f"Unsupported candidate mask shape: {array.shape}.")


def _candidate_sequence_to_dict(candidates: Sequence, metadata_ids: Sequence[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, CandidateMask):
            out[int(candidate.instance_id)] = _select_candidate_score(candidate.masks, candidate.iou_predictions)
        elif isinstance(candidate, Mapping):
            instance_id = int(candidate.get("instance_id", metadata_ids[index] if index < len(metadata_ids) else index + 1))
            mask = candidate.get("mask", candidate.get("masks"))
            out[instance_id] = _select_candidate_score(mask, candidate.get("iou_predictions"))
        else:
            instance_id = int(metadata_ids[index]) if index < len(metadata_ids) else index + 1
            out[instance_id] = _select_candidate_score(candidate)
    return out


def _select_candidate_score(mask, iou_predictions=None) -> np.ndarray:
    score = _to_numpy(mask).astype(np.float32)
    if score.ndim == 2:
        return score
    if score.ndim == 3:
        if score.shape[0] == 1:
            return score[0]
        iou = None if iou_predictions is None else _to_numpy(iou_predictions).reshape(-1)
        if iou is not None and len(iou) == score.shape[0]:
            return score[int(np.argmax(iou))]
        means = score.reshape(score.shape[0], -1).mean(axis=1)
        return score[int(np.argmax(means))]
    if score.ndim == 4 and score.shape[0] == 1:
        return _select_candidate_score(score[0], iou_predictions)
    raise ValueError(f"Expected candidate score [H,W] or [C,H,W], got shape {score.shape}.")


def _looks_like_label_map(array: np.ndarray) -> bool:
    if not np.issubdtype(array.dtype, np.integer):
        return False
    unique = np.unique(array)
    return len(unique) > 2 or (len(unique) == 2 and 0 in unique and unique.max() > 1)


def _metadata_by_instance_id(prompt_metadata) -> dict[int, dict[str, object]]:
    if prompt_metadata is None:
        return {}
    if isinstance(prompt_metadata, PromptSynthesisResult):
        return {
            int(package.instance_id): {"quality_prior": float(package.quality_prior)}
            for package in prompt_metadata.packages
        }
    if isinstance(prompt_metadata, pd.DataFrame):
        rows = prompt_metadata.to_dict(orient="records")
    else:
        rows = list(prompt_metadata)

    metadata: dict[int, dict[str, object]] = {}
    for row in rows:
        if isinstance(row, PromptPackage):
            metadata[int(row.instance_id)] = {"quality_prior": float(row.quality_prior)}
        elif isinstance(row, Mapping):
            instance_id = int(row.get("instance_id", row.get("fiber_id", row.get("pair_id"))))
            metadata[instance_id] = dict(row)
        else:
            raise TypeError(f"Unsupported prompt metadata row type: {type(row)!r}.")
    return metadata


def _ordered_instance_ids(
    metadata_ids: Sequence[int],
    *candidate_dicts: dict[int, np.ndarray],
) -> list[int]:
    seen = set(int(instance_id) for instance_id in metadata_ids)
    for candidate_dict in candidate_dicts:
        seen.update(int(instance_id) for instance_id in candidate_dict)
    return sorted(seen)


def _infer_shape(*candidate_dicts: dict[int, np.ndarray]) -> tuple[int, int]:
    for candidate_dict in candidate_dicts:
        for score in candidate_dict.values():
            if score.ndim != 2:
                raise ValueError(f"Expected selected candidate score to be 2D, got shape {score.shape}.")
            return int(score.shape[0]), int(score.shape[1])
    raise ValueError("At least one candidate mask is required to infer output shape.")


def _score_or_zeros(candidate_dict: dict[int, np.ndarray], instance_id: int, shape: tuple[int, int]) -> np.ndarray:
    score = candidate_dict.get(instance_id)
    if score is None:
        return np.zeros(shape, dtype=np.float32)
    return score.astype(np.float32)


def _validate_shape(instance_id: int, score: np.ndarray, shape: tuple[int, int]) -> None:
    if score.shape != shape:
        raise ValueError(f"Candidate {instance_id} has shape {score.shape}, expected {shape}.")


def _remove_axon_myelin_overlap(
    *,
    axon_mask: np.ndarray,
    myelin_mask: np.ndarray,
    axon_score: np.ndarray,
    myelin_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axon_mask = axon_mask.copy()
    myelin_mask = myelin_mask.copy()
    overlap = axon_mask & myelin_mask
    if not overlap.any():
        return axon_mask, myelin_mask
    axon_wins = overlap & (axon_score >= myelin_score)
    myelin_wins = overlap & ~axon_wins
    axon_mask[myelin_wins] = False
    myelin_mask[axon_wins] = False
    return axon_mask, myelin_mask
