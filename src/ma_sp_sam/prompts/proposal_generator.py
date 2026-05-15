from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from ma_sp_sam.prompts.prompt_synthesizer import InstanceProposal


@dataclass(frozen=True)
class ProposalBatch:
    proposals: list[list[InstanceProposal]]
    label_maps: list[np.ndarray]

    def for_image(self, index: int = 0) -> list[InstanceProposal]:
        return self.proposals[index]


class ProposalGenerator:
    """Generate instance proposals from self-prompt dense outputs."""

    def __init__(
        self,
        *,
        center_threshold: float = 0.5,
        boundary_threshold: float = 0.5,
        foreground_classes: Sequence[int] = (1, 2),
        min_area: int = 8,
        max_proposals: int | None = None,
        use_boundaries: bool = True,
    ) -> None:
        if min_area <= 0:
            raise ValueError("min_area must be positive.")
        if max_proposals is not None and max_proposals <= 0:
            raise ValueError("max_proposals must be positive when set.")
        self.center_threshold = float(center_threshold)
        self.boundary_threshold = float(boundary_threshold)
        self.foreground_classes = tuple(int(value) for value in foreground_classes)
        self.min_area = int(min_area)
        self.max_proposals = max_proposals
        self.use_boundaries = bool(use_boundaries)

    def generate(self, *, semantic_logits, center_heatmap, boundary_maps) -> ProposalBatch:
        semantic = _semantic_prediction_batched(_to_numpy(semantic_logits))
        centers = _center_batched(_to_numpy(center_heatmap), semantic.shape)
        boundaries = _boundary_batched(_to_numpy(boundary_maps), semantic.shape)

        batch_proposals: list[list[InstanceProposal]] = []
        batch_label_maps: list[np.ndarray] = []
        for labels, center, boundary in zip(semantic, centers, boundaries, strict=True):
            proposals, label_map = self._generate_one(labels=labels, center=center, boundary=boundary)
            batch_proposals.append(proposals)
            batch_label_maps.append(label_map)
        return ProposalBatch(proposals=batch_proposals, label_maps=batch_label_maps)

    def _generate_one(
        self,
        *,
        labels: np.ndarray,
        center: np.ndarray,
        boundary: np.ndarray,
    ) -> tuple[list[InstanceProposal], np.ndarray]:
        foreground = np.isin(labels, self.foreground_classes)
        if not foreground.any():
            return [], np.zeros_like(labels, dtype=np.uint16)

        barrier = np.zeros_like(foreground, dtype=bool)
        if self.use_boundaries and boundary.size:
            barrier = np.any(boundary > self.boundary_threshold, axis=0)
        peaks = _center_peaks(center, threshold=self.center_threshold, foreground=foreground)
        if not peaks:
            return [], np.zeros_like(labels, dtype=np.uint16)

        components = measure.label(foreground, connectivity=1)

        proposals: list[InstanceProposal] = []
        label_map = np.zeros_like(labels, dtype=np.uint16)
        next_instance_id = 1
        for component_id in range(1, int(components.max()) + 1):
            component_mask = components == component_id
            component_peaks = [peak for peak in peaks if component_mask[peak]]
            if not component_peaks:
                continue
            split_masks = _split_component_by_nearest_peak(component_mask, component_peaks, barrier=barrier)
            for split_mask, peak in split_masks:
                if int(split_mask.sum()) < self.min_area:
                    continue
                proposal = InstanceProposal(
                    instance_id=next_instance_id,
                    mask=split_mask,
                    score=float(center[peak[0], peak[1]]),
                )
                proposals.append(proposal)
                label_map[split_mask] = next_instance_id
                next_instance_id += 1
                if self.max_proposals is not None and len(proposals) >= self.max_proposals:
                    return proposals, label_map

        return proposals, label_map


def generate_instance_proposals(
    *,
    semantic_logits,
    center_heatmap,
    boundary_maps,
    center_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    foreground_classes: Sequence[int] = (1, 2),
    min_area: int = 8,
    max_proposals: int | None = None,
    use_boundaries: bool = True,
    image_index: int = 0,
) -> list[InstanceProposal]:
    batch = ProposalGenerator(
        center_threshold=center_threshold,
        boundary_threshold=boundary_threshold,
        foreground_classes=foreground_classes,
        min_area=min_area,
        max_proposals=max_proposals,
        use_boundaries=use_boundaries,
    ).generate(
        semantic_logits=semantic_logits,
        center_heatmap=center_heatmap,
        boundary_maps=boundary_maps,
    )
    return batch.for_image(image_index)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _semantic_prediction_batched(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array.astype(np.uint8)[None, ...]
    if array.ndim == 3:
        if array.shape[0] == 3:
            return array.argmax(axis=0).astype(np.uint8)[None, ...]
        return array.astype(np.uint8)
    if array.ndim == 4:
        return array.argmax(axis=1).astype(np.uint8)
    raise ValueError(f"Expected semantic logits [B,3,H,W], [3,H,W], or labels, got shape {array.shape}.")


def _center_batched(array: np.ndarray, semantic_shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    batch, height, width = semantic_shape
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0][None, ...]
    elif array.ndim == 3:
        pass
    elif array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    else:
        raise ValueError(f"Expected center heatmap [B,1,H,W], [1,H,W], or [H,W], got shape {array.shape}.")
    if array.shape != (batch, height, width):
        raise ValueError(f"Center heatmap shape {array.shape} does not match semantic shape {(batch, height, width)}.")
    return array


def _boundary_batched(array: np.ndarray, semantic_shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    batch, height, width = semantic_shape
    if array.ndim == 3 and array.shape[0] >= 2:
        array = array[None, :2]
    elif array.ndim == 4 and array.shape[1] >= 2:
        array = array[:, :2]
    else:
        raise ValueError(f"Expected boundary maps [B,2,H,W] or [2,H,W], got shape {array.shape}.")
    if array.shape != (batch, 2, height, width):
        raise ValueError(f"Boundary map shape {array.shape} does not match semantic shape {(batch, 2, height, width)}.")
    return array


def _center_peaks(
    center: np.ndarray,
    *,
    threshold: float,
    foreground: np.ndarray,
) -> list[tuple[int, int]]:
    peak_mask = (center >= threshold) & foreground
    peak_labels = measure.label(peak_mask, connectivity=1)
    peaks: list[tuple[int, int]] = []
    for peak_id in range(1, int(peak_labels.max()) + 1):
        region = peak_labels == peak_id
        if not region.any():
            continue
        scores = np.where(region, center, -np.inf)
        y, x = np.unravel_index(int(np.argmax(scores)), center.shape)
        peaks.append((int(y), int(x)))
    peaks.sort(key=lambda yx: (yx[0], yx[1]))
    return peaks


def _split_component_by_nearest_peak(
    component_mask: np.ndarray,
    peaks: list[tuple[int, int]],
    *,
    barrier: np.ndarray | None = None,
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    if len(peaks) == 1:
        return [(component_mask, peaks[0])]
    if barrier is not None and barrier.any():
        seeded = _boundary_seeded_split(component_mask, peaks, barrier=barrier)
        if seeded is not None:
            return seeded

    ys, xs = np.where(component_mask)
    peak_array = np.asarray(peaks, dtype=np.int64)
    distances = (ys[:, None] - peak_array[None, :, 0]) ** 2 + (xs[:, None] - peak_array[None, :, 1]) ** 2
    assignments = distances.argmin(axis=1)

    masks: list[tuple[np.ndarray, tuple[int, int]]] = []
    for index, peak in enumerate(peaks):
        split_mask = np.zeros_like(component_mask, dtype=bool)
        owned = assignments == index
        split_mask[ys[owned], xs[owned]] = True
        masks.append((split_mask, peak))
    return masks


def _boundary_seeded_split(
    component_mask: np.ndarray,
    peaks: list[tuple[int, int]],
    *,
    barrier: np.ndarray,
) -> list[tuple[np.ndarray, tuple[int, int]]] | None:
    seed_source = component_mask & ~barrier
    for peak in peaks:
        if component_mask[peak]:
            seed_source[peak] = True
    if not seed_source.any():
        return None

    seed_splits = _split_component_by_nearest_peak(seed_source, peaks, barrier=None)
    seed_labels = np.zeros_like(component_mask, dtype=np.uint16)
    for label, (seed_mask, _) in enumerate(seed_splits, start=1):
        seed_labels[seed_mask] = label
    if not seed_labels.any():
        return None

    _, nearest_indices = ndi.distance_transform_edt(seed_labels == 0, return_indices=True)
    backfilled = seed_labels[tuple(nearest_indices)]
    backfilled = np.where(component_mask, backfilled, 0)

    masks: list[tuple[np.ndarray, tuple[int, int]]] = []
    for label, peak in enumerate(peaks, start=1):
        split_mask = backfilled == label
        if split_mask.any():
            masks.append((split_mask, peak))
    return masks
