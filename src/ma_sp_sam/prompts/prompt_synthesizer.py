from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from skimage.segmentation import find_boundaries


@dataclass(frozen=True)
class InstanceProposal:
    instance_id: int
    mask: np.ndarray
    score: float | None = None


@dataclass(frozen=True)
class PointPrompt:
    instance_id: int
    xy: tuple[float, float]
    label: int
    source: str
    score: float = 1.0
    reference_instance_id: int | None = None


@dataclass(frozen=True)
class BoxPrompt:
    instance_id: int
    xyxy: tuple[int, int, int, int]
    source: str = "instance_proposal"


@dataclass(frozen=True)
class CoarseMaskPrompt:
    instance_id: int
    mask: np.ndarray
    source: str = "semantic_foreground_in_proposal"


@dataclass(frozen=True)
class RingPrompt:
    instance_id: int
    inner_points: np.ndarray
    outer_points: np.ndarray
    source: str = "boundary_maps"

    @property
    def ring_points(self) -> np.ndarray:
        if self.inner_points.size == 0:
            return self.outer_points
        if self.outer_points.size == 0:
            return self.inner_points
        return np.concatenate([self.inner_points, self.outer_points], axis=0)


@dataclass(frozen=True)
class PromptPackage:
    instance_id: int
    positive_points: list[PointPrompt]
    negative_points: list[PointPrompt]
    box_prompt: BoxPrompt
    coarse_mask_prompt: CoarseMaskPrompt
    ring_prompt: RingPrompt
    quality_prior: float
    source: str = "prompt_synthesizer"


@dataclass(frozen=True)
class PromptSynthesisResult:
    packages: list[PromptPackage]
    positive_points: list[PointPrompt]
    negative_points: list[PointPrompt]
    box_prompts: list[BoxPrompt]
    coarse_mask_prompts: list[CoarseMaskPrompt]
    ring_prompts: list[RingPrompt]


class PromptSynthesizer:
    """Convert self-prompt head outputs and proposals into SAM prompt packages."""

    def __init__(
        self,
        *,
        max_negative_points: int = 4,
        negative_margin: int = 8,
        boundary_threshold: float = 0.0,
        ring_points_per_boundary: int = 16,
    ) -> None:
        if max_negative_points < 0:
            raise ValueError("max_negative_points must be non-negative.")
        if negative_margin < 0:
            raise ValueError("negative_margin must be non-negative.")
        if ring_points_per_boundary < 0:
            raise ValueError("ring_points_per_boundary must be non-negative.")
        self.max_negative_points = max_negative_points
        self.negative_margin = negative_margin
        self.boundary_threshold = boundary_threshold
        self.ring_points_per_boundary = ring_points_per_boundary

    def synthesize(
        self,
        *,
        center_heatmap,
        semantic_logits,
        boundary_maps,
        instance_proposals,
    ) -> PromptSynthesisResult:
        center = _squeeze_singleton_spatial(_to_numpy(center_heatmap)).astype(np.float32)
        semantic = _semantic_prediction(_to_numpy(semantic_logits))
        inner_boundary, outer_boundary = _boundary_pair(boundary_maps, threshold=self.boundary_threshold)
        proposals = _normalize_instance_proposals(instance_proposals)

        _validate_same_shape(center, semantic, inner_boundary, outer_boundary)
        proposals = [proposal for proposal in proposals if _proposal_mask(proposal, center.shape).any()]

        centers = {
            proposal.instance_id: _positive_point_for_proposal(center, _proposal_mask(proposal, center.shape))
            for proposal in proposals
        }

        packages: list[PromptPackage] = []
        for proposal in proposals:
            mask = _proposal_mask(proposal, center.shape)
            positive_point = _point_prompt_from_center(proposal.instance_id, centers[proposal.instance_id])
            negative_points = self._negative_points_for_proposal(
                proposal=proposal,
                mask=mask,
                semantic=semantic,
                centers=centers,
            )
            box_prompt = BoxPrompt(instance_id=proposal.instance_id, xyxy=_bbox_xyxy(mask))
            coarse_mask_prompt = CoarseMaskPrompt(
                instance_id=proposal.instance_id,
                mask=_coarse_mask(mask, semantic),
            )
            ring_prompt = RingPrompt(
                instance_id=proposal.instance_id,
                inner_points=_sample_boundary_points(inner_boundary & mask, positive_point.xy, self.ring_points_per_boundary),
                outer_points=_sample_boundary_points(_outer_boundary_candidates(outer_boundary, mask), positive_point.xy, self.ring_points_per_boundary),
            )
            packages.append(
                PromptPackage(
                    instance_id=proposal.instance_id,
                    positive_points=[positive_point],
                    negative_points=negative_points,
                    box_prompt=box_prompt,
                    coarse_mask_prompt=coarse_mask_prompt,
                    ring_prompt=ring_prompt,
                    quality_prior=_quality_prior(proposal, positive_point),
                )
            )

        return PromptSynthesisResult(
            packages=packages,
            positive_points=[point for package in packages for point in package.positive_points],
            negative_points=[point for package in packages for point in package.negative_points],
            box_prompts=[package.box_prompt for package in packages],
            coarse_mask_prompts=[package.coarse_mask_prompt for package in packages],
            ring_prompts=[package.ring_prompt for package in packages],
        )

    def _negative_points_for_proposal(
        self,
        *,
        proposal: InstanceProposal,
        mask: np.ndarray,
        semantic: np.ndarray,
        centers: Mapping[int, tuple[float, float, float, str]],
    ) -> list[PointPrompt]:
        if self.max_negative_points == 0:
            return []

        target_x, target_y, _, _ = centers[proposal.instance_id]
        neighbor_points: list[tuple[float, PointPrompt]] = []
        for other_id, (other_x, other_y, other_score, _) in centers.items():
            if other_id == proposal.instance_id:
                continue
            distance = (other_x - target_x) ** 2 + (other_y - target_y) ** 2
            neighbor_points.append(
                (
                    distance,
                    PointPrompt(
                        instance_id=proposal.instance_id,
                        xy=(other_x, other_y),
                        label=0,
                        source="neighbor_center",
                        score=other_score,
                        reference_instance_id=other_id,
                    ),
                )
            )

        neighbor_points.sort(key=lambda item: item[0])
        negatives = [point for _, point in neighbor_points[: self.max_negative_points]]
        if len(negatives) < self.max_negative_points:
            background = _background_negative_point(mask, semantic, (target_x, target_y), self.negative_margin)
            if background is not None:
                negatives.append(
                    PointPrompt(
                        instance_id=proposal.instance_id,
                        xy=background,
                        label=0,
                        source="background",
                        score=1.0,
                    )
                )
        return negatives[: self.max_negative_points]


def synthesize_prompts(
    *,
    center_heatmap,
    semantic_logits,
    boundary_maps,
    instance_proposals,
    max_negative_points: int = 4,
    negative_margin: int = 8,
    boundary_threshold: float = 0.0,
    ring_points_per_boundary: int = 16,
) -> PromptSynthesisResult:
    synthesizer = PromptSynthesizer(
        max_negative_points=max_negative_points,
        negative_margin=negative_margin,
        boundary_threshold=boundary_threshold,
        ring_points_per_boundary=ring_points_per_boundary,
    )
    return synthesizer.synthesize(
        center_heatmap=center_heatmap,
        semantic_logits=semantic_logits,
        boundary_maps=boundary_maps,
        instance_proposals=instance_proposals,
    )


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _squeeze_singleton_spatial(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 4 and array.shape[0] == 1 and array.shape[1] == 1:
        return array[0, 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D map or singleton map, got shape {array.shape}.")
    return array


def _semantic_prediction(semantic_logits: np.ndarray) -> np.ndarray:
    semantic_logits = np.asarray(semantic_logits)
    if semantic_logits.ndim == 4 and semantic_logits.shape[0] == 1:
        semantic_logits = semantic_logits[0]
    if semantic_logits.ndim == 3:
        return semantic_logits.argmax(axis=0).astype(np.uint8)
    if semantic_logits.ndim == 2:
        return semantic_logits.astype(np.uint8)
    raise ValueError(f"Expected semantic logits [C,H,W] or labels [H,W], got shape {semantic_logits.shape}.")


def _boundary_pair(boundary_maps, *, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(boundary_maps, Mapping):
        inner = _to_numpy(boundary_maps["inner"])
        outer = _to_numpy(boundary_maps["outer"])
    else:
        boundary = _to_numpy(boundary_maps)
        if boundary.ndim == 4 and boundary.shape[0] == 1:
            boundary = boundary[0]
        if boundary.ndim != 3 or boundary.shape[0] < 2:
            raise ValueError(f"Expected boundary maps [2,H,W] or dict with inner/outer, got shape {boundary.shape}.")
        inner, outer = boundary[0], boundary[1]
    inner = _squeeze_singleton_spatial(inner)
    outer = _squeeze_singleton_spatial(outer)
    return inner > threshold, outer > threshold


def _normalize_instance_proposals(instance_proposals) -> list[InstanceProposal]:
    if isinstance(instance_proposals, np.ndarray) or hasattr(instance_proposals, "detach"):
        array = _to_numpy(instance_proposals)
        if array.ndim == 2:
            proposals = []
            for instance_id in sorted(int(value) for value in np.unique(array) if int(value) != 0):
                proposals.append(InstanceProposal(instance_id=instance_id, mask=array == instance_id))
            return proposals
        if array.ndim == 3:
            proposals = []
            for index, mask in enumerate(array, start=1):
                proposals.append(InstanceProposal(instance_id=index, mask=mask.astype(bool)))
            return proposals
        raise ValueError(f"Expected instance proposals [H,W] or [N,H,W], got shape {array.shape}.")

    proposals = []
    for proposal in instance_proposals:
        if isinstance(proposal, InstanceProposal):
            proposals.append(proposal)
        elif isinstance(proposal, Mapping):
            proposals.append(
                InstanceProposal(
                    instance_id=int(proposal["instance_id"]),
                    mask=_to_numpy(proposal["mask"]).astype(bool),
                    score=None if proposal.get("score") is None else float(proposal["score"]),
                )
            )
        else:
            raise TypeError(f"Unsupported proposal type: {type(proposal)!r}.")
    proposals.sort(key=lambda proposal: proposal.instance_id)
    return proposals


def _proposal_mask(proposal: InstanceProposal, shape: tuple[int, int]) -> np.ndarray:
    mask = _to_numpy(proposal.mask).astype(bool)
    if mask.shape != shape:
        raise ValueError(f"Proposal {proposal.instance_id} mask shape {mask.shape} does not match {shape}.")
    return mask


def _validate_same_shape(*arrays: np.ndarray) -> None:
    shapes = {np.asarray(array).shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Prompt inputs must share spatial shape, got {sorted(shapes)}.")


def _positive_point_for_proposal(center: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, str]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Cannot synthesize prompts for an empty proposal mask.")

    masked_scores = np.where(mask, center, -np.inf)
    max_index = int(np.argmax(masked_scores))
    y, x = np.unravel_index(max_index, center.shape)
    score = float(center[y, x])
    if np.isfinite(score) and score > 0:
        return float(x), float(y), score, "center_heatmap"

    return float(xs.mean()), float(ys.mean()), 0.0, "proposal_centroid"


def _point_prompt_from_center(instance_id: int, center: tuple[float, float, float, str]) -> PointPrompt:
    x, y, score, source = center
    return PointPrompt(instance_id=instance_id, xy=(x, y), label=1, source=source, score=score)


def _bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _coarse_mask(mask: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    foreground = mask & np.isin(semantic, [1, 2])
    if not foreground.any():
        foreground = mask
    return foreground.astype(np.uint8)


def _outer_boundary_candidates(outer_boundary: np.ndarray, mask: np.ndarray) -> np.ndarray:
    candidates = outer_boundary & mask
    if candidates.any():
        return candidates
    return find_boundaries(mask, mode="outer")


def _sample_boundary_points(boundary_mask: np.ndarray, center_xy: tuple[float, float], limit: int) -> np.ndarray:
    if limit == 0:
        return np.zeros((0, 2), dtype=np.float32)
    ys, xs = np.where(boundary_mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    center_x, center_y = center_xy
    angles = np.arctan2(ys.astype(np.float32) - center_y, xs.astype(np.float32) - center_x)
    order = np.argsort(angles)
    xs = xs[order]
    ys = ys[order]
    if len(xs) > limit:
        pick = np.linspace(0, len(xs) - 1, num=limit, dtype=int)
        xs = xs[pick]
        ys = ys[pick]
    return np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)


def _background_negative_point(
    mask: np.ndarray,
    semantic: np.ndarray,
    center_xy: tuple[float, float],
    margin: int,
) -> tuple[float, float] | None:
    x_min, y_min, x_max, y_max = _bbox_xyxy(mask)
    h, w = mask.shape
    y0 = max(0, y_min - margin)
    y1 = min(h - 1, y_max + margin)
    x0 = max(0, x_min - margin)
    x1 = min(w - 1, x_max + margin)

    roi_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    roi_semantic = semantic[y0 : y1 + 1, x0 : x1 + 1]
    candidate = (~roi_mask) & (roi_semantic == 0)
    if not candidate.any():
        candidate = ~roi_mask
    if not candidate.any():
        return None

    ys, xs = np.where(candidate)
    xs_global = xs + x0
    ys_global = ys + y0
    center_x, center_y = center_xy
    distances = (xs_global - center_x) ** 2 + (ys_global - center_y) ** 2
    index = int(np.argmax(distances))
    return float(xs_global[index]), float(ys_global[index])


def _quality_prior(proposal: InstanceProposal, positive_point: PointPrompt) -> float:
    if proposal.score is not None:
        return float(proposal.score)
    return float(positive_point.score)
