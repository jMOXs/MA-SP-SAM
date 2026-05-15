"""Self-prompt prediction heads for SAM image embeddings.

The module is intentionally independent from SAM's prompt encoder and mask
decoder. It consumes an image embedding tensor and predicts dense prompt
supervision maps that later modules can turn into point, box, mask, or ring
prompts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SelfPromptOutput:
    """Container for SelfPromptGenerator forward outputs."""

    semantic_logits: torch.Tensor
    axon_center_heatmap: torch.Tensor
    inner_boundary_map: torch.Tensor
    outer_boundary_map: torch.Tensor
    distance_map: torch.Tensor
    prompt_quality_score: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "semantic_logits": self.semantic_logits,
            "axon_center_heatmap": self.axon_center_heatmap,
            "inner_boundary_map": self.inner_boundary_map,
            "outer_boundary_map": self.outer_boundary_map,
            "distance_map": self.distance_map,
            "prompt_quality_score": self.prompt_quality_score,
        }


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_choose_group_count(out_channels, norm_groups), out_channels),
            nn.GELU(),
        )


class DensePredictionHead(nn.Sequential):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, norm_groups: int) -> None:
        super().__init__(
            ConvNormAct(in_channels, hidden_channels, norm_groups),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )


class SelfPromptGenerator(nn.Module):
    """Predict self-prompt supervision maps from a SAM image embedding.

    Parameters
    ----------
    in_channels:
        Channel count of the incoming SAM / micro-SAM image embedding.
    hidden_channels:
        Shared feature width used by the lightweight prediction trunk.
    num_classes:
        Semantic classes. The project convention is 3: background, myelin,
        axon.
    distance_channels:
        Number of distance/radial channels. Use 2 for HoVer-style dx/dy, or K
        for K-angle radial distances.
    num_blocks:
        Number of shared ConvNormAct blocks after the input projection.
    norm_groups:
        Preferred GroupNorm groups. The implementation automatically picks a
        divisor of the channel count.
    dropout:
        Dropout applied before the dense prompt-quality head.
    quality_hidden_channels:
        Kept for backward-compatible construction. The dense V1 quality head
        uses ``hidden_channels`` so it can return ``[B,1,H,W]`` logits.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_classes: int = 3,
        distance_channels: int = 2,
        num_blocks: int = 2,
        norm_groups: int = 8,
        dropout: float = 0.0,
        quality_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if distance_channels <= 0:
            raise ValueError("distance_channels must be positive.")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative.")

        del quality_hidden_channels

        blocks: list[nn.Module] = [ConvNormAct(in_channels, hidden_channels, norm_groups)]
        blocks.extend(ConvNormAct(hidden_channels, hidden_channels, norm_groups) for _ in range(num_blocks))
        self.trunk = nn.Sequential(*blocks)

        self.semantic_head = DensePredictionHead(hidden_channels, hidden_channels, num_classes, norm_groups)
        self.center_head = DensePredictionHead(hidden_channels, hidden_channels, 1, norm_groups)
        self.inner_boundary_head = DensePredictionHead(hidden_channels, hidden_channels, 1, norm_groups)
        self.outer_boundary_head = DensePredictionHead(hidden_channels, hidden_channels, 1, norm_groups)
        self.distance_head = DensePredictionHead(hidden_channels, hidden_channels, distance_channels, norm_groups)

        self.quality_head = nn.Sequential(
            nn.Dropout2d(dropout),
            DensePredictionHead(hidden_channels, hidden_channels, 1, norm_groups),
        )

    def forward(self, image_embedding: torch.Tensor, output_size: tuple[int, int] | None = None) -> SelfPromptOutput:
        """Run the self-prompt heads.

        ``image_embedding`` must be ``[B, C, H, W]``. Dense maps are returned at
        embedding resolution unless ``output_size`` is provided, in which case
        bilinear interpolation resizes them to ``[output_size[0], output_size[1]]``.
        """

        if image_embedding.ndim != 4:
            raise ValueError(
                "SelfPromptGenerator expects a 4D image embedding tensor "
                f"[B, C, H, W], got shape {tuple(image_embedding.shape)}."
            )

        features = self.trunk(image_embedding)
        semantic_logits = self.semantic_head(features)
        axon_center_heatmap = self.center_head(features)
        inner_boundary_map = self.inner_boundary_head(features)
        outer_boundary_map = self.outer_boundary_head(features)
        distance_map = self.distance_head(features)
        prompt_quality_score = self.quality_head(features)

        if output_size is not None:
            semantic_logits = _resize_dense_map(semantic_logits, output_size)
            axon_center_heatmap = _resize_dense_map(axon_center_heatmap, output_size)
            inner_boundary_map = _resize_dense_map(inner_boundary_map, output_size)
            outer_boundary_map = _resize_dense_map(outer_boundary_map, output_size)
            distance_map = _resize_dense_map(distance_map, output_size)
            prompt_quality_score = _resize_dense_map(prompt_quality_score, output_size)

        return SelfPromptOutput(
            semantic_logits=semantic_logits,
            axon_center_heatmap=axon_center_heatmap,
            inner_boundary_map=inner_boundary_map,
            outer_boundary_map=outer_boundary_map,
            distance_map=distance_map,
            prompt_quality_score=prompt_quality_score,
        )


def _choose_group_count(channels: int, preferred_groups: int) -> int:
    max_groups = max(1, min(preferred_groups, channels))
    for groups in range(max_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _resize_dense_map(tensor: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    if len(output_size) != 2:
        raise ValueError("output_size must be a tuple of (height, width).")
    if output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("output_size values must be positive.")
    return F.interpolate(tensor, size=output_size, mode="bilinear", align_corners=False)
