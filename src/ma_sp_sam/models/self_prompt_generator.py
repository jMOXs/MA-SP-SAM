from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SelfPromptGeneratorOutput:
    semantic_logits: torch.Tensor
    center_heatmap: torch.Tensor
    boundary_maps: torch.Tensor
    distance_map: torch.Tensor
    quality_logits: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "semantic_logits": self.semantic_logits,
            "center_heatmap": self.center_heatmap,
            "boundary_maps": self.boundary_maps,
            "distance_map": self.distance_map,
            "quality_logits": self.quality_logits,
        }


class _ConvBlock(nn.Module):
    def __init__(self, channels: int, *, norm_groups: int) -> None:
        super().__init__()
        groups = _valid_group_count(channels, norm_groups)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _PredictionHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class SelfPromptGenerator(nn.Module):
    """Dense self-prompt heads over an image-like feature tensor."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int = 64,
        num_blocks: int = 3,
        norm_groups: int = 8,
        num_semantic_classes: int = 3,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative.")
        if num_semantic_classes <= 0:
            raise ValueError("num_semantic_classes must be positive.")

        groups = _valid_group_count(hidden_channels, norm_groups)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(*[_ConvBlock(hidden_channels, norm_groups=norm_groups) for _ in range(num_blocks)])
        self.semantic_head = _PredictionHead(hidden_channels, num_semantic_classes)
        self.center_head = _PredictionHead(hidden_channels, 1)
        self.boundary_head = _PredictionHead(hidden_channels, 2)
        self.distance_head = _PredictionHead(hidden_channels, 2)
        self.quality_head = _PredictionHead(hidden_channels, 1)

    def forward(self, image: torch.Tensor) -> SelfPromptGeneratorOutput:
        if image.ndim != 4:
            raise ValueError(f"SelfPromptGenerator expects image tensor [B, C, H, W], got shape {tuple(image.shape)}.")
        features = self.trunk(self.stem(image))
        return SelfPromptGeneratorOutput(
            semantic_logits=self.semantic_head(features),
            center_heatmap=self.center_head(features),
            boundary_maps=self.boundary_head(features),
            distance_map=self.distance_head(features),
            quality_logits=self.quality_head(features),
        )


def _valid_group_count(channels: int, requested_groups: int) -> int:
    if requested_groups <= 0:
        return 1
    return max(group for group in range(min(channels, requested_groups), 0, -1) if channels % group == 0)
