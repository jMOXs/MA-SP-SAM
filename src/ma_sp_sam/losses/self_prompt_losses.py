from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ma_sp_sam.models.self_prompt import SelfPromptOutput


def dice_loss_from_logits(
    semantic_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if semantic_logits.ndim != 4:
        raise ValueError("semantic_logits must be [B,C,H,W].")
    if target.ndim != 3:
        raise ValueError("target semantic must be [B,H,W].")
    num_classes = num_classes or semantic_logits.shape[1]
    probabilities = semantic_logits.softmax(dim=1)
    target_one_hot = F.one_hot(target.clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = target_one_hot[:, : probabilities.shape[1]]
    dims = (0, 2, 3)
    intersection = (probabilities * target_one_hot).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + target_one_hot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def semantic_loss(semantic_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(semantic_logits, target.long()) + dice_loss_from_logits(semantic_logits, target.long())


@dataclass(frozen=True)
class SelfPromptLossWeights:
    semantic: float = 1.0
    center: float = 1.0
    boundary: float = 1.0
    distance: float = 1.0


class SelfPromptLoss(nn.Module):
    def __init__(
        self,
        *,
        semantic_weight: float = 1.0,
        center_weight: float = 1.0,
        boundary_weight: float = 1.0,
        distance_weight: float = 1.0,
        center_loss: str = "bce",
    ) -> None:
        super().__init__()
        if center_loss not in {"bce", "mse"}:
            raise ValueError("center_loss must be 'bce' or 'mse'.")
        self.weights = SelfPromptLossWeights(
            semantic=semantic_weight,
            center=center_weight,
            boundary=boundary_weight,
            distance=distance_weight,
        )
        self.center_loss = center_loss
        self.boundary_loss_fn = nn.BCEWithLogitsLoss()
        self.distance_loss_fn = nn.SmoothL1Loss()

    def forward(self, outputs: SelfPromptOutput, targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        semantic_target = targets["semantic"].long()
        center_target = _ensure_channel(targets["center_heatmap"]).float()
        inner_target = _ensure_channel(targets["boundary_inner"]).float()
        outer_target = _ensure_channel(targets["boundary_outer"]).float()
        distance_target = targets["distance_map"].float()
        if distance_target.ndim == 3:
            distance_target = distance_target.unsqueeze(0)

        sem = semantic_loss(outputs.semantic_logits, semantic_target)
        if self.center_loss == "bce":
            center = F.binary_cross_entropy_with_logits(outputs.axon_center_heatmap, center_target)
        else:
            center = F.mse_loss(torch.sigmoid(outputs.axon_center_heatmap), center_target)
        boundary = 0.5 * (
            self.boundary_loss_fn(outputs.inner_boundary_map, inner_target)
            + self.boundary_loss_fn(outputs.outer_boundary_map, outer_target)
        )
        distance = _foreground_smooth_l1(outputs.distance_map, distance_target, semantic_target, self.distance_loss_fn)

        total = (
            self.weights.semantic * sem
            + self.weights.center * center
            + self.weights.boundary * boundary
            + self.weights.distance * distance
        )
        return total, {
            "semantic_loss": sem.detach(),
            "center_loss": center.detach(),
            "boundary_loss": boundary.detach(),
            "distance_loss": distance.detach(),
            "total_loss": total.detach(),
        }


def _ensure_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Expected target map [B,H,W] or [B,1,H,W], got shape {tuple(tensor.shape)}.")


def _foreground_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    semantic_target: torch.Tensor,
    loss_fn: nn.SmoothL1Loss,
) -> torch.Tensor:
    foreground = semantic_target > 0
    if not foreground.any():
        return prediction.sum() * 0.0
    mask = foreground.unsqueeze(1).expand_as(prediction)
    return loss_fn(prediction[mask], target[mask])
