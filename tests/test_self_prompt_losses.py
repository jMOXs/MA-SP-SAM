import pytest


torch = pytest.importorskip("torch")

from ma_sp_sam.losses.self_prompt_losses import SelfPromptLoss
from ma_sp_sam.models.self_prompt import SelfPromptOutput


def _outputs(batch=2, height=6, width=7):
    return SelfPromptOutput(
        semantic_logits=torch.randn(batch, 3, height, width, requires_grad=True),
        axon_center_heatmap=torch.randn(batch, 1, height, width, requires_grad=True),
        inner_boundary_map=torch.randn(batch, 1, height, width, requires_grad=True),
        outer_boundary_map=torch.randn(batch, 1, height, width, requires_grad=True),
        distance_map=torch.randn(batch, 2, height, width, requires_grad=True),
        prompt_quality_score=torch.randn(batch, 1, height, width, requires_grad=True),
    )


def _targets(batch=2, height=6, width=7):
    semantic = torch.zeros(batch, height, width, dtype=torch.long)
    semantic[:, 1:5, 2:6] = 1
    semantic[:, 2:4, 3:5] = 2
    return {
        "semantic": semantic,
        "center_heatmap": torch.zeros(batch, 1, height, width),
        "boundary_inner": torch.zeros(batch, 1, height, width),
        "boundary_outer": torch.zeros(batch, 1, height, width),
        "distance_map": torch.zeros(batch, 2, height, width),
    }


def test_self_prompt_loss_returns_total_and_named_terms_with_gradients():
    outputs = _outputs()
    targets = _targets()
    criterion = SelfPromptLoss()

    total_loss, terms = criterion(outputs, targets)
    total_loss.backward()

    assert total_loss.ndim == 0
    assert {"semantic_loss", "center_loss", "boundary_loss", "distance_loss", "total_loss"} <= set(terms)
    assert torch.isfinite(total_loss)
    assert outputs.semantic_logits.grad is not None
    assert outputs.distance_map.grad is not None


def test_distance_loss_is_finite_when_batch_has_no_foreground():
    outputs = _outputs(batch=1)
    targets = _targets(batch=1)
    targets["semantic"].zero_()

    total_loss, terms = SelfPromptLoss()(outputs, targets)

    assert torch.isfinite(total_loss)
    assert terms["distance_loss"].item() == 0.0
