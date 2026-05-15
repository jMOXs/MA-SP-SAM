import pytest


torch = pytest.importorskip("torch")

from ma_sp_sam.models.self_prompt_generator import SelfPromptGenerator


def test_self_prompt_generator_outputs_dense_heads_at_input_resolution():
    model = SelfPromptGenerator(in_channels=4, hidden_channels=8, num_blocks=1)
    image = torch.randn(2, 4, 11, 13, requires_grad=True)

    output = model(image)

    assert output.semantic_logits.shape == (2, 3, 11, 13)
    assert output.center_heatmap.shape == (2, 1, 11, 13)
    assert output.boundary_maps.shape == (2, 2, 11, 13)
    assert output.distance_map.shape == (2, 2, 11, 13)
    assert output.quality_logits.shape == (2, 1, 11, 13)

    loss = sum(tensor.mean() for tensor in output.as_dict().values())
    loss.backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_self_prompt_generator_rejects_non_image_tensor():
    model = SelfPromptGenerator(in_channels=4, hidden_channels=8, num_blocks=1)

    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        model(torch.randn(4, 11, 13))
