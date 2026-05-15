import pytest


torch = pytest.importorskip("torch")

from ma_sp_sam.models.self_prompt import SelfPromptGenerator, SelfPromptOutput


def test_self_prompt_generator_outputs_expected_shapes_with_upsampling():
    model = SelfPromptGenerator(
        in_channels=256,
        hidden_channels=32,
        num_classes=3,
        distance_channels=2,
        num_blocks=2,
    )
    image_embedding = torch.randn(2, 256, 16, 16)

    output = model(image_embedding, output_size=(64, 64))

    assert isinstance(output, SelfPromptOutput)
    assert output.semantic_logits.shape == (2, 3, 64, 64)
    assert output.axon_center_heatmap.shape == (2, 1, 64, 64)
    assert output.inner_boundary_map.shape == (2, 1, 64, 64)
    assert output.outer_boundary_map.shape == (2, 1, 64, 64)
    assert output.distance_map.shape == (2, 2, 64, 64)
    assert output.prompt_quality_score.shape == (2,)


def test_self_prompt_generator_preserves_embedding_resolution_by_default_and_backprops():
    model = SelfPromptGenerator(in_channels=64, hidden_channels=16, num_classes=3)
    image_embedding = torch.randn(1, 64, 8, 10, requires_grad=True)

    output = model(image_embedding)
    loss = (
        output.semantic_logits.mean()
        + output.axon_center_heatmap.mean()
        + output.inner_boundary_map.mean()
        + output.outer_boundary_map.mean()
        + output.distance_map.mean()
        + output.prompt_quality_score.mean()
    )
    loss.backward()

    assert output.semantic_logits.shape[-2:] == (8, 10)
    assert output.axon_center_heatmap.shape[-2:] == (8, 10)
    assert image_embedding.grad is not None
    assert torch.isfinite(image_embedding.grad).all()


def test_self_prompt_generator_rejects_non_4d_embedding():
    model = SelfPromptGenerator(in_channels=64)

    with pytest.raises(ValueError, match="4D"):
        model(torch.randn(64, 16, 16))
