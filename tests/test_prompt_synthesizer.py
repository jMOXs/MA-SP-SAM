import numpy as np

from ma_sp_sam.prompts.prompt_synthesizer import InstanceProposal, PromptSynthesizer, synthesize_prompts


def _semantic_logits_from_labels(labels: np.ndarray) -> np.ndarray:
    logits = np.full((3, *labels.shape), -4.0, dtype=np.float32)
    for class_id in range(3):
        logits[class_id, labels == class_id] = 4.0
    return logits


def test_synthesizes_prompt_packages_from_label_map_proposals():
    proposals = np.zeros((8, 9), dtype=np.uint16)
    proposals[1:4, 1:4] = 7
    proposals[4:7, 5:8] = 9

    semantic = np.zeros_like(proposals, dtype=np.uint8)
    semantic[1:4, 1:4] = 1
    semantic[2, 2] = 2
    semantic[4:7, 5:8] = 1
    semantic[5, 6] = 2

    center_heatmap = np.zeros_like(proposals, dtype=np.float32)
    center_heatmap[2, 2] = 0.95
    center_heatmap[5, 6] = 0.85

    inner_boundary = np.zeros_like(proposals, dtype=np.float32)
    outer_boundary = np.zeros_like(proposals, dtype=np.float32)
    inner_boundary[2, 1:4] = 1
    inner_boundary[5, 5:8] = 1
    outer_boundary[1, 1:4] = 1
    outer_boundary[6, 5:8] = 1

    result = synthesize_prompts(
        center_heatmap=center_heatmap,
        semantic_logits=_semantic_logits_from_labels(semantic),
        boundary_maps={"inner": inner_boundary, "outer": outer_boundary},
        instance_proposals=proposals,
        max_negative_points=2,
        ring_points_per_boundary=3,
    )

    assert [package.instance_id for package in result.packages] == [7, 9]
    assert [(point.instance_id, point.xy) for point in result.positive_points] == [
        (7, (2.0, 2.0)),
        (9, (6.0, 5.0)),
    ]
    assert [(box.instance_id, box.xyxy) for box in result.box_prompts] == [
        (7, (1, 1, 3, 3)),
        (9, (5, 4, 7, 6)),
    ]
    assert all(mask.mask.dtype == np.uint8 for mask in result.coarse_mask_prompts)
    assert all(mask.mask.sum() > 0 for mask in result.coarse_mask_prompts)
    assert all(ring.inner_points.shape[1] == 2 for ring in result.ring_prompts)
    assert all(ring.outer_points.shape[1] == 2 for ring in result.ring_prompts)
    assert all(len(package.negative_points) >= 1 for package in result.packages)


def test_every_prompt_retains_target_instance_id():
    proposals = np.zeros((5, 6), dtype=np.uint16)
    proposals[1:4, 2:5] = 3
    center_heatmap = np.zeros_like(proposals, dtype=np.float32)
    semantic = np.zeros_like(proposals, dtype=np.uint8)
    semantic[proposals == 3] = 2

    result = PromptSynthesizer(max_negative_points=1).synthesize(
        center_heatmap=center_heatmap,
        semantic_logits=semantic,
        boundary_maps=np.zeros((2, *proposals.shape), dtype=np.float32),
        instance_proposals=proposals,
    )

    assert [point.instance_id for point in result.positive_points] == [3]
    assert [point.instance_id for point in result.negative_points] == [3]
    assert [box.instance_id for box in result.box_prompts] == [3]
    assert [mask.instance_id for mask in result.coarse_mask_prompts] == [3]
    assert [ring.instance_id for ring in result.ring_prompts] == [3]
    package = result.packages[0]
    assert package.instance_id == 3
    assert package.box_prompt.instance_id == 3
    assert package.coarse_mask_prompt.instance_id == 3
    assert package.ring_prompt.instance_id == 3


def test_accepts_explicit_instance_proposal_objects_and_torch_tensors():
    import pytest

    torch = pytest.importorskip("torch")

    mask = np.zeros((6, 6), dtype=bool)
    mask[2:5, 1:4] = True
    proposal = InstanceProposal(instance_id=42, mask=mask)

    center_heatmap = torch.zeros(1, 6, 6)
    center_heatmap[0, 3, 2] = 1
    semantic_logits = torch.zeros(3, 6, 6)
    semantic_logits[2, mask] = 3
    boundary_maps = torch.zeros(2, 6, 6)
    boundary_maps[0, 3, 1:4] = 1
    boundary_maps[1, 2, 1:4] = 1

    result = PromptSynthesizer(ring_points_per_boundary=2).synthesize(
        center_heatmap=center_heatmap,
        semantic_logits=semantic_logits,
        boundary_maps=boundary_maps,
        instance_proposals=[proposal],
    )

    assert result.packages[0].instance_id == 42
    assert result.positive_points[0].xy == (2.0, 3.0)
    assert result.box_prompts[0].xyxy == (1, 2, 3, 4)
    assert result.ring_prompts[0].inner_points.shape[0] <= 2
    assert result.ring_prompts[0].outer_points.shape[0] <= 2
