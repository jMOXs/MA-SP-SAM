import numpy as np

from ma_sp_sam.prompts.proposal_generator import ProposalGenerator, generate_instance_proposals
from ma_sp_sam.prompts.prompt_synthesizer import PromptSynthesizer


def _semantic_logits_from_labels(labels: np.ndarray) -> np.ndarray:
    logits = np.full((1, 3, *labels.shape), -6.0, dtype=np.float32)
    for class_id in range(3):
        logits[0, class_id, labels == class_id] = 6.0
    return logits


def test_proposal_generator_returns_empty_for_empty_maps():
    labels = np.zeros((7, 8), dtype=np.uint8)
    center_heatmap = np.zeros((1, 1, 7, 8), dtype=np.float32)
    boundary_maps = np.zeros((1, 2, 7, 8), dtype=np.float32)

    batch = ProposalGenerator(center_threshold=0.5).generate(
        semantic_logits=_semantic_logits_from_labels(labels),
        center_heatmap=center_heatmap,
        boundary_maps=boundary_maps,
    )

    assert batch.proposals == [[]]
    assert batch.label_maps[0].shape == labels.shape
    assert batch.label_maps[0].max() == 0


def test_proposal_generator_splits_multiple_centers_in_one_foreground_region():
    labels = np.zeros((8, 10), dtype=np.uint8)
    labels[1:6, 1:9] = 1
    labels[2, 2] = 2
    labels[2, 7] = 2
    center_heatmap = np.zeros((1, 1, 8, 10), dtype=np.float32)
    center_heatmap[0, 0, 2, 2] = 0.95
    center_heatmap[0, 0, 2, 7] = 0.9
    boundary_maps = np.zeros((1, 2, 8, 10), dtype=np.float32)

    batch = ProposalGenerator(center_threshold=0.5, min_area=2).generate(
        semantic_logits=_semantic_logits_from_labels(labels),
        center_heatmap=center_heatmap,
        boundary_maps=boundary_maps,
    )
    proposals = batch.proposals[0]

    assert [proposal.instance_id for proposal in proposals] == [1, 2]
    assert len(proposals) == 2
    assert sum(bool(proposal.mask[2, 2]) for proposal in proposals) == 1
    assert sum(bool(proposal.mask[2, 7]) for proposal in proposals) == 1
    assert not np.logical_and(proposals[0].mask, proposals[1].mask).any()
    assert np.logical_or(proposals[0].mask, proposals[1].mask).sum() == np.count_nonzero(labels)


def test_generated_proposals_connect_to_prompt_synthesizer():
    labels = np.zeros((7, 9), dtype=np.uint8)
    labels[1:5, 2:6] = 1
    labels[3, 4] = 2
    center_heatmap = np.zeros((1, 1, 7, 9), dtype=np.float32)
    center_heatmap[0, 0, 3, 4] = 1.0
    boundary_maps = np.zeros((1, 2, 7, 9), dtype=np.float32)
    semantic_logits = _semantic_logits_from_labels(labels)

    proposals = generate_instance_proposals(
        semantic_logits=semantic_logits[0],
        center_heatmap=center_heatmap[0],
        boundary_maps=boundary_maps[0],
        center_threshold=0.5,
    )

    result = PromptSynthesizer(max_negative_points=1).synthesize(
        semantic_logits=semantic_logits[0],
        center_heatmap=center_heatmap[0],
        boundary_maps=boundary_maps[0],
        instance_proposals=proposals,
    )

    assert [package.instance_id for package in result.packages] == [1]
    assert result.positive_points[0].xy == (4.0, 3.0)
