import numpy as np
import pytest


torch = pytest.importorskip("torch")

from ma_sp_sam.refinement.pair_refinement import PairRefinementInput, PairRefinementModule as SamPairRefinementModule
from ma_sp_sam.sam.sam_adapter import SAMMaskPrediction


def test_sam_pair_refinement_derives_axon_myelin_and_pair_table():
    semantic = np.zeros((8, 8), dtype=np.uint8)
    semantic[1:5, 1:5] = 1
    semantic[2:4, 2:4] = 2
    candidate = np.zeros((2, 8, 8), dtype=bool)
    candidate[0, 0:2, 0:2] = True
    candidate[1, 1:5, 1:5] = True
    prediction = SAMMaskPrediction(
        instance_id=7,
        masks=candidate,
        scores=np.asarray([0.1, 0.9], dtype=np.float32),
        logits=None,
        best_index=1,
        prompt_metadata={"instance_id": 7},
    )

    output = SamPairRefinementModule().refine(
        PairRefinementInput(
            sample_id="synthetic",
            semantic_pred=semantic,
            proposal_label_map=np.zeros_like(semantic, dtype=np.uint16),
            sam_predictions=[prediction],
            prompt_packages=[],
            image_shape=semantic.shape,
        )
    )

    assert int((output.fiber_instance == 7).sum()) == 16
    assert int((output.axon_instance == 7).sum()) == 4
    assert int((output.myelin_instance == 7).sum()) == 12
    row = output.pair_table.iloc[0]
    assert row["instance_id"] == 7
    assert row["source_prompt_id"] == 7
    assert row["selected_candidate_index"] == 1
    assert row["axon_area"] == 4
    assert row["myelin_area"] == 12
    assert row["fiber_area"] == 16
    assert row["g_ratio"] == 0.5
    assert row["flags"] == ""


def test_sam_pair_refinement_flags_missing_classes_and_fragmentation():
    semantic = np.zeros((7, 9), dtype=np.uint8)
    semantic[1:4, 1:4] = 1
    semantic[5, 1] = 2
    semantic[5, 3] = 2
    semantic[1, 7] = 1
    semantic[3, 7] = 1

    mask1 = np.zeros((7, 9), dtype=bool)
    mask1[1:4, 1:4] = True
    mask2 = np.zeros((7, 9), dtype=bool)
    mask2[5, 1] = True
    mask2[5, 3] = True
    mask3 = np.zeros((7, 9), dtype=bool)
    mask3[1, 7] = True
    mask3[3, 7] = True
    predictions = [
        SAMMaskPrediction(1, mask1[None], np.asarray([0.9], dtype=np.float32), None, 0, {}),
        SAMMaskPrediction(2, mask2[None], np.asarray([0.8], dtype=np.float32), None, 0, {}),
        SAMMaskPrediction(3, mask3[None], np.asarray([0.7], dtype=np.float32), None, 0, {}),
    ]

    output = SamPairRefinementModule(g_ratio_min=0.2, g_ratio_max=0.95).refine(
        PairRefinementInput(
            sample_id="flags",
            semantic_pred=semantic,
            proposal_label_map=np.zeros_like(semantic, dtype=np.uint16),
            sam_predictions=predictions,
            prompt_packages=[],
            image_shape=semantic.shape,
        )
    )

    flags = dict(zip(output.pair_table["instance_id"], output.pair_table["flags"]))
    assert "missing_axon" in flags[1]
    assert "missing_myelin" in flags[2]
    assert "multi_axon_component" in flags[2]
    assert "fragmented_myelin" in flags[3]
    assert "g_ratio_out_of_range" in flags[2]
