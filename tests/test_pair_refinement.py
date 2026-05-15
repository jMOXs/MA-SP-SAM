import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from ma_sp_sam.models.pair_refinement import PairRefinementModule, PairRefinementOutput
from ma_sp_sam.models.sam_adapter import CandidateMask
from ma_sp_sam.refinement.pair_refinement import PairRefinementInput, PairRefinementModule as SamPairRefinementModule
from ma_sp_sam.sam.sam_adapter import SAMMaskPrediction


def test_pair_refinement_enforces_non_overlap_and_fiber_union():
    axon = np.zeros((6, 6), dtype=np.float32)
    myelin = np.zeros((6, 6), dtype=np.float32)
    fiber = np.zeros((6, 6), dtype=np.float32)
    axon[2:4, 2:4] = 0.9
    myelin[1:5, 1:5] = 0.8
    fiber[1:5, 1:5] = 0.9

    output = PairRefinementModule(mask_threshold=0.5, min_axon_area=1, min_myelin_area=1).refine(
        axon_candidate_masks={5: axon},
        myelin_candidate_masks={5: myelin},
        fiber_candidate_masks={5: fiber},
        prompt_metadata=[{"instance_id": 5, "quality_prior": 0.7}],
    )

    assert isinstance(output, PairRefinementOutput)
    assert not np.any((output.axon_i == 5) & (output.myelin_i == 5))
    assert np.array_equal(output.fiber_i > 0, (output.axon_i > 0) | (output.myelin_i > 0))
    assert np.array_equal(output.pair_id_i, output.fiber_i)
    assert output.pair_table.loc[0, "fiber_id"] == 5
    assert output.pair_table.loc[0, "flags"] == ""


def test_pair_refinement_flags_orphans_and_g_ratio_outliers():
    axon_only = np.zeros((5, 5), dtype=np.float32)
    axon_only[1:3, 1:3] = 1

    myelin_only = np.zeros((5, 5), dtype=np.float32)
    myelin_only[2:5, 2:5] = 1

    high_g_axon = np.zeros((5, 5), dtype=np.float32)
    high_g_myelin = np.zeros((5, 5), dtype=np.float32)
    high_g_axon[0:4, 0:4] = 1
    high_g_myelin[4, 4] = 1

    output = PairRefinementModule(mask_threshold=0.5, min_axon_area=1, min_myelin_area=1, g_ratio_max=0.8).refine(
        axon_candidate_masks={1: axon_only, 2: np.zeros((5, 5), dtype=np.float32), 3: high_g_axon},
        myelin_candidate_masks={1: np.zeros((5, 5), dtype=np.float32), 2: myelin_only, 3: high_g_myelin},
        fiber_candidate_masks={
            1: axon_only,
            2: myelin_only,
            3: np.maximum(high_g_axon, high_g_myelin),
        },
        prompt_metadata=pd.DataFrame(
            [
                {"instance_id": 1, "quality_prior": 0.9},
                {"instance_id": 2, "quality_prior": 0.8},
                {"instance_id": 3, "quality_prior": 0.7},
            ]
        ),
    )

    flags = dict(zip(output.pair_table["fiber_id"], output.pair_table["flags"]))
    assert "orphan_axon" in flags[1]
    assert "missing_myelin" in flags[1]
    assert "orphan_myelin" in flags[2]
    assert "missing_axon" in flags[2]
    assert "g_ratio_out_of_range" in flags[3]


def test_pair_refinement_accepts_candidate_mask_objects_and_uses_best_channel():
    bad = torch.zeros(4, 4)
    good = torch.zeros(4, 4)
    good[1:3, 1:3] = 2.0
    myelin = torch.zeros(4, 4)
    myelin[0, 1:3] = 2.0
    fiber = torch.maximum(good, myelin)

    axon_candidate = CandidateMask(
        instance_id=11,
        masks=torch.stack([bad, good]),
        low_res_logits=torch.stack([bad, good]),
        iou_predictions=torch.tensor([0.1, 0.95]),
    )
    myelin_candidate = CandidateMask(
        instance_id=11,
        masks=myelin.unsqueeze(0),
        low_res_logits=myelin.unsqueeze(0),
        iou_predictions=torch.tensor([0.8]),
    )
    fiber_candidate = CandidateMask(
        instance_id=11,
        masks=fiber.unsqueeze(0),
        low_res_logits=fiber.unsqueeze(0),
        iou_predictions=torch.tensor([0.9]),
    )

    output = PairRefinementModule(mask_threshold=0.5, min_axon_area=1, min_myelin_area=1).refine(
        axon_candidate_masks=[axon_candidate],
        myelin_candidate_masks=[myelin_candidate],
        fiber_candidate_masks=[fiber_candidate],
        prompt_metadata=None,
    )

    assert output.axon_i.sum() == 44
    assert int((output.axon_i == 11).sum()) == 4
    assert int((output.myelin_i == 11).sum()) == 2
    assert int((output.fiber_i == 11).sum()) == 6
    assert output.pair_table.loc[0, "fiber_id"] == 11


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
