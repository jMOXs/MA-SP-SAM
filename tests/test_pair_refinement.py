import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from ma_sp_sam.models.pair_refinement import PairRefinementModule, PairRefinementOutput
from ma_sp_sam.models.sam_adapter import CandidateMask


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
