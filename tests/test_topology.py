import numpy as np

from ma_sp_sam.refinement.topology import (
    bbox_from_mask,
    compute_g_ratio,
    connected_component_count,
    resolve_overlaps_by_score,
)


def test_compute_g_ratio_and_bbox_and_component_count():
    mask = np.zeros((6, 7), dtype=bool)
    mask[1:3, 2:5] = True
    mask[5, 6] = True

    assert compute_g_ratio(axon_area=4, fiber_area=16) == 0.5
    assert compute_g_ratio(axon_area=0, fiber_area=0) == 0.0
    assert bbox_from_mask(mask) == (2, 1, 6, 5)
    assert connected_component_count(mask) == 2


def test_resolve_overlaps_prefers_higher_score_then_smaller_area():
    large = np.zeros((6, 6), dtype=bool)
    large[1:5, 1:5] = True
    small = np.zeros((6, 6), dtype=bool)
    small[2:4, 2:4] = True
    high = np.zeros((6, 6), dtype=bool)
    high[3:6, 3:6] = True

    label_map = resolve_overlaps_by_score(
        candidate_masks=[large, small, high],
        scores=[0.7, 0.7, 0.9],
        instance_ids=[1, 2, 3],
    )

    assert label_map[2, 2] == 2
    assert label_map[4, 4] == 3
    assert label_map[1, 1] == 1
