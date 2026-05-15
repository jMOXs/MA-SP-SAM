import numpy as np

from ma_sp_sam.labels.paired_instances import build_paired_instances


def test_builds_aligned_fiber_axon_and_myelin_instances():
    semantic = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 2, 1, 0, 1],
            [0, 0, 0, 0, 2],
        ],
        dtype=np.uint8,
    )

    bundle = build_paired_instances(semantic, min_axon_area=1, min_myelin_area=1)

    assert bundle.fiber_instance.max() == 2
    assert set(bundle.pair_table["fiber_id"].tolist()) == {1, 2}
    assert np.array_equal(bundle.axon_instance > 0, semantic == 2)
    assert np.array_equal(bundle.myelin_instance > 0, semantic == 1)
    assert bundle.pair_table.loc[bundle.pair_table["fiber_id"] == 1, "g_ratio"].iloc[0] == 0.5


def test_flags_low_quality_and_multi_axon_components():
    semantic = np.array(
        [
            [2, 1, 2],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    bundle = build_paired_instances(
        semantic,
        min_axon_area=3,
        min_myelin_area=1,
        g_ratio_min=0.7,
        g_ratio_max=0.9,
    )

    flags = bundle.pair_table.loc[0, "flags"].split(";")
    assert "small_axon" in flags
    assert "multi_axon_component" in flags
    assert "g_ratio_out_of_range" in flags
