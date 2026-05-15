import numpy as np

from ma_sp_sam.prompts import PromptSynthesizer
from ma_sp_sam.sam.prompt_converter import convert_prompt_package_to_sam_inputs


def _prompt_packages():
    proposals = np.zeros((8, 9), dtype=np.uint16)
    proposals[1:4, 1:4] = 7
    proposals[4:7, 5:8] = 9
    center = np.zeros_like(proposals, dtype=np.float32)
    center[2, 2] = 1.0
    center[5, 6] = 0.9
    semantic = np.zeros_like(proposals, dtype=np.uint8)
    semantic[proposals > 0] = 2
    boundary = np.zeros((2, *proposals.shape), dtype=np.float32)
    boundary[0, 2, 1:4] = 1.0
    boundary[1, 1, 1:4] = 1.0
    return PromptSynthesizer(max_negative_points=1, ring_points_per_boundary=2).synthesize(
        center_heatmap=center,
        semantic_logits=semantic,
        boundary_maps=boundary,
        instance_proposals=proposals,
    ).packages


def test_convert_prompt_package_to_sam_inputs_defaults_to_points_and_box_only():
    package = _prompt_packages()[0]

    sam_inputs = convert_prompt_package_to_sam_inputs(package, image_shape=(8, 9))

    assert sam_inputs["point_coords"].shape == (2, 2)
    assert sam_inputs["point_labels"].tolist() == [1, 0]
    assert sam_inputs["point_coords"][0].tolist() == [2.0, 2.0]
    assert sam_inputs["box"].tolist() == [1.0, 1.0, 3.0, 3.0]
    assert sam_inputs["mask_input"] is None
    assert sam_inputs["metadata"]["instance_id"] == 7
    assert sam_inputs["metadata"]["coarse_mask_shape"] == (8, 9)
    assert "ring_points" in sam_inputs["metadata"]
    assert sam_inputs["metadata"]["ring_points"].shape[1] == 2


def test_convert_prompt_package_to_sam_inputs_can_resize_low_res_mask_prior():
    package = _prompt_packages()[0]

    sam_inputs = convert_prompt_package_to_sam_inputs(
        package,
        image_shape=(8, 9),
        use_mask_input=True,
        mask_input_size=(256, 256),
    )

    assert sam_inputs["mask_input"].shape == (1, 256, 256)
    assert sam_inputs["mask_input"].dtype == np.float32
    assert sam_inputs["metadata"]["coarse_mask_shape"] == (8, 9)
    assert "coarse_mask" in sam_inputs["metadata"]


def test_importing_sam_package_without_segment_anything_does_not_crash():
    import ma_sp_sam.sam as sam

    assert hasattr(sam, "convert_prompt_package_to_sam_inputs")
