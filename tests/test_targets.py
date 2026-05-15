import numpy as np

from ma_sp_sam.labels.targets import (
    center_heatmap_from_instances,
    gaussian_center_heatmap_from_instances,
    hover_distance_map_from_instances,
)


def test_gaussian_center_heatmap_keeps_instance_centers_hot():
    axon_instance = np.zeros((9, 9), dtype=np.uint16)
    axon_instance[2:5, 3:6] = 1
    axon_instance[6:8, 1:3] = 2

    point_heatmap = center_heatmap_from_instances(axon_instance)
    gaussian = gaussian_center_heatmap_from_instances(axon_instance, sigma=1.0)

    assert point_heatmap.sum() == 2
    assert gaussian.shape == axon_instance.shape
    assert gaussian.dtype == np.float32
    assert gaussian.max() == 1.0
    assert gaussian[3, 4] == 1.0
    assert gaussian[7, 2] == 1.0
    assert gaussian[3, 3] > gaussian[0, 0]


def test_hover_distance_map_points_to_axon_center_inside_fiber():
    fiber_instance = np.zeros((7, 7), dtype=np.uint16)
    axon_instance = np.zeros_like(fiber_instance)
    fiber_instance[1:6, 1:6] = 1
    axon_instance[3, 3] = 1

    distance = hover_distance_map_from_instances(fiber_instance=fiber_instance, axon_instance=axon_instance)

    assert distance.shape == (2, 7, 7)
    assert distance.dtype == np.float32
    assert np.allclose(distance[:, 3, 3], [0.0, 0.0])
    assert distance[0, 3, 5] > 0
    assert distance[0, 3, 1] < 0
    assert distance[1, 5, 3] > 0
    assert distance[1, 1, 3] < 0
    assert np.allclose(distance[:, 0, 0], [0.0, 0.0])
