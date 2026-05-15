import numpy as np

from ma_sp_sam.viz.overlays import (
    AXON_COLOR,
    MYELIN_COLOR,
    colorize_instance_labels,
    make_axon_myelin_overlay,
    make_paired_instance_preview,
)


def test_make_axon_myelin_overlay_colors_both_masks():
    image = np.zeros((2, 3), dtype=np.uint8)
    axon_instance = np.array([[0, 1, 0], [0, 0, 2]], dtype=np.uint16)
    myelin_instance = np.array([[3, 0, 0], [0, 4, 0]], dtype=np.uint16)

    overlay = np.asarray(
        make_axon_myelin_overlay(
            image,
            axon_instance=axon_instance,
            myelin_instance=myelin_instance,
            alpha=1.0,
            draw_boundaries=False,
        )
    )

    assert overlay[0, 1].tolist() == AXON_COLOR.astype(np.uint8).tolist()
    assert overlay[1, 2].tolist() == AXON_COLOR.astype(np.uint8).tolist()
    assert overlay[0, 0].tolist() == MYELIN_COLOR.astype(np.uint8).tolist()
    assert overlay[1, 1].tolist() == MYELIN_COLOR.astype(np.uint8).tolist()
    assert overlay[0, 2].tolist() == [0, 0, 0]


def test_colorize_instance_labels_assigns_distinct_nonblack_colors():
    labels = np.array([[0, 1, 2], [3, 0, 0]], dtype=np.uint16)

    preview = np.asarray(colorize_instance_labels(labels))

    assert preview[0, 0].tolist() == [0, 0, 0]
    assert preview[0, 1].tolist() != [0, 0, 0]
    assert preview[0, 2].tolist() != [0, 0, 0]
    assert preview[1, 0].tolist() != [0, 0, 0]
    assert preview[0, 1].tolist() != preview[0, 2].tolist()
    assert preview[0, 2].tolist() != preview[1, 0].tolist()


def test_paired_instance_preview_uses_same_hue_family_for_pair_id():
    axon = np.array([[0, 1, 0], [0, 0, 2]], dtype=np.uint16)
    myelin = np.array([[0, 1, 0], [0, 2, 0]], dtype=np.uint16)

    preview = np.asarray(make_paired_instance_preview(axon_instance=axon, myelin_instance=myelin))

    assert preview[0, 1].tolist() != [0, 0, 0]
    assert preview[1, 1].tolist() != [0, 0, 0]
    assert preview[0, 1].tolist() != preview[1, 2].tolist()
    assert preview[1, 1].tolist() != preview[0, 1].tolist()
