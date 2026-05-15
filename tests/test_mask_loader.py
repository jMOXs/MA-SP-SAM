import numpy as np
from PIL import Image

from ma_sp_sam.data.astih_splits_index import SampleRecord
from ma_sp_sam.data.mask_loader import load_semantic_sample


def _save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def _record(tmp_path, **paths):
    image_path = tmp_path / "sample_TEM.png"
    _save(image_path, np.zeros((3, 5), dtype=np.uint8))
    return SampleRecord(
        dataset="TEM2",
        split="train",
        sample_id="sample_TEM",
        image_path=image_path,
        axonmyelin_mask_path=paths.get("axonmyelin"),
        axon_mask_path=paths.get("axon"),
        myelin_mask_path=paths.get("myelin"),
        auxiliary_mask_paths=paths.get("auxiliary", {}),
    )


def test_decodes_composite_axonmyelin_values(tmp_path):
    composite = tmp_path / "sample_TEM_seg-axonmyelin-manual.png"
    _save(composite, [[0, 126, 127, 128, 255]])

    sample = load_semantic_sample(_record(tmp_path, axonmyelin=composite))

    assert sample.semantic.tolist() == [[0, 1, 1, 1, 2]]
    assert sorted(sample.metadata["composite_values"]) == [0, 126, 127, 128, 255]


def test_independent_axon_and_myelin_masks_override_composite(tmp_path):
    composite = tmp_path / "sample_TEM_seg-axonmyelin-manual.png"
    axon = tmp_path / "sample_TEM_seg-axon-manual.png"
    myelin = tmp_path / "sample_TEM_seg-myelin-manual.png"
    _save(composite, np.zeros((3, 5), dtype=np.uint8))
    _save(axon, [[0, 0, 255, 0, 0], [0, 0, 255, 0, 0], [0, 0, 0, 0, 0]])
    _save(myelin, [[255, 255, 0, 0, 0], [0, 0, 0, 255, 0], [0, 0, 0, 0, 0]])

    sample = load_semantic_sample(_record(tmp_path, axonmyelin=composite, axon=axon, myelin=myelin))

    assert sample.semantic.tolist() == [
        [1, 1, 2, 0, 0],
        [0, 0, 2, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    assert sample.metadata["label_source"] == "independent_masks"
