from pathlib import Path

from PIL import Image

from ma_sp_sam.data.astih_splits_index import index_astih_splits


def _write_png(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (4, 4), value).save(path)


def test_indexes_official_astih_splits_without_creating_new_split(tmp_path):
    splits = tmp_path / "splits"
    tem1_train = splits / "TEM1" / "train"
    tem1_test = splits / "TEM1" / "test"
    tem2_train = splits / "TEM2" / "train"

    _write_png(tem1_train / "sub-a_sample-1_TEM.png")
    _write_png(tem1_train / "sub-a_sample-1_TEM_seg-axon-manual.png", 255)
    _write_png(tem1_train / "sub-a_sample-1_TEM_seg-myelin-manual.png", 255)
    _write_png(tem1_train / "sub-a_sample-1_TEM_seg-axonmyelin-manual.png", 127)

    _write_png(tem1_test / "sub-b_sample-1_TEM.png")
    _write_png(tem1_test / "sub-b_sample-1_TEM_seg-axonmyelin-manual.png", 255)

    _write_png(tem2_train / "sub-c_sample-1_TEM.png")
    _write_png(tem2_train / "sub-c_sample-1_TEM_seg-axonmyelin-manual.png", 127)
    _write_png(tem2_train / "sub-c_sample-1_TEM_seg-uaxon-manual.png", 255)

    records = index_astih_splits(splits, datasets=("TEM1", "TEM2"))

    by_sample = {record.sample_id: record for record in records}
    assert set(by_sample) == {"sub-a_sample-1_TEM", "sub-b_sample-1_TEM", "sub-c_sample-1_TEM"}
    assert by_sample["sub-a_sample-1_TEM"].dataset == "TEM1"
    assert by_sample["sub-a_sample-1_TEM"].split == "train"
    assert by_sample["sub-a_sample-1_TEM"].axon_mask_path is not None
    assert by_sample["sub-a_sample-1_TEM"].myelin_mask_path is not None
    assert by_sample["sub-c_sample-1_TEM"].axon_mask_path is None
    assert by_sample["sub-c_sample-1_TEM"].myelin_mask_path is None
    assert by_sample["sub-c_sample-1_TEM"].auxiliary_mask_paths["uaxon"].name.endswith("_seg-uaxon-manual.png")
