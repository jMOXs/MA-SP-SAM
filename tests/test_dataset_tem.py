from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ma_sp_sam.data.astih_splits_index import SampleRecord, write_manifest
from ma_sp_sam.data.dataset_tem import DatasetTEM


def _save(path: Path, array, dtype=np.uint8):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=dtype)).save(path)


def test_dataset_tem_loads_processed_labels_and_pair_table(tmp_path):
    image_path = tmp_path / "splits" / "TEM1" / "train" / "sample_TEM.png"
    _save(image_path, np.zeros((2, 3), dtype=np.uint8))
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest_path,
        [
            SampleRecord(
                dataset="TEM1",
                split="train",
                sample_id="sample_TEM",
                image_path=image_path,
                axonmyelin_mask_path=None,
                axon_mask_path=None,
                myelin_mask_path=None,
                auxiliary_mask_paths={},
            )
        ],
    )

    processed = tmp_path / "processed"
    sample_dir = processed / "TEM1" / "train" / "sample_TEM"
    _save(sample_dir / "semantic.png", [[0, 1, 2], [0, 0, 0]])
    _save(sample_dir / "fiber_instance.tif", [[0, 1, 1], [0, 0, 0]], dtype=np.uint16)
    _save(sample_dir / "axon_instance.tif", [[0, 0, 1], [0, 0, 0]], dtype=np.uint16)
    _save(sample_dir / "myelin_instance.tif", [[0, 1, 0], [0, 0, 0]], dtype=np.uint16)
    pd.DataFrame([{"fiber_id": 1, "axon_area": 1, "myelin_area": 1, "fiber_area": 2, "flags": ""}]).to_csv(
        sample_dir / "pair_table.csv", index=False
    )

    dataset = DatasetTEM(manifest_path, processed_root=processed, dataset="TEM1", split="train")
    item = dataset[0]

    assert len(dataset) == 1
    assert item["sample_id"] == "sample_TEM"
    assert item["image"].shape == (2, 3)
    assert item["semantic"].tolist() == [[0, 1, 2], [0, 0, 0]]
    assert item["pair_table"].loc[0, "fiber_id"] == 1


def test_dataset_tem_can_return_training_tensors(tmp_path):
    torch = pytest.importorskip("torch")

    image_path = tmp_path / "splits" / "TEM1" / "train" / "tensor_TEM.png"
    _save(image_path, [[0, 255, 128], [64, 32, 16]])
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest_path,
        [
            SampleRecord(
                dataset="TEM1",
                split="train",
                sample_id="tensor_TEM",
                image_path=image_path,
                axonmyelin_mask_path=None,
                axon_mask_path=None,
                myelin_mask_path=None,
                auxiliary_mask_paths={},
            )
        ],
    )

    processed = tmp_path / "processed"
    sample_dir = processed / "TEM1" / "train" / "tensor_TEM"
    _save(sample_dir / "semantic.png", [[0, 1, 2], [0, 0, 0]])
    _save(sample_dir / "fiber_instance.tif", [[0, 1, 1], [0, 0, 0]], dtype=np.uint16)
    _save(sample_dir / "axon_instance.tif", [[0, 0, 1], [0, 0, 0]], dtype=np.uint16)
    _save(sample_dir / "myelin_instance.tif", [[0, 1, 0], [0, 0, 0]], dtype=np.uint16)
    pd.DataFrame([{"fiber_id": 1, "axon_area": 1, "myelin_area": 1, "fiber_area": 2, "flags": ""}]).to_csv(
        sample_dir / "pair_table.csv", index=False
    )

    item = DatasetTEM(
        manifest_path,
        processed_root=processed,
        dataset="TEM1",
        split="train",
        return_tensors=True,
    )[0]

    assert item["image"].shape == (1, 2, 3)
    assert item["image"].dtype == torch.float32
    assert item["semantic"].shape == (2, 3)
    assert item["semantic"].dtype == torch.long
    assert item["center_heatmap"].shape == (1, 2, 3)
    assert item["boundary_inner"].shape == (1, 2, 3)
    assert item["boundary_outer"].shape == (1, 2, 3)
    assert item["distance_map"].shape == (2, 2, 3)
