from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ma_sp_sam.data.astih_splits_index import SampleRecord, read_manifest
from ma_sp_sam.labels.targets import (
    boundary_maps_from_bundle,
    center_heatmap_from_instances,
    gaussian_center_heatmap_from_instances,
    hover_distance_map_from_instances,
)
from ma_sp_sam.labels.paired_instances import PairedLabelBundle
from ma_sp_sam.utils.io import read_array, read_image, read_mask


class DatasetTEM:
    """Lightweight TEM dataset API over ASTIH manifest and processed labels."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        processed_root: str | Path,
        dataset: str | None = None,
        split: str | None = None,
        require_processed: bool = True,
        return_tensors: bool = False,
        center_target: str = "gaussian",
        center_sigma: float = 3.0,
        include_targets: bool = True,
    ) -> None:
        records = read_manifest(manifest_path)
        if dataset is not None:
            records = [record for record in records if record.dataset == dataset]
        if split is not None:
            records = [record for record in records if record.split == split]
        self.processed_root = Path(processed_root)
        if require_processed:
            records = [record for record in records if (self._sample_dir(record) / "pair_table.csv").exists()]
        self.records = records
        self.return_tensors = return_tensors
        self.center_target = center_target
        self.center_sigma = center_sigma
        self.include_targets = bool(include_targets)

    def __len__(self) -> int:
        return len(self.records)

    def _sample_dir(self, record: SampleRecord) -> Path:
        return self.processed_root / record.dataset / record.split / record.sample_id

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample_dir = self._sample_dir(record)
        semantic = read_mask(sample_dir / "semantic.png")
        fiber_instance = read_array(sample_dir / "fiber_instance.tif")
        axon_instance = read_array(sample_dir / "axon_instance.tif")
        myelin_instance = read_array(sample_dir / "myelin_instance.tif")
        pair_table = pd.read_csv(sample_dir / "pair_table.csv")
        bundle = PairedLabelBundle(
            semantic=semantic,
            fiber_instance=fiber_instance,
            axon_instance=axon_instance,
            myelin_instance=myelin_instance,
            pair_table=pair_table,
        )
        item = {
            "dataset": record.dataset,
            "split": record.split,
            "sample_id": record.sample_id,
            "image": read_image(record.image_path),
            "semantic": semantic,
            "fiber_instance": fiber_instance,
            "axon_instance": axon_instance,
            "myelin_instance": myelin_instance,
            "pair_table": pair_table,
        }
        if self.include_targets:
            inner_boundary, outer_boundary = boundary_maps_from_bundle(bundle)
            if self.center_target == "point":
                center_heatmap = center_heatmap_from_instances(axon_instance)
            elif self.center_target == "gaussian":
                center_heatmap = gaussian_center_heatmap_from_instances(axon_instance, sigma=self.center_sigma)
            else:
                raise ValueError("center_target must be 'gaussian' or 'point'.")
            distance_map = hover_distance_map_from_instances(
                fiber_instance=fiber_instance,
                axon_instance=axon_instance,
            )
            item.update(
                {
                    "center_heatmap": center_heatmap,
                    "boundary_inner": inner_boundary,
                    "boundary_outer": outer_boundary,
                    "distance_map": distance_map,
                }
            )
        if self.return_tensors:
            item.update(_tensor_targets(item))
        return item


def _tensor_targets(item: dict[str, Any]) -> dict[str, Any]:
    import torch

    image = _image_to_chw_float(item["image"])
    targets = {
        "image": torch.from_numpy(image),
        "semantic": torch.from_numpy(item["semantic"].astype("int64")),
        "fiber_instance": torch.from_numpy(item["fiber_instance"].astype("int64")),
        "axon_instance": torch.from_numpy(item["axon_instance"].astype("int64")),
        "myelin_instance": torch.from_numpy(item["myelin_instance"].astype("int64")),
    }
    if "center_heatmap" in item:
        targets.update(
            {
                "center_heatmap": torch.from_numpy(item["center_heatmap"].astype("float32"))[None, ...],
                "boundary_inner": torch.from_numpy(item["boundary_inner"].astype("float32"))[None, ...],
                "boundary_outer": torch.from_numpy(item["boundary_outer"].astype("float32"))[None, ...],
                "distance_map": torch.from_numpy(item["distance_map"].astype("float32")),
            }
        )
    return targets


def _image_to_chw_float(image) -> "np.ndarray":
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.ndim != 2:
        raise ValueError(f"Expected grayscale or RGB image, got shape {array.shape}.")
    array = array.astype("float32")
    if array.size and array.max() > 1.0:
        array = array / 255.0
    return array[None, ...]
