from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ma_sp_sam.data.astih_splits_index import SampleRecord, read_manifest
from ma_sp_sam.labels.targets import boundary_maps_from_bundle, center_heatmap_from_instances
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
        inner_boundary, outer_boundary = boundary_maps_from_bundle(bundle)
        return {
            "dataset": record.dataset,
            "split": record.split,
            "sample_id": record.sample_id,
            "image": read_image(record.image_path),
            "semantic": semantic,
            "fiber_instance": fiber_instance,
            "axon_instance": axon_instance,
            "myelin_instance": myelin_instance,
            "pair_table": pair_table,
            "center_heatmap": center_heatmap_from_instances(axon_instance),
            "boundary_inner": inner_boundary,
            "boundary_outer": outer_boundary,
        }
