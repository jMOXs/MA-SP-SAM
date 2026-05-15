from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ma_sp_sam.data.astih_splits_index import SampleRecord
from ma_sp_sam.utils.io import read_image, read_mask


MYELIN_VALUES = {126, 127, 128}
AXON_VALUES = {255}


@dataclass(frozen=True)
class SemanticSample:
    image: np.ndarray
    semantic: np.ndarray
    ignored: np.ndarray
    metadata: dict[str, Any]


def _decode_composite_mask(mask: np.ndarray) -> np.ndarray:
    semantic = np.zeros(mask.shape, dtype=np.uint8)
    semantic[np.isin(mask, list(MYELIN_VALUES))] = 1
    semantic[np.isin(mask, list(AXON_VALUES))] = 2
    return semantic


def _load_optional_binary(path: Path | None, shape: tuple[int, ...]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=bool)
    return read_mask(path) > 0


def load_semantic_sample(record: SampleRecord) -> SemanticSample:
    image = read_image(record.image_path)

    composite_values: list[int] = []
    composite = None
    if record.axonmyelin_mask_path is not None:
        composite = read_mask(record.axonmyelin_mask_path)
        composite_values = [int(value) for value in np.unique(composite)]

    if record.axon_mask_path is not None and record.myelin_mask_path is not None:
        shape = read_mask(record.axon_mask_path).shape
        axon = _load_optional_binary(record.axon_mask_path, shape)
        myelin = _load_optional_binary(record.myelin_mask_path, shape)
        semantic = np.zeros(shape, dtype=np.uint8)
        semantic[myelin] = 1
        semantic[axon] = 2
        label_source = "independent_masks"
    elif composite is not None:
        semantic = _decode_composite_mask(composite)
        label_source = "composite_mask"
    else:
        raise FileNotFoundError(f"No usable manual segmentation mask for {record.sample_id}")

    ignored = np.zeros(semantic.shape, dtype=bool)
    for aux_path in record.auxiliary_mask_paths.values():
        aux = read_mask(aux_path)
        if aux.shape != semantic.shape:
            raise ValueError(f"Auxiliary mask shape mismatch for {record.sample_id}: {aux_path}")
        ignored |= aux > 0

    known_values = {0, *MYELIN_VALUES, *AXON_VALUES}
    unknown_values = [value for value in composite_values if value not in known_values]
    metadata = {
        "dataset": record.dataset,
        "split": record.split,
        "sample_id": record.sample_id,
        "label_source": label_source,
        "composite_values": composite_values,
        "unknown_composite_values": unknown_values,
        "auxiliary_masks": sorted(record.auxiliary_mask_paths.keys()),
    }
    return SemanticSample(image=image, semantic=semantic, ignored=ignored, metadata=metadata)
