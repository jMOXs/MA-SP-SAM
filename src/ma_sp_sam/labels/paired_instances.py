from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure

from ma_sp_sam.utils.io import write_png, write_tiff_u16


@dataclass(frozen=True)
class PairedLabelBundle:
    semantic: np.ndarray
    fiber_instance: np.ndarray
    axon_instance: np.ndarray
    myelin_instance: np.ndarray
    pair_table: pd.DataFrame


def _component_count(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    return int(measure.label(mask, connectivity=1).max())


def _flags_for_instance(
    axon_area: int,
    myelin_area: int,
    fiber_area: int,
    axon_components: int,
    myelin_components: int,
    min_axon_area: int,
    min_myelin_area: int,
    g_ratio: float,
    g_ratio_min: float,
    g_ratio_max: float,
) -> list[str]:
    flags: list[str] = []
    if axon_area == 0:
        flags.append("missing_axon")
    if myelin_area == 0:
        flags.append("missing_myelin")
    if axon_area < min_axon_area:
        flags.append("small_axon")
    if myelin_area < min_myelin_area:
        flags.append("small_myelin")
    if axon_components > 1:
        flags.append("multi_axon_component")
    if myelin_components > 1:
        flags.append("multi_myelin_component")
    if fiber_area > 0 and axon_area > 0 and not (g_ratio_min <= g_ratio <= g_ratio_max):
        flags.append("g_ratio_out_of_range")
    if any(flag in flags for flag in ("missing_axon", "missing_myelin", "multi_axon_component")):
        flags.append("ambiguous_fiber")
    return flags


def build_paired_instances(
    semantic: np.ndarray,
    *,
    min_axon_area: int = 8,
    min_myelin_area: int = 8,
    g_ratio_min: float = 0.2,
    g_ratio_max: float = 0.95,
    connectivity: int = 1,
) -> PairedLabelBundle:
    """Derive aligned fiber, axon, and myelin instance IDs from semantic labels."""
    if semantic.ndim != 2:
        raise ValueError("Expected a 2D semantic mask")

    semantic = np.asarray(semantic, dtype=np.uint8)
    foreground = np.isin(semantic, [1, 2])
    fiber_instance = measure.label(foreground, connectivity=connectivity).astype(np.uint16)
    axon_instance = np.zeros_like(fiber_instance, dtype=np.uint16)
    myelin_instance = np.zeros_like(fiber_instance, dtype=np.uint16)

    rows = []
    for region in measure.regionprops(fiber_instance):
        fiber_id = int(region.label)
        region_slice = region.slice
        fiber_local = fiber_instance[region_slice] == fiber_id
        semantic_local = semantic[region_slice]
        axon_mask = fiber_local & (semantic_local == 2)
        myelin_mask = fiber_local & (semantic_local == 1)
        axon_instance[region_slice][axon_mask] = fiber_id
        myelin_instance[region_slice][myelin_mask] = fiber_id

        axon_area = int(axon_mask.sum())
        myelin_area = int(myelin_mask.sum())
        fiber_area = int(fiber_local.sum())
        g_ratio = float(np.sqrt(axon_area / fiber_area)) if fiber_area else 0.0
        axon_components = _component_count(axon_mask)
        myelin_components = _component_count(myelin_mask)
        flags = _flags_for_instance(
            axon_area,
            myelin_area,
            fiber_area,
            axon_components,
            myelin_components,
            min_axon_area,
            min_myelin_area,
            g_ratio,
            g_ratio_min,
            g_ratio_max,
        )

        rows.append(
            {
                "fiber_id": fiber_id,
                "axon_id": fiber_id if axon_area else 0,
                "myelin_id": fiber_id if myelin_area else 0,
                "axon_area": axon_area,
                "myelin_area": myelin_area,
                "fiber_area": fiber_area,
                "g_ratio": g_ratio,
                "axon_components": axon_components,
                "myelin_components": myelin_components,
                "flags": ";".join(dict.fromkeys(flags)),
            }
        )

    pair_table = pd.DataFrame(
        rows,
        columns=[
            "fiber_id",
            "axon_id",
            "myelin_id",
            "axon_area",
            "myelin_area",
            "fiber_area",
            "g_ratio",
            "axon_components",
            "myelin_components",
            "flags",
        ],
    )
    return PairedLabelBundle(
        semantic=semantic,
        fiber_instance=fiber_instance,
        axon_instance=axon_instance,
        myelin_instance=myelin_instance,
        pair_table=pair_table,
    )


def export_paired_label_bundle(bundle: PairedLabelBundle, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_png(out / "semantic.png", bundle.semantic.astype(np.uint8))
    write_tiff_u16(out / "fiber_instance.tif", bundle.fiber_instance)
    write_tiff_u16(out / "axon_instance.tif", bundle.axon_instance)
    write_tiff_u16(out / "myelin_instance.tif", bundle.myelin_instance)
    bundle.pair_table.to_csv(out / "pair_table.csv", index=False)
