from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ma_sp_sam.prompts import PromptPackage
from ma_sp_sam.refinement.topology import (
    compute_g_ratio,
    connected_component_count,
    make_pair_table,
    resolve_overlaps_by_score,
)
from ma_sp_sam.sam.sam_adapter import SAMMaskPrediction, best_mask
from ma_sp_sam.utils.io import write_tiff_u16


@dataclass(frozen=True)
class PairRefinementInput:
    sample_id: str
    semantic_pred: np.ndarray
    proposal_label_map: np.ndarray
    sam_predictions: Sequence[SAMMaskPrediction]
    prompt_packages: Sequence[PromptPackage]
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class RefinedInstanceRecord:
    instance_id: int
    source_prompt_id: int
    selected_candidate_index: int
    sam_score: float
    fiber_area: int
    axon_area: int
    myelin_area: int
    g_ratio: float
    flags: str


@dataclass(frozen=True)
class PairRefinementOutput:
    fiber_instance: np.ndarray
    axon_instance: np.ndarray
    myelin_instance: np.ndarray
    pair_table: pd.DataFrame
    records: list[RefinedInstanceRecord]


class PairRefinementModule:
    def __init__(self, *, g_ratio_min: float = 0.2, g_ratio_max: float = 0.95) -> None:
        if not 0 <= g_ratio_min <= g_ratio_max:
            raise ValueError("Expected 0 <= g_ratio_min <= g_ratio_max.")
        self.g_ratio_min = float(g_ratio_min)
        self.g_ratio_max = float(g_ratio_max)

    def __call__(self, refinement_input: PairRefinementInput) -> PairRefinementOutput:
        return self.refine(refinement_input)

    def refine(self, refinement_input: PairRefinementInput) -> PairRefinementOutput:
        semantic = np.asarray(refinement_input.semantic_pred, dtype=np.uint8)
        if semantic.shape != tuple(refinement_input.image_shape):
            raise ValueError(f"semantic_pred shape {semantic.shape} does not match image_shape {refinement_input.image_shape}.")

        selected = [_selected_candidate(prediction, refinement_input.image_shape) for prediction in refinement_input.sam_predictions]
        non_empty = [item for item in selected if item["mask"].any()]
        if non_empty:
            fiber_instance = resolve_overlaps_by_score(
                candidate_masks=[item["mask"] for item in non_empty],
                scores=[item["score"] for item in non_empty],
                instance_ids=[item["instance_id"] for item in non_empty],
            )
        else:
            fiber_instance = np.zeros(refinement_input.image_shape, dtype=np.uint16)

        axon_instance = np.zeros(refinement_input.image_shape, dtype=np.uint16)
        myelin_instance = np.zeros(refinement_input.image_shape, dtype=np.uint16)
        records: list[RefinedInstanceRecord] = []
        by_id = {int(item["instance_id"]): item for item in selected}
        for instance_id in sorted(by_id):
            item = by_id[instance_id]
            fiber_mask = fiber_instance == instance_id
            flags: list[str] = []
            if not item["mask"].any():
                flags.append("empty_sam_mask")
            axon_mask = fiber_mask & (semantic == 2)
            myelin_mask = fiber_mask & (semantic == 1)
            axon_instance[axon_mask] = instance_id
            myelin_instance[myelin_mask] = instance_id

            axon_area = int(axon_mask.sum())
            myelin_area = int(myelin_mask.sum())
            fiber_area = int(fiber_mask.sum())
            g_ratio = compute_g_ratio(axon_area, fiber_area)
            flags.extend(_topology_flags(axon_mask, myelin_mask, axon_area, myelin_area, fiber_area, g_ratio, self.g_ratio_min, self.g_ratio_max))
            records.append(
                RefinedInstanceRecord(
                    instance_id=instance_id,
                    source_prompt_id=int(item["source_prompt_id"]),
                    selected_candidate_index=int(item["selected_candidate_index"]),
                    sam_score=float(item["score"]),
                    fiber_area=fiber_area,
                    axon_area=axon_area,
                    myelin_area=myelin_area,
                    g_ratio=g_ratio,
                    flags=";".join(dict.fromkeys(flags)),
                )
            )

        pair_table = make_pair_table(records)
        return PairRefinementOutput(
            fiber_instance=fiber_instance,
            axon_instance=axon_instance,
            myelin_instance=myelin_instance,
            pair_table=pair_table,
            records=records,
        )


def save_pair_refinement_output(output: PairRefinementOutput, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_tiff_u16(out / "refined_fiber_instance.tif", output.fiber_instance)
    write_tiff_u16(out / "refined_axon_instance.tif", output.axon_instance)
    write_tiff_u16(out / "refined_myelin_instance.tif", output.myelin_instance)
    output.pair_table.to_csv(out / "refined_pair_table.csv", index=False)


def _selected_candidate(prediction: SAMMaskPrediction, image_shape: tuple[int, int]) -> dict[str, object]:
    mask = best_mask(prediction).astype(bool)
    if mask.size == 0:
        mask = np.zeros(image_shape, dtype=bool)
    if mask.shape != tuple(image_shape):
        raise ValueError(f"SAM mask shape {mask.shape} does not match image_shape {image_shape}.")
    score = float(prediction.scores[prediction.best_index]) if prediction.scores.size else 0.0
    return {
        "instance_id": int(prediction.instance_id),
        "source_prompt_id": int(prediction.prompt_metadata.get("instance_id", prediction.instance_id)),
        "selected_candidate_index": int(prediction.best_index),
        "score": score,
        "mask": mask,
    }


def _topology_flags(
    axon_mask: np.ndarray,
    myelin_mask: np.ndarray,
    axon_area: int,
    myelin_area: int,
    fiber_area: int,
    g_ratio: float,
    g_ratio_min: float,
    g_ratio_max: float,
) -> list[str]:
    flags: list[str] = []
    if axon_area == 0:
        flags.append("missing_axon")
    if myelin_area == 0:
        flags.append("missing_myelin")
    if axon_area > 0 and connected_component_count(axon_mask) > 1:
        flags.append("multi_axon_component")
    if myelin_area > 0 and connected_component_count(myelin_mask) > 1:
        flags.append("fragmented_myelin")
    if fiber_area > 0 and not (g_ratio_min <= g_ratio <= g_ratio_max):
        flags.append("g_ratio_out_of_range")
    return flags
