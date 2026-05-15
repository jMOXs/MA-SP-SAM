from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Sequence

import numpy as np

from ma_sp_sam.prompts import PromptPackage
from ma_sp_sam.sam.prompt_converter import convert_prompt_package_to_sam_inputs


SEGMENT_ANYTHING_MISSING = "segment-anything is not installed. Install it to use SAMAdapter."


@dataclass(frozen=True)
class SAMMaskPrediction:
    instance_id: int
    masks: np.ndarray
    scores: np.ndarray
    logits: np.ndarray | None
    prompt_metadata: dict


class SAMAdapter:
    """Inference-only adapter around Meta Segment Anything's SamPredictor."""

    def __init__(self, checkpoint, model_type: str = "vit_b", device=None) -> None:
        segment_anything = _import_segment_anything()
        if model_type not in segment_anything.sam_model_registry:
            raise ValueError(
                f"Unknown SAM model_type '{model_type}'. "
                f"Available: {sorted(segment_anything.sam_model_registry)}"
            )
        self.checkpoint = None if checkpoint is None else str(checkpoint)
        self.model_type = model_type
        self.device = device
        sam = segment_anything.sam_model_registry[model_type](checkpoint=self.checkpoint)
        if hasattr(sam, "to"):
            sam = sam.to(device=device)
        self.sam = sam
        self.predictor = segment_anything.SamPredictor(sam)

    def is_available(self) -> bool:
        return self.predictor is not None

    def predict_from_package(self, image, package: PromptPackage, multimask_output: bool = True) -> SAMMaskPrediction:
        predictions = self.predict_from_packages(image, [package], multimask_output=multimask_output)
        return predictions[0]

    def predict_from_packages(
        self,
        image,
        packages: Sequence[PromptPackage],
        multimask_output: bool = True,
    ) -> list[SAMMaskPrediction]:
        rgb = image_to_uint8_rgb(image)
        self.predictor.set_image(rgb)
        image_shape = rgb.shape[:2]
        predictions: list[SAMMaskPrediction] = []
        for package in packages:
            sam_inputs = convert_prompt_package_to_sam_inputs(package, image_shape=image_shape)
            masks, scores, logits = self.predictor.predict(
                point_coords=sam_inputs["point_coords"],
                point_labels=sam_inputs["point_labels"],
                box=sam_inputs["box"],
                mask_input=sam_inputs["mask_input"],
                multimask_output=multimask_output,
            )
            predictions.append(
                SAMMaskPrediction(
                    instance_id=package.instance_id,
                    masks=np.asarray(masks),
                    scores=np.asarray(scores, dtype=np.float32),
                    logits=None if logits is None else np.asarray(logits),
                    prompt_metadata=sam_inputs["metadata"],
                )
            )
        return predictions


def image_to_uint8_rgb(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Expected image [H,W], [H,W,3], [1,H,W], or [3,H,W], got shape {array.shape}.")

    array = array.astype(np.float32)
    if array.size and float(np.nanmax(array)) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _import_segment_anything():
    try:
        return importlib.import_module("segment_anything")
    except ImportError as exc:
        raise RuntimeError(SEGMENT_ANYTHING_MISSING) from exc
