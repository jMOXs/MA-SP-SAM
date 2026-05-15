from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ma_sp_sam.prompts import CoarseMaskPrompt, PointPrompt, PromptPackage, PromptSynthesisResult


DEFAULT_SAM_ROOT = Path("/sydata/js/segment_anything")
DEFAULT_MICRO_SAM_ROOT = Path("/sydata/js/new_micro_sam/micro-sam")


@dataclass(frozen=True)
class CandidateMask:
    instance_id: int
    masks: torch.Tensor
    low_res_logits: torch.Tensor
    iou_predictions: torch.Tensor


@dataclass(frozen=True)
class SAMAdapterOutput:
    candidates: list[CandidateMask]
    masks: torch.Tensor
    low_res_logits: torch.Tensor
    iou_predictions: torch.Tensor
    instance_ids: list[int]


class SAMAdapter(nn.Module):
    """Thin adapter around SAM / micro-SAM prompt encoder and mask decoder."""

    def __init__(
        self,
        sam_model: nn.Module,
        *,
        freeze_image_encoder: bool = True,
        train_mask_decoder: bool = True,
        train_prompt_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.sam = sam_model
        self.configure_trainability(
            freeze_image_encoder=freeze_image_encoder,
            train_mask_decoder=train_mask_decoder,
            train_prompt_encoder=train_prompt_encoder,
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        backend: str = "sam",
        model_type: str = "vit_b",
        checkpoint_path: str | Path | None = None,
        source_root: str | Path | None = None,
        segment_anything_root: str | Path = DEFAULT_SAM_ROOT,
        device: str | torch.device = "cpu",
        freeze_image_encoder: bool = True,
        train_mask_decoder: bool = True,
        train_prompt_encoder: bool = False,
        allow_download: bool = False,
        **model_kwargs,
    ) -> "SAMAdapter":
        backend_key = backend.lower()
        if backend_key in {"sam", "segment_anything", "vit_b"}:
            sam = _load_segment_anything_model(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                source_root=source_root or segment_anything_root,
                device=device,
                **model_kwargs,
            )
        elif backend_key in {"micro_sam", "microsam", "μsam"}:
            sam = _load_micro_sam_model(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                source_root=source_root or DEFAULT_MICRO_SAM_ROOT,
                segment_anything_root=segment_anything_root,
                device=device,
                allow_download=allow_download,
                **model_kwargs,
            )
        else:
            raise ValueError("backend must be one of 'sam' or 'micro_sam'.")

        return cls(
            sam,
            freeze_image_encoder=freeze_image_encoder,
            train_mask_decoder=train_mask_decoder,
            train_prompt_encoder=train_prompt_encoder,
        )

    @property
    def device(self) -> torch.device:
        try:
            return next(self.sam.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def configure_trainability(
        self,
        *,
        freeze_image_encoder: bool,
        train_mask_decoder: bool,
        train_prompt_encoder: bool = False,
    ) -> None:
        _set_module_trainable(self.sam.image_encoder, trainable=not freeze_image_encoder)
        _set_module_trainable(self.sam.prompt_encoder, trainable=train_prompt_encoder)
        _set_module_trainable(self.sam.mask_decoder, trainable=train_mask_decoder)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of SAM-preprocessed images with the wrapped image encoder."""
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected images [B,C,H,W] or [C,H,W], got shape {tuple(images.shape)}.")

        images = images.to(self.device)
        inputs = torch.stack([self.sam.preprocess(image) for image in images], dim=0)
        if any(param.requires_grad for param in self.sam.image_encoder.parameters()):
            return self.sam.image_encoder(inputs)
        with torch.no_grad():
            return self.sam.image_encoder(inputs)

    def forward(
        self,
        *,
        image_embeddings: torch.Tensor,
        prompts: PromptSynthesisResult | Sequence[PromptPackage],
        prompt_image_size: tuple[int, int] | None = None,
        output_size: tuple[int, int] | None = None,
        multimask_output: bool = True,
        use_mask_prompt: bool = True,
    ) -> SAMAdapterOutput:
        packages = _prompt_packages(prompts)
        if not packages:
            empty = torch.empty(0, device=self.device)
            return SAMAdapterOutput([], empty, empty, empty, [])

        image_embeddings = image_embeddings.to(self.device)
        if image_embeddings.ndim == 3:
            image_embeddings = image_embeddings.unsqueeze(0)
        if image_embeddings.ndim != 4:
            raise ValueError(f"Expected image_embeddings [B,C,H,W] or [C,H,W], got shape {tuple(image_embeddings.shape)}.")

        prompt_tensors = self._packages_to_sam_inputs(
            packages,
            prompt_image_size=prompt_image_size,
            use_mask_prompt=use_mask_prompt,
        )
        prompt_batch = prompt_tensors.boxes.shape[0]
        image_embeddings = _expand_image_embeddings(image_embeddings, prompt_batch)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(prompt_tensors.point_coords, prompt_tensors.point_labels),
            boxes=prompt_tensors.boxes,
            masks=prompt_tensors.mask_inputs,
        )
        low_res_logits, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        masks = _resize_masks(low_res_logits, output_size)
        instance_ids = [package.instance_id for package in packages]
        candidates = [
            CandidateMask(
                instance_id=instance_id,
                masks=masks[index],
                low_res_logits=low_res_logits[index],
                iou_predictions=iou_predictions[index],
            )
            for index, instance_id in enumerate(instance_ids)
        ]
        return SAMAdapterOutput(
            candidates=candidates,
            masks=masks,
            low_res_logits=low_res_logits,
            iou_predictions=iou_predictions,
            instance_ids=instance_ids,
        )

    def _packages_to_sam_inputs(
        self,
        packages: Sequence[PromptPackage],
        *,
        prompt_image_size: tuple[int, int] | None,
        use_mask_prompt: bool,
    ) -> "_PromptTensors":
        target_size = _prompt_encoder_input_size(self.sam.prompt_encoder)
        source_size = prompt_image_size or _infer_prompt_image_size(packages) or target_size
        scale_x = float(target_size[1]) / float(source_size[1])
        scale_y = float(target_size[0]) / float(source_size[0])

        max_points = max(len(package.positive_points) + len(package.negative_points) for package in packages)
        max_points = max(max_points, 1)
        point_coords = torch.zeros(len(packages), max_points, 2, dtype=torch.float32, device=self.device)
        point_labels = -torch.ones(len(packages), max_points, dtype=torch.int64, device=self.device)
        boxes = torch.zeros(len(packages), 4, dtype=torch.float32, device=self.device)
        masks = []

        for package_index, package in enumerate(packages):
            points = [*package.positive_points, *package.negative_points]
            for point_index, point in enumerate(points):
                point_coords[package_index, point_index] = _scale_xy(point, scale_x=scale_x, scale_y=scale_y)
                point_labels[package_index, point_index] = int(point.label)

            x0, y0, x1, y1 = package.box_prompt.xyxy
            boxes[package_index] = torch.tensor(
                [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y],
                dtype=torch.float32,
                device=self.device,
            )
            if use_mask_prompt:
                masks.append(_coarse_mask_to_tensor(package.coarse_mask_prompt, self.sam.prompt_encoder, self.device))

        mask_inputs = torch.stack(masks, dim=0) if use_mask_prompt else None
        return _PromptTensors(point_coords=point_coords, point_labels=point_labels, boxes=boxes, mask_inputs=mask_inputs)


@dataclass(frozen=True)
class _PromptTensors:
    point_coords: torch.Tensor
    point_labels: torch.Tensor
    boxes: torch.Tensor
    mask_inputs: torch.Tensor | None


def _load_segment_anything_model(
    *,
    model_type: str,
    checkpoint_path: str | Path | None,
    source_root: str | Path,
    device: str | torch.device,
    **model_kwargs,
) -> nn.Module:
    with _temporary_sys_path(source_root):
        from segment_anything import sam_model_registry

    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown SAM model_type '{model_type}'. Available: {sorted(sam_model_registry)}")
    sam = sam_model_registry[model_type](checkpoint=None if checkpoint_path is None else str(checkpoint_path), **model_kwargs)
    sam.to(device=device)
    return sam


def _load_micro_sam_model(
    *,
    model_type: str,
    checkpoint_path: str | Path | None,
    source_root: str | Path,
    segment_anything_root: str | Path,
    device: str | torch.device,
    allow_download: bool,
    **model_kwargs,
) -> nn.Module:
    if checkpoint_path is None and not allow_download:
        raise ValueError("micro_sam loading without checkpoint_path may download weights; pass allow_download=True to permit it.")
    with _temporary_sys_path(segment_anything_root, source_root):
        from micro_sam.util import get_sam_model

    _, sam = get_sam_model(
        model_type=model_type,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        device=device,
        return_sam=True,
        **model_kwargs,
    )
    return sam


@contextmanager
def _temporary_sys_path(*paths: str | Path):
    old_path = list(sys.path)
    for path in reversed([str(Path(p)) for p in paths if p is not None]):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = old_path


def _set_module_trainable(module: nn.Module, *, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable
    module.train(trainable)


def _prompt_packages(prompts: PromptSynthesisResult | Sequence[PromptPackage]) -> list[PromptPackage]:
    if isinstance(prompts, PromptSynthesisResult):
        return list(prompts.packages)
    return list(prompts)


def _prompt_encoder_input_size(prompt_encoder: nn.Module) -> tuple[int, int]:
    size = getattr(prompt_encoder, "input_image_size", None)
    if size is None:
        raise AttributeError("prompt_encoder must expose input_image_size.")
    return int(size[0]), int(size[1])


def _prompt_encoder_mask_input_size(prompt_encoder: nn.Module) -> tuple[int, int]:
    size = getattr(prompt_encoder, "mask_input_size", None)
    if size is None:
        h, w = getattr(prompt_encoder, "image_embedding_size")
        return int(4 * h), int(4 * w)
    return int(size[0]), int(size[1])


def _infer_prompt_image_size(packages: Sequence[PromptPackage]) -> tuple[int, int] | None:
    for package in packages:
        mask = package.coarse_mask_prompt.mask
        if mask is not None:
            mask = np.asarray(mask)
            if mask.ndim >= 2:
                return int(mask.shape[-2]), int(mask.shape[-1])
    return None


def _scale_xy(point: PointPrompt, *, scale_x: float, scale_y: float) -> torch.Tensor:
    x, y = point.xy
    return torch.tensor([x * scale_x, y * scale_y], dtype=torch.float32)


def _coarse_mask_to_tensor(mask_prompt: CoarseMaskPrompt, prompt_encoder: nn.Module, device: torch.device) -> torch.Tensor:
    mask = torch.as_tensor(np.asarray(mask_prompt.mask), dtype=torch.float32, device=device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    else:
        raise ValueError(f"Expected coarse mask [H,W] or [1,H,W], got shape {tuple(mask.shape)}.")
    mask = F.interpolate(mask, size=_prompt_encoder_mask_input_size(prompt_encoder), mode="bilinear", align_corners=False)
    return mask.squeeze(0)


def _expand_image_embeddings(image_embeddings: torch.Tensor, prompt_batch: int) -> torch.Tensor:
    if image_embeddings.shape[0] == prompt_batch:
        return image_embeddings
    if image_embeddings.shape[0] == 1:
        return image_embeddings.expand(prompt_batch, -1, -1, -1)
    raise ValueError(
        "image_embeddings batch must be 1 or match number of prompt packages, "
        f"got {image_embeddings.shape[0]} embeddings for {prompt_batch} prompts."
    )


def _resize_masks(low_res_logits: torch.Tensor, output_size: tuple[int, int] | None) -> torch.Tensor:
    if output_size is None:
        return low_res_logits
    return F.interpolate(low_res_logits, size=output_size, mode="bilinear", align_corners=False)
