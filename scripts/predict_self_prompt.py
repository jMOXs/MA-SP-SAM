#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
from PIL import Image

from ma_sp_sam.data.dataset_tem import DatasetTEM
from ma_sp_sam.eval.proposal_quality import (
    proposal_f1_at_iou,
    proposal_precision_at_iou,
    proposal_recall_at_iou,
)
from ma_sp_sam.models import SelfPromptGenerator
from ma_sp_sam.prompts.prompt_synthesizer import PromptSynthesizer
from ma_sp_sam.prompts.proposal_generator import ProposalGenerator
from ma_sp_sam.utils.io import load_yaml, write_tiff_u16
from ma_sp_sam.viz.self_prompt import save_heatmap_png, save_proposal_overlay, save_semantic_prediction_png
from train_self_prompt import SyntheticSelfPromptDataset


SUMMARY_FIELDS = [
    "dataset",
    "split",
    "sample_id",
    "num_proposals",
    "num_positive_points",
    "num_negative_points",
    "num_box_prompts",
    "num_ring_prompts",
    "mean_quality_score",
    "foreground_pixel_ratio",
    "gt_fibers",
    "proposal_to_gt_ratio",
    "proposal_recall50",
    "proposal_precision50",
    "proposal_f1_50",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SelfPromptGenerator inference and proposal QC.")
    parser.add_argument("--checkpoint", default="checkpoints/self_prompt/best.pt")
    parser.add_argument("--config", default="configs/train/self_prompt.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset", default="TEM1")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs/self_prompt_predictions")
    args = parser.parse_args()

    run_prediction(
        checkpoint_path=_resolve(args.checkpoint),
        config_path=_resolve(args.config),
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        device_name=args.device,
        out_root=_resolve(args.out),
    )


def run_prediction(
    *,
    checkpoint_path: Path,
    config_path: Path,
    dataset_name: str,
    split: str,
    limit: int | None,
    device_name: str | None,
    out_root: Path,
) -> list[dict[str, Any]]:
    config = load_yaml(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_config = {**checkpoint_config, **config}
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(model_config)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset = _build_dataset(model_config, dataset_name=dataset_name, split=split)
    rows: list[dict[str, Any]] = []
    out_root.mkdir(parents=True, exist_ok=True)
    max_items = len(dataset) if limit is None else min(int(limit), len(dataset))
    generator = ProposalGenerator(
        center_threshold=float(model_config.get("center_threshold", 0.5)),
        boundary_threshold=float(model_config.get("boundary_threshold", 0.5)),
        min_area=int(model_config.get("proposal_min_area", 1)),
    )
    synthesizer = PromptSynthesizer(boundary_threshold=float(model_config.get("boundary_threshold", 0.5)))

    with torch.no_grad():
        for index in range(max_items):
            item = dataset[index]
            row_dataset = str(item.get("dataset", dataset_name if dataset_name else model_config.get("dataset", "TEM1")))
            row_split = str(item.get("split", split))
            if model_config.get("synthetic", False):
                row_dataset = dataset_name
                row_split = split
            sample_id = str(item.get("sample_id", f"sample_{index:04d}"))
            sample_dir = out_root / row_dataset / row_split / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            image = item["image"].unsqueeze(0).to(device)
            outputs = model(image)
            semantic_logits = outputs.semantic_logits[0].detach().cpu()
            semantic_pred = semantic_logits.argmax(dim=0).numpy().astype(np.uint8)
            center_heatmap = torch.sigmoid(outputs.axon_center_heatmap[0, 0]).detach().cpu().numpy()
            inner_boundary = torch.sigmoid(outputs.inner_boundary_map[0, 0]).detach().cpu().numpy()
            outer_boundary = torch.sigmoid(outputs.outer_boundary_map[0, 0]).detach().cpu().numpy()
            boundary_maps = np.stack([inner_boundary, outer_boundary], axis=0).astype(np.float32)
            quality_score = torch.sigmoid(outputs.prompt_quality_score[0]).detach().cpu().numpy()

            proposal_batch = generator.generate(
                semantic_logits=semantic_logits.numpy(),
                center_heatmap=center_heatmap,
                boundary_maps=boundary_maps,
            )
            proposals = proposal_batch.proposals[0]
            proposal_label_map = proposal_batch.label_maps[0]
            prompts = synthesizer.synthesize(
                center_heatmap=center_heatmap,
                semantic_logits=semantic_logits.numpy(),
                boundary_maps=boundary_maps,
                instance_proposals=proposals,
            )

            save_semantic_prediction_png(sample_dir / "semantic_pred.png", semantic_pred)
            Image.fromarray(semantic_pred.astype(np.uint8)).save(sample_dir / "semantic_pred_labels.tif")
            save_heatmap_png(sample_dir / "center_heatmap.png", center_heatmap)
            save_heatmap_png(sample_dir / "inner_boundary_pred.png", inner_boundary)
            save_heatmap_png(sample_dir / "outer_boundary_pred.png", outer_boundary)
            write_tiff_u16(sample_dir / "proposal_labels.tif", proposal_label_map)
            save_proposal_overlay(sample_dir / "proposal_overlay.png", item["image"].numpy(), proposal_label_map)
            prompt_summary = _prompt_summary(
                dataset=row_dataset,
                split=row_split,
                sample_id=sample_id,
                prompts=prompts,
                proposals=proposals,
            )
            (sample_dir / "prompt_summary.json").write_text(
                json.dumps(prompt_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            rows.append(
                _summary_row(
                    dataset=row_dataset,
                    split=row_split,
                    sample_id=sample_id,
                    prompts=prompts,
                    proposals=proposals,
                    quality_score=quality_score,
                    semantic_pred=semantic_pred,
                    item=item,
                    proposal_label_map=proposal_label_map,
                )
            )

    summary_path = out_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {summary_path}")
    return rows


def _build_model(config: dict[str, Any]) -> SelfPromptGenerator:
    return SelfPromptGenerator(
        in_channels=int(config.get("in_channels", 1)),
        hidden_channels=int(config.get("hidden_channels", 64)),
        num_blocks=int(config.get("num_blocks", 2)),
        num_classes=3,
        distance_channels=2,
    )


def _build_dataset(config: dict[str, Any], *, dataset_name: str, split: str):
    if config.get("synthetic", False):
        return SyntheticSelfPromptDataset(
            samples=int(config.get("synthetic_samples", 4)),
            height=int(config.get("synthetic_height", 32)),
            width=int(config.get("synthetic_width", 32)),
        )
    return DatasetTEM(
        _resolve(config["manifest_path"]),
        processed_root=_resolve(config["processed_root"]),
        dataset=dataset_name or config.get("dataset", "TEM1"),
        split=split,
        return_tensors=True,
        center_target="gaussian",
        center_sigma=float(config.get("center_sigma", 3.0)),
    )


def _summary_row(
    *,
    dataset: str,
    split: str,
    sample_id: str,
    prompts,
    proposals,
    quality_score: np.ndarray,
    semantic_pred: np.ndarray,
    item: dict[str, Any],
    proposal_label_map: np.ndarray,
) -> dict[str, Any]:
    gt_fiber = _maybe_numpy(item.get("fiber_instance"))
    gt_fibers = _count_instances(gt_fiber) if gt_fiber is not None else None
    recall = precision = f1 = None
    if gt_fiber is not None:
        recall = proposal_recall_at_iou(proposal_label_map, gt_fiber, iou_threshold=0.5)
        precision = proposal_precision_at_iou(proposal_label_map, gt_fiber, iou_threshold=0.5)
        f1 = proposal_f1_at_iou(proposal_label_map, gt_fiber, iou_threshold=0.5)

    return {
        "dataset": dataset,
        "split": split,
        "sample_id": sample_id,
        "num_proposals": len(proposals),
        "num_positive_points": len(prompts.positive_points),
        "num_negative_points": len(prompts.negative_points),
        "num_box_prompts": len(prompts.box_prompts),
        "num_ring_prompts": len(prompts.ring_prompts),
        "mean_quality_score": float(np.mean(quality_score)),
        "foreground_pixel_ratio": float(np.mean(semantic_pred > 0)),
        "gt_fibers": "" if gt_fibers is None else gt_fibers,
        "proposal_to_gt_ratio": "" if not gt_fibers else float(len(proposals) / gt_fibers),
        "proposal_recall50": "" if recall is None else recall,
        "proposal_precision50": "" if precision is None else precision,
        "proposal_f1_50": "" if f1 is None else f1,
    }


def _prompt_summary(*, dataset: str, split: str, sample_id: str, prompts, proposals) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "sample_id": sample_id,
        "num_proposals": len(proposals),
        "num_positive_points": len(prompts.positive_points),
        "num_negative_points": len(prompts.negative_points),
        "num_box_prompts": len(prompts.box_prompts),
        "num_ring_prompts": len(prompts.ring_prompts),
        "packages": [
            {
                "instance_id": package.instance_id,
                "quality_prior": package.quality_prior,
                "positive_points": [point.xy for point in package.positive_points],
                "negative_points": [point.xy for point in package.negative_points],
                "box": package.box_prompt.xyxy,
                "num_ring_points": int(package.ring_prompt.ring_points.shape[0]),
            }
            for package in prompts.packages
        ],
    }


def _maybe_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _count_instances(label_map: np.ndarray | None) -> int:
    if label_map is None:
        return 0
    return int(sum(1 for value in np.unique(label_map) if int(value) != 0))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
