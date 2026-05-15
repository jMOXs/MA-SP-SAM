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

from ma_sp_sam.prompts.prompt_synthesizer import PromptSynthesizer
from ma_sp_sam.prompts.proposal_generator import ProposalGenerator
from ma_sp_sam.sam.sam_adapter import SAMAdapter, SAMMaskPrediction, SEGMENT_ANYTHING_MISSING, best_mask
from ma_sp_sam.utils.io import load_yaml, write_tiff_u16
from predict_self_prompt import _build_dataset, _build_model, _resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM from Self-Prompt packages.")
    parser.add_argument("--self-prompt-checkpoint", default="checkpoints/self_prompt/best.pt")
    parser.add_argument("--self-prompt-config", default="configs/train/self_prompt.yaml")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--dataset", default="TEM1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs/sam_predictions")
    parser.add_argument("--use-mask-input", action="store_true")
    parser.add_argument("--save-all-candidates", action="store_true")
    args = parser.parse_args()

    try:
        run_sam_prediction(
            self_prompt_checkpoint=_resolve(args.self_prompt_checkpoint),
            self_prompt_config=_resolve(args.self_prompt_config),
            sam_checkpoint=_resolve(args.sam_checkpoint),
            sam_model_type=args.sam_model_type,
            dataset_name=args.dataset,
            split=args.split,
            limit=args.limit,
            device_name=args.device,
            out_root=_resolve(args.out),
            use_mask_input=args.use_mask_input,
            save_all_candidates=args.save_all_candidates,
        )
    except RuntimeError as exc:
        if SEGMENT_ANYTHING_MISSING in str(exc):
            print(SEGMENT_ANYTHING_MISSING, file=sys.stderr)
            raise SystemExit(2) from None
        raise


def run_sam_prediction(
    *,
    self_prompt_checkpoint: Path,
    self_prompt_config: Path,
    sam_checkpoint: Path,
    sam_model_type: str,
    dataset_name: str,
    split: str,
    limit: int | None,
    device_name: str | None,
    out_root: Path,
    use_mask_input: bool = False,
    save_all_candidates: bool = False,
) -> list[dict[str, Any]]:
    config = load_yaml(self_prompt_config)
    checkpoint = torch.load(self_prompt_checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_config = {**checkpoint_config, **config}
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(model_config)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset = _build_dataset(model_config, dataset_name=dataset_name, split=split)
    sam_adapter = SAMAdapter(checkpoint=sam_checkpoint, model_type=sam_model_type, device=device)
    proposal_generator = ProposalGenerator(
        center_threshold=float(model_config.get("center_threshold", 0.5)),
        boundary_threshold=float(model_config.get("boundary_threshold", 0.5)),
        min_area=int(model_config.get("proposal_min_area", 1)),
    )
    synthesizer = PromptSynthesizer(boundary_threshold=float(model_config.get("boundary_threshold", 0.5)))

    out_root.mkdir(parents=True, exist_ok=True)
    max_items = len(dataset) if limit is None else min(int(limit), len(dataset))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(max_items):
            item = dataset[index]
            row_dataset = str(item.get("dataset", dataset_name))
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
            center_heatmap = torch.sigmoid(outputs.axon_center_heatmap[0, 0]).detach().cpu().numpy()
            inner_boundary = torch.sigmoid(outputs.inner_boundary_map[0, 0]).detach().cpu().numpy()
            outer_boundary = torch.sigmoid(outputs.outer_boundary_map[0, 0]).detach().cpu().numpy()
            boundary_maps = np.stack([inner_boundary, outer_boundary], axis=0).astype(np.float32)

            proposal_batch = proposal_generator.generate(
                semantic_logits=semantic_logits.numpy(),
                center_heatmap=center_heatmap,
                boundary_maps=boundary_maps,
            )
            proposals = proposal_batch.proposals[0]
            packages = synthesizer.synthesize(
                center_heatmap=center_heatmap,
                semantic_logits=semantic_logits.numpy(),
                boundary_maps=boundary_maps,
                instance_proposals=proposals,
            ).packages
            sam_predictions = sam_adapter.predict_from_packages(
                item["image"],
                packages,
                multimask_output=True,
                use_mask_input=use_mask_input,
            )
            candidate_label_map = _best_candidate_label_map(sam_predictions, image_shape=tuple(item["semantic"].shape[-2:]))
            write_tiff_u16(sample_dir / "sam_candidate_masks.tif", candidate_label_map)
            _write_candidates_npz(sample_dir / "sam_candidates.npz", sam_predictions)
            if save_all_candidates:
                _write_candidate_arrays(sample_dir, sam_predictions)
            _write_scores_csv(sample_dir / "sam_scores.csv", sam_predictions)
            summary = _sam_prompt_summary(
                dataset=row_dataset,
                split=row_split,
                sample_id=sample_id,
                packages=packages,
                sam_predictions=sam_predictions,
            )
            (sample_dir / "sam_prompt_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            rows.append(
                {
                    "dataset": row_dataset,
                    "split": row_split,
                    "sample_id": sample_id,
                    "num_packages": len(packages),
                    "num_sam_predictions": len(sam_predictions),
                    "num_candidate_pixels": int(np.count_nonzero(candidate_label_map)),
                    "use_mask_input": bool(use_mask_input),
                }
            )
    _write_summary_csv(out_root / "summary.csv", rows)
    print(f"Wrote {len(rows)} SAM prediction rows to {out_root / 'summary.csv'}")
    return rows


def _best_candidate_label_map(
    predictions: list[SAMMaskPrediction],
    *,
    image_shape: tuple[int, int],
) -> np.ndarray:
    label_map = np.zeros(image_shape, dtype=np.uint16)
    for prediction in predictions:
        if prediction.masks.size == 0:
            continue
        mask = best_mask(prediction).astype(bool)
        if mask.shape != image_shape:
            raise ValueError(f"SAM mask shape {mask.shape} does not match image shape {image_shape}.")
        label_map[mask] = int(prediction.instance_id)
    return label_map


def _write_candidates_npz(path: Path, predictions: list[SAMMaskPrediction]) -> None:
    instance_ids = np.asarray([prediction.instance_id for prediction in predictions], dtype=np.int64)
    scores = np.empty(len(predictions), dtype=object)
    masks = np.empty(len(predictions), dtype=object)
    best_indices = np.asarray([prediction.best_index for prediction in predictions], dtype=np.int64)
    for index, prediction in enumerate(predictions):
        scores[index] = prediction.scores
        masks[index] = prediction.masks
    np.savez(path, instance_ids=instance_ids, masks=masks, scores=scores, best_indices=best_indices)


def _write_candidate_arrays(sample_dir: Path, predictions: list[SAMMaskPrediction]) -> None:
    for prediction in predictions:
        np.save(sample_dir / f"instance_{prediction.instance_id}_masks.npy", prediction.masks)
        np.save(sample_dir / f"instance_{prediction.instance_id}_scores.npy", prediction.scores)


def _write_scores_csv(path: Path, predictions: list[SAMMaskPrediction]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id", "candidate_index", "score", "best_index"])
        writer.writeheader()
        for prediction in predictions:
            for index, score in enumerate(prediction.scores.tolist()):
                writer.writerow(
                    {
                        "instance_id": prediction.instance_id,
                        "candidate_index": index,
                        "score": float(score),
                        "best_index": prediction.best_index,
                    }
                )


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "split",
        "sample_id",
        "num_packages",
        "num_sam_predictions",
        "num_candidate_pixels",
        "use_mask_input",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sam_prompt_summary(
    *,
    dataset: str,
    split: str,
    sample_id: str,
    packages,
    sam_predictions: list[SAMMaskPrediction],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "sample_id": sample_id,
        "num_packages": len(packages),
        "num_sam_predictions": len(sam_predictions),
        "packages": [
            {
                "instance_id": package.instance_id,
                "quality_prior": float(package.quality_prior),
                "box": package.box_prompt.xyxy,
                "num_positive_points": len(package.positive_points),
                "num_negative_points": len(package.negative_points),
            }
            for package in packages
        ],
        "sam_predictions": [
            {
                "instance_id": prediction.instance_id,
                "scores": prediction.scores.tolist(),
                "num_masks": int(prediction.masks.shape[0]),
                "best_index": prediction.best_index,
                "prompt_metadata": _json_safe_metadata(prediction.prompt_metadata),
            }
            for prediction in sam_predictions
        ],
    }


def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            safe[key] = value.tolist()
        else:
            safe[key] = value
    return safe


if __name__ == "__main__":
    main()
