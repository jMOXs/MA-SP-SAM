#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ma_sp_sam.refinement import PairRefinementInput, PairRefinementModule, save_pair_refinement_output
from ma_sp_sam.sam.sam_adapter import SAMMaskPrediction
from ma_sp_sam.utils.io import read_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine SAM candidate masks into paired axon/myelin instances.")
    parser.add_argument("--sam-pred-root", default="outputs/sam_predictions")
    parser.add_argument("--self-prompt-root", default="outputs/self_prompt_predictions")
    parser.add_argument("--processed-root", default="data/processed/astih_tem")
    parser.add_argument("--dataset", default="TEM1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", default="outputs/refined_predictions")
    args = parser.parse_args()

    rows = refine_directory(
        sam_pred_root=_resolve(args.sam_pred_root),
        self_prompt_root=_resolve(args.self_prompt_root),
        processed_root=_resolve(args.processed_root),
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        out_root=_resolve(args.out),
    )
    print(f"Wrote {len(rows)} refined rows to {_resolve(args.out) / 'summary.csv'}")


def refine_directory(
    *,
    sam_pred_root: Path,
    self_prompt_root: Path,
    processed_root: Path,
    dataset: str,
    split: str,
    limit: int | None,
    out_root: Path,
) -> list[dict[str, object]]:
    del processed_root
    module = PairRefinementModule()
    rows: list[dict[str, object]] = []
    sample_dirs = sorted(path for path in (sam_pred_root / dataset / split).iterdir() if path.is_dir())
    if limit is not None:
        sample_dirs = sample_dirs[: int(limit)]
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        semantic = _read_semantic_prediction(self_prompt_root / dataset / split / sample_id)
        predictions = _read_sam_predictions(sample_dir, image_shape=semantic.shape)
        proposal_label_map = _read_optional_label_map(self_prompt_root / dataset / split / sample_id / "proposal_labels.tif", semantic.shape)
        output = module.refine(
            PairRefinementInput(
                sample_id=sample_id,
                semantic_pred=semantic,
                proposal_label_map=proposal_label_map,
                sam_predictions=predictions,
                prompt_packages=[],
                image_shape=semantic.shape,
            )
        )
        out_dir = out_root / dataset / split / sample_id
        save_pair_refinement_output(output, out_dir)
        rows.append(_summary_row(dataset, split, sample_id, output))

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset",
            "split",
            "sample_id",
            "num_refined_instances",
            "num_missing_axon",
            "num_missing_myelin",
            "num_multi_axon",
            "mean_g_ratio",
            "median_g_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_sam_predictions(sample_dir: Path, *, image_shape: tuple[int, int]) -> list[SAMMaskPrediction]:
    npz_path = sample_dir / "sam_candidates.npz"
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        instance_ids = data["instance_ids"].astype(np.int64).tolist()
        masks = data["masks"]
        scores = data["scores"]
        best_indices = data["best_indices"] if "best_indices" in data else np.zeros(len(instance_ids), dtype=np.int64)
        predictions = []
        for index, instance_id in enumerate(instance_ids):
            pred_masks = np.asarray(masks[index]).astype(bool)
            pred_scores = np.asarray(scores[index], dtype=np.float32)
            predictions.append(
                SAMMaskPrediction(
                    instance_id=int(instance_id),
                    masks=pred_masks,
                    scores=pred_scores,
                    logits=None,
                    best_index=int(best_indices[index]),
                    prompt_metadata={"instance_id": int(instance_id)},
                )
            )
        return predictions

    label_path = sample_dir / "sam_candidate_masks.tif"
    if label_path.exists():
        labels = read_array(label_path)
        predictions = []
        for instance_id in sorted(int(value) for value in np.unique(labels) if int(value) != 0):
            predictions.append(
                SAMMaskPrediction(
                    instance_id=instance_id,
                    masks=(labels == instance_id)[None, ...],
                    scores=np.asarray([1.0], dtype=np.float32),
                    logits=None,
                    best_index=0,
                    prompt_metadata={"instance_id": instance_id},
                )
            )
        return predictions

    return []


def _read_semantic_prediction(sample_dir: Path) -> np.ndarray:
    for name in ("semantic_pred.tif", "semantic_pred.png", "semantic.png"):
        path = sample_dir / name
        if path.exists():
            with Image.open(path) as img:
                array = np.asarray(img)
            return _decode_semantic_array(array)
    raise FileNotFoundError(f"No semantic prediction found under {sample_dir}.")


def _decode_semantic_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array.astype(np.uint8)
    semantic = np.zeros(array.shape[:2], dtype=np.uint8)
    rgb = array[..., :3]
    semantic[np.all(rgb == np.asarray([180, 60, 220], dtype=np.uint8), axis=-1)] = 1
    semantic[np.all(rgb == np.asarray([40, 220, 80], dtype=np.uint8), axis=-1)] = 2
    return semantic


def _read_optional_label_map(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if path.exists():
        return read_array(path).astype(np.uint16)
    return np.zeros(shape, dtype=np.uint16)


def _summary_row(dataset: str, split: str, sample_id: str, output) -> dict[str, object]:
    table = output.pair_table
    flags = table["flags"].astype(str) if not table.empty else []
    valid_g = table.loc[table["fiber_area"] > 0, "g_ratio"] if not table.empty else []
    return {
        "dataset": dataset,
        "split": split,
        "sample_id": sample_id,
        "num_refined_instances": int(len(table)),
        "num_missing_axon": int(sum("missing_axon" in value for value in flags)),
        "num_missing_myelin": int(sum("missing_myelin" in value for value in flags)),
        "num_multi_axon": int(sum("multi_axon_component" in value for value in flags)),
        "mean_g_ratio": float(np.mean(valid_g)) if len(valid_g) else "",
        "median_g_ratio": float(np.median(valid_g)) if len(valid_g) else "",
    }


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
