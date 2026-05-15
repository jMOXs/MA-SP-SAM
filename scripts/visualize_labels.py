#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ma_sp_sam.data.astih_splits_index import read_manifest
from ma_sp_sam.utils.io import read_array, read_image, read_mask
from ma_sp_sam.viz.overlays import colorize_instance_labels, make_paired_instance_preview, save_axon_myelin_overlay


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _raw_image_lookup(manifest_path: Path) -> dict[tuple[str, str, str], Path]:
    if not manifest_path.exists():
        return {}
    return {
        (record.dataset, record.split, record.sample_id): record.image_path
        for record in read_manifest(manifest_path)
    }


def _iter_sample_dirs(processed: Path, dataset: str | None, split: str | None, sample: str | None):
    datasets = [dataset] if dataset else [path.name for path in sorted(processed.iterdir()) if path.is_dir()]
    for dataset_name in datasets:
        dataset_dir = processed / dataset_name
        if not dataset_dir.exists():
            continue
        splits = [split] if split else [path.name for path in sorted(dataset_dir.iterdir()) if path.is_dir()]
        for split_name in splits:
            split_dir = dataset_dir / split_name
            if not split_dir.exists():
                continue
            if sample:
                sample_dir = split_dir / sample
                if sample_dir.exists():
                    yield sample_dir
            else:
                for sample_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                    if (sample_dir / "axon_instance.tif").exists() and (sample_dir / "myelin_instance.tif").exists():
                        yield sample_dir


def _fallback_image(sample_dir: Path) -> np.ndarray:
    semantic_path = sample_dir / "semantic.png"
    if semantic_path.exists():
        semantic = read_mask(semantic_path)
        return np.where(semantic > 0, 90, 15).astype(np.uint8)
    axon = read_array(sample_dir / "axon_instance.tif")
    return np.zeros(axon.shape, dtype=np.uint8)


def visualize_sample(
    sample_dir: Path,
    manifest_lookup: dict[tuple[str, str, str], Path],
    overlay_name: str,
    alpha: float,
    write_overlay: bool,
    write_instance_previews: bool,
) -> list[Path]:
    dataset = sample_dir.parent.parent.name
    split = sample_dir.parent.name
    sample_id = sample_dir.name
    image_path = manifest_lookup.get((dataset, split, sample_id))
    image = read_image(image_path) if image_path and image_path.exists() else _fallback_image(sample_dir)

    pair_table_path = sample_dir / "pair_table.csv"
    pair_table = pd.read_csv(pair_table_path) if pair_table_path.exists() else None
    axon_instance = read_array(sample_dir / "axon_instance.tif")
    myelin_instance = read_array(sample_dir / "myelin_instance.tif")
    fiber_instance = read_array(sample_dir / "fiber_instance.tif") if (sample_dir / "fiber_instance.tif").exists() else None

    outputs: list[Path] = []
    if write_overlay:
        output_path = sample_dir / overlay_name
        save_axon_myelin_overlay(
            output_path,
            image,
            axon_instance=axon_instance,
            myelin_instance=myelin_instance,
            fiber_instance=fiber_instance,
            pair_table=pair_table,
            alpha=alpha,
        )
        outputs.append(output_path)

    if write_instance_previews:
        axon_preview = sample_dir / "axon_instance_preview.png"
        myelin_preview = sample_dir / "myelin_instance_preview.png"
        paired_preview = sample_dir / "paired_instance_preview.png"
        colorize_instance_labels(axon_instance).save(axon_preview)
        colorize_instance_labels(myelin_instance).save(myelin_preview)
        make_paired_instance_preview(axon_instance=axon_instance, myelin_instance=myelin_instance).save(paired_preview)
        outputs.extend([axon_preview, myelin_preview, paired_preview])

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create axon/myelin visual overlays from processed labels.")
    parser.add_argument("--processed", default="data/processed/astih_tem")
    parser.add_argument("--manifest", default="data/manifests/astih_tem_manifest.jsonl")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-name", default="axon_myelin_overlay.png")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--overlay-only", action="store_true")
    parser.add_argument("--instance-only", action="store_true")
    args = parser.parse_args()

    processed = _resolve(args.processed)
    manifest_lookup = _raw_image_lookup(_resolve(args.manifest))
    sample_dirs = list(_iter_sample_dirs(processed, args.dataset, args.split, args.sample))
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]

    write_overlay = not args.instance_only
    write_instance_previews = not args.overlay_only
    for index, sample_dir in enumerate(sample_dirs, start=1):
        output_paths = visualize_sample(
            sample_dir,
            manifest_lookup,
            args.out_name,
            args.alpha,
            write_overlay=write_overlay,
            write_instance_previews=write_instance_previews,
        )
        print(f"[{index}/{len(sample_dirs)}] wrote {', '.join(str(path) for path in output_paths)}")


if __name__ == "__main__":
    main()
