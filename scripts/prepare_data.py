#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ma_sp_sam.data.aimseg_index import index_aimseg_archives, write_aimseg_manifest
from ma_sp_sam.data.astih_splits_index import index_astih_splits, read_manifest, write_manifest
from ma_sp_sam.data.mask_loader import load_semantic_sample
from ma_sp_sam.labels.paired_instances import build_paired_instances, export_paired_label_bundle
from ma_sp_sam.utils.io import load_yaml
from ma_sp_sam.utils.paths import resolve_path
from ma_sp_sam.viz.overlays import save_qc_overlay


def _resolve(path: str | Path) -> Path:
    return resolve_path(PROJECT_ROOT, path)


def _load_records(config: dict, datasets: list[str] | None) -> list:
    manifest_path = _resolve(config["manifest_path"])
    if manifest_path.exists():
        records = read_manifest(manifest_path)
    else:
        records = index_astih_splits(
            _resolve(config["splits_root"]),
            datasets=config.get("datasets", ["TEM1", "TEM2"]),
            split_names=config.get("splits", ["train", "test"]),
        )
    if datasets:
        records = [record for record in records if record.dataset in set(datasets)]
    return records


def stage_index(config: dict, datasets: list[str] | None) -> None:
    selected = datasets or config.get("datasets", ["TEM1", "TEM2"])
    records = index_astih_splits(
        _resolve(config["splits_root"]),
        datasets=selected,
        split_names=config.get("splits", ["train", "test"]),
    )
    manifest_path = _resolve(config["manifest_path"])
    write_manifest(manifest_path, records)
    counts = Counter((record.dataset, record.split) for record in records)
    print(f"Wrote {len(records)} ASTIH records to {manifest_path}")
    for key in sorted(counts):
        print(f"{key[0]} {key[1]}: {counts[key]}")


def stage_build_labels(config: dict, args: argparse.Namespace) -> None:
    label_config = load_yaml(_resolve(args.label_config))
    records = _load_records(config, args.datasets)
    if args.limit is not None:
        records = records[: args.limit]

    processed_root = _resolve(config["processed_root"])
    for index, record in enumerate(records, start=1):
        sample = load_semantic_sample(record)
        bundle = build_paired_instances(sample.semantic, **label_config)
        sample_dir = processed_root / record.dataset / record.split / record.sample_id
        export_paired_label_bundle(bundle, sample_dir)
        (sample_dir / "metadata.json").write_text(json.dumps(sample.metadata, indent=2), encoding="utf-8")
        if args.export_qc:
            save_qc_overlay(sample_dir / "qc_overlay.png", sample.image, bundle)
        print(f"[{index}/{len(records)}] {record.dataset}/{record.split}/{record.sample_id}: {len(bundle.pair_table)} fibers")


def stage_index_aimseg(config: dict) -> None:
    records = index_aimseg_archives(_resolve(config["aimseg_root"]))
    manifest_path = _resolve(config["manifest_path"])
    write_aimseg_manifest(manifest_path, records)
    counts = Counter(record.dataset for record in records)
    print(f"Wrote {len(records)} AimSeg records to {manifest_path}")
    for dataset, count in sorted(counts.items()):
        print(f"{dataset}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MA-SP-SAM data artifacts.")
    parser.add_argument("--config", default="configs/data/astih_tem.yaml")
    parser.add_argument("--label-config", default="configs/labels/paired_instance.yaml")
    parser.add_argument("--stage", choices=["index", "build-labels", "index-aimseg"], required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--export-qc", action="store_true")
    args = parser.parse_args()

    config = load_yaml(_resolve(args.config))
    if args.stage == "index":
        stage_index(config, args.datasets)
    elif args.stage == "build-labels":
        stage_build_labels(config, args)
    elif args.stage == "index-aimseg":
        stage_index_aimseg(config)


if __name__ == "__main__":
    main()
