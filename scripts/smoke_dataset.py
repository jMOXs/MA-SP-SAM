#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ma_sp_sam.data.dataset_tem import DatasetTEM


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the TEM dataset API.")
    parser.add_argument("--manifest", default="data/manifests/astih_tem_manifest.jsonl")
    parser.add_argument("--processed", default="data/processed/astih_tem")
    parser.add_argument("--dataset", default="TEM1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    processed = Path(args.processed)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    if not processed.is_absolute():
        processed = PROJECT_ROOT / processed

    dataset = DatasetTEM(manifest, processed_root=processed, dataset=args.dataset, split=args.split)
    print(f"Dataset records: {len(dataset)}")
    for idx in range(min(args.batch_size, len(dataset))):
        item = dataset[idx]
        print(
            f"{idx}: {item['dataset']}/{item['split']}/{item['sample_id']} "
            f"image={item['image'].shape} semantic={item['semantic'].shape} "
            f"fibers={len(item['pair_table'])}"
        )


if __name__ == "__main__":
    main()
