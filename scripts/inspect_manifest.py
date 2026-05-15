#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ma_sp_sam.data.aimseg_index import index_aimseg_archives
from ma_sp_sam.data.astih_splits_index import index_astih_splits
from ma_sp_sam.utils.io import load_yaml
from ma_sp_sam.utils.paths import resolve_path


def _resolve(path: str | Path) -> Path:
    return resolve_path(PROJECT_ROOT, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect configured manifests or source directories.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_yaml(_resolve(args.config))
    if "splits_root" in config:
        splits_root = _resolve(config["splits_root"])
        print(f"ASTIH splits root: {splits_root}")
        print(f"Datasets: {', '.join(config.get('datasets', []))}")
        print(f"Splits: {', '.join(config.get('splits', []))}")
        if args.dry_run:
            return
        records = index_astih_splits(splits_root, config.get("datasets", ["TEM1", "TEM2"]), config.get("splits", ["train", "test"]))
        counts = Counter((record.dataset, record.split) for record in records)
        print(f"Records: {len(records)}")
        for key in sorted(counts):
            print(f"{key[0]} {key[1]}: {counts[key]}")
    elif "aimseg_root" in config:
        aimseg_root = _resolve(config["aimseg_root"])
        print(f"AimSeg root: {aimseg_root}")
        records = index_aimseg_archives(aimseg_root)
        counts = Counter(record.dataset for record in records)
        print(f"Records: {len(records)}")
        for dataset, count in sorted(counts.items()):
            print(f"{dataset}: {count}")
    else:
        raise ValueError(f"Unsupported config shape: {args.config}")


if __name__ == "__main__":
    main()
