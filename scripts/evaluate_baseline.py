#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ma_sp_sam.eval.baseline import evaluate_baseline_directory


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic baseline predictions via connected components.")
    parser.add_argument("--pred", required=True, help="Directory containing semantic prediction masks.")
    parser.add_argument("--gt", default="data/processed/astih_tem", help="Processed GT root with semantic.png per sample.")
    parser.add_argument("--out", default="outputs/reports/baseline_eval.csv", help="Output CSV path.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--fiber-iou-threshold", type=float, default=0.5)
    parser.add_argument("--class-iou-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of GT samples to evaluate.")
    args = parser.parse_args()

    rows = evaluate_baseline_directory(
        pred_root=_resolve(args.pred),
        gt_root=_resolve(args.gt),
        out_csv=_resolve(args.out),
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        fiber_iou_threshold=args.fiber_iou_threshold,
        class_iou_threshold=args.class_iou_threshold,
    )
    print(f"Wrote {len(rows)} sample rows to {_resolve(args.out)}")


if __name__ == "__main__":
    main()
