#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ma_sp_sam.eval.label_quality import summarize_pair_table


def iter_pair_tables(processed_root: Path):
    for pair_table_path in sorted(processed_root.glob("*/*/*/pair_table.csv")):
        sample_dir = pair_table_path.parent
        yield sample_dir, pd.read_csv(pair_table_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed paired labels.")
    parser.add_argument("--processed", default="data/processed/astih_tem")
    parser.add_argument("--report", default="outputs/reports/astih_label_qc.csv")
    parser.add_argument("--with-baseline", action="store_true")
    parser.add_argument("--export-overlays", action="store_true")
    args = parser.parse_args()

    processed_root = Path(args.processed)
    if not processed_root.is_absolute():
        processed_root = PROJECT_ROOT / processed_root

    rows = []
    for sample_dir, pair_table in iter_pair_tables(processed_root):
        summary = summarize_pair_table(pair_table)
        summary["dataset"] = sample_dir.parent.parent.name
        summary["split"] = sample_dir.parent.name
        summary["sample_id"] = sample_dir.name
        rows.append(summary)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(report_path, index=False)
    print(f"Wrote {len(df)} sample summaries to {report_path}")
    if args.with_baseline:
        print("Label-only baseline summary is represented by derived pair-table self-consistency metrics.")
    if args.export_overlays:
        print("QC overlays are exported during build-labels with --export-qc.")


if __name__ == "__main__":
    main()
