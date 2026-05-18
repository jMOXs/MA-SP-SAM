from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


SUMMARY_FIELDS = [
    "dataset",
    "split",
    "sample_id",
    "num_proposals",
    "proposal_recall50",
    "proposal_precision50",
    "proposal_f1_50",
    "foreground_pixel_ratio",
    "num_sam_predictions",
    "num_refined_instances",
    "fiber_iou50_recall",
    "fiber_iou50_precision",
    "mean_g_ratio",
    "axon_dice",
    "myelin_dice",
    "pair_accuracy_proxy",
    "g_ratio_mae",
    "num_missing_axon",
    "num_missing_myelin",
]


def write_v1_summary(
    self_prompt_summary: str | Path | None,
    sam_summary: str | Path | None,
    refined_summary: str | Path | None,
    out_csv: str | Path,
) -> list[dict[str, str]]:
    """Merge V1 stage summaries into a compact per-sample report."""
    self_rows = _read_rows(self_prompt_summary)
    sam_rows = _read_rows(sam_summary)
    refined_rows = _read_rows(refined_summary)
    keys = _ordered_keys(self_rows, sam_rows, refined_rows)
    by_self = {_row_key(row): row for row in self_rows}
    by_sam = {_row_key(row): row for row in sam_rows}
    by_refined = {_row_key(row): row for row in refined_rows}

    rows: list[dict[str, str]] = []
    for key in keys:
        self_row = by_self.get(key, {})
        sam_row = by_sam.get(key, {})
        refined_row = by_refined.get(key, {})
        rows.append(
            {
                "dataset": key[0],
                "split": key[1],
                "sample_id": key[2],
                "num_proposals": self_row.get("num_proposals", ""),
                "proposal_recall50": self_row.get("proposal_recall50", ""),
                "proposal_precision50": self_row.get("proposal_precision50", ""),
                "proposal_f1_50": self_row.get("proposal_f1_50", ""),
                "foreground_pixel_ratio": self_row.get("foreground_pixel_ratio", ""),
                "num_sam_predictions": sam_row.get("num_sam_predictions", ""),
                "num_refined_instances": refined_row.get("num_refined_instances", ""),
                "fiber_iou50_recall": refined_row.get("fiber_iou50_recall", ""),
                "fiber_iou50_precision": refined_row.get("fiber_iou50_precision", ""),
                "mean_g_ratio": refined_row.get("mean_g_ratio", ""),
                "axon_dice": refined_row.get("axon_dice", ""),
                "myelin_dice": refined_row.get("myelin_dice", ""),
                "pair_accuracy_proxy": refined_row.get("pair_accuracy_proxy", ""),
                "g_ratio_mae": refined_row.get("g_ratio_mae", ""),
                "num_missing_axon": refined_row.get("num_missing_axon", ""),
                "num_missing_myelin": refined_row.get("num_missing_myelin", ""),
            }
        )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_rows(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _ordered_keys(*row_groups: Iterable[dict[str, str]]) -> list[tuple[str, str, str]]:
    seen: set[tuple[str, str, str]] = set()
    keys: list[tuple[str, str, str]] = []
    for rows in row_groups:
        for row in rows:
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def _row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row.get("dataset", ""), row.get("split", ""), row.get("sample_id", "")
