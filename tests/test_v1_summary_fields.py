import csv
from pathlib import Path

from ma_sp_sam.reporting.v1_summary import SUMMARY_FIELDS, write_v1_summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["dataset", "split", "sample_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_v1_summary_includes_dataset_split_and_qc_fields(tmp_path):
    self_csv = tmp_path / "self_prompt.csv"
    sam_csv = tmp_path / "sam.csv"
    refined_csv = tmp_path / "refined.csv"
    out_csv = tmp_path / "summary.csv"
    _write_csv(
        self_csv,
        [
            {
                "dataset": "TEM1",
                "split": "test",
                "sample_id": "s001",
                "num_proposals": 4,
                "proposal_recall50": 0.75,
                "proposal_precision50": 0.6,
                "proposal_f1_50": 0.666,
                "foreground_pixel_ratio": 0.2,
            }
        ],
    )
    _write_csv(
        sam_csv,
        [
            {
                "dataset": "TEM1",
                "split": "test",
                "sample_id": "s001",
                "num_sam_predictions": 4,
            }
        ],
    )
    _write_csv(
        refined_csv,
        [
            {
                "dataset": "TEM1",
                "split": "test",
                "sample_id": "s001",
                "num_refined_instances": 3,
                "fiber_iou50_recall": 0.5,
                "fiber_iou50_precision": 0.75,
                "axon_dice": 0.8,
                "myelin_dice": 0.7,
                "pair_accuracy_proxy": 0.6,
                "g_ratio_mae": 0.1,
                "mean_g_ratio": 0.62,
                "num_missing_axon": 1,
                "num_missing_myelin": 0,
            }
        ],
    )

    rows = write_v1_summary(self_csv, sam_csv, refined_csv, out_csv)
    saved_rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8")))

    assert "dataset" in SUMMARY_FIELDS
    assert "split" in SUMMARY_FIELDS
    assert "proposal_precision50" in SUMMARY_FIELDS
    assert "num_missing_myelin" in SUMMARY_FIELDS
    assert rows == saved_rows
    row = saved_rows[0]
    assert row["dataset"] == "TEM1"
    assert row["split"] == "test"
    assert row["proposal_precision50"] == "0.6"
    assert row["foreground_pixel_ratio"] == "0.2"
    assert row["fiber_iou50_recall"] == "0.5"
    assert row["pair_accuracy_proxy"] == "0.6"
    assert row["num_missing_axon"] == "1"
