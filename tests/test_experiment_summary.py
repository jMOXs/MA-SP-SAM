import csv
import json
from pathlib import Path

from ma_sp_sam.reporting.experiment_summary import write_experiment_summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_status(path: Path, *, name: str, dataset: str, split: str, status: str = "success") -> None:
    path.write_text(
        json.dumps(
            {
                "name": name,
                "dataset": dataset,
                "split": split,
                "mode": "skip_sam",
                "status": status,
                "error": "",
                "started_at": "2026-05-18T00:00:00+08:00",
                "finished_at": "2026-05-18T00:00:01+08:00",
            }
        ),
        encoding="utf-8",
    )


def test_experiment_summary_merges_experiments_and_aggregates_metrics(tmp_path):
    root = tmp_path / "experiments"
    exp_a = root / "tem1_internal"
    exp_b = root / "tem2_external"
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)
    _write_status(exp_a / "run_status.json", name="tem1_internal", dataset="TEM1", split="test")
    _write_status(exp_b / "run_status.json", name="tem2_external", dataset="TEM2", split="test")
    _write_csv(
        exp_a / "summary.csv",
        [
            {
                "dataset": "TEM1",
                "split": "test",
                "sample_id": "a",
                "num_proposals": 3,
                "proposal_recall50": 0.5,
                "proposal_precision50": 0.25,
                "proposal_f1_50": 0.333,
                "num_sam_predictions": 3,
                "num_refined_instances": 2,
                "fiber_iou50_recall": 0.5,
                "fiber_iou50_precision": 1.0,
                "axon_dice": 0.8,
                "myelin_dice": 0.7,
                "pair_accuracy_proxy": 0.5,
                "g_ratio_mae": 0.1,
                "mean_g_ratio": 0.6,
                "num_missing_axon": 1,
                "num_missing_myelin": 0,
            },
            {
                "dataset": "TEM1",
                "split": "test",
                "sample_id": "b",
                "num_proposals": 5,
                "proposal_recall50": 1.0,
                "proposal_precision50": 0.5,
                "proposal_f1_50": 0.667,
                "num_sam_predictions": 5,
                "num_refined_instances": 4,
                "fiber_iou50_recall": 1.0,
                "fiber_iou50_precision": 0.5,
                "axon_dice": 0.6,
                "myelin_dice": 0.9,
                "pair_accuracy_proxy": 1.0,
                "g_ratio_mae": 0.3,
                "mean_g_ratio": 0.7,
                "num_missing_axon": 0,
                "num_missing_myelin": 1,
            },
        ],
    )
    _write_csv(
        exp_b / "summary.csv",
        [
            {
                "dataset": "TEM2",
                "split": "test",
                "sample_id": "c",
                "num_proposals": 1,
                "proposal_recall50": "",
                "proposal_precision50": "",
                "proposal_f1_50": "",
                "num_sam_predictions": 1,
                "num_refined_instances": 1,
                "fiber_iou50_recall": "",
                "fiber_iou50_precision": "",
                "axon_dice": "",
                "myelin_dice": "",
                "pair_accuracy_proxy": "",
                "g_ratio_mae": "",
                "mean_g_ratio": 0.5,
                "num_missing_axon": 0,
                "num_missing_myelin": 0,
            }
        ],
    )

    rows, metrics = write_experiment_summary(root)
    summary_rows = list(csv.DictReader((root / "summary_all.csv").open("r", encoding="utf-8")))
    metric_rows = list(csv.DictReader((root / "metrics_by_experiment.csv").open("r", encoding="utf-8")))

    assert rows == summary_rows
    assert len(summary_rows) == 3
    assert summary_rows[0]["experiment_name"] == "tem1_internal"
    assert summary_rows[2]["dataset"] == "TEM2"
    assert metrics == metric_rows
    axon_rows = [row for row in metric_rows if row["experiment_name"] == "tem1_internal" and row["metric"] == "axon_dice"]
    assert axon_rows[0]["count"] == "2"
    assert axon_rows[0]["mean"] == "0.7"
    refined_rows = [
        row
        for row in metric_rows
        if row["experiment_name"] == "tem2_external" and row["metric"] == "num_refined_instances"
    ]
    assert refined_rows[0]["count"] == "1"
    assert refined_rows[0]["mean"] == "1.0"
