from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev


SUMMARY_FIELDS = [
    "experiment_name",
    "status",
    "dataset",
    "split",
    "sample_id",
    "num_proposals",
    "proposal_recall50",
    "proposal_precision50",
    "proposal_f1_50",
    "num_sam_predictions",
    "num_refined_instances",
    "fiber_iou50_recall",
    "fiber_iou50_precision",
    "axon_dice",
    "myelin_dice",
    "pair_accuracy_proxy",
    "g_ratio_mae",
    "mean_g_ratio",
    "num_missing_axon",
    "num_missing_myelin",
]

AGGREGATE_METRICS = [
    "axon_dice",
    "myelin_dice",
    "fiber_iou50_recall",
    "fiber_iou50_precision",
    "pair_accuracy_proxy",
    "g_ratio_mae",
    "proposal_recall50",
    "proposal_precision50",
    "num_refined_instances",
]

METRIC_FIELDS = ["experiment_name", "metric", "mean", "median", "std", "count"]

STATUS_FIELDS = [
    "experiment_name",
    "status",
    "dataset",
    "split",
    "mode",
    "error",
    "started_at",
    "finished_at",
]


def write_experiment_summary(
    experiments_root: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    root = Path(experiments_root)
    status_rows = _collect_status_rows(root)
    rows = _collect_summary_rows(root)
    metrics = _aggregate_rows(rows)
    _write_csv(root / "summary_all.csv", SUMMARY_FIELDS, rows)
    _write_csv(root / "metrics_by_experiment.csv", METRIC_FIELDS, metrics)
    _write_csv(root / "experiment_status.csv", STATUS_FIELDS, status_rows)
    return rows, metrics, status_rows


def _collect_status_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not root.exists():
        return rows
    for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        status = _read_status(experiment_dir / "run_status.json", experiment_dir.name)
        rows.append(
            {
                "experiment_name": status.get("name", experiment_dir.name),
                "status": status.get("status", ""),
                "dataset": status.get("dataset", ""),
                "split": status.get("split", ""),
                "mode": status.get("mode", ""),
                "error": status.get("error", ""),
                "started_at": status.get("started_at", ""),
                "finished_at": status.get("finished_at", ""),
            }
        )
    return rows


def _collect_summary_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not root.exists():
        return rows
    for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        status = _read_status(experiment_dir / "run_status.json", experiment_dir.name)
        summary_path = experiment_dir / "summary.csv"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            for source in csv.DictReader(f):
                row = {field: "" for field in SUMMARY_FIELDS}
                row["experiment_name"] = status.get("name", experiment_dir.name)
                row["status"] = status.get("status", "")
                row["dataset"] = source.get("dataset") or status.get("dataset", "")
                row["split"] = source.get("split") or status.get("split", "")
                for field in SUMMARY_FIELDS:
                    if field in {"experiment_name", "status", "dataset", "split"}:
                        continue
                    row[field] = source.get(field, "")
                rows.append(row)
    return rows


def _aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    success_rows = [row for row in rows if row.get("status") in {"", "success"}]
    experiments = sorted({row["experiment_name"] for row in success_rows})
    metric_rows: list[dict[str, str]] = []
    for experiment in experiments:
        experiment_rows = [row for row in success_rows if row["experiment_name"] == experiment]
        for metric in AGGREGATE_METRICS:
            values = [_to_float(row.get(metric, "")) for row in experiment_rows]
            clean = [value for value in values if value is not None]
            if not clean:
                continue
            metric_rows.append(
                {
                    "experiment_name": experiment,
                    "metric": metric,
                    "mean": _format_float(mean(clean)),
                    "median": _format_float(median(clean)),
                    "std": _format_float(pstdev(clean) if len(clean) > 1 else 0.0),
                    "count": str(len(clean)),
                }
            )
    return metric_rows


def _read_status(path: Path, fallback_name: str) -> dict[str, str]:
    if not path.exists():
        return {"name": fallback_name, "status": "", "dataset": "", "split": ""}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: "" if value is None else str(value) for key, value in data.items()}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _format_float(value: float) -> str:
    return str(round(float(value), 12))
