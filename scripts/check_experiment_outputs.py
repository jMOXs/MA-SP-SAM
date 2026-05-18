#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


KEY_METRICS = [
    "axon_dice",
    "myelin_dice",
    "g_ratio_mae",
    "num_refined_instances",
    "proposal_recall50",
    "proposal_precision50",
    "fiber_iou50_recall",
    "fiber_iou50_precision",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ASTIH V1 experiment outputs.")
    parser.add_argument("--experiments-root", default="outputs/experiments")
    args = parser.parse_args()
    report = check_experiment_outputs(Path(args.experiments_root))
    print(f"{report['status']}: {report['num_pass']} pass, {report['num_warn']} warn, {report['num_fail']} fail")
    raise SystemExit(1 if report["status"] == "FAIL" else 0)


def check_experiment_outputs(experiments_root: str | Path) -> dict[str, Any]:
    root = Path(experiments_root)
    root.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, str]] = []

    status_path = root / "experiment_status.csv"
    summary_path = root / "summary_all.csv"
    metrics_path = root / "metrics_by_experiment.csv"
    checks.append(_file_check("experiment_status_csv", status_path))
    checks.append(_file_check("summary_all_csv", summary_path))
    checks.append(_file_check("metrics_by_experiment_csv", metrics_path))

    status_rows = _read_csv(status_path) if status_path.exists() else []
    for row in status_rows:
        if row.get("status") != "success":
            continue
        name = row.get("experiment_name", "")
        exp_dir = root / name
        checks.append(_file_check(f"{name}:summary_csv", exp_dir / "summary.csv"))
        checks.append(_file_check(f"{name}:refined_summary_csv", exp_dir / "refined" / "summary.csv"))

    if summary_path.exists():
        checks.append(_metric_completeness_check(_read_csv(summary_path)))

    status = _overall_status(checks)
    report = {
        "status": status,
        "num_pass": sum(check["status"] == "pass" for check in checks),
        "num_warn": sum(check["status"] == "warn" for check in checks),
        "num_fail": sum(check["status"] == "fail" for check in checks),
        "checks": checks,
    }
    (root / "output_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _file_check(name: str, path: Path) -> dict[str, str]:
    if path.exists():
        return {"name": name, "status": "pass", "message": f"Found {path}"}
    return {"name": name, "status": "fail", "message": f"Missing {path}"}


def _metric_completeness_check(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {"name": "metric_completeness", "status": "warn", "message": "summary_all.csv has no sample rows."}
    total = 0
    missing = 0
    for row in rows:
        for metric in KEY_METRICS:
            if metric not in row:
                continue
            total += 1
            if _is_missing(row.get(metric, "")):
                missing += 1
    if total == 0:
        return {"name": "metric_completeness", "status": "warn", "message": "No key metrics were found in summary_all.csv."}
    ratio = missing / total
    status = "warn" if ratio > 0.5 else "pass"
    return {
        "name": "metric_completeness",
        "status": status,
        "message": f"Missing/NaN key metric ratio: {ratio:.3f} ({missing}/{total}).",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return True
    try:
        return not math.isfinite(float(text))
    except ValueError:
        return False


def _overall_status(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "FAIL"
    if any(check["status"] == "warn" for check in checks):
        return "WARN"
    return "PASS"


if __name__ == "__main__":
    main()
