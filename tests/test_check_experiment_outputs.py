import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_check_experiment_outputs_fails_when_summary_all_is_missing(tmp_path):
    root = tmp_path / "experiments"
    root.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "check_experiment_outputs.py"),
            "--experiments-root",
            str(root),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    report = json.loads((root / "output_check.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert any(check["name"] == "summary_all_csv" and check["status"] == "fail" for check in report["checks"])


def test_check_experiment_outputs_writes_warning_report_without_nonzero_exit(tmp_path):
    root = tmp_path / "experiments"
    exp_dir = root / "tem1_internal"
    (exp_dir / "refined").mkdir(parents=True)
    _write_csv(
        root / "experiment_status.csv",
        [
            {
                "experiment_name": "tem1_internal",
                "status": "success",
                "dataset": "TEM1",
                "split": "test",
                "mode": "full",
                "error": "",
                "started_at": "2026-05-18T00:00:00+08:00",
                "finished_at": "2026-05-18T00:00:01+08:00",
            }
        ],
    )
    _write_csv(
        root / "summary_all.csv",
        [
            {
                "experiment_name": "tem1_internal",
                "sample_id": "s001",
                "axon_dice": "",
                "myelin_dice": "",
                "g_ratio_mae": "",
                "num_refined_instances": "2",
            }
        ],
    )
    _write_csv(root / "metrics_by_experiment.csv", [{"experiment_name": "tem1_internal", "metric": "axon_dice", "mean": "", "median": "", "std": "", "count": "0"}])
    _write_csv(exp_dir / "summary.csv", [{"sample_id": "s001", "num_refined_instances": "2"}])
    _write_csv(exp_dir / "refined" / "summary.csv", [{"sample_id": "s001", "num_refined_instances": "2"}])

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "check_experiment_outputs.py"),
            "--experiments-root",
            str(root),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "WARN" in result.stdout
    report = json.loads((root / "output_check.json").read_text(encoding="utf-8"))
    assert report["status"] == "WARN"
    assert any(check["name"] == "metric_completeness" and check["status"] == "warn" for check in report["checks"])
