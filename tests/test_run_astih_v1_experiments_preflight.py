import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_config(path: Path, *, work_root: Path, mode: str = "full") -> None:
    processed_root = path.parent / "processed"
    (processed_root / "TEM1" / "test").mkdir(parents=True, exist_ok=True)
    self_prompt_config = path.parent / "self_prompt.yaml"
    self_prompt_config.write_text(f"processed_root: {processed_root}\n", encoding="utf-8")
    path.write_text(
        f"""
self_prompt_checkpoint: {path.parent / "missing_self_prompt.pt"}
self_prompt_config: {self_prompt_config}
sam_checkpoint: {path.parent / "missing_sam.pth"}
sam_model_type: vit_b
limit: 1
device: cpu
work_root: {work_root}
experiments:
  - name: tem1_preflight
    dataset: TEM1
    split: test
    mode: {mode}
""",
        encoding="utf-8",
    )


def test_astih_v1_preflight_only_writes_preflight_and_status_without_pipeline(tmp_path):
    work_root = tmp_path / "experiments"
    config_path = tmp_path / "astih_v1.yaml"
    _write_config(config_path, work_root=work_root)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_astih_v1_experiments.py"),
            "--config",
            str(config_path),
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    exp_dir = work_root / "tem1_preflight"
    status = json.loads((exp_dir / "run_status.json").read_text(encoding="utf-8"))
    preflight = json.loads((exp_dir / "preflight.json").read_text(encoding="utf-8"))
    assert status["status"] == "preflight_failed"
    assert "self_prompt_checkpoint" in status["error"]
    assert any(check["status"] == "fail" for check in preflight)
    assert not (exp_dir / "self_prompt").exists()


def test_astih_v1_no_strict_preflight_records_fails_but_does_not_mark_preflight_failed(tmp_path):
    work_root = tmp_path / "experiments"
    config_path = tmp_path / "astih_v1.yaml"
    _write_config(config_path, work_root=work_root, mode="skip_sam")

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_astih_v1_experiments.py"),
            "--config",
            str(config_path),
            "--no-strict-preflight",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    exp_dir = work_root / "tem1_preflight"
    status = json.loads((exp_dir / "run_status.json").read_text(encoding="utf-8"))
    preflight = json.loads((exp_dir / "preflight.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "No such file" in status["error"] or "missing_self_prompt" in status["error"]
    assert any(check["status"] == "fail" for check in preflight)


def test_astih_v1_preflight_resolves_processed_root_from_self_prompt_config(tmp_path):
    work_root = tmp_path / "experiments"
    config_path = tmp_path / "astih_v1.yaml"
    config_path.write_text(
        f"""
self_prompt_checkpoint: {tmp_path / "missing_self_prompt.pt"}
self_prompt_config: configs/train/self_prompt.yaml
sam_checkpoint: {tmp_path / "missing_sam.pth"}
work_root: {work_root}
experiments:
  - name: tem1_preflight
    dataset: TEM1
    split: test
    mode: full
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_astih_v1_experiments.py"),
            "--config",
            str(config_path),
            "--preflight-only",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    preflight = json.loads((work_root / "tem1_preflight" / "preflight.json").read_text(encoding="utf-8"))
    processed_root = [check for check in preflight if check["name"] == "processed_root"][0]
    assert str(PROJECT_ROOT / "data" / "processed" / "astih_tem") in processed_root["message"]
