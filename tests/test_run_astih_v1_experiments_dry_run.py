import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_astih_v1_experiment_runner_dry_run_prints_plan_without_execution(tmp_path):
    work_root = tmp_path / "experiments"
    config_path = tmp_path / "astih_v1.yaml"
    config_path.write_text(
        f"""
self_prompt_checkpoint: {tmp_path / "missing_self_prompt.pt"}
self_prompt_config: {tmp_path / "missing_self_prompt.yaml"}
sam_checkpoint: {tmp_path / "missing_sam.pth"}
sam_model_type: vit_b
limit: 1
device: cpu
work_root: {work_root}
experiments:
  - name: tem1_internal
    dataset: TEM1
    split: test
    mode: full
  - name: tem2_external
    dataset: TEM2
    split: test
    mode: full
  - name: tem1_skip_sam_smoke
    dataset: TEM1
    split: test
    mode: skip_sam
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_astih_v1_experiments.py"),
            "--config",
            str(config_path),
            "--only",
            "tem1_internal,tem1_skip_sam_smoke",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "tem1_internal" in result.stdout
    assert "tem1_skip_sam_smoke" in result.stdout
    assert "tem2_external" not in result.stdout
    assert not work_root.exists()
