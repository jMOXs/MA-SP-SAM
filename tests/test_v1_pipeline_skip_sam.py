import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from ma_sp_sam.models import SelfPromptGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v1_pipeline_skip_sam_uses_existing_sam_predictions(tmp_path):
    work_dir = tmp_path / "v1"
    checkpoint = tmp_path / "self_prompt.pt"
    config_path = tmp_path / "self_prompt.yaml"
    config = {
        "synthetic": True,
        "synthetic_samples": 1,
        "synthetic_height": 12,
        "synthetic_width": 12,
        "in_channels": 1,
        "hidden_channels": 8,
        "num_blocks": 1,
        "center_threshold": 0.0,
        "boundary_threshold": 1.1,
    }
    model = SelfPromptGenerator(in_channels=1, hidden_channels=8, num_blocks=1)
    torch.save({"model_state": model.state_dict(), "config": config}, checkpoint)
    config_path.write_text("\n".join(f"{key}: {value}" for key, value in config.items()) + "\n", encoding="utf-8")

    sam_dir = work_dir / "sam" / "TEM1" / "test" / "synthetic_0000"
    sam_dir.mkdir(parents=True)
    masks = np.ones((1, 1, 12, 12), dtype=bool)
    np.savez(
        sam_dir / "sam_candidates.npz",
        instance_ids=np.asarray([1], dtype=np.int64),
        masks=masks,
        scores=np.asarray([[0.9]], dtype=np.float32),
        best_indices=np.asarray([0], dtype=np.int64),
    )
    (work_dir / "sam").mkdir(exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_v1_pipeline.py"),
            "--self-prompt-checkpoint",
            str(checkpoint),
            "--self-prompt-config",
            str(config_path),
            "--sam-checkpoint",
            str(tmp_path / "unused_sam.pth"),
            "--dataset",
            "TEM1",
            "--split",
            "test",
            "--limit",
            "1",
            "--device",
            "cpu",
            "--work-dir",
            str(work_dir),
            "--skip-sam",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert (work_dir / "self_prompt" / "summary.csv").exists()
    assert (work_dir / "refined" / "summary.csv").exists()
    summary_path = work_dir / "summary.csv"
    assert summary_path.exists()
    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    assert rows[0]["sample_id"] == "synthetic_0000"
    assert "num_proposals" in rows[0]
    assert "num_refined_instances" in rows[0]
