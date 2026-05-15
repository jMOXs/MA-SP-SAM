import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from ma_sp_sam.models import SelfPromptGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_predict_self_prompt_runs_from_synthetic_checkpoint(tmp_path):
    checkpoint = tmp_path / "best.pt"
    out_dir = tmp_path / "predictions"
    config_path = tmp_path / "self_prompt_predict.yaml"
    config = {
        "synthetic": True,
        "synthetic_samples": 2,
        "synthetic_height": 16,
        "synthetic_width": 16,
        "in_channels": 1,
        "hidden_channels": 8,
        "num_blocks": 1,
        "center_threshold": 0.0,
        "boundary_threshold": 1.1,
    }
    model = SelfPromptGenerator(in_channels=1, hidden_channels=8, num_blocks=1)
    torch.save({"model_state": model.state_dict(), "config": config}, checkpoint)
    config_path.write_text(
        "\n".join(f"{key}: {value}" for key, value in config.items()) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "predict_self_prompt.py"),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config_path),
            "--split",
            "test",
            "--dataset",
            "synthetic",
            "--limit",
            "2",
            "--device",
            "cpu",
            "--out",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    summary_path = out_dir / "summary.csv"
    assert summary_path.exists()
    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    assert len(rows) == 2
    assert {"proposal_recall50", "proposal_precision50", "proposal_f1_50"} <= set(rows[0])

    sample_dirs = sorted(path for path in out_dir.rglob("*") if path.is_dir() and (path / "prompt_summary.json").exists())
    assert sample_dirs
    sample_dir = sample_dirs[0]
    assert (sample_dir / "semantic_pred.png").exists()
    assert (sample_dir / "center_heatmap.png").exists()
    assert (sample_dir / "inner_boundary_pred.png").exists()
    assert (sample_dir / "outer_boundary_pred.png").exists()
    assert (sample_dir / "proposal_labels.tif").exists()
    prompt_summary = json.loads((sample_dir / "prompt_summary.json").read_text(encoding="utf-8"))
    assert "num_proposals" in prompt_summary
