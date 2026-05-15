import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


torch = pytest.importorskip("torch")

from ma_sp_sam.models import SelfPromptGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from refine_sam_predictions import _read_semantic_prediction


def test_predict_self_prompt_saves_semantic_pred_labels_tif(tmp_path):
    checkpoint = tmp_path / "best.pt"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "self_prompt"
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

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "predict_self_prompt.py"),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config_path),
            "--dataset",
            "TEM1",
            "--split",
            "test",
            "--limit",
            "1",
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

    labels_path = out_dir / "TEM1" / "test" / "synthetic_0000" / "semantic_pred_labels.tif"
    assert labels_path.exists()
    with Image.open(labels_path) as img:
        labels = np.asarray(img)
    assert labels.dtype == np.uint8
    assert set(np.unique(labels).tolist()).issubset({0, 1, 2})


def test_refinement_reads_semantic_pred_labels_before_color_png(tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(sample_dir / "semantic_pred.png")
    labels = np.full((4, 4), 2, dtype=np.uint8)
    Image.fromarray(labels).save(sample_dir / "semantic_pred_labels.tif")

    read = _read_semantic_prediction(sample_dir)

    assert read.tolist() == labels.tolist()
