import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_refine_sam_predictions_script_runs_on_synthetic_files(tmp_path):
    dataset = "TEM1"
    split = "test"
    sample_id = "sample_synthetic"
    sam_dir = tmp_path / "sam_predictions" / dataset / split / sample_id
    self_prompt_dir = tmp_path / "self_prompt_predictions" / dataset / split / sample_id
    processed_dir = tmp_path / "processed"
    out_dir = tmp_path / "refined"
    sam_dir.mkdir(parents=True)
    self_prompt_dir.mkdir(parents=True)

    semantic = np.zeros((8, 8), dtype=np.uint8)
    semantic[1:5, 1:5] = 1
    semantic[2:4, 2:4] = 2
    Image.fromarray(semantic).save(self_prompt_dir / "semantic_pred.png")

    masks = np.zeros((1, 1, 8, 8), dtype=bool)
    masks[0, 0, 1:5, 1:5] = True
    scores = np.asarray([[0.9]], dtype=np.float32)
    np.savez(
        sam_dir / "sam_candidates.npz",
        instance_ids=np.asarray([3], dtype=np.int64),
        masks=masks,
        scores=scores,
        best_indices=np.asarray([0], dtype=np.int64),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "refine_sam_predictions.py"),
            "--sam-pred-root",
            str(tmp_path / "sam_predictions"),
            "--self-prompt-root",
            str(tmp_path / "self_prompt_predictions"),
            "--processed-root",
            str(processed_dir),
            "--dataset",
            dataset,
            "--split",
            split,
            "--limit",
            "1",
            "--out",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    sample_out = out_dir / dataset / split / sample_id
    assert "Wrote" in result.stdout
    assert (sample_out / "refined_fiber_instance.tif").exists()
    assert (sample_out / "refined_axon_instance.tif").exists()
    assert (sample_out / "refined_myelin_instance.tif").exists()
    pair_table = sample_out / "refined_pair_table.csv"
    assert pair_table.exists()
    rows = list(csv.DictReader(pair_table.open("r", encoding="utf-8")))
    assert rows[0]["instance_id"] == "3"
    assert rows[0]["g_ratio"] == "0.5"
    assert (out_dir / "summary.csv").exists()
