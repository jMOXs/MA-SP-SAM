import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _save(path: Path, array, dtype=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if dtype is not None:
        arr = arr.astype(dtype)
    Image.fromarray(arr).save(path)


def test_refine_sam_predictions_summary_includes_gt_qc_metrics(tmp_path):
    dataset = "TEM1"
    split = "test"
    sample_id = "sample_gt"
    semantic = np.zeros((8, 8), dtype=np.uint8)
    semantic[1:5, 1:5] = 1
    semantic[2:4, 2:4] = 2

    self_prompt_dir = tmp_path / "self_prompt" / dataset / split / sample_id
    sam_dir = tmp_path / "sam" / dataset / split / sample_id
    processed_dir = tmp_path / "processed" / dataset / split / sample_id
    out_dir = tmp_path / "refined"
    _save(self_prompt_dir / "semantic_pred_labels.tif", semantic, dtype=np.uint8)
    sam_dir.mkdir(parents=True)
    masks = np.zeros((1, 1, 8, 8), dtype=bool)
    masks[0, 0, 1:5, 1:5] = True
    np.savez(
        sam_dir / "sam_candidates.npz",
        instance_ids=np.asarray([1], dtype=np.int64),
        masks=masks,
        scores=np.asarray([[0.9]], dtype=np.float32),
        best_indices=np.asarray([0], dtype=np.int64),
    )

    _save(processed_dir / "semantic.png", semantic, dtype=np.uint8)
    _save(processed_dir / "fiber_instance.tif", (semantic > 0).astype(np.uint16), dtype=np.uint16)
    _save(processed_dir / "axon_instance.tif", (semantic == 2).astype(np.uint16), dtype=np.uint16)
    _save(processed_dir / "myelin_instance.tif", (semantic == 1).astype(np.uint16), dtype=np.uint16)
    pd.DataFrame(
        [
            {
                "fiber_id": 1,
                "axon_area": 4,
                "myelin_area": 12,
                "fiber_area": 16,
                "g_ratio": 0.5,
                "flags": "",
            }
        ]
    ).to_csv(processed_dir / "pair_table.csv", index=False)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "refine_sam_predictions.py"),
            "--sam-pred-root",
            str(tmp_path / "sam"),
            "--self-prompt-root",
            str(tmp_path / "self_prompt"),
            "--processed-root",
            str(tmp_path / "processed"),
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

    rows = list(csv.DictReader((out_dir / "summary.csv").open("r", encoding="utf-8")))
    row = rows[0]
    assert row["axon_dice"] == "1.0"
    assert row["myelin_dice"] == "1.0"
    assert row["g_ratio_mae"] == "0.0"
    assert row["pair_accuracy_proxy"] == "1.0"
