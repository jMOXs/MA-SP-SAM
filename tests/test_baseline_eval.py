import numpy as np
import pandas as pd
from PIL import Image

from ma_sp_sam.eval.baseline import evaluate_baseline_directory, evaluate_semantic_prediction


def test_evaluate_semantic_prediction_returns_perfect_scores_for_identical_mask():
    semantic = np.array(
        [
            [0, 1, 1, 0],
            [0, 2, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 2, 0],
        ],
        dtype=np.uint8,
    )

    result = evaluate_semantic_prediction(semantic, semantic, sample_id="sample")

    assert result["axon_dice"] == 1.0
    assert result["myelin_dice"] == 1.0
    assert result["fiber_ap50"] == 1.0
    assert result["pair_accuracy"] == 1.0
    assert result["g_ratio_mae"] == 0.0
    assert result["gt_fibers"] == 2
    assert result["pred_fibers"] == 2


def test_evaluate_semantic_prediction_penalizes_missing_myelin_pair():
    gt = np.array(
        [
            [0, 1, 1],
            [0, 2, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    pred = np.array(
        [
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    result = evaluate_semantic_prediction(pred, gt, sample_id="sample")

    assert result["axon_dice"] == 1.0
    assert result["myelin_dice"] == 0.0
    assert result["fiber_ap50"] == 0.0
    assert result["pair_accuracy"] == 0.0


def test_evaluate_baseline_directory_writes_sample_rows(tmp_path):
    gt_root = tmp_path / "processed"
    pred_root = tmp_path / "pred"
    sample_dir = gt_root / "TEM1" / "test" / "sample_TEM"
    sample_dir.mkdir(parents=True)
    semantic = np.array([[0, 1, 2], [0, 0, 0]], dtype=np.uint8)
    Image.fromarray(semantic).save(sample_dir / "semantic.png")
    Image.fromarray(np.array([[0, 1, 1], [0, 0, 0]], dtype=np.uint16)).save(sample_dir / "fiber_instance.tif")
    Image.fromarray(np.array([[0, 0, 1], [0, 0, 0]], dtype=np.uint16)).save(sample_dir / "axon_instance.tif")
    Image.fromarray(np.array([[0, 1, 0], [0, 0, 0]], dtype=np.uint16)).save(sample_dir / "myelin_instance.tif")
    pd.DataFrame(
        [{"fiber_id": 1, "axon_area": 1, "myelin_area": 1, "fiber_area": 2, "g_ratio": np.sqrt(0.5), "flags": ""}]
    ).to_csv(sample_dir / "pair_table.csv", index=False)

    pred_root.mkdir()
    Image.fromarray(semantic).save(pred_root / "sample_TEM.png")
    out_csv = tmp_path / "baseline.csv"

    rows = evaluate_baseline_directory(pred_root=pred_root, gt_root=gt_root, out_csv=out_csv, dataset="TEM1", split="test")

    assert len(rows) == 1
    df = pd.read_csv(out_csv)
    assert df.loc[0, "sample_id"] == "sample_TEM"
    assert df.loc[0, "fiber_ap50"] == 1.0
    assert df.loc[0, "pair_accuracy"] == 1.0


def test_evaluate_baseline_directory_respects_limit(tmp_path):
    gt_root = tmp_path / "processed"
    pred_root = tmp_path / "pred"
    pred_root.mkdir()
    semantic = np.array([[0, 1, 2]], dtype=np.uint8)

    for sample_id in ["sample_1", "sample_2"]:
        sample_dir = gt_root / "TEM1" / "test" / sample_id
        sample_dir.mkdir(parents=True)
        Image.fromarray(semantic).save(sample_dir / "semantic.png")
        Image.fromarray(semantic).save(pred_root / f"{sample_id}.png")

    rows = evaluate_baseline_directory(
        pred_root=pred_root,
        gt_root=gt_root,
        out_csv=tmp_path / "limited.csv",
        dataset="TEM1",
        split="test",
        limit=1,
    )

    assert len(rows) == 1
