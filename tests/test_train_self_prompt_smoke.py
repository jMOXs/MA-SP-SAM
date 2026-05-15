import subprocess
import sys


def test_train_self_prompt_script_runs_one_synthetic_epoch(tmp_path):
    config_path = tmp_path / "self_prompt_synthetic.yaml"
    checkpoint_dir = tmp_path / "checkpoints"
    config_path.write_text(
        f"""
synthetic: true
synthetic_samples: 4
synthetic_height: 16
synthetic_width: 16
checkpoint_dir: {checkpoint_dir}
in_channels: 1
hidden_channels: 8
num_blocks: 1
lr: 0.001
weight_decay: 0.0
batch_size: 2
epochs: 1
proposal_summary_interval: 1
loss_weights:
  semantic: 1.0
  center: 1.0
  boundary: 1.0
  distance: 1.0
center_threshold: 0.5
boundary_threshold: 0.5
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_self_prompt.py",
            "--config",
            str(config_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
        cwd="/sydata/js/MA-SP-SAM",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "epoch=1" in result.stdout
    assert "proposal_summary" in result.stdout
    assert (checkpoint_dir / "best.pt").exists()
