from pathlib import Path

from ma_sp_sam.experiments.preflight import check_experiment_preflight


def _base_config(tmp_path: Path, *, mode: str = "full") -> dict[str, object]:
    processed_root = tmp_path / "processed"
    (processed_root / "TEM1" / "test").mkdir(parents=True)
    self_prompt_config = tmp_path / "self_prompt.yaml"
    self_prompt_config.write_text(f"processed_root: {processed_root}\n", encoding="utf-8")
    checkpoint = tmp_path / "self_prompt.pt"
    checkpoint.write_bytes(b"checkpoint")
    sam_checkpoint = tmp_path / "sam.pth"
    sam_checkpoint.write_bytes(b"sam")
    return {
        "name": "tem1_internal",
        "dataset": "TEM1",
        "split": "test",
        "mode": mode,
        "self_prompt_checkpoint": str(checkpoint),
        "self_prompt_config": str(self_prompt_config),
        "sam_checkpoint": str(sam_checkpoint),
        "work_dir": str(tmp_path / "work"),
        "segment_anything_required": False,
    }


def _by_name(checks):
    return {check["name"]: check for check in checks}


def test_preflight_fails_when_self_prompt_checkpoint_is_missing(tmp_path):
    config = _base_config(tmp_path)
    Path(config["self_prompt_checkpoint"]).unlink()

    checks = _by_name(check_experiment_preflight(config))

    assert checks["self_prompt_checkpoint"]["status"] == "fail"


def test_preflight_fails_when_full_mode_sam_checkpoint_is_missing(tmp_path):
    config = _base_config(tmp_path)
    Path(config["sam_checkpoint"]).unlink()

    checks = _by_name(check_experiment_preflight(config))

    assert checks["sam_checkpoint"]["status"] == "fail"


def test_preflight_fails_when_skip_sam_predictions_are_missing(tmp_path):
    config = _base_config(tmp_path, mode="skip_sam")

    checks = _by_name(check_experiment_preflight(config))

    assert checks["skip_sam_predictions"]["status"] == "fail"


def test_preflight_passes_skip_sam_predictions_when_directory_exists(tmp_path):
    config = _base_config(tmp_path, mode="skip_sam")
    (Path(config["work_dir"]) / "sam" / "TEM1" / "test").mkdir(parents=True)

    checks = _by_name(check_experiment_preflight(config))

    assert checks["skip_sam_predictions"]["status"] == "pass"
