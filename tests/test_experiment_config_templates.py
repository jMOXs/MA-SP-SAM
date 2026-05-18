from pathlib import Path

from ma_sp_sam.utils.io import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_astih_limit_config_templates_are_loadable():
    limit2 = load_yaml(PROJECT_ROOT / "configs" / "experiments" / "astih_v1_limit2.yaml")
    limit10 = load_yaml(PROJECT_ROOT / "configs" / "experiments" / "astih_v1_limit10.yaml")

    assert limit2["limit"] == 2
    assert limit2["device"] == "cuda"
    assert limit10["limit"] == 10
    assert limit10["device"] == "cuda"
    assert {experiment["mode"] for experiment in limit2["experiments"]} == {"full"}
    assert {experiment["mode"] for experiment in limit10["experiments"]} == {"full"}


def test_experiment_configs_do_not_contain_weight_or_data_files():
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    files = [path for path in (PROJECT_ROOT / "configs").rglob("*") if path.is_file()]

    assert all(path.suffix.lower() not in forbidden_suffixes for path in files)
