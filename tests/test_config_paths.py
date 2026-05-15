from ma_sp_sam.utils.paths import resolve_path


def test_resolve_path_accepts_yaml_numeric_path_values(tmp_path):
    assert resolve_path(tmp_path, 8351731) == tmp_path / "8351731"
