from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from ma_sp_sam.utils.io import load_yaml


def check_experiment_preflight(resolved_config: dict[str, Any], *, strict: bool = True) -> list[dict[str, str]]:
    """Check experiment inputs before starting a V1 run."""
    del strict
    checks: list[dict[str, str]] = []
    self_prompt_checkpoint = _path_or_none(resolved_config.get("self_prompt_checkpoint"))
    self_prompt_config = _path_or_none(resolved_config.get("self_prompt_config"))
    dataset = str(resolved_config.get("dataset", ""))
    split = str(resolved_config.get("split", ""))
    mode = str(resolved_config.get("mode", "full"))

    checks.append(_exists_check("self_prompt_checkpoint", self_prompt_checkpoint, "Self-Prompt checkpoint"))
    config_check = _exists_check("self_prompt_config", self_prompt_config, "Self-Prompt config")
    checks.append(config_check)

    processed_root = _processed_root(resolved_config, self_prompt_config if config_check["status"] == "pass" else None)
    if processed_root is None:
        checks.append(_check("processed_root", "warn", "processed_root is not configured."))
    else:
        checks.append(_exists_check("processed_root", processed_root, "Processed label root"))
        if processed_root.exists():
            processed_split_dir = processed_root / dataset / split
            checks.append(_exists_check("processed_split_dir", processed_split_dir, f"Processed directory for {dataset}/{split}"))

    if mode == "full":
        sam_checkpoint = _path_or_none(resolved_config.get("sam_checkpoint"))
        checks.append(_exists_check("sam_checkpoint", sam_checkpoint, "SAM checkpoint"))
        segment_required = bool(resolved_config.get("segment_anything_required", True))
        segment_status = "pass" if importlib.util.find_spec("segment_anything") is not None else ("fail" if segment_required else "warn")
        checks.append(
            _check(
                "segment_anything",
                segment_status,
                "segment_anything is importable."
                if segment_status == "pass"
                else "segment_anything is not installed. Install it for full SAM experiments.",
            )
        )
    elif mode == "skip_sam":
        work_dir = _path_or_none(resolved_config.get("work_dir"))
        sam_dir = None if work_dir is None else work_dir / "sam" / dataset / split
        checks.append(_exists_check("skip_sam_predictions", sam_dir, f"Existing SAM predictions for {dataset}/{split}"))
    else:
        checks.append(_check("mode", "fail", f"Unsupported experiment mode: {mode}"))
    return checks


def has_failed_checks(checks: list[dict[str, str]]) -> bool:
    return any(check.get("status") == "fail" for check in checks)


def failed_check_summary(checks: list[dict[str, str]]) -> str:
    failures = [f"{check['name']}: {check['message']}" for check in checks if check.get("status") == "fail"]
    return "; ".join(failures)


def _processed_root(resolved_config: dict[str, Any], self_prompt_config: Path | None) -> Path | None:
    explicit = _path_or_none(resolved_config.get("processed_root"))
    if explicit is not None:
        return explicit
    if self_prompt_config is None or not self_prompt_config.exists():
        return None
    config = load_yaml(self_prompt_config)
    return _path_or_none(config.get("processed_root"))


def _exists_check(name: str, path: Path | None, label: str) -> dict[str, str]:
    if path is None:
        return _check(name, "fail", f"{label} is not configured.")
    if path.exists():
        return _check(name, "pass", f"{label} exists: {path}")
    return _check(name, "fail", f"{label} does not exist: {path}")


def _check(name: str, status: str, message: str) -> dict[str, str]:
    return {"name": name, "status": status, "message": message}


def _path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()
