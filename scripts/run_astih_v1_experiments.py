#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ma_sp_sam.reporting import write_experiment_summary
from ma_sp_sam.experiments.preflight import check_experiment_preflight, failed_check_summary, has_failed_checks
from ma_sp_sam.utils.io import load_yaml
from run_v1_pipeline import run_v1_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured ASTIH V1 experiments.")
    parser.add_argument("--config", default="configs/experiments/astih_v1.yaml")
    parser.add_argument("--only", default=None, help="Comma-separated experiment names to run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-strict-preflight", action="store_true")
    args = parser.parse_args()

    try:
        run_experiments(
            config_path=_resolve(args.config),
            only=_parse_only(args.only),
            dry_run=args.dry_run,
            preflight_only=args.preflight_only,
            strict_preflight=not args.no_strict_preflight,
        )
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None


def run_experiments(
    *,
    config_path: Path,
    only: set[str] | None = None,
    dry_run: bool = False,
    preflight_only: bool = False,
    strict_preflight: bool = True,
) -> list[dict[str, Any]]:
    config = load_yaml(config_path)
    all_experiments = config.get("experiments", [])
    _validate_only_names(all_experiments, only)
    experiments = _selected_experiments(all_experiments, only)
    disabled = _disabled_experiments(all_experiments, only)
    work_root = _resolve(config.get("work_root", "outputs/experiments"))

    if dry_run:
        print(f"ASTIH V1 experiment dry run: {len(experiments)} experiment(s)")
        for experiment in experiments:
            resolved = _resolved_config(config, experiment, work_root)
            print(
                f"- {resolved['name']}: dataset={resolved['dataset']} "
                f"split={resolved['split']} mode={resolved['mode']} work_dir={resolved['work_dir']}"
            )
        for experiment in disabled:
            print(f"- disabled: {experiment['name']}")
        return []

    statuses: list[dict[str, Any]] = []
    for experiment in experiments:
        resolved = _resolved_config(config, experiment, work_root)
        work_dir = Path(resolved["work_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        _write_yaml(work_dir / "resolved_config.yaml", resolved)
        status = _initial_status(resolved)
        print(
            f"experiment start: {resolved['name']} dataset={resolved['dataset']} "
            f"split={resolved['split']} mode={resolved['mode']} work_dir={work_dir}",
            flush=True,
        )
        preflight = check_experiment_preflight(resolved)
        _write_json(work_dir / "preflight.json", preflight)
        preflight_failed = has_failed_checks(preflight)
        print(_preflight_message(resolved["name"], preflight), flush=True)
        if preflight_only:
            if strict_preflight and preflight_failed:
                status["status"] = "preflight_failed"
                status["error"] = failed_check_summary(preflight)
            else:
                status["status"] = "preflight_passed"
            status["finished_at"] = _now()
            _write_json(work_dir / "run_status.json", status)
            statuses.append(status)
            print(f"experiment {resolved['name']} status={status['status']}", flush=True)
            continue
        if strict_preflight and preflight_failed:
            status["status"] = "preflight_failed"
            status["error"] = failed_check_summary(preflight)
            status["finished_at"] = _now()
            _write_json(work_dir / "run_status.json", status)
            statuses.append(status)
            print(f"experiment {resolved['name']} status=preflight_failed error={status['error']}", flush=True)
            continue
        try:
            run_v1_pipeline(
                self_prompt_checkpoint=Path(resolved["self_prompt_checkpoint"]),
                self_prompt_config=Path(resolved["self_prompt_config"]),
                sam_checkpoint=None if not resolved.get("sam_checkpoint") else Path(resolved["sam_checkpoint"]),
                sam_model_type=str(resolved.get("sam_model_type", "vit_b")),
                dataset=str(resolved["dataset"]),
                split=str(resolved["split"]),
                limit=None if resolved.get("limit") is None else int(resolved["limit"]),
                device=resolved.get("device"),
                work_dir=work_dir,
                use_mask_input=bool(resolved.get("use_mask_input", False)),
                skip_sam=str(resolved["mode"]) == "skip_sam",
            )
            status["status"] = "success"
        except KeyboardInterrupt:
            status["status"] = "interrupted"
            status["error"] = "Interrupted by user."
            raise
        except Exception as exc:  # noqa: BLE001 - experiment runner records failures and continues.
            status["status"] = "failed"
            status["error"] = str(exc)
        finally:
            status["finished_at"] = _now()
            _write_json(work_dir / "run_status.json", status)
            statuses.append(status)
            print(f"experiment {resolved['name']} status={status['status']} error={status['error']}", flush=True)

    write_experiment_summary(work_root)
    print(f"experiment summary written: {work_root}", flush=True)
    return statuses


def _selected_experiments(experiments: list[dict[str, Any]], only: set[str] | None) -> list[dict[str, Any]]:
    selected = []
    for experiment in experiments:
        name = str(experiment.get("name", ""))
        if not bool(experiment.get("enabled", True)):
            continue
        if only is not None and name not in only:
            continue
        selected.append(experiment)
    return selected


def _disabled_experiments(experiments: list[dict[str, Any]], only: set[str] | None) -> list[dict[str, Any]]:
    disabled = []
    for experiment in experiments:
        name = str(experiment.get("name", ""))
        if bool(experiment.get("enabled", True)):
            continue
        if only is not None and name not in only:
            continue
        disabled.append(experiment)
    return disabled


def _preflight_message(name: str, checks: list[dict[str, str]]) -> str:
    counts = {
        "pass": sum(check["status"] == "pass" for check in checks),
        "warn": sum(check["status"] == "warn" for check in checks),
        "fail": sum(check["status"] == "fail" for check in checks),
    }
    return f"preflight {name}: pass={counts['pass']} warn={counts['warn']} fail={counts['fail']}"


def _validate_only_names(experiments: list[dict[str, Any]], only: set[str] | None) -> None:
    if only is None:
        return
    available = {str(experiment.get("name", "")) for experiment in experiments}
    unknown = sorted(only - available)
    if unknown:
        raise ValueError(
            "Unknown experiment name(s): "
            + ", ".join(unknown)
            + "\nAvailable experiment name(s): "
            + ", ".join(sorted(available))
        )


def _resolved_config(config: dict[str, Any], experiment: dict[str, Any], work_root: Path) -> dict[str, Any]:
    name = str(experiment["name"])
    mode = str(experiment.get("mode", "full"))
    if mode not in {"full", "skip_sam"}:
        raise ValueError(f"Unsupported experiment mode for {name}: {mode}")
    self_prompt_config = _optional_path_str(experiment.get("self_prompt_config", config.get("self_prompt_config", "")))
    processed_root = _resolved_processed_root(
        explicit=experiment.get("processed_root", config.get("processed_root", "")),
        self_prompt_config=self_prompt_config,
    )
    resolved = {
        "name": name,
        "dataset": str(experiment.get("dataset", config.get("dataset", "TEM1"))),
        "split": str(experiment.get("split", config.get("split", "test"))),
        "mode": mode,
        "enabled": bool(experiment.get("enabled", True)),
        "self_prompt_checkpoint": _optional_path_str(experiment.get("self_prompt_checkpoint", config.get("self_prompt_checkpoint", ""))),
        "self_prompt_config": self_prompt_config,
        "sam_checkpoint": _optional_path_str(experiment.get("sam_checkpoint", config.get("sam_checkpoint", ""))),
        "sam_model_type": str(experiment.get("sam_model_type", config.get("sam_model_type", "vit_b"))),
        "segment_anything_required": bool(experiment.get("segment_anything_required", config.get("segment_anything_required", True))),
        "processed_root": processed_root,
        "limit": experiment.get("limit", config.get("limit")),
        "device": experiment.get("device", config.get("device")),
        "use_mask_input": bool(experiment.get("use_mask_input", config.get("use_mask_input", False))),
        "work_dir": str(work_root / name),
    }
    return resolved


def _initial_status(resolved: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": resolved["name"],
        "dataset": resolved["dataset"],
        "split": resolved["split"],
        "mode": resolved["mode"],
        "status": "running",
        "error": "",
        "started_at": _now(),
        "finished_at": "",
    }


def _parse_only(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optional_path_str(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(_resolve(value))


def _resolved_processed_root(*, explicit: Any, self_prompt_config: str) -> str:
    if explicit not in (None, ""):
        return _optional_path_str(explicit)
    if not self_prompt_config:
        return ""
    config_path = Path(self_prompt_config)
    if not config_path.exists():
        return ""
    config = load_yaml(config_path)
    return _optional_path_str(config.get("processed_root", ""))


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
