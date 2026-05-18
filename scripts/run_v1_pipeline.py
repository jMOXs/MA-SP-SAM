#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ma_sp_sam.reporting import write_v1_summary
from ma_sp_sam.sam.sam_adapter import SEGMENT_ANYTHING_MISSING
from ma_sp_sam.utils.io import load_yaml
from predict_sam_from_self_prompt import run_sam_prediction
from predict_self_prompt import run_prediction
from refine_sam_predictions import refine_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MA-SP-SAM End-to-End V1 pipeline.")
    parser.add_argument("--self-prompt-checkpoint", required=True)
    parser.add_argument("--self-prompt-config", default="configs/train/self_prompt.yaml")
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--dataset", default="TEM1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--work-dir", default="outputs/v1_pipeline")
    parser.add_argument("--use-mask-input", action="store_true")
    parser.add_argument("--skip-sam", action="store_true")
    args = parser.parse_args()
    if not args.skip_sam and not args.sam_checkpoint:
        parser.error("--sam-checkpoint is required unless --skip-sam is used")

    try:
        run_v1_pipeline(
            self_prompt_checkpoint=_resolve(args.self_prompt_checkpoint),
            self_prompt_config=_resolve(args.self_prompt_config),
            sam_checkpoint=None if args.sam_checkpoint is None else _resolve(args.sam_checkpoint),
            sam_model_type=args.sam_model_type,
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
            device=args.device,
            work_dir=_resolve(args.work_dir),
            use_mask_input=args.use_mask_input,
            skip_sam=args.skip_sam,
        )
    except RuntimeError as exc:
        if SEGMENT_ANYTHING_MISSING in str(exc):
            print(SEGMENT_ANYTHING_MISSING, file=sys.stderr)
            raise SystemExit(2) from None
        raise


def run_v1_pipeline(
    *,
    self_prompt_checkpoint: Path,
    self_prompt_config: Path,
    sam_checkpoint: Path | None,
    sam_model_type: str,
    dataset: str,
    split: str,
    limit: int | None,
    device: str | None,
    work_dir: Path,
    use_mask_input: bool = False,
    skip_sam: bool = False,
) -> list[dict[str, str]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    self_prompt_dir = work_dir / "self_prompt"
    sam_dir = work_dir / "sam"
    refined_dir = work_dir / "refined"

    config = load_yaml(self_prompt_config)
    run_prediction(
        checkpoint_path=self_prompt_checkpoint,
        config_path=self_prompt_config,
        dataset_name=dataset,
        split=split,
        limit=limit,
        device_name=device,
        out_root=self_prompt_dir,
    )

    if skip_sam:
        if not sam_dir.exists():
            raise FileNotFoundError(f"--skip-sam requires existing SAM predictions under {sam_dir}.")
    else:
        if sam_checkpoint is None:
            raise ValueError("--sam-checkpoint is required unless --skip-sam is used")
        run_sam_prediction(
            self_prompt_checkpoint=self_prompt_checkpoint,
            self_prompt_config=self_prompt_config,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
            dataset_name=dataset,
            split=split,
            limit=limit,
            device_name=device,
            out_root=sam_dir,
            use_mask_input=use_mask_input,
            save_all_candidates=False,
        )

    processed_root = _processed_root_from_config(config)
    refine_directory(
        sam_pred_root=sam_dir,
        self_prompt_root=self_prompt_dir,
        processed_root=processed_root,
        dataset=dataset,
        split=split,
        limit=limit,
        out_root=refined_dir,
    )

    rows = write_v1_summary(
        self_prompt_summary=self_prompt_dir / "summary.csv",
        sam_summary=sam_dir / "summary.csv",
        refined_summary=refined_dir / "summary.csv",
        out_csv=work_dir / "summary.csv",
    )
    print(f"Wrote {len(rows)} V1 summary rows to {work_dir / 'summary.csv'}")
    return rows


def _processed_root_from_config(config: dict[str, Any]) -> Path:
    processed_root = config.get("processed_root", "data/processed/astih_tem")
    return _resolve(processed_root)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
