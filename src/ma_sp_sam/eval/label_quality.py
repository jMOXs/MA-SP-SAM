from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from ma_sp_sam.labels.qc import add_flag_columns


def summarize_pair_table(pair_table: pd.DataFrame) -> dict[str, int | float]:
    if pair_table.empty:
        return {
            "instances": 0,
            "flagged_instances": 0,
            "axon_area_total": 0,
            "myelin_area_total": 0,
            "fiber_area_total": 0,
            "g_ratio_mean": 0.0,
        }

    flags = pair_table["flags"].fillna("").astype(str)
    if "g_ratio" in pair_table:
        g_ratio_mean = float(pair_table["g_ratio"].mean())
    else:
        ratios = np.sqrt(pair_table["axon_area"] / pair_table["fiber_area"]).replace([np.inf, -np.inf], 0)
        g_ratio_mean = float(ratios.fillna(0).mean())

    summary: dict[str, int | float] = {
        "instances": int(len(pair_table)),
        "flagged_instances": int((flags != "").sum()),
        "axon_area_total": int(pair_table["axon_area"].sum()),
        "myelin_area_total": int(pair_table["myelin_area"].sum()),
        "fiber_area_total": int(pair_table["fiber_area"].sum()),
        "g_ratio_mean": g_ratio_mean,
    }
    return add_flag_columns(summary, pair_table)


def summarize_processed_sample(sample_dir: str | Path) -> dict[str, int | float | str]:
    sample_path = Path(sample_dir)
    pair_table = pd.read_csv(sample_path / "pair_table.csv")
    summary = summarize_pair_table(pair_table)
    summary["sample_id"] = sample_path.name
    summary["dataset"] = sample_path.parent.parent.name
    summary["split"] = sample_path.parent.name
    return summary
