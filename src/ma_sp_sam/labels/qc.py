from __future__ import annotations

from collections import Counter

import pandas as pd


def count_flags(pair_table: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    if pair_table.empty or "flags" not in pair_table:
        return counts
    for flag_text in pair_table["flags"].fillna(""):
        for flag in str(flag_text).split(";"):
            if flag:
                counts[flag] += 1
    return counts


def add_flag_columns(summary: dict[str, int | float], pair_table: pd.DataFrame) -> dict[str, int | float]:
    for flag, count in count_flags(pair_table).items():
        summary[flag] = count
    return summary
