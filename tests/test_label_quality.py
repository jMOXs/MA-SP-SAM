import pandas as pd

from ma_sp_sam.eval.label_quality import summarize_pair_table


def test_summarizes_pair_table_flags_and_area_totals():
    pair_table = pd.DataFrame(
        [
            {"fiber_id": 1, "axon_area": 4, "myelin_area": 12, "fiber_area": 16, "flags": ""},
            {"fiber_id": 2, "axon_area": 1, "myelin_area": 5, "fiber_area": 6, "flags": "small_axon;ambiguous_fiber"},
        ]
    )

    summary = summarize_pair_table(pair_table)

    assert summary["instances"] == 2
    assert summary["flagged_instances"] == 1
    assert summary["small_axon"] == 1
    assert summary["ambiguous_fiber"] == 1
    assert summary["axon_area_total"] == 5
