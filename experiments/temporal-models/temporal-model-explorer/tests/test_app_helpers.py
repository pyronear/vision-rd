import pandas as pd

from temporal_model_explorer.app import decision_agreement, pivot_decisions


def test_pivot_decisions_one_row_per_sequence():
    df = pd.DataFrame(
        [
            {"key": "zip_1", "label": "smoke", "model": "a", "decision": "keep"},
            {"key": "zip_1", "label": "smoke", "model": "b", "decision": "discard"},
        ]
    )
    wide = pivot_decisions(df)
    row = wide[wide["key"] == "zip_1"].iloc[0]
    assert row["a"] == "keep" and row["b"] == "discard"
    assert row["label"] == "smoke"


def test_decision_agreement():
    df = pd.DataFrame(
        [
            {"key": "s1", "model": "a", "decision": "keep"},
            {"key": "s1", "model": "b", "decision": "keep"},
            {"key": "s2", "model": "a", "decision": "keep"},
            {"key": "s2", "model": "b", "decision": "discard"},
        ]
    )
    agree = decision_agreement(df)
    assert bool(agree["s1"]) is True
    assert bool(agree["s2"]) is False
