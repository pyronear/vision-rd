import pandas as pd

from temporal_model_explorer.outcomes import (
    compute_outcome,
    decision_from_output,
    filter_results,
    max_probability,
)


def test_decision_from_output():
    assert decision_from_output(True) == "keep"
    assert decision_from_output(False) == "discard"


def test_max_probability_picks_max_over_kept_tubes():
    details = {
        "tubes": {
            "kept": [{"probability": 0.2}, {"probability": 0.8}, {"probability": None}]
        }
    }
    assert max_probability(details) == 0.8


def test_max_probability_none_when_no_probs():
    assert max_probability({"tubes": {"kept": [{"probability": None}]}}) is None
    assert max_probability({}) is None


def test_compute_outcome_all_branches():
    assert compute_outcome("keep", "smoke") == "kept-smoke"
    assert compute_outcome("discard", "smoke") == "discarded-smoke"
    assert compute_outcome("keep", "fp") == "kept-fp"
    assert compute_outcome("discard", "fp") == "discarded-fp"
    assert compute_outcome("keep", "unknown") == "n/a"


def test_filter_results_by_decision_and_label():
    df = pd.DataFrame(
        [
            {"decision": "keep", "label": "smoke", "outcome": "kept-smoke"},
            {"decision": "discard", "label": "fp", "outcome": "discarded-fp"},
        ]
    )
    out = filter_results(df, decision="discard")
    assert list(out["label"]) == ["fp"]
    assert len(filter_results(df, label="smoke")) == 1
    assert len(filter_results(df, errors_only=True)) == 0


def test_filter_results_errors_only_returns_errors():
    df = pd.DataFrame(
        [
            {"decision": "discard", "label": "smoke", "outcome": "discarded-smoke"},
            {"decision": "discard", "label": "fp", "outcome": "discarded-fp"},
        ]
    )
    out = filter_results(df, errors_only=True)
    assert list(out["outcome"]) == ["discarded-smoke"]
