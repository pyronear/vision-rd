"""Unit tests for aggregation-rule analysis helpers."""

import json

import numpy as np
import pytest

from bbox_tube_temporal.aggregation_analysis import (
    aggregate_score,
    load_predictions,
)


def test_load_predictions_returns_records_sorted_by_sequence_id(tmp_path):
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "sequence_id": "b",
                    "label": "smoke",
                    "tube_logits": [1.0, 2.0],
                    "num_tubes_kept": 2,
                    "is_positive": True,
                    "score": 2.0,
                },
                {
                    "sequence_id": "a",
                    "label": "fp",
                    "tube_logits": [],
                    "num_tubes_kept": 0,
                    "is_positive": False,
                    "score": -float("inf"),
                },
            ]
        )
    )

    records = load_predictions(predictions_path)

    assert [r["sequence_id"] for r in records] == ["a", "b"]
    assert records[0]["label"] == "fp"
    assert records[1]["tube_logits"] == [1.0, 2.0]


def test_load_predictions_preserves_empty_tube_logits(tmp_path):
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "sequence_id": "only",
                    "label": "fp",
                    "tube_logits": [],
                    "num_tubes_kept": 0,
                    "is_positive": False,
                    "score": -float("inf"),
                }
            ]
        )
    )

    records = load_predictions(predictions_path)

    assert len(records) == 1
    assert records[0]["tube_logits"] == []


def test_load_predictions_accepts_numpy_inf_serialization(tmp_path):
    predictions_path = tmp_path / "predictions.json"
    # evaluate_packaged writes "-Infinity" via json.dump(..., allow_nan=True)
    predictions_path.write_text(
        '[{"sequence_id": "x", "label": "fp", "tube_logits": [], '
        '"num_tubes_kept": 0, "is_positive": false, "score": -Infinity}]'
    )

    records = load_predictions(predictions_path)

    assert records[0]["score"] == -np.inf


def test_aggregate_score_max_picks_largest_logit():
    assert aggregate_score([1.0, 5.0, 2.0], rule="max", k=1) == 5.0


def test_aggregate_score_max_on_empty_is_neg_inf():
    assert aggregate_score([], rule="max", k=1) == -np.inf


def test_aggregate_score_top_k_mean_averages_highest_k():
    # top-2 of [1, 5, 3, 4] = [5, 4], mean = 4.5
    assert aggregate_score([1.0, 5.0, 3.0, 4.0], rule="top_k_mean", k=2) == 4.5


def test_aggregate_score_top_k_mean_on_fewer_tubes_than_k_is_neg_inf():
    # Not enough tubes to form a top-k → conservative: sequence is negative.
    assert aggregate_score([5.0], rule="top_k_mean", k=2) == -np.inf


def test_aggregate_score_top_k_mean_on_empty_is_neg_inf():
    assert aggregate_score([], rule="top_k_mean", k=2) == -np.inf


def test_aggregate_score_rejects_unknown_rule():
    with pytest.raises(ValueError, match="unknown rule"):
        aggregate_score([1.0], rule="bogus", k=1)


def test_aggregate_score_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k must be >= 1"):
        aggregate_score([1.0], rule="max", k=0)
