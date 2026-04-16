"""Unit tests for aggregation-rule analysis helpers."""

import json

import numpy as np

from bbox_tube_temporal.aggregation_analysis import load_predictions


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
