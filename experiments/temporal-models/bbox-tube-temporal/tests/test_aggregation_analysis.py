"""Unit tests for aggregation-rule analysis helpers."""

import json

import numpy as np
import pytest

from bbox_tube_temporal.aggregation_analysis import (
    aggregate_score,
    build_scores_and_labels,
    find_threshold_for_recall,
    load_predictions,
    metrics_at_threshold,
    summarize_rule,
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


def test_find_threshold_returns_lowest_positive_score_for_full_recall():
    y_true = np.array([1, 1, 0])
    scores = np.array([3.0, 5.0, 1.0])

    # recall = 1.0 requires threshold <= 3.0; largest such threshold equals 3.0
    assert find_threshold_for_recall(y_true, scores, target_recall=1.0) == 3.0


def test_find_threshold_allows_dropping_one_positive_at_recall_050():
    y_true = np.array([1, 1, 0])
    scores = np.array([3.0, 5.0, 1.0])

    # recall = 0.5 only needs 1 of 2 positives; largest threshold = 5.0
    assert find_threshold_for_recall(y_true, scores, target_recall=0.5) == 5.0


def test_find_threshold_handles_neg_inf_positive_scores():
    # A positive sequence with no tubes (score = -inf) cannot be recovered
    # except by threshold = -inf, which we represent explicitly.
    y_true = np.array([1, 1])
    scores = np.array([-np.inf, 4.0])

    assert find_threshold_for_recall(y_true, scores, target_recall=1.0) == -np.inf
    assert find_threshold_for_recall(y_true, scores, target_recall=0.5) == 4.0


def test_find_threshold_raises_when_no_positives():
    y_true = np.array([0, 0])
    scores = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="no positives"):
        find_threshold_for_recall(y_true, scores, target_recall=0.95)


def test_find_threshold_raises_on_invalid_target():
    y_true = np.array([1])
    scores = np.array([1.0])

    with pytest.raises(ValueError, match=r"target_recall must be in \(0, 1\]"):
        find_threshold_for_recall(y_true, scores, target_recall=0.0)
    with pytest.raises(ValueError, match=r"target_recall must be in \(0, 1\]"):
        find_threshold_for_recall(y_true, scores, target_recall=1.5)


def test_metrics_at_threshold_all_correct():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([2.0, 3.0, 0.0, -1.0])

    m = metrics_at_threshold(y_true, scores, threshold=1.0)

    assert m == {
        "threshold": 1.0,
        "tp": 2,
        "fp": 0,
        "fn": 0,
        "tn": 2,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "fpr": 0.0,
    }


def test_metrics_at_threshold_all_false_positives():
    y_true = np.array([0, 0])
    scores = np.array([5.0, 5.0])

    m = metrics_at_threshold(y_true, scores, threshold=1.0)

    assert m["tp"] == 0 and m["fp"] == 2 and m["fn"] == 0 and m["tn"] == 0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0  # no positives exist
    assert m["f1"] == 0.0
    assert m["fpr"] == 1.0


def test_metrics_at_threshold_handles_neg_inf_threshold():
    y_true = np.array([1, 0])
    scores = np.array([-np.inf, -np.inf])

    # threshold = -inf => everything predicted positive
    m = metrics_at_threshold(y_true, scores, threshold=-np.inf)

    assert m["tp"] == 1 and m["fp"] == 1 and m["tn"] == 0 and m["fn"] == 0


def test_metrics_at_threshold_no_positives_no_negatives_safe():
    y_true = np.array([], dtype=int)
    scores = np.array([], dtype=float)

    m = metrics_at_threshold(y_true, scores, threshold=0.0)

    assert m["tp"] == 0 and m["fp"] == 0 and m["fn"] == 0 and m["tn"] == 0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["fpr"] == 0.0


def _record(sid: str, label: str, logits: list[float]) -> dict:
    return {
        "sequence_id": sid,
        "label": label,
        "tube_logits": logits,
        "num_tubes_kept": len(logits),
        "is_positive": False,
        "score": max(logits) if logits else -float("inf"),
    }


def test_build_scores_and_labels_max_rule():
    records = [
        _record("a", "smoke", [1.0, 3.0]),
        _record("b", "fp", [0.5]),
        _record("c", "smoke", []),
    ]

    y, s = build_scores_and_labels(records, rule="max", k=1)

    assert y.tolist() == [1, 0, 1]
    assert s[0] == 3.0
    assert s[1] == 0.5
    assert s[2] == -np.inf


def test_build_scores_and_labels_top_k_mean():
    records = [
        _record("a", "smoke", [1.0, 3.0, 5.0]),  # top-2 mean = 4
        _record("b", "fp", [2.0]),  # too few tubes → -inf
    ]

    y, s = build_scores_and_labels(records, rule="top_k_mean", k=2)

    assert y.tolist() == [1, 0]
    assert s[0] == 4.0
    assert s[1] == -np.inf


def test_summarize_rule_returns_threshold_and_metrics():
    records = [
        _record("p1", "smoke", [3.0]),
        _record("p2", "smoke", [5.0]),
        _record("n1", "fp", [1.0]),
        _record("n2", "fp", [4.0]),
    ]

    result = summarize_rule(records, rule="max", k=1, target_recall=1.0)

    # Threshold = 3.0 catches both positives
    assert result["rule"] == "max"
    assert result["k"] == 1
    assert result["target_recall"] == 1.0
    assert result["threshold"] == 3.0
    assert result["tp"] == 2
    assert result["fp"] == 1  # n2 at score 4.0 >= 3.0
    assert result["fn"] == 0
    assert result["tn"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1.0
