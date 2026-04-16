"""Unit tests for protocol_eval record + metrics helpers."""

import math

from bbox_tube_temporal.protocol_eval import (
    SequenceRecord,
    compute_metrics,
)


def _rec(label, is_positive, *, score=0.0, trigger=None, ttd=None, num_kept=0):
    return SequenceRecord(
        sequence_id=f"seq_{label}_{is_positive}",
        label=label,
        is_positive=is_positive,
        trigger_frame_index=trigger,
        score=score,
        num_tubes_kept=num_kept,
        tube_logits=[],
        ttd_seconds=ttd,
    )


def test_compute_metrics_all_correct():
    records = [
        _rec("smoke", True, score=1.0, trigger=5, ttd=30.0),
        _rec("smoke", True, score=0.9, trigger=3, ttd=10.0),
        _rec("fp", False, score=-1.0),
        _rec("fp", False, score=-0.5),
    ]

    m = compute_metrics("my-model", records)

    assert m["model_name"] == "my-model"
    assert m["num_sequences"] == 4
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["tn"] == 2
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["fpr"] == 0.0
    assert m["mean_ttd_seconds"] == 20.0
    assert m["median_ttd_seconds"] == 20.0


def test_compute_metrics_all_wrong():
    records = [
        _rec("smoke", False, score=-1.0),
        _rec("fp", True, score=0.9, trigger=1),
    ]

    m = compute_metrics("my-model", records)

    assert m["tp"] == 0
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["fpr"] == 1.0
    assert m["mean_ttd_seconds"] is None
    assert m["median_ttd_seconds"] is None


def test_compute_metrics_rounds_to_four_decimals():
    records = [
        _rec("smoke", True, score=1.0, trigger=0, ttd=1.23456),
        _rec("smoke", False, score=-1.0),
        _rec("smoke", False, score=-1.0),
        _rec("fp", True, score=0.5, trigger=0),
        _rec("fp", False, score=-1.0),
    ]

    m = compute_metrics("my-model", records)

    assert m["precision"] == round(1 / 2, 4)
    assert m["recall"] == round(1 / 3, 4)
    assert m["f1"] == round(2 * (1 / 2) * (1 / 3) / ((1 / 2) + (1 / 3)), 4)
    assert m["fpr"] == round(1 / 2, 4)
    assert m["mean_ttd_seconds"] == 1.2
    assert m["median_ttd_seconds"] == 1.2


def test_compute_metrics_auc_handles_minus_inf_scores():
    records = [
        _rec("smoke", False, score=-math.inf),
        _rec("smoke", True, score=2.0, trigger=0),
        _rec("fp", False, score=-math.inf),
        _rec("fp", False, score=-1.0),
    ]

    m = compute_metrics("my-model", records)

    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert not math.isnan(m["pr_auc"])
    assert not math.isnan(m["roc_auc"])


def test_compute_metrics_auc_zero_when_only_one_class():
    records = [
        _rec("fp", False, score=-1.0),
        _rec("fp", True, score=1.0, trigger=0),
    ]

    m = compute_metrics("my-model", records)

    assert m["roc_auc"] == 0.0
    assert m["pr_auc"] == 0.0


def test_compute_metrics_empty_records():
    m = compute_metrics("my-model", [])

    assert m["num_sequences"] == 0
    assert m["tp"] == m["fp"] == m["fn"] == m["tn"] == 0
    assert m["precision"] == m["recall"] == m["f1"] == m["fpr"] == 0.0
    assert m["mean_ttd_seconds"] is None
    assert m["median_ttd_seconds"] is None
    assert m["pr_auc"] == 0.0
    assert m["roc_auc"] == 0.0


def test_compute_metrics_ttd_median_three_values():
    records = [
        _rec("smoke", True, score=1.0, trigger=0, ttd=10.0),
        _rec("smoke", True, score=1.0, trigger=0, ttd=30.0),
        _rec("smoke", True, score=1.0, trigger=0, ttd=50.0),
    ]

    m = compute_metrics("my-model", records)

    assert m["mean_ttd_seconds"] == 30.0
    assert m["median_ttd_seconds"] == 30.0
