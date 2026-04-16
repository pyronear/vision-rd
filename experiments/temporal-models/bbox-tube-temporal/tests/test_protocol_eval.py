"""Unit tests for protocol_eval record + metrics helpers."""

import math
from datetime import datetime
from pathlib import Path

from pyrocore import Frame, TemporalModelOutput

from bbox_tube_temporal.protocol_eval import (
    SequenceRecord,
    _compute_ttd_seconds,
    build_record,
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


def _frame(stem="f", ts=None):
    return Frame(frame_id=stem, image_path=Path(f"/tmp/{stem}.jpg"), timestamp=ts)


# ── build_record ────────────────────────────────────────────────────────


def test_build_record_extracts_score_and_tube_logits_from_details():
    output = TemporalModelOutput(
        is_positive=True,
        trigger_frame_index=0,
        details={"tube_logits": [1.5, 0.3, -0.2], "num_tubes_kept": 3},
    )

    rec = build_record(
        sequence_dir=Path("/data/wildfire/seq_a"),
        label="smoke",
        frames=[_frame("f0")],
        output=output,
    )

    assert rec.sequence_id == "seq_a"
    assert rec.label == "smoke"
    assert rec.is_positive is True
    assert rec.trigger_frame_index == 0
    assert rec.tube_logits == [1.5, 0.3, -0.2]
    assert rec.num_tubes_kept == 3
    assert rec.score == 1.5  # max of the three logits


def test_build_record_empty_tube_logits_yields_minus_inf_score():
    output = TemporalModelOutput(
        is_positive=False,
        trigger_frame_index=None,
        details={"tube_logits": [], "num_tubes_kept": 0},
    )

    rec = build_record(
        sequence_dir=Path("/data/fp/seq_b"),
        label="fp",
        frames=[_frame("f0")],
        output=output,
    )

    assert rec.score == float("-inf")
    assert rec.num_tubes_kept == 0
    assert rec.tube_logits == []


def test_build_record_copies_details_dict():
    """build_record should snapshot output.details, not alias it."""
    details = {"tube_logits": [0.9], "num_tubes_kept": 1, "extra": "foo"}
    output = TemporalModelOutput(
        is_positive=True, trigger_frame_index=0, details=details
    )

    rec = build_record(
        sequence_dir=Path("/data/wildfire/seq_c"),
        label="smoke",
        frames=[_frame("f0")],
        output=output,
    )

    details["extra"] = "mutated"
    assert rec.details["extra"] == "foo"


# ── _compute_ttd_seconds edge cases ──────────────────────────────────────


def test_compute_ttd_returns_none_when_not_a_tp():
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    t1 = datetime(2024, 1, 1, 10, 0, 30)
    frames = [_frame("f0", t0), _frame("f1", t1)]

    # FN: ground_truth=True, predicted=False
    assert (
        _compute_ttd_seconds(
            ground_truth=True, predicted=False, trigger_frame_index=None, frames=frames
        )
        is None
    )
    # FP: ground_truth=False, predicted=True
    assert (
        _compute_ttd_seconds(
            ground_truth=False, predicted=True, trigger_frame_index=1, frames=frames
        )
        is None
    )


def test_compute_ttd_returns_none_when_trigger_out_of_range():
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    frames = [_frame("f0", t0)]  # only 1 frame

    assert (
        _compute_ttd_seconds(
            ground_truth=True, predicted=True, trigger_frame_index=5, frames=frames
        )
        is None
    )


def test_compute_ttd_returns_none_when_timestamp_missing():
    frames = [_frame("f0", None), _frame("f1", None)]  # no timestamps parsed

    assert (
        _compute_ttd_seconds(
            ground_truth=True, predicted=True, trigger_frame_index=1, frames=frames
        )
        is None
    )


def test_compute_ttd_returns_none_for_empty_frames():
    assert (
        _compute_ttd_seconds(
            ground_truth=True, predicted=True, trigger_frame_index=0, frames=[]
        )
        is None
    )


def test_compute_ttd_computes_seconds_for_valid_tp():
    t0 = datetime(2024, 1, 1, 10, 0, 0)
    t1 = datetime(2024, 1, 1, 10, 0, 45)
    frames = [_frame("f0", t0), _frame("f1", t1)]

    result = _compute_ttd_seconds(
        ground_truth=True, predicted=True, trigger_frame_index=1, frames=frames
    )

    assert result == 45.0


# ── compute_metrics TTD filter regression ────────────────────────────────


def test_compute_metrics_ignores_ttd_for_non_tp_records():
    """TTD is filtered to TPs even if non-TP records carry a ttd_seconds.

    Protects against misuse where a caller constructs a record with
    label!="smoke" or is_positive=False but a non-None ttd_seconds.
    """
    records = [
        _rec("smoke", True, score=1.0, trigger=0, ttd=10.0),
        _rec("smoke", False, score=-1.0, ttd=999.0),  # FN with bogus ttd
        _rec("fp", True, score=0.5, trigger=0, ttd=888.0),  # FP with bogus ttd
        _rec("fp", False, score=-1.0, ttd=777.0),  # TN with bogus ttd
    ]

    m = compute_metrics("my-model", records)

    # Only the one TP record's ttd should be counted
    assert m["mean_ttd_seconds"] == 10.0
    assert m["median_ttd_seconds"] == 10.0
