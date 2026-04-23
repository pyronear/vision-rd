"""Tests for data_quality_sequential.review."""

from pathlib import Path

from data_quality_sequential.dataset import SequenceRef
from data_quality_sequential.review import (
    Prediction,
    ReviewEntry,
    ReviewSet,
    build_review_sets,
)


def _ref(name: str, ground_truth: bool) -> SequenceRef:
    return SequenceRef(
        name=name,
        split="val",
        ground_truth=ground_truth,
        frame_paths=[Path(f"/tmp/{name}_2023-01-01T00-00-00.jpg")],
    )


def _pred(name: str, predicted: bool, trigger: int | None = None) -> Prediction:
    return Prediction(
        sequence_name=name,
        predicted=predicted,
        trigger_frame_index=trigger,
    )


def test_build_review_sets_partitions_fp_and_fn() -> None:
    refs = [
        _ref("wf_a", True),
        _ref("wf_b", True),
        _ref("fp_a", False),
        _ref("fp_b", False),
    ]
    preds = [
        _pred("wf_a", predicted=True, trigger=3),  # TP
        _pred("wf_b", predicted=False),  # FN
        _pred("fp_a", predicted=True, trigger=0),  # FP
        _pred("fp_b", predicted=False),  # TN
    ]

    fp, fn = build_review_sets(refs, preds, split="val", model_name="m")

    assert isinstance(fp, ReviewSet) and isinstance(fn, ReviewSet)
    assert fp.kind == "fp"
    assert fn.kind == "fn"
    assert [e.sequence_name for e in fp.entries] == ["fp_a"]
    assert [e.sequence_name for e in fn.entries] == ["wf_b"]


def test_review_entries_are_sorted_alphabetically() -> None:
    refs = [
        _ref("wf_z", True),
        _ref("wf_a", True),
        _ref("fp_z", False),
        _ref("fp_a", False),
    ]
    preds = [
        _pred("wf_z", False),
        _pred("wf_a", False),
        _pred("fp_z", True, trigger=5),
        _pred("fp_a", True, trigger=0),
    ]

    fp, fn = build_review_sets(refs, preds, split="val", model_name="m")

    assert [e.sequence_name for e in fp.entries] == ["fp_a", "fp_z"]
    assert [e.sequence_name for e in fn.entries] == ["wf_a", "wf_z"]


def test_review_entries_carry_model_and_split_metadata() -> None:
    refs = [_ref("fp_a", False)]
    preds = [_pred("fp_a", True, trigger=2)]

    fp, _ = build_review_sets(refs, preds, split="test", model_name="my-model")

    [entry] = fp.entries
    assert entry == ReviewEntry(
        sequence_name="fp_a",
        split="test",
        model_name="my-model",
        ground_truth=False,
        predicted=True,
        trigger_frame_index=2,
    )


def test_missing_predictions_are_skipped_not_fatal() -> None:
    refs = [_ref("wf_a", True), _ref("fp_a", False)]
    preds = [_pred("wf_a", False)]  # fp_a has no prediction

    fp, fn = build_review_sets(refs, preds, split="val", model_name="m")

    # wf_a is FN; fp_a simply doesn't appear.
    assert [e.sequence_name for e in fn.entries] == ["wf_a"]
    assert fp.entries == []
