"""FP / FN extraction from a model's per-sequence predictions.

Given a list of :class:`~data_quality_sequential.dataset.SequenceRef` ground
truths and a list of :class:`Prediction`s, splits disagreements into two
:class:`ReviewSet`s:

* FP — predicted positive, ground truth negative.
* FN — predicted negative, ground truth positive.

Both sets are emitted unranked in stable alphabetical order by sequence name.
Ranking is deliberately deferred (see
``docs/specs/2026-04-23-sequential-label-audit-design.md`` §8).
"""

from dataclasses import dataclass
from typing import Literal

from .dataset import SequenceRef


@dataclass
class Prediction:
    """One model's verdict on a single sequence."""

    sequence_name: str
    predicted: bool
    trigger_frame_index: int | None = None


@dataclass
class ReviewEntry:
    """One sequence flagged for human review."""

    sequence_name: str
    split: str
    model_name: str
    ground_truth: bool
    predicted: bool
    trigger_frame_index: int | None


@dataclass
class ReviewSet:
    """All :class:`ReviewEntry`s of one kind for one (model, split)."""

    kind: Literal["fp", "fn"]
    split: str
    model_name: str
    entries: list[ReviewEntry]


def build_review_sets(
    refs: list[SequenceRef],
    predictions: list[Prediction],
    *,
    split: str,
    model_name: str,
) -> tuple[ReviewSet, ReviewSet]:
    """Return ``(fp_set, fn_set)`` for one ``(model, split)``.

    Sequences without a matching :class:`Prediction` are silently skipped.
    Within each set, entries are sorted alphabetically by ``sequence_name``.
    """
    pred_by_name = {p.sequence_name: p for p in predictions}

    fp_entries: list[ReviewEntry] = []
    fn_entries: list[ReviewEntry] = []

    for ref in refs:
        pred = pred_by_name.get(ref.name)
        if pred is None:
            continue
        entry = ReviewEntry(
            sequence_name=ref.name,
            split=split,
            model_name=model_name,
            ground_truth=ref.ground_truth,
            predicted=pred.predicted,
            trigger_frame_index=pred.trigger_frame_index,
        )
        if pred.predicted and not ref.ground_truth:
            fp_entries.append(entry)
        elif not pred.predicted and ref.ground_truth:
            fn_entries.append(entry)

    fp_entries.sort(key=lambda e: e.sequence_name)
    fn_entries.sort(key=lambda e: e.sequence_name)

    return (
        ReviewSet(kind="fp", split=split, model_name=model_name, entries=fp_entries),
        ReviewSet(kind="fn", split=split, model_name=model_name, entries=fn_entries),
    )
