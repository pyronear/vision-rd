# TTD frame-index migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace time-to-detection (TTD) values computed from unreliable frame filename timestamps with TTD in frame index (`trigger_frame_index`) across the leaderboard and all four per-experiment evaluators.

**Architecture:** Mechanical rename + replacement. The `trigger_frame_index` field already exists on `TemporalModelOutput` (pyrocore) and on each per-experiment `SequenceResult` / dict row. Every `_compute_ttd_seconds` / `_extract_ttd_seconds` helper is replaced with a direct read of that field. No new shared helper — the computation collapses to an identity for TPs. One docstring line on `TemporalModelOutput.trigger_frame_index` codifies the convention.

**Spec:** [`docs/specs/2026-04-20-ttd-frame-index-migration-design.md`](../specs/2026-04-20-ttd-frame-index-migration-design.md)

**Tech Stack:** Python 3.11+, uv, pytest, ruff, DVC. Per-sub-project Makefiles. Repo conventions: no Claude co-author in commit trailers; stage files explicitly (never `git add -A`); `uv`-native commands only (no `uv pip`).

**Conventions used throughout:**

- All commands run from **within the relevant sub-project directory** (e.g., `cd experiments/temporal-models/temporal-model-leaderboard/`). Each sub-project is its own uv project.
- Lint/format/test use `make lint`, `make format`, `make test` (wired in each sub-project's Makefile).
- Commit messages follow the repo's `type(scope): subject` style (see recent `git log --oneline -20`).

---

## Task 1: Document TTD convention on `TemporalModelOutput`

**Files:**

- Modify: `lib/pyrocore/src/pyrocore/types.py`
- Test: none (docstring only)

- [ ] **Step 1: Update the `trigger_frame_index` docstring**

Open `lib/pyrocore/src/pyrocore/types.py`. Current lines 24-37 define `TemporalModelOutput`. The `trigger_frame_index` attribute docstring currently reads:

```
        trigger_frame_index: Index of the frame where the model decided positive
            (for time-to-detection computation), or ``None`` if negative.
```

Replace it with:

```
        trigger_frame_index: Index of the frame (0-based) where the model
            decided positive, or ``None`` if negative. Time-to-detection
            in frames equals this value for a true positive. Do not
            compute TTD by subtracting frame filename timestamps — they
            are unreliable in the pyro-dataset test set.
```

No code changes.

- [ ] **Step 2: Lint and format**

Run (from `lib/pyrocore/`):

```bash
make lint
make format
```

Expected: both pass silently (only touched a docstring).

- [ ] **Step 3: Run tests**

Run: `make test`

Expected: all tests pass unchanged.

- [ ] **Step 4: Commit**

```bash
git add lib/pyrocore/src/pyrocore/types.py
git commit -m "docs(pyrocore): document TTD convention on trigger_frame_index"
```

---

## Task 2: Migrate `temporal-model-leaderboard` to frame-index TTD

**Files:**

- Modify: `experiments/temporal-models/temporal-model-leaderboard/src/temporal_model_leaderboard/types.py`
- Modify: `experiments/temporal-models/temporal-model-leaderboard/src/temporal_model_leaderboard/runner.py`
- Modify: `experiments/temporal-models/temporal-model-leaderboard/src/temporal_model_leaderboard/metrics.py`
- Modify: `experiments/temporal-models/temporal-model-leaderboard/src/temporal_model_leaderboard/leaderboard.py`
- Modify: `experiments/temporal-models/temporal-model-leaderboard/tests/test_runner.py`
- Modify: `experiments/temporal-models/temporal-model-leaderboard/tests/test_metrics.py`

All commands below run from `experiments/temporal-models/temporal-model-leaderboard/`.

- [ ] **Step 1: Update `tests/test_metrics.py` to the frame-index contract**

Replace the whole file with:

```python
"""Tests for temporal_model_leaderboard.metrics."""

from temporal_model_leaderboard.metrics import compute_metrics
from temporal_model_leaderboard.types import SequenceResult


def _make_result(gt: bool, pred: bool, ttd: int | None = None) -> SequenceResult:
    return SequenceResult(
        sequence_id="test",
        ground_truth=gt,
        predicted=pred,
        ttd_frames=ttd,
    )


class TestComputeMetrics:
    def test_perfect_predictions(self) -> None:
        results = [
            _make_result(gt=True, pred=True, ttd=1),
            _make_result(gt=True, pred=True, ttd=3),
            _make_result(gt=False, pred=False),
            _make_result(gt=False, pred=False),
        ]
        m = compute_metrics("test-model", results)

        assert m.tp == 2
        assert m.fp == 0
        assert m.fn == 0
        assert m.tn == 2
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.fpr == 0.0
        assert m.mean_ttd_frames == 2.0
        assert m.median_ttd_frames == 2.0

    def test_all_false_positives(self) -> None:
        results = [
            _make_result(gt=False, pred=True),
            _make_result(gt=False, pred=True),
        ]
        m = compute_metrics("test", results)

        assert m.tp == 0
        assert m.fp == 2
        assert m.precision == 0.0
        assert m.fpr == 1.0

    def test_all_false_negatives(self) -> None:
        results = [
            _make_result(gt=True, pred=False),
            _make_result(gt=True, pred=False),
        ]
        m = compute_metrics("test", results)

        assert m.fn == 2
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_mixed(self) -> None:
        results = [
            _make_result(gt=True, pred=True, ttd=4),
            _make_result(gt=True, pred=False),
            _make_result(gt=False, pred=True),
            _make_result(gt=False, pred=False),
        ]
        m = compute_metrics("test", results)

        assert m.tp == 1
        assert m.fp == 1
        assert m.fn == 1
        assert m.tn == 1
        assert m.precision == 0.5
        assert m.recall == 0.5
        assert m.mean_ttd_frames == 4.0

    def test_empty_results(self) -> None:
        m = compute_metrics("test", [])

        assert m.num_sequences == 0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.mean_ttd_frames is None
        assert m.median_ttd_frames is None

    def test_ttd_none_excluded(self) -> None:
        results = [
            _make_result(gt=True, pred=True, ttd=None),
            _make_result(gt=True, pred=True, ttd=2),
        ]
        m = compute_metrics("test", results)

        assert m.mean_ttd_frames == 2.0
        assert m.median_ttd_frames == 2.0

    def test_ttd_all_none(self) -> None:
        results = [
            _make_result(gt=True, pred=True, ttd=None),
        ]
        m = compute_metrics("test", results)

        assert m.mean_ttd_frames is None
        assert m.median_ttd_frames is None

    def test_median_even_count(self) -> None:
        results = [
            _make_result(gt=True, pred=True, ttd=1),
            _make_result(gt=True, pred=True, ttd=2),
            _make_result(gt=True, pred=True, ttd=3),
            _make_result(gt=True, pred=True, ttd=4),
        ]
        m = compute_metrics("test", results)

        assert m.median_ttd_frames == 2.5

    def test_model_name_preserved(self) -> None:
        m = compute_metrics("my-model", [])
        assert m.model_name == "my-model"

    def test_num_sequences(self) -> None:
        results = [_make_result(gt=True, pred=True, ttd=0)] * 5
        m = compute_metrics("test", results)
        assert m.num_sequences == 5
```

- [ ] **Step 2: Update `tests/test_runner.py` to the frame-index contract**

In `tests/test_runner.py`, replace occurrences of `ttd_seconds` with `ttd_frames` and update the numeric expectations. Specifically:

- Line 65 (`assert wf_result.ttd_seconds == 60.0  # frame 2 - frame 0`):
  Replace with: `assert wf_result.ttd_frames == 2  # trigger at frame 2`
- Line 69 (`assert fp_result.ttd_seconds is None  # not a TP`):
  Replace `ttd_seconds` with `ttd_frames` (keep the `is None` assertion).
- Line 79 (`assert all(r.ttd_seconds is None for r in results)`):
  Replace `ttd_seconds` with `ttd_frames`.
- Line 117 (`assert wf_result.ttd_seconds == 0.0`):
  Replace with: `assert wf_result.ttd_frames == 0`
- Line 131 (`assert wf_result.ttd_seconds is None`):
  Replace `ttd_seconds` with `ttd_frames`.

- [ ] **Step 3: Run tests and verify they fail**

```bash
uv run pytest tests/ -v
```

Expected: `test_metrics.py` and `test_runner.py` tests fail with errors like `AttributeError: 'SequenceResult' object has no attribute 'ttd_frames'` or `TypeError: SequenceResult.__init__() got an unexpected keyword argument 'ttd_frames'`. This confirms the tests now exercise the new contract.

- [ ] **Step 4: Update `src/temporal_model_leaderboard/types.py`**

Replace the file contents with:

```python
"""Result types for leaderboard evaluation."""

from dataclasses import dataclass, field


@dataclass
class SequenceResult:
    """Evaluation result for a single sequence.

    Attributes:
        sequence_id: Unique identifier (typically the sequence directory name).
        ground_truth: ``True`` if wildfire (positive), ``False`` if false positive.
        predicted: The model's binary classification decision.
        ttd_frames: Time-to-detection in frames (0-based trigger index)
            for true positives, or ``None`` if not a TP or the model did
            not report a trigger frame.
    """

    sequence_id: str
    ground_truth: bool
    predicted: bool
    ttd_frames: int | None = None


@dataclass
class ModelMetrics:
    """Aggregated metrics for a single model on the test set.

    Attributes:
        model_name: Human-readable model identifier.
        num_sequences: Total number of sequences evaluated.
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        tn: True negatives.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
        f1: Harmonic mean of precision and recall.
        fpr: FP / (FP + TN).
        mean_ttd_frames: Mean time-to-detection across TPs, or ``None``.
        median_ttd_frames: Median time-to-detection across TPs, or ``None``.
    """

    model_name: str
    num_sequences: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fpr: float
    mean_ttd_frames: float | None = None
    median_ttd_frames: float | None = None


@dataclass
class LeaderboardEntry:
    """One row of the leaderboard: a model with its metrics and per-sequence details."""

    metrics: ModelMetrics
    sequence_results: list[SequenceResult] = field(default_factory=list)
```

- [ ] **Step 5: Update `src/temporal_model_leaderboard/metrics.py`**

Replace the file contents with:

```python
"""Sequence-level classification metrics for leaderboard evaluation."""

import statistics

from .types import ModelMetrics, SequenceResult


def compute_metrics(
    model_name: str,
    results: list[SequenceResult],
) -> ModelMetrics:
    """Compute precision, recall, F1, FPR, and TTD from sequence results.

    TTD is reported in frame indices (0-based). See ``pyrocore.TemporalModelOutput``
    for the convention.

    Args:
        model_name: Human-readable model identifier.
        results: Per-sequence evaluation results.

    Returns:
        Aggregated :class:`ModelMetrics`.
    """
    tp = sum(1 for r in results if r.ground_truth and r.predicted)
    fp = sum(1 for r in results if not r.ground_truth and r.predicted)
    fn = sum(1 for r in results if r.ground_truth and not r.predicted)
    tn = sum(1 for r in results if not r.ground_truth and not r.predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    n_negative_gt = fp + tn
    fpr = fp / n_negative_gt if n_negative_gt > 0 else 0.0

    ttd_values = [
        r.ttd_frames
        for r in results
        if r.ground_truth and r.predicted and r.ttd_frames is not None
    ]
    mean_ttd = (
        round(sum(ttd_values) / len(ttd_values), 1) if ttd_values else None
    )
    median_ttd = (
        round(statistics.median(ttd_values), 1) if ttd_values else None
    )

    return ModelMetrics(
        model_name=model_name,
        num_sequences=len(results),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        fpr=round(fpr, 4),
        mean_ttd_frames=mean_ttd,
        median_ttd_frames=median_ttd,
    )
```

- [ ] **Step 6: Update `src/temporal_model_leaderboard/runner.py`**

Replace the file contents with:

```python
"""Evaluate a single TemporalModel on the pyro-dataset test set."""

import logging
from pathlib import Path

from pyrocore import TemporalModel
from tqdm import tqdm

from .dataset import get_sorted_frames, list_sequences
from .types import SequenceResult

logger = logging.getLogger(__name__)


def evaluate_model(
    model: TemporalModel,
    test_dir: Path,
) -> list[SequenceResult]:
    """Run a model on every test sequence and collect results.

    For each sequence:

    1. Discover sorted frame paths via :func:`get_sorted_frames`.
    2. Call ``model.load_sequence`` then ``model.predict`` to obtain the
       model's decision and trigger frame index.
    3. Record TTD in frames (= ``trigger_frame_index``) for true positives.

    Sequences with no images are skipped with a warning.

    Args:
        model: A :class:`~pyrocore.TemporalModel` instance.
        test_dir: Path to the test split root (e.g.,
            ``.../sequential_test/test``).

    Returns:
        List of :class:`SequenceResult`, one per evaluated sequence.
    """
    sequences = list_sequences(test_dir)
    logger.info("Found %d sequences in %s", len(sequences), test_dir)

    results: list[SequenceResult] = []

    for seq_path, ground_truth in tqdm(sequences, desc="eval", unit="seq"):
        frame_paths = get_sorted_frames(seq_path)
        if not frame_paths:
            logger.warning("Skipping %s: no images found", seq_path.name)
            continue

        frames = model.load_sequence(frame_paths)
        output = model.predict(frames)

        ttd_frames = (
            output.trigger_frame_index
            if ground_truth
            and output.is_positive
            and output.trigger_frame_index is not None
            else None
        )

        results.append(
            SequenceResult(
                sequence_id=seq_path.name,
                ground_truth=ground_truth,
                predicted=output.is_positive,
                ttd_frames=ttd_frames,
            )
        )

    logger.info("Evaluated %d sequences", len(results))
    return results
```

- [ ] **Step 7: Update `src/temporal_model_leaderboard/leaderboard.py`**

Locate line 9 (the `_LOWER_IS_BETTER` set) and replace:

```python
_LOWER_IS_BETTER = {"fpr", "mean_ttd_seconds", "median_ttd_seconds"}
```

with:

```python
_LOWER_IS_BETTER = {"fpr", "mean_ttd_frames", "median_ttd_frames"}
```

Locate the `headers` list (lines 50-59) and replace the last two entries:

```python
        "Mean TTD (s)",
        "Median TTD (s)",
```

with:

```python
        "Mean TTD (frames)",
        "Median TTD (frames)",
```

Locate the row-building block (lines 61-79) and update the two TTD cells:

```python
                f"{m.mean_ttd_seconds:.1f}" if m.mean_ttd_seconds is not None else "-",
                (
                    f"{m.median_ttd_seconds:.1f}"
                    if m.median_ttd_seconds is not None
                    else "-"
                ),
```

Replace with:

```python
                f"{m.mean_ttd_frames:.1f}" if m.mean_ttd_frames is not None else "-",
                (
                    f"{m.median_ttd_frames:.1f}"
                    if m.median_ttd_frames is not None
                    else "-"
                ),
```

- [ ] **Step 8: Run tests and verify they pass**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 9: Lint and format**

```bash
make lint
make format
```

Expected: both pass.

- [ ] **Step 10: Commit**

```bash
git add experiments/temporal-models/temporal-model-leaderboard/src/temporal_model_leaderboard/types.py \
        experiments/temporal-models/temporal-model-leaderboard/src/temporal_model_leaderboard/runner.py \
        experiments/temporal-models/temporal-model-leaderboard/src/temporal_model_leaderboard/metrics.py \
        experiments/temporal-models/temporal-model-leaderboard/src/temporal_model_leaderboard/leaderboard.py \
        experiments/temporal-models/temporal-model-leaderboard/tests/test_runner.py \
        experiments/temporal-models/temporal-model-leaderboard/tests/test_metrics.py
git commit -m "refactor(leaderboard): TTD in frames instead of seconds"
```

---

## Task 3: Migrate `bbox-tube-temporal` protocol_eval to frame-index TTD

**Files:**

- Modify: `experiments/temporal-models/bbox-tube-temporal/src/bbox_tube_temporal/protocol_eval.py`
- Modify: `experiments/temporal-models/bbox-tube-temporal/tests/test_protocol_eval.py`

All commands below run from `experiments/temporal-models/bbox-tube-temporal/`.

- [ ] **Step 1: Update `tests/test_protocol_eval.py`**

Make these targeted edits:

**1.1** — Update the import block (top of file, currently imports `_compute_ttd_seconds`):

```python
from bbox_tube_temporal.protocol_eval import (
    SequenceRecord,
    _compute_ttd_seconds,
    build_record,
    compute_metrics,
)
```

Replace with:

```python
from bbox_tube_temporal.protocol_eval import (
    SequenceRecord,
    build_record,
    compute_metrics,
)
```

**1.2** — Update the `_rec` helper signature (lines 17-27). Current:

```python
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
```

Replace with:

```python
def _rec(label, is_positive, *, score=0.0, trigger=None, ttd=None, num_kept=0):
    return SequenceRecord(
        sequence_id=f"seq_{label}_{is_positive}",
        label=label,
        is_positive=is_positive,
        trigger_frame_index=trigger,
        score=score,
        num_tubes_kept=num_kept,
        tube_logits=[],
        ttd_frames=ttd,
    )
```

**1.3** — Update every `ttd=<float>` argument to an int matching the trigger and every assertion of `mean_ttd_seconds` / `median_ttd_seconds` to `mean_ttd_frames` / `median_ttd_frames`, preserving the ratios but reinterpreting as frame indices. Concrete edits:

- `test_compute_metrics_all_correct` (lines 30-51): change `trigger=5, ttd=30.0` → `trigger=5, ttd=5` and `trigger=3, ttd=10.0` → `trigger=3, ttd=3`. Update the final two assertions:
  - `assert m["mean_ttd_seconds"] == 20.0` → `assert m["mean_ttd_frames"] == 4.0`
  - `assert m["median_ttd_seconds"] == 20.0` → `assert m["median_ttd_frames"] == 4.0`
- `test_compute_metrics_all_wrong`: rename the two final `_seconds` keys to `_frames`.
- `test_compute_metrics_rounds_to_four_decimals`: change `ttd=1.23456` → `ttd=1`. Update:
  - `assert m["mean_ttd_seconds"] == 1.2` → `assert m["mean_ttd_frames"] == 1.0`
  - `assert m["median_ttd_seconds"] == 1.2` → `assert m["median_ttd_frames"] == 1.0`
- `test_compute_metrics_empty_records`: rename the `_seconds` keys to `_frames`.
- `test_compute_metrics_ttd_median_three_values`: change `ttd=10.0, 30.0, 50.0` → `ttd=1, 3, 5`. Update:
  - `assert m["mean_ttd_seconds"] == 30.0` → `assert m["mean_ttd_frames"] == 3.0`
  - `assert m["median_ttd_seconds"] == 30.0` → `assert m["median_ttd_frames"] == 3.0`
- `test_compute_metrics_ignores_ttd_for_non_tp_records` (lines 315-332): change `ttd=10.0` (TP) → `ttd=1`; `ttd=999.0, 888.0, 777.0` bogus values → `ttd=99, 88, 77` (still distinct bogus integers). Update:
  - `assert m["mean_ttd_seconds"] == 10.0` → `assert m["mean_ttd_frames"] == 1.0`
  - `assert m["median_ttd_seconds"] == 10.0` → `assert m["median_ttd_frames"] == 1.0`

**1.4** — Delete the whole `# ── _compute_ttd_seconds edge cases ──────────────────────────────────────` section (lines 244-298 in the current file, covering tests `test_compute_ttd_returns_none_when_not_a_tp`, `test_compute_ttd_returns_none_when_trigger_out_of_range`, `test_compute_ttd_returns_none_when_timestamp_missing`, `test_compute_ttd_returns_none_for_empty_frames`, `test_compute_ttd_computes_seconds_for_valid_tp`). These tests exercise the `_compute_ttd_seconds` helper that we're deleting in the next step.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_protocol_eval.py -v
```

Expected: failures at import time (`ImportError: cannot import name '_compute_ttd_seconds'`) **or** `TypeError: SequenceRecord.__init__() got an unexpected keyword argument 'ttd_frames'`, confirming the tests now exercise the new contract.

- [ ] **Step 3: Update `src/bbox_tube_temporal/protocol_eval.py`**

Replace the file contents with:

```python
"""Protocol-level evaluation records + metrics for bbox-tube-temporal.

Operates on the output of ``BboxTubeTemporalModel.predict`` (the
pyrocore ``TemporalModel`` protocol) rather than on pre-built tube
patches.

Field names and rounding match the leaderboard's
``temporal_model_leaderboard.metrics.compute_metrics`` so numbers
produced here are directly comparable.
"""

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pyrocore import Frame, TemporalModelOutput
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class SequenceRecord:
    """One sequence's protocol-eval result.

    Attributes:
        sequence_id: Sequence directory name.
        label: ``"smoke"`` or ``"fp"``.
        is_positive: Model's binary decision.
        trigger_frame_index: Decision frame (0-based), or ``None`` if negative.
        score: Sequence-level score used for PR/ROC
            (``max(tube_logits)`` per the ``max_logit`` aggregation
            rule baked into the packaged config). ``-inf`` when no
            tubes survived filtering.
        num_tubes_kept: Tubes that passed the inference-time filter.
        tube_logits: Per-tube logits (in kept-tube order).
        ttd_frames: Time-to-detect in frames for TPs (= ``trigger_frame_index``),
            else ``None``.
        details: Passthrough of ``TemporalModelOutput.details``.
    """

    sequence_id: str
    label: str
    is_positive: bool
    trigger_frame_index: int | None
    score: float
    num_tubes_kept: int
    tube_logits: list[float]
    ttd_frames: int | None = None
    details: dict = field(default_factory=dict)


def _score_from_tube_logits(tube_logits: list[float]) -> float:
    """max(logits), or ``-inf`` for an empty tube list."""
    return max(tube_logits) if tube_logits else -math.inf


def build_record(
    *,
    sequence_dir: Path,
    label: str,
    frames: list[Frame],
    output: TemporalModelOutput,
) -> SequenceRecord:
    """Bundle a per-sequence eval record from the model's output + frames.

    ``frames`` is accepted (for interface symmetry with the previous
    timestamp-based TTD computation) but no longer read — TTD is taken
    directly from ``output.trigger_frame_index`` per the pyrocore
    convention.
    """
    kept = output.details.get("tubes", {}).get("kept", [])
    tube_logits = [float(t["logit"]) for t in kept]
    ground_truth = label == "smoke"
    ttd_frames = (
        output.trigger_frame_index
        if ground_truth
        and output.is_positive
        and output.trigger_frame_index is not None
        else None
    )
    return SequenceRecord(
        sequence_id=sequence_dir.name,
        label=label,
        is_positive=output.is_positive,
        trigger_frame_index=output.trigger_frame_index,
        score=_score_from_tube_logits(tube_logits),
        num_tubes_kept=len(kept),
        tube_logits=tube_logits,
        ttd_frames=ttd_frames,
        details=dict(output.details),
    )


def compute_metrics(model_name: str, records: list[SequenceRecord]) -> dict:
    """Aggregate leaderboard-style metrics + PR/ROC AUCs over records.

    Returns a plain dict so it serializes with ``json.dumps`` directly.
    Field names / rounding match
    ``temporal_model_leaderboard.metrics.ModelMetrics``.
    """
    tp = sum(1 for r in records if r.label == "smoke" and r.is_positive)
    fp = sum(1 for r in records if r.label == "fp" and r.is_positive)
    fn = sum(1 for r in records if r.label == "smoke" and not r.is_positive)
    tn = sum(1 for r in records if r.label == "fp" and not r.is_positive)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    n_neg = fp + tn
    fpr = fp / n_neg if n_neg > 0 else 0.0

    ttd_values = [
        r.ttd_frames
        for r in records
        if r.label == "smoke" and r.is_positive and r.ttd_frames is not None
    ]
    mean_ttd = (
        round(sum(ttd_values) / len(ttd_values), 1) if ttd_values else None
    )
    median_ttd = (
        round(statistics.median(ttd_values), 1) if ttd_values else None
    )

    y_true = np.array([1 if r.label == "smoke" else 0 for r in records])
    scores = np.array([r.score for r in records], dtype=float)
    # sklearn rejects ±inf — clip to finite range before AUC computation.
    scores_finite = np.clip(scores, np.finfo(float).min, np.finfo(float).max)
    pr_auc = (
        float(average_precision_score(y_true, scores_finite))
        if y_true.sum() > 0
        else 0.0
    )
    roc_auc = (
        float(roc_auc_score(y_true, scores_finite))
        if 0 < y_true.sum() < len(y_true)
        else 0.0
    )

    return {
        "model_name": model_name,
        "num_sequences": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "mean_ttd_frames": mean_ttd,
        "median_ttd_frames": median_ttd,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
    }
```

- [ ] **Step 4: Check for other consumers of `ttd_seconds` / `_compute_ttd_seconds`**

Run from the repo root:

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
```

Then from within `experiments/temporal-models/bbox-tube-temporal/`:

```bash
uv run python -c "import bbox_tube_temporal"
```

Also search for any residual references:

Run: `git grep -n "ttd_seconds\|_compute_ttd_seconds" experiments/temporal-models/bbox-tube-temporal/`

Expected: no matches. If any match appears (e.g., in a script or notebook), update that file in the same task (rename `ttd_seconds` → `ttd_frames`; delete timestamp-subtraction code). The bbox-tube-temporal package has a `scripts/` directory and a `docs/` directory; any code-level hits must be fixed, but string matches in DVC-tracked JSON outputs in `data/` should be ignored (they'll be regenerated in Task 7).

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_protocol_eval.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Run the full test suite**

```bash
make test
```

Expected: pass.

- [ ] **Step 7: Lint and format**

```bash
make lint
make format
```

Expected: both pass.

- [ ] **Step 8: Commit**

```bash
git add experiments/temporal-models/bbox-tube-temporal/src/bbox_tube_temporal/protocol_eval.py \
        experiments/temporal-models/bbox-tube-temporal/tests/test_protocol_eval.py
git commit -m "refactor(bbox-tube-temporal): TTD in frames instead of seconds"
```

If Step 4 flagged additional files, include them in the `git add` list.

---

## Task 4: Migrate `pyro-detector-baseline` to frame-index TTD

**Files:**

- Modify: `experiments/temporal-models/pyro-detector-baseline/src/pyro_detector_baseline/evaluator.py`
- Modify: `experiments/temporal-models/pyro-detector-baseline/scripts/evaluate.py`
- Modify: `experiments/temporal-models/pyro-detector-baseline/scripts/sweep.py`
- Modify: `experiments/temporal-models/pyro-detector-baseline/tests/test_evaluator.py`

All commands below run from `experiments/temporal-models/pyro-detector-baseline/`.

**Preconditions verified:** `scripts/predict.py:107-121` already writes `confirmed_frame_index` into each result row (see `tracking_results.json` shape). Evaluator just needs to read that instead of the timestamp pair.

- [ ] **Step 1: Update `tests/test_evaluator.py`**

Make these targeted edits:

**1.1** — In `test_ttd_computed` (lines 75-92), replace the whole test with:

```python
    def test_ttd_computed(self):
        results = [
            _result(True, True, confirmed_frame_index=2),
            _result(True, True, confirmed_frame_index=4),
        ]
        m = compute_metrics(results)
        assert m["mean_ttd_frames"] == 3.0
        assert m["median_ttd_frames"] == 3.0
```

**1.2** — In `test_ttd_none_when_no_tp` (lines 94-98), replace the assertion keys:

```python
    def test_ttd_none_when_no_tp(self):
        results = [_result(True, False)]
        m = compute_metrics(results)
        assert m["mean_ttd_frames"] is None
        assert m["median_ttd_frames"] is None
```

**1.3** — Locate the `_result` helper at the top of the file. Its current signature takes `first_ts` and `conf_ts` keyword arguments. Replace the helper with:

```python
def _result(gt: bool, pred: bool, *, confirmed_frame_index: int | None = None) -> dict:
    return {
        "is_positive_gt": gt,
        "is_positive_pred": pred,
        "confirmed_frame_index": confirmed_frame_index,
    }
```

(If the existing helper passes more fields through, keep them — but remove any `first_timestamp` / `confirmed_timestamp` fields.)

**1.4** — If any other test in the file references `first_ts`, `conf_ts`, `first_timestamp`, `confirmed_timestamp`, `mean_ttd_seconds`, or `median_ttd_seconds`, update those assertions the same way: replace the timestamps with a `confirmed_frame_index` int and replace `_seconds` → `_frames` in metric key lookups. Run `grep -n "timestamp\|ttd_seconds" tests/test_evaluator.py` in the sub-project to find all occurrences.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_evaluator.py -v
```

Expected: `test_ttd_computed` fails with `KeyError: 'mean_ttd_frames'` or similar.

- [ ] **Step 3: Update `src/pyro_detector_baseline/evaluator.py`**

**3.1** — Delete `_extract_ttd_seconds` entirely (lines 30-47 in the current file).

**3.2** — Remove the unused `datetime` import from the top of the file (it was only used by the deleted helper). The `import statistics` line stays.

**3.3** — In `compute_metrics`, replace the TTD computation block (lines 84-86 in the current file):

```python
    ttd_seconds = _extract_ttd_seconds(results)
    mean_ttd = sum(ttd_seconds) / len(ttd_seconds) if ttd_seconds else None
    median_ttd = statistics.median(ttd_seconds) if ttd_seconds else None
```

with:

```python
    ttd_frames = [
        r["confirmed_frame_index"]
        for r in results
        if r["is_positive_gt"]
        and r["is_positive_pred"]
        and r.get("confirmed_frame_index") is not None
    ]
    mean_ttd = sum(ttd_frames) / len(ttd_frames) if ttd_frames else None
    median_ttd = statistics.median(ttd_frames) if ttd_frames else None
```

**3.4** — In the `compute_metrics` return dict (lines 107-110), replace:

```python
        "mean_ttd_seconds": (round(mean_ttd, 1) if mean_ttd is not None else None),
        "median_ttd_seconds": (
            round(median_ttd, 1) if median_ttd is not None else None
        ),
```

with:

```python
        "mean_ttd_frames": (round(mean_ttd, 1) if mean_ttd is not None else None),
        "median_ttd_frames": (
            round(median_ttd, 1) if median_ttd is not None else None
        ),
```

**3.5** — In the `compute_metrics` docstring (lines 50-63), replace:

```
    Args:
        results: List of per-sequence result dicts, each containing at least
            ``is_positive_gt``, ``is_positive_pred``, ``confirmed_timestamp``,
            and ``first_timestamp``.

    Returns:
        Dict with keys: ``num_sequences``, ``num_positive_gt``,
        ``num_negative_gt``, ``tp``, ``fp``, ``fn``, ``tn``, ``precision``,
        ``recall``, ``f1``, ``fpr``, ``mean_ttd_seconds``,
        ``median_ttd_seconds``.  TTD values are ``None`` when there are no
        true positives.
```

with:

```
    Args:
        results: List of per-sequence result dicts, each containing at least
            ``is_positive_gt``, ``is_positive_pred``, and
            ``confirmed_frame_index`` (0-based trigger frame index or None).

    Returns:
        Dict with keys: ``num_sequences``, ``num_positive_gt``,
        ``num_negative_gt``, ``tp``, ``fp``, ``fn``, ``tn``, ``precision``,
        ``recall``, ``f1``, ``fpr``, ``mean_ttd_frames``,
        ``median_ttd_frames``.  TTD values are ``None`` when there are no
        true positives.
```

**3.6** — In `compute_single_frame_baseline` (lines 114-138), the baseline currently constructs synthetic records with `confirmed_timestamp` / `first_timestamp`. Rewrite to use `confirmed_frame_index`:

```python
def compute_single_frame_baseline(results: list[dict]) -> dict:
    """Compute baseline metrics where any detection triggers an alarm.

    Simulates a naive strategy with no temporal filtering: a sequence is
    predicted positive if it contains at least one frame where the
    predictor returned a nonzero confidence. The baseline "triggers"
    on the first frame (index 0), so TTD is always 0 when positive.

    Args:
        results: List of per-sequence result dicts (same format as
            :func:`compute_metrics`).

    Returns:
        Metrics dict (same keys as :func:`compute_metrics`).
    """
    baseline_results = []
    for r in results:
        is_positive = r["num_detections_total"] > 0
        baseline_results.append(
            {
                "is_positive_gt": r["is_positive_gt"],
                "is_positive_pred": is_positive,
                "confirmed_frame_index": 0 if is_positive else None,
            }
        )
    return compute_metrics(baseline_results)
```

**3.7** — In `plot_ttd_histogram` (lines 222-248), update the data source and labels:

```python
def plot_ttd_histogram(results: list[dict], output_path: Path) -> None:
    """Histogram of time-to-detection for true-positive wildfire sequences.

    Skips plotting entirely if there are no true positives.
    """
    sns.set_theme(style="whitegrid")
    ttd_frames = [
        r["confirmed_frame_index"]
        for r in results
        if r["is_positive_gt"]
        and r["is_positive_pred"]
        and r.get("confirmed_frame_index") is not None
    ]

    if not ttd_frames:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(ttd_frames, bins=20, kde=True, ax=ax)
    mean_val = sum(ttd_frames) / len(ttd_frames)
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        label=f"Mean: {mean_val:.1f} frames",
    )
    ax.set_xlabel("Time to Detection (frames)")
    ax.set_ylabel("Count")
    ax.set_title("Time to Detection Distribution (True Positives)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
```

- [ ] **Step 4: Update `scripts/evaluate.py`**

Open `scripts/evaluate.py`. Locate lines 96-100 (the `if predictor_metrics["mean_ttd_seconds"] is not None:` block). Rename the dict keys there:

- `predictor_metrics["mean_ttd_seconds"]` → `predictor_metrics["mean_ttd_frames"]`
- `predictor_metrics["median_ttd_seconds"]` → `predictor_metrics["median_ttd_frames"]`
- If any display/log string contains the literal `"seconds"` or `"s"` (unit suffix), change it to `"frames"`.

Run `grep -n "ttd_seconds\|TTD.*s\|seconds" scripts/evaluate.py` from the sub-project to find all hits. Update each.

- [ ] **Step 5: Update `scripts/sweep.py`**

Open `scripts/sweep.py`. Locate lines 217-218:

```python
        ttd = row.get("mean_ttd_seconds")
        ttd_str = f"{ttd:.0f}" if ttd is not None else "N/A"
```

Replace with:

```python
        ttd = row.get("mean_ttd_frames")
        ttd_str = f"{ttd:.1f}" if ttd is not None else "N/A"
```

Run `grep -n "ttd_seconds" scripts/sweep.py` to confirm no remaining hits.

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Smoke-check that scripts still import cleanly**

```bash
uv run python -c "import pyro_detector_baseline.evaluator; print('ok')"
uv run python scripts/evaluate.py --help >/dev/null && echo "evaluate ok"
uv run python scripts/sweep.py --help >/dev/null && echo "sweep ok"
```

Expected: each prints "ok" / "... ok" with no import errors. (If a script lacks `--help`, replace with `python -m py_compile scripts/<name>.py`.)

- [ ] **Step 8: Lint and format**

```bash
make lint
make format
```

Expected: both pass.

- [ ] **Step 9: Commit**

```bash
git add experiments/temporal-models/pyro-detector-baseline/src/pyro_detector_baseline/evaluator.py \
        experiments/temporal-models/pyro-detector-baseline/scripts/evaluate.py \
        experiments/temporal-models/pyro-detector-baseline/scripts/sweep.py \
        experiments/temporal-models/pyro-detector-baseline/tests/test_evaluator.py
git commit -m "refactor(pyro-detector-baseline): TTD in frames instead of seconds"
```

---

## Task 5: Migrate `tracking-fsm-baseline` to frame-index TTD

**Files:**

- Modify: `experiments/temporal-models/tracking-fsm-baseline/src/tracking_fsm_baseline/evaluator.py`
- Modify: `experiments/temporal-models/tracking-fsm-baseline/scripts/evaluate.py`
- Modify: `experiments/temporal-models/tracking-fsm-baseline/scripts/sweep.py`
- Modify: `experiments/temporal-models/tracking-fsm-baseline/scripts/ablation.py`
- Modify: `experiments/temporal-models/tracking-fsm-baseline/tests/test_evaluator.py`

All commands below run from `experiments/temporal-models/tracking-fsm-baseline/`.

**Preconditions verified:** `scripts/track.py:164-174` already writes `confirmed_frame_index` into the SequenceResult dataclass, which is then `dataclasses.asdict()`-ed into the row dict. Evaluator just needs to read that instead of the timestamp pair.

- [ ] **Step 1: Update `tests/test_evaluator.py`**

Replace each TTD-related test:

**1.1** — `test_ttd_computation` (lines 56-73): replace with:

```python
    def test_ttd_computation(self):
        results = [
            {
                "is_positive_gt": True,
                "is_positive_pred": True,
                "confirmed_frame_index": 2,
            },
            {
                "is_positive_gt": True,
                "is_positive_pred": True,
                "confirmed_frame_index": 6,
            },
        ]
        m = compute_metrics(results)
        assert m["mean_ttd_frames"] == 4.0
        assert m["median_ttd_frames"] == 4.0
```

**1.2** — `test_median_even_count` (lines 75-104): replace with:

```python
    def test_median_even_count(self):
        results = [
            {
                "is_positive_gt": True,
                "is_positive_pred": True,
                "confirmed_frame_index": 2,
            },
            {
                "is_positive_gt": True,
                "is_positive_pred": True,
                "confirmed_frame_index": 4,
            },
            {
                "is_positive_gt": True,
                "is_positive_pred": True,
                "confirmed_frame_index": 6,
            },
            {
                "is_positive_gt": True,
                "is_positive_pred": True,
                "confirmed_frame_index": 8,
            },
        ]
        m = compute_metrics(results)
        # Median of [2, 4, 6, 8] = (4 + 6) / 2 = 5
        assert m["median_ttd_frames"] == 5.0
```

**1.3** — If the `_result` helper at the top of the file passes `confirmed_timestamp` / `first_timestamp`, replace those fields with `confirmed_frame_index: int | None = None`. Run `grep -n "timestamp" tests/test_evaluator.py` to find occurrences.

**1.4** — Any other assertion using `mean_ttd_seconds` or `median_ttd_seconds` keys: rename to `mean_ttd_frames` / `median_ttd_frames`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_evaluator.py -v
```

Expected: failures with `KeyError: 'mean_ttd_frames'` or similar.

- [ ] **Step 3: Update `src/tracking_fsm_baseline/evaluator.py`**

Apply the same shape of changes as Task 4 Step 3 for this file:

**3.1** — Delete `_extract_ttd_seconds` entirely (lines 33-56 of the current file).

**3.2** — Remove the unused `datetime` import if it becomes unused after the deletion (keep it if other code in the file uses it — `grep -n datetime src/tracking_fsm_baseline/evaluator.py` to check).

**3.3** — In `compute_metrics`, replace the TTD block (lines 93-95) with:

```python
    ttd_frames = [
        r["confirmed_frame_index"]
        for r in results
        if r["is_positive_gt"]
        and r["is_positive_pred"]
        and r.get("confirmed_frame_index") is not None
    ]
    mean_ttd = sum(ttd_frames) / len(ttd_frames) if ttd_frames else None
    median_ttd = statistics.median(ttd_frames) if ttd_frames else None
```

**3.4** — In the `compute_metrics` return dict, rename `mean_ttd_seconds` → `mean_ttd_frames` and `median_ttd_seconds` → `median_ttd_frames`.

**3.5** — Update the `compute_metrics` docstring to replace the timestamp references with `confirmed_frame_index` (same pattern as Task 4 Step 3.5).

**3.6** — If the file contains a `plot_ttd_histogram` or analogous plotting helper, apply the same changes as Task 4 Step 3.7 (read `confirmed_frame_index`, update labels to "frames").

**3.7** — Grep the whole file for remaining `ttd_seconds`, `first_timestamp`, `confirmed_timestamp`, `total_seconds`. Fix or delete every hit.

- [ ] **Step 4: Update `scripts/evaluate.py`, `scripts/sweep.py`, `scripts/ablation.py`**

For each of the three scripts, run from the sub-project:

```bash
grep -n "ttd_seconds\|Mean TTD\|Median TTD\|TTD.*s\b" scripts/<name>.py
```

For each matching line:

- Rename dict keys (`mean_ttd_seconds` → `mean_ttd_frames`, `median_ttd_seconds` → `median_ttd_frames`).
- Update display strings: `"{ttd:.0f}s"` or similar → `"{ttd:.1f}"` (drop the "s" suffix or replace with "frames").
- Update table column headers (`"Mean TTD"` can stay, but any `(s)` / `(seconds)` annotations should become `(frames)`).

Known hits (from prior grep):

- `scripts/ablation.py:142`: `ttd = row.get("mean_ttd_seconds")` → rename and update format string below.

- [ ] **Step 5: Run tests and smoke-check imports**

```bash
uv run pytest tests/ -v
uv run python -c "import tracking_fsm_baseline.evaluator; print('ok')"
for s in evaluate sweep ablation; do uv run python -m py_compile scripts/$s.py && echo "$s ok"; done
```

Expected: all tests pass; three `ok` prints.

- [ ] **Step 6: Lint and format**

```bash
make lint
make format
```

- [ ] **Step 7: Commit**

```bash
git add experiments/temporal-models/tracking-fsm-baseline/src/tracking_fsm_baseline/evaluator.py \
        experiments/temporal-models/tracking-fsm-baseline/scripts/evaluate.py \
        experiments/temporal-models/tracking-fsm-baseline/scripts/sweep.py \
        experiments/temporal-models/tracking-fsm-baseline/scripts/ablation.py \
        experiments/temporal-models/tracking-fsm-baseline/tests/test_evaluator.py
git commit -m "refactor(tracking-fsm-baseline): TTD in frames instead of seconds"
```

---

## Task 6: Migrate `mtb-change-detection` to frame-index TTD

**Files:**

- Modify: `experiments/temporal-models/mtb-change-detection/src/mtb_change_detection/evaluator.py`
- Modify: `experiments/temporal-models/mtb-change-detection/scripts/evaluate.py`
- Modify: `experiments/temporal-models/mtb-change-detection/scripts/sweep.py`
- Modify: `experiments/temporal-models/mtb-change-detection/tests/test_evaluator.py`

All commands below run from `experiments/temporal-models/mtb-change-detection/`.

**Preconditions verified:** `scripts/track.py:214-224` already writes `confirmed_frame_index` into each row dict.

- [ ] **Step 1: Update `tests/test_evaluator.py`**

**1.1** — Rewrite the `_result` helper at the top of the file:

```python
def _result(gt: bool, pred: bool, *, confirmed_frame_index: int | None = None) -> dict:
    return {
        "is_positive_gt": gt,
        "is_positive_pred": pred,
        "confirmed_frame_index": (
            confirmed_frame_index if pred else None
        ),
    }
```

(Replaces the existing helper that uses `confirmed_timestamp` / `first_timestamp`.)

**1.2** — `test_ttd_computation` (lines 56-73): replace with:

```python
    def test_ttd_computation(self):
        results = [
            {
                "is_positive_gt": True,
                "is_positive_pred": True,
                "confirmed_frame_index": 2,
            },
            {
                "is_positive_gt": True,
                "is_positive_pred": True,
                "confirmed_frame_index": 6,
            },
        ]
        m = compute_metrics(results)
        assert m["mean_ttd_frames"] == 4.0
        assert m["median_ttd_frames"] == 4.0
```

**1.3** — For `TestComputeYoloOnlyBaseline` tests (starting around line 76), if any row dict contains `confirmed_timestamp` or `first_timestamp`, remove those keys and add `confirmed_frame_index: None` (or an integer if the test simulates a positive). Also replace any `mean_ttd_seconds` / `median_ttd_seconds` assertions with `_frames` equivalents.

Run `grep -n "timestamp\|ttd_seconds" tests/test_evaluator.py` in the sub-project and fix every hit.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_evaluator.py -v
```

Expected: failures reference the new frame-index contract.

- [ ] **Step 3: Update `src/mtb_change_detection/evaluator.py`**

Apply the same shape of changes as Task 4 Step 3 (delete `_extract_ttd_seconds`, switch to `confirmed_frame_index`, rename dict keys, update docstring, update `plot_ttd_histogram` if present, remove unused `datetime` import if applicable). Concrete line references (current file):

- Delete `_extract_ttd_seconds` (lines 33-56).
- Replace lines 93-95 (TTD block) with the frame-index comprehension shown in Task 4 Step 3.3.
- Rename `mean_ttd_seconds`/`median_ttd_seconds` → `mean_ttd_frames`/`median_ttd_frames` in the `compute_metrics` return dict (lines 116-117).
- Update `plot_ttd_histogram` (lines 305-322) the same way as Task 4 Step 3.7.
- Update the `compute_metrics` docstring (lines 58-71).

- [ ] **Step 4: Update `scripts/evaluate.py` and `scripts/sweep.py`**

**4.1** — `scripts/evaluate.py`: locate lines 96-100 (the `tracking_metrics["mean_ttd_seconds"]` block). Rename dict keys and any display strings as in Task 4 Step 4.

**4.2** — `scripts/sweep.py`: locate lines 312-313:

```python
        ttd = row.get("mean_ttd_seconds")
        ttd_str = f"{ttd:.0f}" if ttd is not None else "N/A"
```

Replace with:

```python
        ttd = row.get("mean_ttd_frames")
        ttd_str = f"{ttd:.1f}" if ttd is not None else "N/A"
```

Run `grep -n "ttd_seconds" scripts/sweep.py scripts/evaluate.py` to confirm no remaining hits.

- [ ] **Step 5: Run tests and smoke-check imports**

```bash
uv run pytest tests/ -v
uv run python -c "import mtb_change_detection.evaluator; print('ok')"
for s in evaluate sweep; do uv run python -m py_compile scripts/$s.py && echo "$s ok"; done
```

- [ ] **Step 6: Lint and format**

```bash
make lint
make format
```

- [ ] **Step 7: Commit**

```bash
git add experiments/temporal-models/mtb-change-detection/src/mtb_change_detection/evaluator.py \
        experiments/temporal-models/mtb-change-detection/scripts/evaluate.py \
        experiments/temporal-models/mtb-change-detection/scripts/sweep.py \
        experiments/temporal-models/mtb-change-detection/tests/test_evaluator.py
git commit -m "refactor(mtb-change-detection): TTD in frames instead of seconds"
```

---

## Task 7: Regenerate DVC outputs

**Files:** DVC-tracked artifacts under each sub-project's `data/08_reporting/` and `data/07_model_output/` trees.

**Precondition:** Tasks 1-6 are committed and tests pass in each sub-project.

**Note on scope:** We only need to re-run DVC stages that depend on the modified `src/**/evaluator.py`, `src/**/metrics.py`, `src/**/protocol_eval.py`, or `src/**/runner.py` files. Upstream tracking / prediction stages are unaffected (they already emit `confirmed_frame_index`). DVC's dependency graph will determine what to rerun — just invoke `dvc repro` in each sub-project and let it decide.

- [ ] **Step 1: Regenerate leaderboard outputs**

```bash
cd experiments/temporal-models/temporal-model-leaderboard
uv run dvc repro
```

Expected: `data/07_model_output/*/results.json` and `data/08_reporting/leaderboard.{json,txt}` are recomputed. Old files now contain `mean_ttd_frames` / `median_ttd_frames` instead of `*_seconds`.

- [ ] **Step 2: Inspect the regenerated leaderboard**

```bash
cat data/08_reporting/leaderboard.txt
```

Expected: columns now show "Mean TTD (frames)" / "Median TTD (frames)"; numeric values are small floats (e.g., `0.0` – `15.0` range), no longer seconds.

Quick sanity check — the regenerated median should be roughly `old_median_seconds / <per-source mean spacing>` ≈ the per-source numbers from `docs/specs/2026-04-20-ttd-frame-index-migration-design.md` expressed as frame indices. Exact values will differ, but orders of magnitude should match. If a model has a median of `>20 frames`, that is worth flagging (it corresponds to >10 minutes at 30s cadence and may reveal a separate bug).

- [ ] **Step 3: Regenerate per-experiment outputs**

For each of the four experiments, in order (independent — can be parallelized):

```bash
for exp in bbox-tube-temporal pyro-detector-baseline tracking-fsm-baseline mtb-change-detection; do
  ( cd "experiments/temporal-models/$exp" && uv run dvc repro )
done
```

Expected: `data/08_reporting/**/metrics.json` files in each experiment now contain `mean_ttd_frames` / `median_ttd_frames`. If any experiment's `dvc repro` triggers model re-inference (GPU-heavy), stop and investigate — this is unexpected given the preconditions, and likely means the evaluator.py file is listed as a `deps:` of the upstream tracking stage in `dvc.yaml`. If that's the case, adjust `dvc.yaml` to not list the evaluator as a dependency of the tracking stage (it's a dependency of the metrics stage).

- [ ] **Step 4: Stage regenerated DVC-tracked JSON outputs**

DVC-tracked files live under `data/` and are git-ignored; DVC tracks them via `.dvc` meta-files and each sub-project's `dvc.lock`. Only those meta-files need to be staged.

Run `git status` and explicitly list every changed `dvc.lock` and `*.dvc` path. Example:

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git status
# Inspect the output; for each changed dvc.lock or *.dvc file, stage it by name:
git add experiments/temporal-models/temporal-model-leaderboard/dvc.lock
git add experiments/temporal-models/temporal-model-leaderboard/data/08_reporting/leaderboard.json.dvc
# ... repeat for each of the 5 sub-projects as needed
```

Do **not** use `git add -A` or wildcard globs. If `git status` shows raw files under `data/` (not a `.dvc` or `dvc.lock`), investigate — it means a stage output is tracked by git instead of DVC, which is unexpected and should be resolved before continuing.

- [ ] **Step 5: Commit regenerated outputs**

```bash
git commit -m "chore(experiments): regenerate leaderboard and metrics in frames"
```

---

## Task 8: Update READMEs

**Files:**

- Modify: `experiments/temporal-models/temporal-model-leaderboard/README.md`
- Modify: `experiments/temporal-models/pyro-detector-baseline/README.md`
- Modify: `experiments/temporal-models/mtb-change-detection/README.md`
- Modify: `experiments/temporal-models/tracking-fsm-baseline/README.md`

- [ ] **Step 1: Update `temporal-model-leaderboard/README.md`**

**1.1** — Replace the leaderboard table header line (line 7):

```
| Rank | Model | Precision | Recall | F1 | FPR | Mean TTD (s) | Median TTD (s) |
```

with:

```
| Rank | Model | Precision | Recall | F1 | FPR | Mean TTD (frames) | Median TTD (frames) |
```

**1.2** — Replace the table rows (lines 9-13) with regenerated numeric values from the current `data/08_reporting/leaderboard.txt` (produced in Task 7). Preserve the Rank / Model / links columns; substitute the TTD values with the new frame-index ones.

**1.3** — Replace the metrics bullet (line 30):

```
- **Mean / Median TTD** -- time-to-detection in seconds for true positives (time from first frame to trigger frame)
```

with:

```
- **Mean / Median TTD** -- time-to-detection in **frames** for true positives (0-based `trigger_frame_index` where the model first decides positive). Frames are nominally 30s apart in production, but filename timestamps in the sequential test set are unreliable, so we report in frames directly.
```

**1.4** — In the "Data" section, locate the bullet:

```
- Max 20 frames per sequence, 30s apart
```

Replace with:

```
- Up to tens of frames per sequence; production cadence is nominally 30s per frame, but filename timestamps in this test set are not reliable (see the TTD note above)
```

**1.5** — Update the `*Last updated: ...*` line to today's date if present (line 15).

- [ ] **Step 2: Update each of the three experiment READMEs**

For each of `pyro-detector-baseline`, `mtb-change-detection`, `tracking-fsm-baseline`:

**2.1** — Locate table headers containing `Mean TTD` / `Median TTD`. If the cell lacks a unit suffix, leave the header alone but **replace the numeric values** in the table rows with the regenerated frame-index numbers from the experiment's `data/08_reporting/**/metrics.json`.

**2.2** — If the README has a "Key findings" or prose section that quotes TTD numbers (e.g., "Mean TTD: 56s" in `tracking-fsm-baseline/README.md:84` and "Mean TTD: 58s (median: 15s)" in `mtb-change-detection/README.md:73`), rewrite the line with the frame-index values. Example transformation:

Before:
```
- **Mean TTD: 56s** (median: 30s)
```

After:
```
- **Mean TTD: 1.9 frames** (median: 1.0 frames) — nominal cadence is 30s/frame; see pyrocore's `TemporalModelOutput.trigger_frame_index` for the convention
```

Use the actual regenerated numbers, not the example.

- [ ] **Step 3: Cross-check for residual `ttd_seconds` / `seconds` references**

From the repo root:

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git grep -n "ttd_seconds\|TTD.*seconds" -- experiments lib
```

Expected: no hits in code (`.py`) or READMEs (`.md`) under `experiments/` or `lib/`. If any appear, fix them and include in the commit.

- [ ] **Step 4: Final repo-wide verification**

From the repo root:

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git grep -n "ttd_seconds\|_compute_ttd_seconds\|_extract_ttd_seconds" -- experiments lib
```

Expected: no hits under `experiments/` or `lib/`. Intentional hits in `docs/specs/2026-04-20-ttd-frame-index-migration-design.md` (which describes the bug) are expected and OK — that's why the path spec excludes `docs/`.

Also run all sub-projects' tests once more to catch any cross-sub-project breakage (`pyrocore` docstring shouldn't break anything, but verify):

```bash
for dir in lib/pyrocore \
           experiments/temporal-models/temporal-model-leaderboard \
           experiments/temporal-models/bbox-tube-temporal \
           experiments/temporal-models/pyro-detector-baseline \
           experiments/temporal-models/tracking-fsm-baseline \
           experiments/temporal-models/mtb-change-detection; do
  echo "=== $dir ==="
  ( cd "$dir" && uv run pytest tests/ -v ) || exit 1
done
```

Expected: all pass.

- [ ] **Step 5: Commit README updates**

```bash
git add experiments/temporal-models/temporal-model-leaderboard/README.md \
        experiments/temporal-models/pyro-detector-baseline/README.md \
        experiments/temporal-models/mtb-change-detection/README.md \
        experiments/temporal-models/tracking-fsm-baseline/README.md
git commit -m "docs(experiments): update TTD column headers and values to frames"
```

- [ ] **Step 6: Open PR**

Ask your human partner whether to open the PR; if yes, use `gh pr create` with a body summarizing:

- Motivation: buggy pyro-dataset filename timestamps produced inflated / misleading TTD seconds on the leaderboard.
- Fix: TTD is now reported in frame index (0-based `trigger_frame_index`) across the leaderboard and four per-experiment evaluators. Convention documented on `pyrocore.TemporalModelOutput`.
- Impact: leaderboard numbers change materially; interpretation changes from "seconds to detect" to "frame index of trigger" (multiply by 30 for nominal seconds).

Commit and PR must **not** include any Claude / Anthropic co-author trailers.
