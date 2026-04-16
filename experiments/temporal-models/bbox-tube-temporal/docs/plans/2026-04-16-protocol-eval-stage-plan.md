# Protocol-level eval stage — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `evaluate_packaged` DVC stage inside the `bbox-tube-temporal` experiment that evaluates each packaged `model.zip` via `pyrocore.TemporalModel.predict` on the `train` and `val` splits, and emits leaderboard-schema metrics plus PR/ROC curves.

**Architecture:** New script `scripts/evaluate_packaged.py` loads a packaged `BboxTubeTemporalModel` via `from_archive`, iterates split sequences using existing `bbox_tube_temporal.data` helpers, calls `model.load_sequence` + `model.predict` per sequence, then aggregates results via a new `bbox_tube_temporal.protocol_eval` module. Plotting helpers extracted into `bbox_tube_temporal.eval_plots` and shared with the existing `scripts/evaluate.py`. One DVC foreach stage with 4 instances (2 packaged variants × 2 splits).

**Tech Stack:** Python 3.11, uv, pytest, matplotlib, scikit-learn, pyrocore (`TemporalModel`, `Frame`, `TemporalModelOutput`), DVC.

**Spec:** `docs/specs/2026-04-16-protocol-eval-stage-design.md`.

---

## Conventions (apply to every task)

- **Working directory:** `experiments/temporal-models/bbox-tube-temporal/`. All `uv run` / path references are relative to it unless otherwise stated.
- **Imports:** top of module only. No function-local imports. No `# noqa` to silence them.
- **Commits:** always `git add <explicit paths>` (no `git add -A`, no wildcards). No Claude/Anthropic co-author trailer.
- **Commit prefix:** match the repo style — `feat(bbox-tube-temporal): ...`, `refactor(bbox-tube-temporal): ...`, `test(bbox-tube-temporal): ...`, `chore(bbox-tube-temporal): ...`.
- **Dependencies:** `uv add <pkg>` (never `uv pip install`). sklearn and matplotlib are already project deps — verify with `uv run python -c "import sklearn, matplotlib"`; no `uv add` needed if that succeeds.
- **Device/YOLO weights:** no CI-time integration test touches the real `model.zip` — driver tests monkeypatch `BboxTubeTemporalModel`.

---

## File structure

**Create:**
- `src/bbox_tube_temporal/eval_plots.py` — shared plot helpers (confusion matrix, PR curve, ROC curve). ~60 LoC.
- `src/bbox_tube_temporal/protocol_eval.py` — `SequenceRecord` dataclass + `build_record` + `compute_metrics` + TTD helper. ~90 LoC.
- `scripts/evaluate_packaged.py` — CLI driver. ~120 LoC.
- `tests/test_eval_plots.py` — smoke tests that each plot helper writes a non-empty PNG.
- `tests/test_protocol_eval.py` — unit tests on metrics + TTD edge cases.
- `tests/test_evaluate_packaged_driver.py` — driver smoke test with monkeypatched model.

**Modify:**
- `scripts/evaluate.py` — delete inline `plot_confusion_matrix` + PR/ROC plotting, import from `eval_plots` instead. Behavior unchanged.
- `dvc.yaml` — append `evaluate_packaged` foreach stage.

---

## Task 1: Create `eval_plots.py` with the three plot helpers

**Files:**
- Create: `src/bbox_tube_temporal/eval_plots.py`
- Test: `tests/test_eval_plots.py`

**Rationale:** `scripts/evaluate.py` already has `plot_confusion_matrix` (lines 31–57) plus inline PR/ROC plotting (lines 183–202). To let `evaluate_packaged.py` reuse them without duplicating code, extract all three into a tightly-scoped module. Pure functions, no global state.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_plots.py`:

```python
"""Smoke tests for shared plot helpers."""

import numpy as np

from bbox_tube_temporal.eval_plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


def test_plot_confusion_matrix_writes_nonempty_png(tmp_path):
    matrix = np.array([[10, 2], [3, 15]], dtype=float)
    out = tmp_path / "cm.png"

    plot_confusion_matrix(matrix, out, title="smoke test", normalized=False)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_confusion_matrix_normalized_writes_nonempty_png(tmp_path):
    matrix = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    out = tmp_path / "cm_norm.png"

    plot_confusion_matrix(matrix, out, title="smoke test (norm)", normalized=True)

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_pr_curve_writes_nonempty_png(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    out = tmp_path / "pr.png"

    plot_pr_curve(y_true, scores, out, title="PR")

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_roc_curve_writes_nonempty_png(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    out = tmp_path / "roc.png"

    plot_roc_curve(y_true, scores, out, title="ROC")

    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_plots.py -v`
Expected: `ImportError` / `ModuleNotFoundError` for `bbox_tube_temporal.eval_plots`.

- [ ] **Step 3: Implement `eval_plots.py`**

Create `src/bbox_tube_temporal/eval_plots.py` (copies the existing logic from `scripts/evaluate.py:31-57, 183-202` verbatim, plus titles + AUC labels become parameters):

```python
"""Shared matplotlib plot helpers for classification eval output."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    normalized: bool,
) -> None:
    """Render a 2x2 confusion matrix to ``output_path``.

    Args:
        matrix: 2x2 array. Rows = actual (fp, smoke). Cols = predicted.
        output_path: PNG path to write.
        title: Figure title.
        normalized: If True, format cells as percentages; else as integers.
    """
    labels = ["fp", "smoke"]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    vmax = float(matrix.max()) if matrix.size else 0.0
    threshold = vmax * 0.5
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value * 100:.1f}%" if normalized else f"{int(value)}"
            color = "white" if value > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    title: str = "PR",
) -> None:
    """Render a precision-recall curve. Title includes AP if computable."""
    ap = (
        float(average_precision_score(y_true, scores))
        if y_true.sum() > 0
        else 0.0
    )
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"{title} (AP={ap:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    title: str = "ROC",
) -> None:
    """Render a ROC curve. Title includes AUC if computable."""
    auc = (
        float(roc_auc_score(y_true, scores))
        if 0 < y_true.sum() < len(y_true)
        else 0.0
    )
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{title} (AUC={auc:.3f})")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_plots.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Lint + format**

Run: `uv run ruff check src/bbox_tube_temporal/eval_plots.py tests/test_eval_plots.py && uv run ruff format src/bbox_tube_temporal/eval_plots.py tests/test_eval_plots.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/bbox_tube_temporal/eval_plots.py tests/test_eval_plots.py
git commit -m "feat(bbox-tube-temporal): add shared eval_plots module

Extracts confusion-matrix / PR-curve / ROC-curve plotting into a
standalone helper module so both the raw-classifier evaluate stage
and the upcoming evaluate_packaged stage share one implementation.
No behavior change yet — scripts/evaluate.py still holds its own
copy; the next task rewires it."
```

---

## Task 2: Refactor `scripts/evaluate.py` to use `eval_plots`

**Files:**
- Modify: `scripts/evaluate.py`

**Rationale:** Remove the duplicated plot code so there's a single source of truth. Behavior must stay identical so existing stage outputs / DVC hashes for `evaluate.py` don't shift more than the refactor demands.

- [ ] **Step 1: Replace the plot imports + remove `plot_confusion_matrix` from `evaluate.py`**

Edit `scripts/evaluate.py`:

Replace the import block at the top (lines 10–25):

```python
import argparse
import json
import shutil
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from bbox_tube_temporal.dataset import TubePatchDataset
from bbox_tube_temporal.eval_plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from bbox_tube_temporal.lit_temporal import LitTemporalClassifier
```

(Note: drop `matplotlib.pyplot`, `precision_recall_curve`, `roc_curve` imports since they're now only used inside `eval_plots`.)

Delete the `plot_confusion_matrix` function body (currently lines 31–57).

- [ ] **Step 2: Replace the inline PR/ROC plot blocks with calls to the helpers**

Replace the PR block (currently lines 183–192) with:

```python
    plot_pr_curve(
        labels,
        probs,
        args.output_dir / "pr_curve.png",
        title="PR",
    )
```

Replace the ROC block (currently lines 194–202) with:

```python
    plot_roc_curve(
        labels,
        probs,
        args.output_dir / "roc_curve.png",
        title="ROC",
    )
```

- [ ] **Step 3: Run existing tests to confirm no regressions**

Run: `uv run pytest tests/ -v -x`
Expected: all tests pass (same set as before — the refactor does not add behavior).

- [ ] **Step 4: Lint + format**

Run: `uv run ruff check scripts/evaluate.py && uv run ruff format scripts/evaluate.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate.py
git commit -m "refactor(bbox-tube-temporal): route evaluate.py plots through eval_plots

Drops the inline plot_confusion_matrix function and inline PR/ROC
blocks in favor of the shared bbox_tube_temporal.eval_plots helpers.
No behavior change — output files and titles are identical."
```

---

## Task 3: Create `protocol_eval.py` with record + metrics logic

**Files:**
- Create: `src/bbox_tube_temporal/protocol_eval.py`
- Test: `tests/test_protocol_eval.py`

**Rationale:** Pure record-building + metrics math, fully unit-testable, independent of matplotlib / YOLO / pyrocore-the-abstract. Mirrors the split the leaderboard project already uses (`metrics.py`, `runner.py`).

**Leaderboard parity checked (from `temporal_model_leaderboard/metrics.py`):** rates rounded to 4 dp, TTD rounded to 1 dp, TTD returns `None` when no TPs; precision/recall/f1 each return `0.0` when their denominator is zero; `fpr = fp / (fp + tn)` with the `0.0` fallback when no negatives.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_protocol_eval.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_protocol_eval.py -v`
Expected: `ImportError` for `bbox_tube_temporal.protocol_eval`.

- [ ] **Step 3: Implement `protocol_eval.py`**

Create `src/bbox_tube_temporal/protocol_eval.py`:

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
from datetime import datetime
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
        trigger_frame_index: Decision frame, or ``None`` if negative.
        score: Sequence-level score used for PR/ROC
            (``max(tube_logits)`` per the ``max_logit`` aggregation
            rule baked into the packaged config). ``-inf`` when no
            tubes survived filtering.
        num_tubes_kept: Tubes that passed the inference-time filter.
        tube_logits: Per-tube logits (in kept-tube order).
        ttd_seconds: Time-to-detect for TPs, else ``None``.
        details: Passthrough of ``TemporalModelOutput.details``.
    """

    sequence_id: str
    label: str
    is_positive: bool
    trigger_frame_index: int | None
    score: float
    num_tubes_kept: int
    tube_logits: list[float]
    ttd_seconds: float | None = None
    details: dict = field(default_factory=dict)


def _score_from_tube_logits(tube_logits: list[float]) -> float:
    """max(logits), or ``-inf`` for an empty tube list."""
    return max(tube_logits) if tube_logits else -math.inf


def _compute_ttd_seconds(
    *,
    ground_truth: bool,
    predicted: bool,
    trigger_frame_index: int | None,
    frames: list[Frame],
) -> float | None:
    """TTD only for TPs with a valid trigger frame + timestamps.

    Mirrors ``temporal_model_leaderboard.runner._compute_ttd`` verbatim
    so the leaderboard and this stage agree on edge cases.
    """
    if not (ground_truth and predicted and trigger_frame_index is not None):
        return None
    first_ts: datetime | None = frames[0].timestamp if frames else None
    trigger_ts: datetime | None = (
        frames[trigger_frame_index].timestamp
        if trigger_frame_index < len(frames)
        else None
    )
    if first_ts is None or trigger_ts is None:
        return None
    return (trigger_ts - first_ts).total_seconds()


def build_record(
    *,
    sequence_dir: Path,
    label: str,
    frames: list[Frame],
    output: TemporalModelOutput,
) -> SequenceRecord:
    """Bundle a per-sequence eval record from the model's output + frames."""
    tube_logits = list(output.details.get("tube_logits", []))
    ttd_seconds = _compute_ttd_seconds(
        ground_truth=(label == "smoke"),
        predicted=output.is_positive,
        trigger_frame_index=output.trigger_frame_index,
        frames=frames,
    )
    return SequenceRecord(
        sequence_id=sequence_dir.name,
        label=label,
        is_positive=output.is_positive,
        trigger_frame_index=output.trigger_frame_index,
        score=_score_from_tube_logits(tube_logits),
        num_tubes_kept=int(output.details.get("num_tubes_kept", 0)),
        tube_logits=tube_logits,
        ttd_seconds=ttd_seconds,
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

    ttd_values = [r.ttd_seconds for r in records if r.ttd_seconds is not None]
    mean_ttd = round(sum(ttd_values) / len(ttd_values), 1) if ttd_values else None
    median_ttd = round(statistics.median(ttd_values), 1) if ttd_values else None

    y_true = np.array([1 if r.label == "smoke" else 0 for r in records])
    scores = np.array([r.score for r in records], dtype=float)
    pr_auc = (
        float(average_precision_score(y_true, scores))
        if y_true.sum() > 0
        else 0.0
    )
    roc_auc = (
        float(roc_auc_score(y_true, scores))
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
        "mean_ttd_seconds": mean_ttd,
        "median_ttd_seconds": median_ttd,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_protocol_eval.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Lint + format**

Run: `uv run ruff check src/bbox_tube_temporal/protocol_eval.py tests/test_protocol_eval.py && uv run ruff format src/bbox_tube_temporal/protocol_eval.py tests/test_protocol_eval.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/bbox_tube_temporal/protocol_eval.py tests/test_protocol_eval.py
git commit -m "feat(bbox-tube-temporal): add protocol_eval records + metrics

Introduces SequenceRecord + build_record + compute_metrics, the
pure functions the upcoming evaluate_packaged stage will use to
aggregate TemporalModelOutputs into a leaderboard-schema metrics
dict (tp/fp/fn/tn/P/R/F1/FPR/TTD plus PR-AUC/ROC-AUC). Field
names + rounding match temporal_model_leaderboard.metrics so the
numbers are directly comparable. Covered by unit tests — no script
wiring yet."
```

---

## Task 4: Create `scripts/evaluate_packaged.py` driver + smoke test

**Files:**
- Create: `scripts/evaluate_packaged.py`
- Create: `tests/test_evaluate_packaged_driver.py`

**Rationale:** Thin orchestrator: load packaged model → iterate sequences → collect records → compute metrics → write outputs. Strict per-sequence error policy (any crash aborts). No real `model.zip` in CI — driver test monkeypatches `BboxTubeTemporalModel`.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_evaluate_packaged_driver.py`:

```python
"""End-to-end smoke test for scripts/evaluate_packaged.py.

Monkeypatches BboxTubeTemporalModel so the driver never touches YOLO
or a real classifier — purely exercises the iteration / aggregation /
output-writing path.
"""

import json
import sys
from pathlib import Path

import pytest
from pyrocore import TemporalModelOutput

from bbox_tube_temporal import model as model_module


def _write_jpg(path: Path) -> None:
    """Write a minimal 1x1 JPEG placeholder.

    Driver never decodes the image (predict is monkeypatched), so a
    plausible-looking 1-byte payload is fine.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff")


def _make_sequence(split_dir: Path, category: str, seq_name: str, n_frames: int):
    seq_dir = split_dir / category / seq_name
    for i in range(n_frames):
        _write_jpg(seq_dir / "images" / f"cam_2024-01-01T10-00-{i:02d}.jpg")
    return seq_dir


class _FakeModel:
    """Stand-in for BboxTubeTemporalModel.

    load_sequence: defers to pyrocore's default (Frame per path).
    predict: returns a canned positive-or-not output based on the
    seq name prefix.
    """

    def load_sequence(self, frames):
        from pyrocore import Frame  # local import only in this fake test helper

        return [
            Frame(frame_id=p.stem, image_path=p, timestamp=None) for p in frames
        ]

    def predict(self, frames):
        # Decide based on how many frames we got — purely to vary outputs.
        is_pos = len(frames) >= 3
        return TemporalModelOutput(
            is_positive=is_pos,
            trigger_frame_index=(len(frames) - 1) if is_pos else None,
            details={
                "num_tubes_kept": 1 if is_pos else 0,
                "tube_logits": [2.5] if is_pos else [],
            },
        )


def test_evaluate_packaged_writes_expected_outputs(tmp_path, monkeypatch):
    sequences_dir = tmp_path / "sequences"
    output_dir = tmp_path / "out"
    _make_sequence(sequences_dir, "wildfire", "wf_seq_a", n_frames=4)  # TP
    _make_sequence(sequences_dir, "wildfire", "wf_seq_b", n_frames=2)  # FN
    _make_sequence(sequences_dir, "fp", "fp_seq_c", n_frames=4)  # FP
    _make_sequence(sequences_dir, "fp", "fp_seq_d", n_frames=1)  # TN

    monkeypatch.setattr(
        model_module.BboxTubeTemporalModel,
        "from_archive",
        classmethod(lambda cls, path, device=None: _FakeModel()),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_packaged.py",
            "--model-zip",
            str(tmp_path / "placeholder.zip"),
            "--sequences-dir",
            str(sequences_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "fake-variant-fake-split",
        ],
    )

    # Import driver here so monkeypatched argv takes effect.
    from scripts import evaluate_packaged

    evaluate_packaged.main()

    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "predictions.json").is_file()
    assert (output_dir / "dropped.json").is_file()
    assert (output_dir / "confusion_matrix.png").is_file()
    assert (output_dir / "confusion_matrix_normalized.png").is_file()
    assert (output_dir / "pr_curve.png").is_file()
    assert (output_dir / "roc_curve.png").is_file()
    assert (output_dir / "errors").is_dir()

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["model_name"] == "fake-variant-fake-split"
    assert metrics["num_sequences"] == 4
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert "pr_auc" in metrics and "roc_auc" in metrics

    predictions = json.loads((output_dir / "predictions.json").read_text())
    assert len(predictions) == 4
    assert {p["sequence_id"] for p in predictions} == {
        "wf_seq_a",
        "wf_seq_b",
        "fp_seq_c",
        "fp_seq_d",
    }


def test_evaluate_packaged_strict_errors_abort(tmp_path, monkeypatch):
    """Any predict() exception must bubble out — strict policy."""
    sequences_dir = tmp_path / "sequences"
    output_dir = tmp_path / "out"
    _make_sequence(sequences_dir, "wildfire", "wf_seq", n_frames=3)

    class _BrokenModel(_FakeModel):
        def predict(self, frames):
            raise RuntimeError("simulated inference crash")

    monkeypatch.setattr(
        model_module.BboxTubeTemporalModel,
        "from_archive",
        classmethod(lambda cls, path, device=None: _BrokenModel()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_packaged.py",
            "--model-zip",
            str(tmp_path / "placeholder.zip"),
            "--sequences-dir",
            str(sequences_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "fake",
        ],
    )
    from scripts import evaluate_packaged

    with pytest.raises(RuntimeError, match="simulated inference crash"):
        evaluate_packaged.main()


def test_evaluate_packaged_skips_sequences_without_images(tmp_path, monkeypatch):
    """No images/ subdir → logged under dropped.json, not evaluated."""
    sequences_dir = tmp_path / "sequences"
    output_dir = tmp_path / "out"
    # Good sequence
    _make_sequence(sequences_dir, "wildfire", "wf_seq_ok", n_frames=3)
    # Bad sequence: directory exists but no images/ subdir
    (sequences_dir / "fp" / "fp_seq_bad").mkdir(parents=True)

    monkeypatch.setattr(
        model_module.BboxTubeTemporalModel,
        "from_archive",
        classmethod(lambda cls, path, device=None: _FakeModel()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_packaged.py",
            "--model-zip",
            str(tmp_path / "placeholder.zip"),
            "--sequences-dir",
            str(sequences_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "fake",
        ],
    )
    from scripts import evaluate_packaged

    evaluate_packaged.main()

    dropped = json.loads((output_dir / "dropped.json").read_text())
    assert len(dropped) == 1
    assert dropped[0]["sequence_id"] == "fp_seq_bad"
    assert dropped[0]["reason"] == "no_images"

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["num_sequences"] == 1  # only the good one counts
```

**Note on the import inside `_FakeModel.load_sequence`:** this is a test-helper class inside a test module, not production code. The project rule about no function-local imports applies to `src/`, not to test-only `_Fake*` helpers where the import would otherwise force top-level coupling between the fake and the real class. That said, the `from pyrocore import Frame` import can live at the top of the test file — move it there and drop the in-method import so the test file stays clean. Update the test as:

```python
from pyrocore import Frame, TemporalModelOutput
```

...and change `_FakeModel.load_sequence` to use the top-level `Frame` symbol. The snippet above has it inside the method for illustration — put it at the top when you write the file.

- [ ] **Step 2: Run the test to confirm the driver is missing**

Run: `uv run pytest tests/test_evaluate_packaged_driver.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.evaluate_packaged'`.

- [ ] **Step 3: Ensure `scripts/` is importable**

Check: `ls scripts/__init__.py`
- If missing: `touch scripts/__init__.py`, stage it with `git add scripts/__init__.py` (commit in step 7 along with everything else).
- If present: skip.

Rationale: the test imports `from scripts import evaluate_packaged`. Other scripts are run via CLI and never imported, so `__init__.py` may be absent.

- [ ] **Step 4: Implement `scripts/evaluate_packaged.py`**

Create `scripts/evaluate_packaged.py`:

```python
"""Evaluate a packaged bbox-tube-temporal model.zip via the pyrocore.TemporalModel protocol.

Loads the archive with BboxTubeTemporalModel.from_archive, iterates
sequences in the given split directory, calls model.load_sequence +
model.predict per sequence, and writes leaderboard-schema metrics
plus PR/ROC curves and per-sequence predictions to --output-dir.

Strict error policy: any per-sequence exception aborts the run.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from bbox_tube_temporal.data import get_sorted_frames, is_wf_sequence, list_sequences
from bbox_tube_temporal.eval_plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from bbox_tube_temporal.model import BboxTubeTemporalModel
from bbox_tube_temporal.protocol_eval import (
    SequenceRecord,
    build_record,
    compute_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zip", type=Path, required=True)
    parser.add_argument("--sequences-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        required=True,
        help="Label embedded in metrics.json (e.g. 'vit_dinov2_finetune-val').",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device selection (cuda/mps/cpu). Defaults to auto.",
    )
    return parser.parse_args()


def _record_to_json(rec: SequenceRecord) -> dict:
    """Serialise a record for predictions.json (drops the verbose details blob)."""
    return {
        "sequence_id": rec.sequence_id,
        "label": rec.label,
        "is_positive": rec.is_positive,
        "trigger_frame_index": rec.trigger_frame_index,
        "score": rec.score if rec.score != float("-inf") else None,
        "num_tubes_kept": rec.num_tubes_kept,
        "tube_logits": rec.tube_logits,
        "ttd_seconds": rec.ttd_seconds,
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "errors").mkdir(exist_ok=True)

    model = BboxTubeTemporalModel.from_archive(args.model_zip, device=args.device)

    sequences = list_sequences(args.sequences_dir)
    records: list[SequenceRecord] = []
    dropped: list[dict] = []

    for seq_dir in sequences:
        frame_paths = get_sorted_frames(seq_dir)
        if not frame_paths:
            dropped.append({"sequence_id": seq_dir.name, "reason": "no_images"})
            continue
        label = "smoke" if is_wf_sequence(seq_dir) else "fp"
        frames = model.load_sequence(frame_paths)
        output = model.predict(frames)  # strict: any exception aborts
        records.append(
            build_record(
                sequence_dir=seq_dir,
                label=label,
                frames=frames,
                output=output,
            )
        )

    metrics = compute_metrics(args.model_name, records)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output_dir / "dropped.json").write_text(json.dumps(dropped, indent=2))
    predictions = sorted(
        (_record_to_json(r) for r in records),
        key=lambda p: (p["score"] is None, -(p["score"] or 0.0)),
    )
    (args.output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))

    y_true = np.array([1 if r.label == "smoke" else 0 for r in records])
    scores = np.array([r.score for r in records], dtype=float)

    cm_counts = np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
        dtype=float,
    )
    neg_total = metrics["tn"] + metrics["fp"]
    pos_total = metrics["tp"] + metrics["fn"]
    cm_norm = np.array(
        [
            [
                metrics["tn"] / neg_total if neg_total else 0.0,
                metrics["fp"] / neg_total if neg_total else 0.0,
            ],
            [
                metrics["fn"] / pos_total if pos_total else 0.0,
                metrics["tp"] / pos_total if pos_total else 0.0,
            ],
        ],
        dtype=float,
    )

    plot_confusion_matrix(
        cm_counts,
        args.output_dir / "confusion_matrix.png",
        title=f"{args.model_name} (counts)",
        normalized=False,
    )
    plot_confusion_matrix(
        cm_norm,
        args.output_dir / "confusion_matrix_normalized.png",
        title=f"{args.model_name} (row-normalized)",
        normalized=True,
    )
    plot_pr_curve(y_true, scores, args.output_dir / "pr_curve.png", title=args.model_name)
    plot_roc_curve(y_true, scores, args.output_dir / "roc_curve.png", title=args.model_name)

    print(json.dumps(metrics, indent=2))
    print(
        f"[{args.model_name}] kept={len(records)} dropped={len(dropped)} "
        f"P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the driver tests to verify they pass**

Run: `uv run pytest tests/test_evaluate_packaged_driver.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 6: Run full test suite to catch regressions**

Run: `uv run pytest tests/ -v -x`
Expected: all tests pass.

- [ ] **Step 7: Lint + format**

Run: `uv run ruff check scripts/evaluate_packaged.py tests/test_evaluate_packaged_driver.py && uv run ruff format scripts/evaluate_packaged.py tests/test_evaluate_packaged_driver.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add scripts/evaluate_packaged.py tests/test_evaluate_packaged_driver.py
# Only if scripts/__init__.py was newly created in Step 3:
# git add scripts/__init__.py
git commit -m "feat(bbox-tube-temporal): evaluate_packaged driver script

Wires BboxTubeTemporalModel.from_archive through list_sequences +
protocol_eval.build_record + compute_metrics to produce
leaderboard-schema metrics.json, predictions.json, dropped.json,
confusion matrices, and PR/ROC curves for a packaged model.zip.
Strict per-sequence error policy: any predict() crash aborts the
run. Smoke-tested end-to-end with a monkeypatched fake model;
DVC stage wiring comes in the next commit."
```

---

## Task 5: Wire the DVC stage

**Files:**
- Modify: `dvc.yaml`

**Rationale:** Declare `evaluate_packaged` as a foreach stage with 4 instances (2 variants × 2 splits). Outputs land under `data/08_reporting/{split}/packaged/{variant}/`.

- [ ] **Step 1: Append the stage to `dvc.yaml`**

Append (after the existing `evaluate_*` / `package` / `compare_variants` stages — order doesn't matter for DVC, but grouping with other eval stages keeps the file readable):

```yaml
  evaluate_packaged:
    foreach:
      - {variant: vit_dinov2_finetune, split: train}
      - {variant: vit_dinov2_finetune, split: val}
      - {variant: gru_convnext_finetune, split: train}
      - {variant: gru_convnext_finetune, split: val}
    do:
      cmd: >-
        uv run python scripts/evaluate_packaged.py
        --model-zip data/06_models/${item.variant}/model.zip
        --sequences-dir data/01_raw/datasets/${item.split}
        --output-dir data/08_reporting/${item.split}/packaged/${item.variant}
        --model-name ${item.variant}-${item.split}
      deps:
        - scripts/evaluate_packaged.py
        - src/bbox_tube_temporal/data.py
        - src/bbox_tube_temporal/eval_plots.py
        - src/bbox_tube_temporal/inference.py
        - src/bbox_tube_temporal/model.py
        - src/bbox_tube_temporal/model_input.py
        - src/bbox_tube_temporal/protocol_eval.py
        - src/bbox_tube_temporal/tubes.py
        - data/06_models/${item.variant}/model.zip
        - data/01_raw/datasets/${item.split}
      outs:
        - data/08_reporting/${item.split}/packaged/${item.variant}/errors
        - data/08_reporting/${item.split}/packaged/${item.variant}/predictions.json:
            cache: false
        - data/08_reporting/${item.split}/packaged/${item.variant}/dropped.json:
            cache: false
      metrics:
        - data/08_reporting/${item.split}/packaged/${item.variant}/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item.split}/packaged/${item.variant}/pr_curve.png
        - data/08_reporting/${item.split}/packaged/${item.variant}/roc_curve.png
        - data/08_reporting/${item.split}/packaged/${item.variant}/confusion_matrix.png
        - data/08_reporting/${item.split}/packaged/${item.variant}/confusion_matrix_normalized.png
```

- [ ] **Step 2: Validate the DVC file parses and the stages expand**

Run: `uv run dvc stage list | grep evaluate_packaged`
Expected output (4 lines):
```
evaluate_packaged@{variant:vit_dinov2_finetune,split:train}
evaluate_packaged@{variant:vit_dinov2_finetune,split:val}
evaluate_packaged@{variant:gru_convnext_finetune,split:train}
evaluate_packaged@{variant:gru_convnext_finetune,split:val}
```

If `dvc` rejects the YAML, re-read the file with `uv run dvc stage list` (with no filter) to get the full error. Likely causes: indentation drift when appending, mis-typed `${item.variant}` vs `${item[variant]}`.

- [ ] **Step 3: Dry-run the DVC DAG to verify wiring**

Run: `uv run dvc status evaluate_packaged`
Expected: either
- "new" for all four stages (they've never run), OR
- changed-dep listings (if `model.zip` is already built).

Should **not** show "missing deps" for any of the src files or the model.zip paths if those exist locally.

- [ ] **Step 4: Commit**

```bash
git add dvc.yaml
git commit -m "chore(bbox-tube-temporal): add evaluate_packaged DVC stage

Declares a foreach stage that runs evaluate_packaged.py on the two
packaged variants (vit_dinov2_finetune, gru_convnext_finetune)
against both train and val splits, emitting metrics.json,
predictions.json, dropped.json, confusion matrices, and PR/ROC
curves under data/08_reporting/{split}/packaged/{variant}/."
```

---

## Task 6: Manual smoke run + leaderboard parity check

**Files:** none — this is a verification task.

**Rationale:** Unit tests cover the math; this step exercises the real pipeline end-to-end to confirm the numbers are sensible and to capture a baseline for the future activity discussed in the spec motivation.

- [ ] **Step 1: Run the stage for the `vit_dinov2_finetune` × `val` instance**

This depends on `data/06_models/vit_dinov2_finetune/model.zip` and `data/01_raw/datasets/val` being present locally. If either is missing, `dvc pull` them first:

```bash
uv run dvc pull data/06_models/vit_dinov2_finetune/model.zip data/01_raw/datasets/val
```

Then:

```bash
uv run dvc repro 'evaluate_packaged@{variant:vit_dinov2_finetune,split:val}'
```

Expected: completes without error. Prints a metrics summary line to stdout.

- [ ] **Step 2: Inspect the resulting metrics**

Run: `uv run python -c "import json; print(json.dumps(json.load(open('data/08_reporting/val/packaged/vit_dinov2_finetune/metrics.json')), indent=2))"`

Eyeball check:
- `num_sequences` > 0 and roughly matches the val split size
- `tp + fn` = positive count, `fp + tn` = negative count, consistent
- `precision` / `recall` / `f1` / `fpr` are in [0, 1]
- `mean_ttd_seconds` / `median_ttd_seconds` are either a positive float or `null` (never NaN, never negative)
- `pr_auc` / `roc_auc` are in [0, 1]

- [ ] **Step 3: Spot-check `predictions.json`**

Open `data/08_reporting/val/packaged/vit_dinov2_finetune/predictions.json`:
- Sorted with highest scores first (null scores last)
- Each entry has all expected keys
- `sequence_id`s match real directory names from `data/01_raw/datasets/val/{wildfire,fp}/`

- [ ] **Step 4: Sanity-compare against leaderboard numbers**

The leaderboard reports `vit_dinov2_finetune` (on the test split) at `P=0.8136, R=0.9664, F1=0.8834`. The val-split numbers will differ because it's a different split, but the raw-classifier val F1 from the README is 0.971 — our `val`-through-the-protocol F1 should land **between** these two numbers. If it lands outside `[0.80, 0.97]`, re-check the driver (likely wiring error).

No automated assertion here — this step is an informed sanity check.

- [ ] **Step 5: Run the remaining three instances**

```bash
uv run dvc repro evaluate_packaged
```

Expected: all four instances complete. Populates `data/08_reporting/{train,val}/packaged/{vit_dinov2_finetune,gru_convnext_finetune}/`.

- [ ] **Step 6: Commit the dvc.lock updates + tracked outputs**

```bash
git status   # inspect what's changed
git add dvc.lock
# DVC may have created new .gitignore files under data/08_reporting/{train,val}/packaged/<variant>/
# Inspect the `git status` output above and add each .gitignore by explicit path, e.g.:
# git add data/08_reporting/val/packaged/vit_dinov2_finetune/.gitignore
# Do NOT add the PNGs / JSONs themselves — DVC tracks them.
git commit -m "chore(bbox-tube-temporal): record evaluate_packaged dvc.lock"
```

---

## Self-review

- **Spec coverage:**
  - "Evaluate on train and val" → Task 5 foreach covers both splits.
  - "Two packaged variants" → Task 5 foreach lists both.
  - "Leaderboard-schema metrics + PR/ROC" → Task 3 `compute_metrics` (names/rounding pinned to leaderboard; PR/ROC added).
  - "Max(tube_logits) score for PR/ROC" → Task 3 `_score_from_tube_logits` with `-inf` fallback; tested.
  - "TTD matching leaderboard convention" → Task 3 `_compute_ttd_seconds` mirrors the leaderboard's `_compute_ttd` verbatim.
  - "Strict per-sequence error policy" → Task 4 does not wrap `predict` in try/except; tested in `test_evaluate_packaged_strict_errors_abort`.
  - "Plot helpers extracted into eval_plots.py" → Tasks 1–2.
  - "Output dir `data/08_reporting/{split}/packaged/{variant}/`" → Task 5.
  - "No new `params:` on the stage" → Task 5 `do:` has no `params:` block.
  - "Empty `errors/`" / "empty `dropped.json`" as DVC outs → Task 4 always creates them; Task 5 declares both.

- **Placeholder scan:** Clean. No TBD, no "implement later", no "similar to Task N". The one in-task aside about the `_ = dataclasses` trap is explicit about being deleted before commit; if a worker leaves it, the ruff step in Task 4 Step 7 will fail.

- **Type consistency:** `SequenceRecord` (Task 3) fields — `sequence_id`, `label`, `is_positive`, `trigger_frame_index`, `score`, `num_tubes_kept`, `tube_logits`, `ttd_seconds`, `details` — used consistently in `build_record`, `compute_metrics`, the driver, and `_record_to_json`. `compute_metrics` output dict keys match leaderboard `ModelMetrics` plus `pr_auc` / `roc_auc`. `plot_pr_curve` / `plot_roc_curve` / `plot_confusion_matrix` signatures match call sites in both `scripts/evaluate.py` (post-refactor) and `scripts/evaluate_packaged.py`.

No issues found.
