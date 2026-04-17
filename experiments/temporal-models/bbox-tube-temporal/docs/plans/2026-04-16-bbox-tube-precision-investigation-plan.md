# Bbox-tube-temporal precision investigation — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute Tracks A (offline aggregation-rule sweeps), F (error-visualization notebook), and C (inference-config ablations) from the precision-investigation spec, and produce a summary report recommending whether any drop-in fix clears the precision target.

**Architecture:** Pure aggregation / threshold helpers live in a new `bbox_tube_temporal.aggregation_analysis` module, unit-tested in isolation. A one-off CLI `scripts/analyze_aggregation_rules.py` orchestrates loading `predictions.json`, running the sweeps, and rendering a markdown report at `data/08_reporting/aggregation_ablation.md`. A new Jupyter notebook (`notebooks/04-error-analysis.ipynb`) provides qualitative inspection of FN/FP sequences after Track A's quantitative output is in hand. Track C then reuses the same analysis on fresh `evaluate_packaged` outputs generated via `dvc exp run -S` parameter overrides — no new DVC stages or production-code changes.

**Tech Stack:** Python 3.11, uv, pytest, numpy, scikit-learn, DVC.

**Spec:** `docs/specs/2026-04-16-bbox-tube-precision-investigation.md`.

---

## Conventions (apply to every task)

- **Working directory:** `experiments/temporal-models/bbox-tube-temporal/`. All `uv run` / path references are relative to it unless otherwise stated.
- **Imports:** top of module only. No function-local imports. No `# noqa` to silence them.
- **Commits:** always `git add <explicit paths>` (no `git add -A`, no wildcards). No Claude/Anthropic co-author trailer.
- **Commit prefix:** match the repo style — `feat(bbox-tube-temporal): ...`, `test(bbox-tube-temporal): ...`, `chore(bbox-tube-temporal): ...`.
- **Dependencies:** `uv add <pkg>` (never `uv pip install`). numpy and scikit-learn are already project deps — verify with `uv run python -c "import numpy, sklearn"`; no `uv add` needed if that succeeds.
- **Out of scope for this plan:** Track B is conditional per the spec and only pursued if Tracks A + C + F leave a ≥ 3pp precision gap; if even that falls short, stop and write a new spec rather than expanding scope here.

---

## File structure

**Create:**
- `src/bbox_tube_temporal/aggregation_analysis.py` — pure helpers: prediction loading, aggregation rules, threshold search, metric derivation. ~140 LoC.
- `scripts/analyze_aggregation_rules.py` — CLI driver. Iterates variants × splits × rules. ~130 LoC.
- `tests/test_aggregation_analysis.py` — unit tests on the pure helpers.
- `notebooks/04-error-analysis.ipynb` — qualitative FN/FP inspection notebook (Track F).

**Produce (outputs, not tracked):**
- `data/08_reporting/aggregation_ablation.md` — Track A report.
- `data/08_reporting/aggregation_ablation_track_c.md` — Track C comparative report.

**Modify:**
- None in production code. Track C triggers DVC re-runs but does not change committed params/config.

---

# Track A — Offline aggregation-rule sweeps

## Task 1: Create the `aggregation_analysis` module skeleton with the prediction loader

**Files:**
- Create: `src/bbox_tube_temporal/aggregation_analysis.py`
- Test: `tests/test_aggregation_analysis.py`

**Rationale:** A `predictions.json` record carries `{sequence_id, label, tube_logits, num_tubes_kept, ...}`. All downstream analysis needs two arrays: `y_true` (1 for `label == "smoke"`, 0 for `"fp"`) and a per-sequence score derived from `tube_logits` via some aggregation rule. Isolate that extraction.

- [ ] **Step 1: Write the failing test**

Create `tests/test_aggregation_analysis.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregation_analysis.py -v`
Expected: `ModuleNotFoundError: No module named 'bbox_tube_temporal.aggregation_analysis'`.

- [ ] **Step 3: Implement `load_predictions`**

Create `src/bbox_tube_temporal/aggregation_analysis.py`:

```python
"""Offline analysis of sequence-level aggregation rules over per-tube logits.

Reads predictions.json files produced by scripts/evaluate_packaged.py and
derives sequence-level scores under alternative aggregation rules (max,
top-k-mean). Supports threshold sweeps, target-recall search, and
confusion-matrix derivation.
"""

import json
from pathlib import Path
from typing import Any


def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Load per-sequence prediction records, sorted by sequence_id.

    Accepts the JSON written by scripts/evaluate_packaged.py, which uses
    json.dump with default settings (so +/-Infinity serializes as the
    non-strict "Infinity" / "-Infinity" literals that json.loads handles).
    """
    records = json.loads(predictions_path.read_text())
    records.sort(key=lambda r: r["sequence_id"])
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aggregation_analysis.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/aggregation_analysis.py tests/test_aggregation_analysis.py
git commit -m "feat(bbox-tube-temporal): scaffold aggregation_analysis module with prediction loader"
```

---

## Task 2: Implement `aggregate_score` with `max` and `top_k_mean` rules

**Files:**
- Modify: `src/bbox_tube_temporal/aggregation_analysis.py`
- Test: `tests/test_aggregation_analysis.py`

**Rationale:** Two aggregation rules to explore: the current deployment's `max(tube_logits)` and the more conservative "top-k-mean" (require the *average of the top k logits* to exceed threshold — if only one tube fires loudly among quiet siblings, the mean drags below threshold). Picking mean-of-top-k instead of "all top-k above τ" keeps the scalar-score framing so threshold sweeps still work.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_aggregation_analysis.py`:

```python
import pytest

from bbox_tube_temporal.aggregation_analysis import aggregate_score


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregation_analysis.py -v`
Expected: 7 new tests fail with `ImportError: cannot import name 'aggregate_score'`.

- [ ] **Step 3: Implement `aggregate_score`**

Append to `src/bbox_tube_temporal/aggregation_analysis.py`:

```python
import numpy as np

AGGREGATION_RULES = ("max", "top_k_mean")


def aggregate_score(tube_logits: list[float], *, rule: str, k: int) -> float:
    """Aggregate per-tube logits into a single sequence-level score.

    Rules:
        * ``max``: maximum logit across all tubes. ``k`` is ignored.
        * ``top_k_mean``: mean of the k largest logits. If fewer than k
          tubes exist, returns ``-inf`` (sequence cannot clear the rule).

    Empty tube list always returns ``-inf``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if rule not in AGGREGATION_RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {AGGREGATION_RULES}")
    if not tube_logits:
        return -np.inf
    arr = np.asarray(tube_logits, dtype=float)
    if rule == "max":
        return float(arr.max())
    # top_k_mean
    if arr.size < k:
        return -np.inf
    top_k = np.partition(arr, -k)[-k:]
    return float(top_k.mean())
```

Update the module's import block at the top by moving the `import numpy as np` line above the `AGGREGATION_RULES` constant (keep all imports at the top of the file).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aggregation_analysis.py -v`
Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/aggregation_analysis.py tests/test_aggregation_analysis.py
git commit -m "feat(bbox-tube-temporal): add max + top_k_mean aggregation rules"
```

---

## Task 3: Implement `find_threshold_for_recall`

**Files:**
- Modify: `src/bbox_tube_temporal/aggregation_analysis.py`
- Test: `tests/test_aggregation_analysis.py`

**Rationale:** The core Track A experiment: find the smallest threshold τ such that sequence-level recall ≥ target on a given `(y_true, scores)` pair. This mirrors `calibration.calibrate_threshold` but operates on sequence-level scores (not per-tube sigmoid probs) and works on raw logit scores including `-inf`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_aggregation_analysis.py`:

```python
from bbox_tube_temporal.aggregation_analysis import find_threshold_for_recall


def test_find_threshold_returns_lowest_positive_score_for_full_recall():
    y_true = np.array([1, 1, 0])
    scores = np.array([3.0, 5.0, 1.0])

    # recall = 1.0 requires threshold <= 3.0; smallest such threshold equals 3.0
    assert find_threshold_for_recall(y_true, scores, target_recall=1.0) == 3.0


def test_find_threshold_allows_dropping_one_positive_at_recall_050():
    y_true = np.array([1, 1, 0])
    scores = np.array([3.0, 5.0, 1.0])

    # recall = 0.5 only needs 1 of 2 positives; smallest threshold = 5.0
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregation_analysis.py -v`
Expected: 5 new tests fail with `ImportError`.

- [ ] **Step 3: Implement `find_threshold_for_recall`**

Append to `src/bbox_tube_temporal/aggregation_analysis.py`:

```python
def find_threshold_for_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    target_recall: float,
) -> float:
    """Return the largest threshold whose recall ≥ target_recall.

    We count a sequence as predicted-positive when ``score >= threshold``.
    Sorting positive scores ascending, we may drop the ``n_drop`` lowest
    positives while still hitting the recall target; the returned
    threshold is the ``n_drop``-th positive score.

    Returns ``-inf`` if target_recall forces including positives whose
    score is ``-inf``.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall!r}")
    pos_scores = np.sort(scores[y_true == 1])
    if pos_scores.size == 0:
        raise ValueError("no positives in y_true; cannot calibrate recall")
    n_pos = pos_scores.size
    n_drop = int(np.floor(n_pos * (1.0 - target_recall)))
    return float(pos_scores[n_drop])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aggregation_analysis.py -v`
Expected: 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/aggregation_analysis.py tests/test_aggregation_analysis.py
git commit -m "feat(bbox-tube-temporal): add sequence-level threshold search by target recall"
```

---

## Task 4: Implement `metrics_at_threshold`

**Files:**
- Modify: `src/bbox_tube_temporal/aggregation_analysis.py`
- Test: `tests/test_aggregation_analysis.py`

**Rationale:** Given a threshold and `(y_true, scores)`, compute the confusion counts and derived rates (precision, recall, f1, fpr) that the final report will tabulate. Reuses the leaderboard schema vocabulary.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_aggregation_analysis.py`:

```python
from bbox_tube_temporal.aggregation_analysis import metrics_at_threshold


def test_metrics_at_threshold_all_correct():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([2.0, 3.0, 0.0, -1.0])

    m = metrics_at_threshold(y_true, scores, threshold=1.0)

    assert m == {
        "threshold": 1.0,
        "tp": 2, "fp": 0, "fn": 0, "tn": 2,
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "fpr": 0.0,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregation_analysis.py -v`
Expected: 4 new tests fail with `ImportError`.

- [ ] **Step 3: Implement `metrics_at_threshold`**

Append to `src/bbox_tube_temporal/aggregation_analysis.py`:

```python
def metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Compute TP/FP/FN/TN + precision/recall/f1/fpr for a scalar threshold.

    Sequences with ``score >= threshold`` are predicted positive.
    """
    pred = scores >= threshold
    y_true_bool = y_true.astype(bool)
    tp = int((pred & y_true_bool).sum())
    fp = int((pred & ~y_true_bool).sum())
    fn = int((~pred & y_true_bool).sum())
    tn = int((~pred & ~y_true_bool).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aggregation_analysis.py -v`
Expected: 19 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/aggregation_analysis.py tests/test_aggregation_analysis.py
git commit -m "feat(bbox-tube-temporal): add metrics_at_threshold helper"
```

---

## Task 5: Implement `build_scores_and_labels` + `summarize_rule`

**Files:**
- Modify: `src/bbox_tube_temporal/aggregation_analysis.py`
- Test: `tests/test_aggregation_analysis.py`

**Rationale:** Two thin orchestration helpers the driver will call once per (split × variant × rule). `build_scores_and_labels` applies an aggregation rule across every record; `summarize_rule` finds the target-recall threshold and derives metrics there. Keeping them out of the driver so they remain unit-testable.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_aggregation_analysis.py`:

```python
from bbox_tube_temporal.aggregation_analysis import (
    build_scores_and_labels,
    summarize_rule,
)


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
        _record("b", "fp", [2.0]),                # too few tubes → -inf
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aggregation_analysis.py -v`
Expected: 3 new tests fail with `ImportError`.

- [ ] **Step 3: Implement the two functions**

Append to `src/bbox_tube_temporal/aggregation_analysis.py`:

```python
def build_scores_and_labels(
    records: list[dict[str, Any]],
    *,
    rule: str,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``aggregate_score`` across records; return (y_true, scores)."""
    y = np.array([1 if r["label"] == "smoke" else 0 for r in records], dtype=int)
    s = np.array(
        [aggregate_score(r["tube_logits"], rule=rule, k=k) for r in records],
        dtype=float,
    )
    return y, s


def summarize_rule(
    records: list[dict[str, Any]],
    *,
    rule: str,
    k: int,
    target_recall: float,
) -> dict[str, Any]:
    """Run one aggregation rule at one target recall; return a flat row."""
    y, s = build_scores_and_labels(records, rule=rule, k=k)
    threshold = find_threshold_for_recall(y, s, target_recall=target_recall)
    metrics = metrics_at_threshold(y, s, threshold=threshold)
    return {
        "rule": rule,
        "k": k,
        "target_recall": target_recall,
        "n_sequences": int(y.size),
        "n_positive": int(y.sum()),
        **metrics,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aggregation_analysis.py -v`
Expected: 22 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/aggregation_analysis.py tests/test_aggregation_analysis.py
git commit -m "feat(bbox-tube-temporal): add scores/labels builder and rule summarizer"
```

---

## Task 6: Write the driver script `analyze_aggregation_rules.py`

**Files:**
- Create: `scripts/analyze_aggregation_rules.py`

**Rationale:** The CLI that glues it all together. Loads `predictions.json` for each (variant, split), runs a fixed rule grid, prints + writes a markdown report. Deliberately thin — no new logic beyond composition.

- [ ] **Step 1: Create the script**

Create `scripts/analyze_aggregation_rules.py`:

```python
"""Offline aggregation-rule analysis on evaluate_packaged predictions.

For each (variant, split), loads predictions.json and explores
alternative sequence-level aggregation rules over per-tube logits.
Writes a markdown report ranked by precision @ target_recall.

Usage:
    uv run python scripts/analyze_aggregation_rules.py \\
        --reporting-dir data/08_reporting \\
        --output data/08_reporting/aggregation_ablation.md \\
        --target-recall 0.95
"""

import argparse
from pathlib import Path

from bbox_tube_temporal.aggregation_analysis import (
    load_predictions,
    summarize_rule,
)

VARIANTS = ("gru_convnext_finetune", "vit_dinov2_finetune")
SPLITS = ("train", "val")
RULE_GRID = (
    ("max", 1),
    ("top_k_mean", 2),
    ("top_k_mean", 3),
)


def _format_row(row: dict) -> str:
    return (
        f"| {row['variant']} | {row['split']} | {row['rule']} | {row['k']} | "
        f"{row['threshold']:.4f} | {row['precision']:.4f} | "
        f"{row['recall']:.4f} | {row['f1']:.4f} | {row['fpr']:.4f} | "
        f"{row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} |"
    )


def _render_report(rows: list[dict], target_recall: float) -> str:
    header = (
        "# Aggregation-rule ablation\n\n"
        f"Target recall for threshold search: **{target_recall}**.\n\n"
        "One threshold is chosen per (variant, split, rule) to hit the target recall;\n"
        "precision/FPR/etc. are reported at that threshold.\n\n"
        "| variant | split | rule | k | threshold | precision | recall | F1 | FPR | TP | FP | FN | TN |\n"
        "|---------|-------|------|---|-----------|-----------|--------|----|----|----|----|----|----|\n"
    )
    body = "\n".join(_format_row(r) for r in rows)
    return header + body + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reporting-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args()

    rows: list[dict] = []
    for variant in VARIANTS:
        for split in SPLITS:
            predictions_path = (
                args.reporting_dir / split / "packaged" / variant / "predictions.json"
            )
            if not predictions_path.is_file():
                print(f"SKIP missing: {predictions_path}")
                continue
            records = load_predictions(predictions_path)
            for rule, k in RULE_GRID:
                row = summarize_rule(
                    records,
                    rule=rule,
                    k=k,
                    target_recall=args.target_recall,
                )
                row["variant"] = variant
                row["split"] = split
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_report(rows, args.target_recall))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script on existing predictions**

Run:
```bash
uv run python scripts/analyze_aggregation_rules.py \
    --reporting-dir data/08_reporting \
    --output data/08_reporting/aggregation_ablation.md \
    --target-recall 0.95
```

Expected: prints `Wrote N rows to data/08_reporting/aggregation_ablation.md` where N = variants × splits × rules (should be 12 with 2 variants, 2 splits, 3 rules). No exceptions.

- [ ] **Step 3: Inspect the report**

Run: `cat data/08_reporting/aggregation_ablation.md`
Expected: markdown table with 12 rows, one per (variant, split, rule). Note the highest-precision row at recall ≥ 0.95 on **val** for each variant.

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_aggregation_rules.py
git commit -m "feat(bbox-tube-temporal): add Track A aggregation-rule analysis CLI"
```

Do NOT commit the generated `data/08_reporting/aggregation_ablation.md` — it is a DVC-tracked output directory, so committing the markdown would collide with DVC's management. The file may be overwritten on later runs.

---

## Task 7: Decide whether Track A alone clears the precision target

**Files:**
- None modified.

- [ ] **Step 1: Check the val-packaged rows for each variant**

Inspect `data/08_reporting/aggregation_ablation.md`. For each variant, find the row on `val` with the highest precision at recall ≥ 0.95.

- [ ] **Step 2: Write a conclusion note**

Append a short section to `data/08_reporting/aggregation_ablation.md` (either by editing in place or re-running with a small doc addition — editing in place is fine):

- If best val-packaged precision ≥ 0.93: mark Track A **sufficient**, skip Tracks C and the rest of this plan. Write a final summary note and stop.
- If best val-packaged precision is in [0.90, 0.93): Track A delivers a partial drop-in fix. **Continue to Track C** to see if config alignment closes the remaining gap.
- If best val-packaged precision < 0.90: Track A alone is insufficient. **Continue to Track C**; Track B will likely also be needed per the spec (out of scope here).

There is no code to commit in this step; it is a decision checkpoint. Record the decision in the report.

---

# Track F — Error visualization notebook

## Task 8: Create the error-analysis notebook for qualitative FN/FP inspection

**Files:**
- Create: `notebooks/04-error-analysis.ipynb`

**Rationale:** Track A produces quantitative precision numbers but not explanations. Before deciding which config alignment (Track C) is likely to help, a human should look at the actual failing sequences. The notebook filters per-sequence predictions to FN and FP cases at the deployed threshold and renders raw-frame grids so errors can be characterized qualitatively: "these FPs are all thin clouds", "these FNs are all smoke plumes that barely show up", etc.

The notebook is an analyst tool, not production code. Minimal structure, no tests, kept short. Uses `aggregation_analysis.load_predictions` + direct filesystem access to `data/01_raw/datasets/<split>/<label>/<seq>/images/*.jpg`. No YOLO re-runs, no box overlays — scope kept to "show the images, show the numbers".

Runs after Track A (Task 7) so the human has quantitative context, and before Track C so qualitative findings can inform which config ablations are worth running.

- [ ] **Step 1: Scaffold the notebook**

Create `notebooks/04-error-analysis.ipynb` in Jupyter (or any equivalent editor) with the following cells. The source snippets below give each cell's contents verbatim.

**Markdown cell 1**

```markdown
# Error analysis — bbox-tube-temporal

Inspect FN and FP sequences from `evaluate_packaged` output. Renders
raw frames + per-tube logits so error modes can be characterized
qualitatively.

**Configure** the `VARIANT`, `SPLIT`, and `PACKAGED_SUBDIR` constants
in the next cell to switch between baseline and Track C snapshots.
```

**Code cell 2 — config + imports**

```python
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from bbox_tube_temporal.aggregation_analysis import load_predictions

PROJECT_ROOT = Path.cwd()
assert (PROJECT_ROOT / "pyproject.toml").is_file(), (
    f"Run from experiments/temporal-models/bbox-tube-temporal/, not {PROJECT_ROOT}"
)

VARIANT = "gru_convnext_finetune"
SPLIT = "val"
PACKAGED_SUBDIR = "packaged"
MAX_ERRORS_TO_SHOW = 10
FRAMES_PER_ROW = 5

PREDICTIONS_PATH = (
    PROJECT_ROOT / "data/08_reporting" / SPLIT / PACKAGED_SUBDIR / VARIANT / "predictions.json"
)
SEQUENCES_ROOT = PROJECT_ROOT / "data/01_raw/datasets" / SPLIT
```

**Code cell 3 — load + split into FN / FP**

```python
records = load_predictions(PREDICTIONS_PATH)

false_negatives = [r for r in records if r["label"] == "smoke" and not r["is_positive"]]
false_positives = [r for r in records if r["label"] == "fp" and r["is_positive"]]
print(f"FN: {len(false_negatives)}   FP: {len(false_positives)}")
```

**Code cell 4 — rendering helpers**

```python
def _sequence_image_paths(label: str, sequence_id: str) -> list[Path]:
    label_dir = "wildfire" if label == "smoke" else "fp"
    seq_dir = SEQUENCES_ROOT / label_dir / sequence_id / "images"
    return sorted(seq_dir.glob("*.jpg"))


def show_error(record: dict) -> None:
    paths = _sequence_image_paths(record["label"], record["sequence_id"])
    n = len(paths)
    rows = max(1, (n + FRAMES_PER_ROW - 1) // FRAMES_PER_ROW)
    fig, axes = plt.subplots(rows, FRAMES_PER_ROW, figsize=(FRAMES_PER_ROW * 2.2, rows * 2))
    axes_list = axes.flatten() if rows * FRAMES_PER_ROW > 1 else [axes]
    for i, ax in enumerate(axes_list):
        ax.axis("off")
        if i < n:
            ax.imshow(Image.open(paths[i]))
            ax.set_title(f"f{i}", fontsize=7)
    tubes_summary = ", ".join(f"{l:.2f}" for l in record["tube_logits"]) or "(no tubes)"
    fig.suptitle(
        f"{record['sequence_id']}  [{record['label']}]   "
        f"score={record['score']:.2f}  tubes={record['num_tubes_kept']}  "
        f"logits=[{tubes_summary}]",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()
```

**Markdown cell 5**

```markdown
## False Negatives (smoke sequences the model missed)
```

**Code cell 6**

```python
for r in false_negatives[:MAX_ERRORS_TO_SHOW]:
    show_error(r)
```

**Markdown cell 7**

```markdown
## False Positives (fp sequences the model flagged)
```

**Code cell 8**

```python
for r in false_positives[:MAX_ERRORS_TO_SHOW]:
    show_error(r)
```

- [ ] **Step 2: Run the notebook headless to verify it executes cleanly**

From the experiment root:
```bash
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/04-error-analysis.ipynb
```

Expected: all cells execute; output cells contain FN and FP image grids. No exceptions.

- [ ] **Step 3: Commit the notebook**

`make install` set up `nbstripout`, which strips cell outputs before committing — source cells go to git, outputs do not.

```bash
git add notebooks/04-error-analysis.ipynb
git commit -m "feat(bbox-tube-temporal): add error-analysis notebook for FN/FP inspection"
```

- [ ] **Step 4: Skim the notebook interactively and record observations**

Launch Jupyter locally (`uv run jupyter lab`) and open the notebook. Skim both image grids. For each variant, jot 2–3 sentences characterizing the error modes into a new file `data/08_reporting/error_observations.md`, e.g. "FPs on `gru_convnext_finetune` are dominated by thin cirrus clouds and distant cell-tower glints; FNs are short-lived thin plumes in low-contrast dawn light." This qualitative note feeds into Task 13's synthesis — cloud-heavy FPs suggest Track C's `confidence_threshold` alignment will help; building/structure FPs suggest it won't.

```bash
git add data/08_reporting/error_observations.md
git commit -m "docs(bbox-tube-temporal): record qualitative error modes from notebook"
```

---

# Track C — Inference-config ablations

Execute only if Task 7 did not declare Track A sufficient.

## Task 9: Identify the YOLO confidence threshold used for training-label generation

**Files:**
- Create: `data/08_reporting/track_c_prereq_notes.md`

**Rationale:** Track C experiment 1 raises `package.infer.confidence_threshold` from `0.01` to match whatever generated the training `.txt` label files in `data/01_raw/datasets/{train,val}/fp/<seq>/labels/*.txt`. Without a target value, the ablation is a guess. Inspecting pyro-dataset preparation scripts / dataset docs is the prerequisite.

- [ ] **Step 1: Inspect dataset preparation to locate the confidence threshold**

Look for the threshold in the `pyro-dataset` repository or the `pyronear` dataset documentation / README. Starting points:

```bash
# Search the pyro-dataset cached artifacts if any are in the experiment
ls data/01_raw/datasets/val/fp/ | head -3
find data/01_raw/datasets/val/fp/ -maxdepth 3 -name "labels" | head -1
# Inspect a labels file to see the confidence values present
head -10 "$(find data/01_raw/datasets/val/fp -type f -name '*.txt' | head -1)"
```

FP label files are 6-column YOLO format: `class cx cy w h confidence`. The minimum confidence observed in the label files is a lower bound on whatever threshold was used to filter detections during dataset prep.

- [ ] **Step 2: Compute the minimum confidence observed across val/fp labels**

Run a one-liner to scan all FP label confidences:

```bash
uv run python -c "
from pathlib import Path
mins = []
for lf in Path('data/01_raw/datasets/val/fp').rglob('labels/*.txt'):
    for line in lf.read_text().splitlines():
        parts = line.split()
        if len(parts) == 6:
            mins.append(float(parts[5]))
print(f'n_detections={len(mins)}  min_conf={min(mins):.4f}  p01={sorted(mins)[len(mins)//100]:.4f}')
"
```

Expected: prints a minimum confidence value. That is the **lowest threshold consistent with the training label data** (labels contain detections down to that confidence; anything below was filtered out).

- [ ] **Step 3: Record the decision**

Create `data/08_reporting/track_c_prereq_notes.md`:

```markdown
# Track C prerequisite: confidence threshold target

Observed minimum detection confidence in training val FP labels: <VALUE>.
Source scan: `data/01_raw/datasets/val/fp/**/labels/*.txt`.

Track C experiment 1 will set `package.infer.confidence_threshold` to
this observed minimum (rounded down to the nearest 0.05) to align the
deployment tube-construction confidence floor with what the classifier
saw in training.
```

Replace `<VALUE>` with the measured minimum. If the minimum is suspiciously close to 0 (say < 0.02), the labels were likely also built with `conf=0.01` and this ablation is a no-op — note that explicitly in the file and skip to Task 10.

- [ ] **Step 4: Commit**

```bash
git add data/08_reporting/track_c_prereq_notes.md
git commit -m "chore(bbox-tube-temporal): record confidence threshold prereq for Track C"
```

---

## Task 10: Run ablation C1 — align `package.infer.confidence_threshold`

**Files:**
- None modified (params overridden via DVC exp command, not committed).

**Rationale:** Verify whether the 25×-noisier inference detection stream is the driver of the per-tube FPR. Re-runs the `package` stage (which rebuilds `model.zip` with the overridden param) and all dependent `evaluate_packaged` stages.

- [ ] **Step 1: Re-run the pipeline with the overridden threshold**

Let `T` be the value decided in Task 8 (e.g. 0.15). Run:

```bash
uv run dvc exp run -S package.infer.confidence_threshold=<T> --no-commit evaluate_packaged
```

The `--no-commit` flag keeps the experiment out of the git history — we want ablation numbers, not new commits on `dvc.yaml`/`params.yaml`. The experiment artifacts live in `.dvc/tmp/` until promoted.

Expected: the `package` stage reruns for each packaged variant, then `evaluate_packaged` reruns for all 4 (variant × split) combinations. Runtime: ~30 min end-to-end on GPU.

- [ ] **Step 2: Snapshot the new predictions to a sibling directory**

The exp run rewrites `data/08_reporting/{split}/packaged/{variant}/predictions.json` in place. Copy the results to a comparable-but-distinct location before the next ablation overwrites them:

```bash
uv run python -c "
import shutil
from pathlib import Path
base = Path('data/08_reporting')
for split in ('train', 'val'):
    for variant in ('gru_convnext_finetune', 'vit_dinov2_finetune'):
        src = base / split / 'packaged' / variant
        dst = base / split / 'packaged_ablation_c1_conf' / variant
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f'{src} -> {dst}')
"
```

- [ ] **Step 3: Restore the baseline predictions via `dvc checkout`**

Run: `uv run dvc checkout data/08_reporting` (and any upstream outputs it requires).
Expected: the workspace returns to baseline-config predictions, leaving the `packaged_ablation_c1_conf/` snapshots untouched because they are outside any DVC stage's `outs`.

- [ ] **Step 4: Commit a note (snapshots themselves go through DVC tracking)**

Since the snapshots live under `data/08_reporting/`, they fall under the DVC `.gitignore`. Add a README instead:

```bash
uv run python -c "
from pathlib import Path
from textwrap import dedent

note = Path('data/08_reporting/packaged_ablations_README.md')
note.write_text(dedent('''
    # Packaged-eval ablation snapshots

    Directories named ``packaged_ablation_<experiment>_<variant>/`` hold
    copies of ``predictions.json`` and ``metrics.json`` captured from
    DVC experiments that override ``package.*`` params. They are NOT
    tracked by DVC stages; they are reference data for the Track C
    analysis reports in this directory.

    Experiment naming:

    * ``c1_conf`` — ``package.infer.confidence_threshold`` aligned to
      training-label minimum (Task 9).
    * ``c2_len`` — ``package.infer_min_tube_length`` = 4 (Task 11).
''').lstrip())
print(note)
"
git add data/08_reporting/packaged_ablations_README.md
git commit -m "chore(bbox-tube-temporal): document Track C ablation snapshot layout"
```

---

## Task 11: Re-run the aggregation analysis on the C1 snapshot

**Files:**
- Modify: `scripts/analyze_aggregation_rules.py`

**Rationale:** The driver from Task 6 hard-codes `packaged/` as the predictions subdirectory. Track C snapshots live under `packaged_ablation_c*_<name>/`, so extend the driver with a `--packaged-subdir` flag (default `packaged` keeps existing behaviour) before re-running.

- [ ] **Step 1: Extend the driver with `--packaged-subdir`**

Edit `scripts/analyze_aggregation_rules.py`. Locate the argparse block and add an argument; locate the predictions path construction and use the new arg.

Before:
```python
    parser.add_argument("--reporting-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args()
```

After:
```python
    parser.add_argument("--reporting-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument(
        "--packaged-subdir",
        default="packaged",
        help="Sub-directory under {reporting-dir}/{split}/ holding the predictions.",
    )
    args = parser.parse_args()
```

Before:
```python
            predictions_path = (
                args.reporting_dir / split / "packaged" / variant / "predictions.json"
            )
```

After:
```python
            predictions_path = (
                args.reporting_dir / split / args.packaged_subdir / variant / "predictions.json"
            )
```

- [ ] **Step 3: Commit the driver change**

```bash
git add scripts/analyze_aggregation_rules.py
git commit -m "feat(bbox-tube-temporal): allow analyze_aggregation_rules to read alt packaged subdirs"
```

- [ ] **Step 4: Run the analysis on the C1 snapshot**

```bash
uv run python scripts/analyze_aggregation_rules.py \
    --reporting-dir data/08_reporting \
    --output data/08_reporting/aggregation_ablation_c1_conf.md \
    --target-recall 0.95 \
    --packaged-subdir packaged_ablation_c1_conf
```

Expected: 12-row markdown table written. Compare best val precision to Task 6's baseline report.

---

## Task 12: Run ablation C2 — `package.infer_min_tube_length = 4`

Repeat Task 10's template with the second param override.

- [ ] **Step 1: Re-run with the tube-length override**

```bash
uv run dvc exp run -S package.infer_min_tube_length=4 --no-commit evaluate_packaged
```

- [ ] **Step 2: Snapshot to `packaged_ablation_c2_len`**

```bash
uv run python -c "
import shutil
from pathlib import Path
base = Path('data/08_reporting')
for split in ('train', 'val'):
    for variant in ('gru_convnext_finetune', 'vit_dinov2_finetune'):
        src = base / split / 'packaged' / variant
        dst = base / split / 'packaged_ablation_c2_len' / variant
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f'{src} -> {dst}')
"
```

- [ ] **Step 3: Restore baseline**

Run: `uv run dvc checkout data/08_reporting`.

- [ ] **Step 4: Run the aggregation analysis on the C2 snapshot**

```bash
uv run python scripts/analyze_aggregation_rules.py \
    --reporting-dir data/08_reporting \
    --output data/08_reporting/aggregation_ablation_c2_len.md \
    --target-recall 0.95 \
    --packaged-subdir packaged_ablation_c2_len
```

---

## Task 13: Consolidate findings into a final Track-C report

**Files:**
- Create: `data/08_reporting/aggregation_ablation_track_c.md`

- [ ] **Step 1: Aggregate the three reports**

Write `data/08_reporting/aggregation_ablation_track_c.md` manually. It should contain:

1. A headline sentence: "Best val-packaged precision at recall ≥ 0.95 across all configs: <value> under <rule> × <config>."
2. A 3-column comparison table (baseline vs. C1 conf-aligned vs. C2 length-aligned), pulling the best val row for each variant from each of the three reports.
3. A verdict against the spec's success criterion: precision ≥ 0.93 at recall ≥ 0.95 on val-packaged for at least one variant without train-packaged precision dropping below 0.90.
4. Recommendation: ship / continue to Track B / continue to Track D.

There is no code to test here — this is a human-authored synthesis of the numeric outputs from Tasks 6, 11, and 12 plus the qualitative notes from Task 8.

- [ ] **Step 2: Commit the final report**

```bash
git add data/08_reporting/aggregation_ablation_track_c.md
git commit -m "docs(bbox-tube-temporal): Track A/C precision investigation findings"
```

- [ ] **Step 3: Decide next step**

- If the report shows a variant × rule × config meets the spec target: investigation closed. Open a follow-up PR that promotes the chosen config to `params.yaml` (in a separate change so this branch remains an investigation artifact).
- If the report shows the target is missed: write a new spec for Track B (longest-tube-only inference) and/or Track D (YOLO-version alignment + retrain), referencing this report as the motivation.
