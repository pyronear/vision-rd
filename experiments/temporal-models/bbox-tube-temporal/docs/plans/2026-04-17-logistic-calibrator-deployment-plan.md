# Logistic Calibrator Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the already-validated multivariate logistic calibrator into `BboxTubeTemporalModel.predict()` so `vit_dinov2_finetune`'s packaged inference reaches val P=0.974 / R=0.950 / F1=0.962, while renaming the concept from "Platt" to "logistic calibrator" across forward-looking code and specs.

**Architecture:** Two-module split (`logistic_calibrator.py` pure-numpy runtime + `logistic_calibrator_fit.py` sklearn-based fitter). `scripts/package_model.py` gains an `aggregation: "logistic"` branch that runs the full inference pipeline on raw `train` and `val` data, fits the calibrator, calibrates a probability threshold via the existing `calibrate_threshold()`, and embeds `logistic_calibrator.json` in `package.zip`. Runtime loads the JSON and branches in `pick_winner_and_trigger`.

**Tech Stack:** Python 3.11+, uv, pytorch, ultralytics YOLO, scikit-learn (package-time only), pytest, DVC, ruff. Working directory for all commands: `experiments/temporal-models/bbox-tube-temporal/`.

**Spec:** `docs/specs/2026-04-17-logistic-calibrator-deployment-design.md`

---

## File Map

**New files:**
- `src/bbox_tube_temporal/logistic_calibrator.py` — dataclass, JSON I/O, `predict_proba`, `predict_proba_batch`, `extract_features`, `verify_sanity_checks`. No sklearn.
- `src/bbox_tube_temporal/logistic_calibrator_fit.py` — `fit(records) -> LogisticCalibrator`. sklearn at module top.
- `src/bbox_tube_temporal/package_predict.py` — `collect_pipeline_records(yolo_model, classifier, config, raw_dir, device) -> list[dict]` for package-time full-pipeline inference.
- `tests/test_logistic_calibrator.py` — runtime module tests.
- `tests/test_logistic_calibrator_fit.py` — fitter tests (parity check).
- `tests/test_package_predict.py` — package-time inference helper test.

**Modified files:**
- `src/bbox_tube_temporal/package.py` — manifest schema, `ModelPackage.calibrator`, `load_model_package`, `build_model_package`.
- `src/bbox_tube_temporal/inference.py` — `pick_winner_and_trigger` decision branch.
- `src/bbox_tube_temporal/model.py` — thread calibrator through `predict`.
- `scripts/package_model.py` — `aggregation: "logistic"` branch.
- `scripts/analyze_variant.py` — consume shared fitter + feature extractor, rename `platt` → `logistic`, rename output file.
- `params.yaml` — `aggregation: "max_logit"` under `train_gru_convnext_finetune`, `aggregation: "logistic"` under `train_vit_dinov2_finetune`.
- `dvc.yaml` — `package` stage: add `aggregation` param, add raw-datasets + new module deps. `analyze_variant` stage: rename `platt_model.json` → `logistic_calibrator.json`.
- `docs/specs/2026-04-17-automated-variant-analysis.md` — rewrite Platt → logistic with renaming note.
- `data/08_reporting/precision_investigation/summary.md` — append single-line footnote.
- `tests/test_package.py` — extend with calibrator load/parse tests.

---

### Task 1: Create LogisticCalibrator dataclass + JSON round-trip

**Files:**
- Create: `src/bbox_tube_temporal/logistic_calibrator.py`
- Create: `tests/test_logistic_calibrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logistic_calibrator.py
"""Tests for the pure-numpy runtime logistic calibrator module."""

import json
from pathlib import Path

import numpy as np

from bbox_tube_temporal.logistic_calibrator import LogisticCalibrator


def test_json_roundtrip_preserves_weights(tmp_path: Path) -> None:
    cal = LogisticCalibrator(
        features=["logit", "log_len", "mean_conf", "n_tubes"],
        coefficients=np.array([0.69138, 1.672666, 2.449859, -0.01339]),
        intercept=-6.172852,
        sanity_checks=[],
    )
    out = tmp_path / "calibrator.json"
    cal.to_json(out)
    loaded = LogisticCalibrator.from_json(out)

    assert loaded.features == cal.features
    assert np.allclose(loaded.coefficients, cal.coefficients, atol=1e-12)
    assert loaded.intercept == cal.intercept
    assert loaded.sanity_checks == cal.sanity_checks


def test_json_file_shape_matches_contract(tmp_path: Path) -> None:
    cal = LogisticCalibrator(
        features=["logit", "log_len", "mean_conf", "n_tubes"],
        coefficients=np.array([0.1, 0.2, 0.3, 0.4]),
        intercept=-1.0,
        sanity_checks=[{"features": [1.0, 2.0, 3.0, 4.0], "prob": 0.5}],
    )
    out = tmp_path / "cal.json"
    cal.to_json(out)
    payload = json.loads(out.read_text())

    assert set(payload.keys()) == {
        "features", "coefficients", "intercept", "sanity_checks"
    }
    assert payload["features"] == ["logit", "log_len", "mean_conf", "n_tubes"]
    assert payload["coefficients"] == [0.1, 0.2, 0.3, 0.4]
    assert payload["intercept"] == -1.0
    assert payload["sanity_checks"] == [
        {"features": [1.0, 2.0, 3.0, 4.0], "prob": 0.5}
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_logistic_calibrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bbox_tube_temporal.logistic_calibrator'`

- [ ] **Step 3: Implement the runtime module**

```python
# src/bbox_tube_temporal/logistic_calibrator.py
"""Pure-numpy runtime logistic calibrator.

Loads serialized weights fitted by ``logistic_calibrator_fit.fit`` and
applies the multivariate logistic regression at inference time. No
sklearn import: runtime keeps a small dep surface and avoids pickle
version drift.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LogisticCalibrator:
    """Fitted multivariate logistic regression for sequence-level decisions.

    Args:
        features: Ordered list of feature names (must match the order of
            ``coefficients`` and the array returned by
            :func:`extract_features`).
        coefficients: 1-D array, shape ``(len(features),)``.
        intercept: Scalar bias.
        sanity_checks: A list of ``{"features": [...], "prob": float}``
            dicts captured at fit time; used to detect
            serialization/version drift at load time.
    """

    features: list[str]
    coefficients: np.ndarray
    intercept: float
    sanity_checks: list[dict] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        payload = {
            "features": list(self.features),
            "coefficients": [float(c) for c in self.coefficients],
            "intercept": float(self.intercept),
            "sanity_checks": list(self.sanity_checks),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "LogisticCalibrator":
        data = json.loads(path.read_text())
        return cls(
            features=list(data["features"]),
            coefficients=np.asarray(data["coefficients"], dtype=float),
            intercept=float(data["intercept"]),
            sanity_checks=list(data.get("sanity_checks", [])),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_logistic_calibrator.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/logistic_calibrator.py tests/test_logistic_calibrator.py
git commit -m "feat(bbox-tube-temporal): add LogisticCalibrator dataclass with JSON I/O"
```

---

### Task 2: predict_proba + predict_proba_batch

**Files:**
- Modify: `src/bbox_tube_temporal/logistic_calibrator.py`
- Modify: `tests/test_logistic_calibrator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_logistic_calibrator.py`:

```python
def test_predict_proba_matches_textbook_sigmoid() -> None:
    # Known weights: z = 1*2 + 2*3 + 0 = 8; sigmoid(8) ~= 0.99966
    cal = LogisticCalibrator(
        features=["a", "b"],
        coefficients=np.array([1.0, 2.0]),
        intercept=0.0,
        sanity_checks=[],
    )
    prob = cal.predict_proba(np.array([2.0, 3.0]))
    assert abs(prob - 1.0 / (1.0 + np.exp(-8.0))) < 1e-12


def test_predict_proba_batch_matches_single_rows() -> None:
    cal = LogisticCalibrator(
        features=["a", "b"],
        coefficients=np.array([0.5, -0.25]),
        intercept=0.1,
        sanity_checks=[],
    )
    X = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 0.0]])
    batch = cal.predict_proba_batch(X)
    singles = np.array([cal.predict_proba(row) for row in X])
    assert np.allclose(batch, singles, atol=1e-12)
    assert batch.shape == (3,)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_logistic_calibrator.py -v`
Expected: FAIL — `AttributeError: 'LogisticCalibrator' object has no attribute 'predict_proba'`

- [ ] **Step 3: Implement both methods**

Append inside the `LogisticCalibrator` dataclass in `src/bbox_tube_temporal/logistic_calibrator.py`:

```python
    def predict_proba(self, features_row: np.ndarray) -> float:
        """Probability of the positive class for one feature row.

        Args:
            features_row: 1-D array, shape ``(len(self.features),)``.
        """
        z = float(features_row @ self.coefficients) + self.intercept
        return 1.0 / (1.0 + math.exp(-z))

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`predict_proba`.

        Args:
            X: 2-D array, shape ``(n_rows, len(self.features))``.

        Returns:
            1-D array of probabilities, shape ``(n_rows,)``.
        """
        z = X @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-z))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_logistic_calibrator.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/logistic_calibrator.py tests/test_logistic_calibrator.py
git commit -m "feat(bbox-tube-temporal): add predict_proba + batch variant to LogisticCalibrator"
```

---

### Task 3: extract_features helper

**Files:**
- Modify: `src/bbox_tube_temporal/logistic_calibrator.py`
- Modify: `tests/test_logistic_calibrator.py`

**Context:** Must match `scripts/analyze_variant.py` exactly so the runtime features are identical to the fit-time features. Confirmed field names from `scripts/analyze_variant.py:36-47`: tube length is `end_frame - start_frame + 1`, confidence field is `entry["confidence"]` (may be `None`; ignored if so; yields 0.0 if every entry is `None`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_logistic_calibrator.py`:

```python
from bbox_tube_temporal.logistic_calibrator import (
    LogisticCalibrator,
    extract_features,
)


def test_extract_features_matches_analyze_variant_convention() -> None:
    tube = {
        "logit": 2.5,
        "start_frame": 4,
        "end_frame": 12,  # length 9
        "entries": [
            {"confidence": 0.3},
            {"confidence": 0.5},
            {"confidence": None},  # gap entry, must be skipped
            {"confidence": 0.7},
        ],
    }
    feats = extract_features(tube, n_tubes=3)
    # length = 9, log1p(9), mean_conf = (0.3+0.5+0.7)/3 = 0.5
    np.testing.assert_allclose(
        feats,
        np.array([2.5, np.log1p(9), 0.5, 3.0]),
        atol=1e-12,
    )


def test_extract_features_all_none_confidence_defaults_to_zero() -> None:
    tube = {
        "logit": -1.0,
        "start_frame": 0,
        "end_frame": 0,  # length 1
        "entries": [
            {"confidence": None},
            {"confidence": None},
        ],
    }
    feats = extract_features(tube, n_tubes=1)
    np.testing.assert_allclose(
        feats,
        np.array([-1.0, np.log1p(1), 0.0, 1.0]),
        atol=1e-12,
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_logistic_calibrator.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_features'`

- [ ] **Step 3: Implement `extract_features`**

Append at module level (below the dataclass) in `src/bbox_tube_temporal/logistic_calibrator.py`:

```python
FEATURE_NAMES: list[str] = ["logit", "log_len", "mean_conf", "n_tubes"]


def _tube_len(tube: dict) -> int:
    return tube["end_frame"] - tube["start_frame"] + 1


def _tube_mean_conf(tube: dict) -> float:
    confs = [
        e["confidence"]
        for e in tube["entries"]
        if e["confidence"] is not None
    ]
    return sum(confs) / len(confs) if confs else 0.0


def extract_features(tube: dict, n_tubes: int) -> np.ndarray:
    """Build the feature row consumed by :class:`LogisticCalibrator`.

    Matches ``scripts/analyze_variant.py:tube_len`` /
    ``tube_mean_conf`` exactly — the order is
    ``["logit", "log_len", "mean_conf", "n_tubes"]``.
    """
    return np.array(
        [
            tube["logit"],
            math.log1p(_tube_len(tube)),
            _tube_mean_conf(tube),
            float(n_tubes),
        ],
        dtype=float,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_logistic_calibrator.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/logistic_calibrator.py tests/test_logistic_calibrator.py
git commit -m "feat(bbox-tube-temporal): add extract_features helper for logistic calibrator"
```

---

### Task 4: verify_sanity_checks

**Files:**
- Modify: `src/bbox_tube_temporal/logistic_calibrator.py`
- Modify: `tests/test_logistic_calibrator.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_logistic_calibrator.py`:

```python
import pytest


def _reference_sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def test_verify_sanity_checks_passes_on_correct_weights() -> None:
    cal = LogisticCalibrator(
        features=["a", "b"],
        coefficients=np.array([1.0, 2.0]),
        intercept=0.0,
        sanity_checks=[
            {"features": [2.0, 3.0], "prob": _reference_sigmoid(8.0)},
            {"features": [0.0, 0.0], "prob": 0.5},
        ],
    )
    cal.verify_sanity_checks()  # no raise


def test_verify_sanity_checks_raises_on_tampered_weights() -> None:
    cal = LogisticCalibrator(
        features=["a", "b"],
        coefficients=np.array([1.0, 2.0]),
        intercept=0.0,
        sanity_checks=[
            {"features": [2.0, 3.0], "prob": _reference_sigmoid(8.0)},
        ],
    )
    tampered = LogisticCalibrator(
        features=cal.features,
        coefficients=cal.coefficients * 2.0,
        intercept=cal.intercept,
        sanity_checks=cal.sanity_checks,
    )
    with pytest.raises(ValueError, match="sanity check"):
        tampered.verify_sanity_checks()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_logistic_calibrator.py -v`
Expected: FAIL — `AttributeError: 'LogisticCalibrator' object has no attribute 'verify_sanity_checks'`

- [ ] **Step 3: Implement**

Append inside the `LogisticCalibrator` dataclass:

```python
    def verify_sanity_checks(self, atol: float = 1e-6) -> None:
        """Re-run each persisted (features, prob) pair; raise on mismatch.

        Guards against sklearn-version drift or JSON tampering: if the
        numpy inference path no longer reproduces the probabilities
        captured at fit time within ``atol``, we refuse to use the
        calibrator.
        """
        for i, pair in enumerate(self.sanity_checks):
            expected = float(pair["prob"])
            actual = self.predict_proba(np.asarray(pair["features"], dtype=float))
            if abs(actual - expected) > atol:
                raise ValueError(
                    f"logistic calibrator sanity check #{i} failed: "
                    f"expected prob={expected!r} got {actual!r} "
                    f"(features={pair['features']!r})"
                )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_logistic_calibrator.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/logistic_calibrator.py tests/test_logistic_calibrator.py
git commit -m "feat(bbox-tube-temporal): add sanity-check verification to LogisticCalibrator"
```

---

### Task 5: logistic_calibrator_fit.fit with sklearn parity check

**Files:**
- Create: `src/bbox_tube_temporal/logistic_calibrator_fit.py`
- Create: `tests/test_logistic_calibrator_fit.py`

**Context:** Matches `scripts/analyze_variant.py:fit_platt_model` (commit `a86f668`): feature order `[logit, log_len, mean_conf, n_tubes]`, `LogisticRegression(max_iter=1000, C=1.0)`, max-logit tube per sequence, empty-kept-tubes → `[0.0, 0.0, 0.0, 0.0]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logistic_calibrator_fit.py
"""Tests for the sklearn-based fitter used at package time."""

import math

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from bbox_tube_temporal.logistic_calibrator import (
    FEATURE_NAMES,
    LogisticCalibrator,
)
from bbox_tube_temporal.logistic_calibrator_fit import fit


def _record(*, label: str, logit: float, length: int, mean_conf: float) -> dict:
    """Synthetic predictions.json-style record with exactly one kept tube."""
    entries = [{"confidence": mean_conf} for _ in range(length)]
    return {
        "label": label,
        "kept_tubes": [
            {
                "logit": logit,
                "start_frame": 0,
                "end_frame": length - 1,
                "entries": entries,
            }
        ],
    }


def _separable_dataset(n_per_class: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for _ in range(n_per_class):
        records.append(
            _record(
                label="smoke",
                logit=float(rng.normal(3.0, 0.5)),
                length=int(rng.integers(10, 20)),
                mean_conf=float(rng.uniform(0.5, 0.9)),
            )
        )
        records.append(
            _record(
                label="fp",
                logit=float(rng.normal(-1.0, 0.5)),
                length=int(rng.integers(2, 6)),
                mean_conf=float(rng.uniform(0.1, 0.3)),
            )
        )
    return records


def test_fit_returns_logistic_calibrator_with_correct_shape() -> None:
    cal = fit(_separable_dataset(50))

    assert isinstance(cal, LogisticCalibrator)
    assert cal.features == FEATURE_NAMES
    assert cal.coefficients.shape == (4,)
    assert isinstance(cal.intercept, float)
    assert len(cal.sanity_checks) == 3


def test_fit_parity_between_sklearn_and_numpy() -> None:
    records = _separable_dataset(100)
    cal = fit(records)

    # Reconstruct the same feature matrix the fitter used, then compare
    # sklearn.predict_proba vs LogisticCalibrator.predict_proba_batch.
    X = []
    y = []
    for r in records:
        kept = r["kept_tubes"]
        if kept:
            t = max(kept, key=lambda t: t["logit"])
            entries = [e["confidence"] for e in t["entries"] if e["confidence"] is not None]
            X.append([
                t["logit"],
                math.log1p(t["end_frame"] - t["start_frame"] + 1),
                sum(entries) / len(entries) if entries else 0.0,
                len(kept),
            ])
        else:
            X.append([0.0, 0.0, 0.0, 0.0])
        y.append(1 if r["label"] == "smoke" else 0)
    X = np.array(X)

    sklearn_model = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
    sklearn_probs = sklearn_model.predict_proba(X)[:, 1]
    numpy_probs = cal.predict_proba_batch(X)

    assert np.allclose(sklearn_probs, numpy_probs, atol=1e-6)


def test_fit_sanity_checks_pass() -> None:
    cal = fit(_separable_dataset(30))
    cal.verify_sanity_checks()  # no raise


def test_fit_handles_empty_kept_tubes() -> None:
    records = _separable_dataset(20)
    records.append({"label": "fp", "kept_tubes": []})
    cal = fit(records)
    # Empty-kept-tubes row becomes [0, 0, 0, 0] → prob must be a finite number.
    prob = cal.predict_proba(np.array([0.0, 0.0, 0.0, 0.0]))
    assert 0.0 < prob < 1.0


def test_fit_rejects_single_class_dataset() -> None:
    records = [
        _record(label="smoke", logit=1.0, length=5, mean_conf=0.5)
        for _ in range(10)
    ]
    with pytest.raises(ValueError):
        fit(records)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_logistic_calibrator_fit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bbox_tube_temporal.logistic_calibrator_fit'`

- [ ] **Step 3: Implement the fitter**

```python
# src/bbox_tube_temporal/logistic_calibrator_fit.py
"""Package-time fitter for :class:`LogisticCalibrator`.

Imports sklearn at module top. Only the package stage and the
``analyze_variant`` research script import this module; the runtime
inference path stays sklearn-free.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression

from .logistic_calibrator import (
    FEATURE_NAMES,
    LogisticCalibrator,
    extract_features,
)

_EMPTY_FEATURES = np.zeros(len(FEATURE_NAMES), dtype=float)
_N_SANITY_CHECKS = 3
_PARITY_ATOL = 1e-6


def _features_for_record(record: dict) -> np.ndarray:
    kept = record["kept_tubes"]
    if not kept:
        return _EMPTY_FEATURES.copy()
    best = max(kept, key=lambda t: t["logit"])
    return extract_features(best, n_tubes=len(kept))


def fit(records: list[dict]) -> LogisticCalibrator:
    """Fit a :class:`LogisticCalibrator` on ``predictions.json``-style records.

    Each record must carry ``label`` (``"smoke"`` or ``"fp"``) and
    ``kept_tubes`` (a list of tubes with ``logit``, ``start_frame``,
    ``end_frame``, and ``entries[].confidence``). Empty ``kept_tubes``
    lists yield an all-zero feature row (matching
    ``scripts/analyze_variant.py`` behaviour).

    The returned calibrator carries three sanity-check ``(features, prob)``
    pairs and is parity-tested against sklearn before being returned;
    a ``RuntimeError`` is raised if the numpy implementation disagrees
    with sklearn on the training rows.

    Raises:
        ValueError: if the record set contains a single class (sklearn
            cannot fit).
        RuntimeError: if the numpy / sklearn parity check fails.
    """
    X = np.stack([_features_for_record(r) for r in records])
    y = np.array(
        [1 if r["label"] == "smoke" else 0 for r in records], dtype=int
    )

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X, y)

    coefficients = np.asarray(model.coef_[0], dtype=float)
    intercept = float(model.intercept_[0])

    cal = LogisticCalibrator(
        features=list(FEATURE_NAMES),
        coefficients=coefficients,
        intercept=intercept,
        sanity_checks=[],
    )

    sklearn_probs = model.predict_proba(X)[:, 1]
    numpy_probs = cal.predict_proba_batch(X)
    if not np.allclose(sklearn_probs, numpy_probs, atol=_PARITY_ATOL):
        max_diff = float(np.max(np.abs(sklearn_probs - numpy_probs)))
        raise RuntimeError(
            "sklearn / numpy logistic-regression parity failed "
            f"(max |Δp|={max_diff:.3e} > atol={_PARITY_ATOL:.3e})"
        )

    # Pick sanity-check rows spanning the prob range so drift anywhere
    # in the operating band gets caught at load time.
    order = np.argsort(sklearn_probs)
    pick = order[
        np.linspace(0, len(order) - 1, num=_N_SANITY_CHECKS, dtype=int)
    ]
    sanity_checks = [
        {
            "features": [float(v) for v in X[i]],
            "prob": float(sklearn_probs[i]),
        }
        for i in pick
    ]

    return LogisticCalibrator(
        features=cal.features,
        coefficients=cal.coefficients,
        intercept=cal.intercept,
        sanity_checks=sanity_checks,
    )


def fit_from_prediction_records(records: list[dict]) -> LogisticCalibrator:
    """Alias for :func:`fit` — kept as the public name for scripts."""
    return fit(records)


__all__ = ["fit", "fit_from_prediction_records"]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_logistic_calibrator_fit.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/logistic_calibrator_fit.py tests/test_logistic_calibrator_fit.py
git commit -m "feat(bbox-tube-temporal): add sklearn-based logistic calibrator fitter with parity check"
```

---

### Task 6: Migrate analyze_variant.py to the shared fitter + rename output

**Files:**
- Modify: `scripts/analyze_variant.py`
- Modify: `dvc.yaml` (rename `platt_model.json` → `logistic_calibrator.json` in `analyze_variant` outs)
- Delete old artifact (manual step inside this task): `data/08_reporting/variant_analysis/<variant>/platt_model.json` for both variants

**Context:** `scripts/analyze_variant.py` currently defines `fit_platt_model` (lines 166-189) and `evaluate_platt` (lines 192-234) plus inline tube helpers (`tube_len` at line 36, `tube_mean_conf` at line 40). We replace the fit with a call to `logistic_calibrator_fit.fit`, use `extract_features` from the runtime module in `evaluate_*`, rename the output file, and update the report section header. Keep `tube_len` / `tube_mean_conf` as module-local helpers (still used by other analysis rules inside this script).

- [ ] **Step 1: Rename helpers and update fitter call**

In `scripts/analyze_variant.py`:

1. Replace top-level `sklearn` import with runtime + fitter imports:

```python
# Replace:
# from sklearn.linear_model import LogisticRegression
from bbox_tube_temporal.logistic_calibrator import (
    LogisticCalibrator,
    extract_features,
)
from bbox_tube_temporal.logistic_calibrator_fit import fit as fit_logistic_calibrator
```

2. Replace `fit_platt_model` function (lines 166-189) with a thin wrapper that calls the shared fitter. Drop `tuple[..., list[str]]` return:

```python
def fit_logistic_calibrator_for_records(
    train_records: list[dict],
) -> LogisticCalibrator:
    return fit_logistic_calibrator(train_records)
```

3. Replace `evaluate_platt` (lines 192-234) to consume a `LogisticCalibrator`:

```python
def evaluate_calibrator(
    calibrator: LogisticCalibrator,
    records: list[dict],
    threshold: float,
) -> dict:
    tp = fp = fn = tn = 0
    for r in records:
        cls = r["label"]
        kept = r["kept_tubes"]
        if not kept:
            features = np.zeros(len(calibrator.features), dtype=float)
        else:
            best = max(kept, key=lambda t: t["logit"])
            features = extract_features(best, n_tubes=len(kept))
        prob = calibrator.predict_proba(features)
        fires = prob >= threshold
        if fires:
            if cls == "smoke":
                tp += 1
            else:
                fp += 1
        else:
            if cls == "smoke":
                fn += 1
            else:
                tn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r_ = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r_, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
```

4. In `build_report` (grep for the Platt section — current header
   `## 6. Platt re-calibration (fit on train)`), rename the header to
   `## 6. Logistic calibration (fit on train)`, rename the weight-print
   line, and update calls: `fit_platt_model` → `fit_logistic_calibrator_for_records`,
   `evaluate_platt` → `evaluate_calibrator`. Weights are now accessed
   as `cal.coefficients[0]` etc. and `cal.intercept`.

5. In the `main()` JSON-writing block (search for `platt_model.json`):
   change the output filename and use the calibrator's own serialization:

```python
# Replace the manual json.dump({features, coefficients, intercept, ...})
calibrator.to_json(output_dir / "logistic_calibrator.json")
```

6. Rename the loop config label `"platt thr=<x>"` → `"logistic thr=<x>"`
   in the report rows.

- [ ] **Step 2: Update dvc.yaml**

In `dvc.yaml`, inside the `analyze_variant` `outs:` block, rename:

```yaml
      outs:
        - data/08_reporting/variant_analysis/${item}/analysis_report.md:
            cache: false
        - data/08_reporting/variant_analysis/${item}/recommended_config.yaml:
            cache: false
        - data/08_reporting/variant_analysis/${item}/logistic_calibrator.json:
            cache: false
```

(The third entry is the renamed path.)

- [ ] **Step 3: Remove stale artifacts**

```bash
rm -f data/08_reporting/variant_analysis/gru_convnext_finetune/platt_model.json
rm -f data/08_reporting/variant_analysis/vit_dinov2_finetune/platt_model.json
```

- [ ] **Step 4: Re-run analyze_variant to regenerate the renamed artifacts**

Run: `uv run dvc repro -f analyze_variant@gru_convnext_finetune analyze_variant@vit_dinov2_finetune`
Expected: both stages complete; `logistic_calibrator.json` exists under both variant dirs; coefficients match the committed `platt_model.json` values modulo sign conventions (same random seed in sklearn LogisticRegression default solver).

Verify: `cat data/08_reporting/variant_analysis/vit_dinov2_finetune/logistic_calibrator.json | python -c "import json,sys; d=json.load(sys.stdin); print(d['coefficients'], d['intercept'])"`
Expected: coefficients approximately `[0.691, 1.673, 2.450, -0.013]`, intercept ≈ `-6.173`.

- [ ] **Step 5: Lint + tests**

Run: `uv run ruff check scripts/analyze_variant.py && uv run pytest tests/ -v`
Expected: no lint errors; all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_variant.py dvc.yaml dvc.lock \
    data/08_reporting/variant_analysis/gru_convnext_finetune/logistic_calibrator.json \
    data/08_reporting/variant_analysis/vit_dinov2_finetune/logistic_calibrator.json \
    data/08_reporting/variant_analysis/gru_convnext_finetune/analysis_report.md \
    data/08_reporting/variant_analysis/vit_dinov2_finetune/analysis_report.md
git rm -f data/08_reporting/variant_analysis/gru_convnext_finetune/platt_model.json || true
git rm -f data/08_reporting/variant_analysis/vit_dinov2_finetune/platt_model.json || true
git commit -m "refactor(bbox-tube-temporal): migrate analyze_variant to shared logistic calibrator"
```

(If either `git rm` fails because the file was already untracked, proceed — the previous `rm -f` removed it from the working tree.)

---

### Task 7: Extend package.py (manifest + loader + ModelPackage.calibrator)

**Files:**
- Modify: `src/bbox_tube_temporal/package.py`
- Modify: `tests/test_package.py` (extend existing tests)

**Context:** `src/bbox_tube_temporal/package.py` owns the zip schema. Today the manifest has four keys (`format_version`, `variant`, `yolo_weights`, `classifier_checkpoint`, `config`). We add an optional `logistic_calibrator:` key pointing at `logistic_calibrator.json` inside the zip, and a corresponding `ModelPackage.calibrator` field.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_package.py`:

```python
# Near the other imports at the top of the file:
import numpy as np

from bbox_tube_temporal.logistic_calibrator import LogisticCalibrator
from bbox_tube_temporal.package import (
    build_model_package,
    load_model_package,
)


def test_build_package_without_calibrator_has_no_calibrator_entry(
    tmp_path, fake_yolo_weights, fake_classifier_ckpt, simple_config
):
    out = tmp_path / "m.zip"
    build_model_package(
        yolo_weights_path=fake_yolo_weights,
        classifier_ckpt_path=fake_classifier_ckpt,
        config=simple_config,
        variant="test",
        output_path=out,
    )
    pkg = load_model_package(out, extract_dir=tmp_path / "extract")
    assert pkg.calibrator is None


def test_build_package_with_calibrator_round_trips(
    tmp_path, fake_yolo_weights, fake_classifier_ckpt, simple_config
):
    cal = LogisticCalibrator(
        features=["logit", "log_len", "mean_conf", "n_tubes"],
        coefficients=np.array([0.5, 1.5, 2.5, 0.0]),
        intercept=-3.0,
        sanity_checks=[
            {"features": [1.0, 2.0, 0.5, 2.0], "prob": 0.7310585786300049},
        ],
    )
    # Compute the sanity-check prob precisely so verify_sanity_checks passes:
    z = float(np.array([1.0, 2.0, 0.5, 2.0]) @ cal.coefficients) + cal.intercept
    cal = LogisticCalibrator(
        features=cal.features,
        coefficients=cal.coefficients,
        intercept=cal.intercept,
        sanity_checks=[
            {"features": [1.0, 2.0, 0.5, 2.0], "prob": 1.0 / (1.0 + np.exp(-z))},
        ],
    )

    out = tmp_path / "m.zip"
    build_model_package(
        yolo_weights_path=fake_yolo_weights,
        classifier_ckpt_path=fake_classifier_ckpt,
        config=simple_config,
        variant="test",
        output_path=out,
        calibrator=cal,
    )
    pkg = load_model_package(out, extract_dir=tmp_path / "extract")
    assert pkg.calibrator is not None
    assert pkg.calibrator.features == cal.features
    np.testing.assert_allclose(pkg.calibrator.coefficients, cal.coefficients)
    assert pkg.calibrator.intercept == cal.intercept
    pkg.calibrator.verify_sanity_checks()  # no raise
```

If fixtures `fake_yolo_weights`, `fake_classifier_ckpt`, `simple_config` don't exist already in `tests/test_package.py`, add minimal ones (see existing `tests/test_package.py` for any reusable fixtures) — use empty bytes for the .pt files (the loader won't exercise YOLO/classifier in these tests; only the zip-layout/config-round-trip code paths).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_package.py -v -k calibrator`
Expected: FAIL — `TypeError: build_model_package() got an unexpected keyword argument 'calibrator'` (or similar).

- [ ] **Step 3: Extend package.py**

In `src/bbox_tube_temporal/package.py`:

1. Add constant below existing filename constants:

```python
LOGISTIC_CALIBRATOR_FILENAME = "logistic_calibrator.json"
```

2. Add import at top:

```python
from .logistic_calibrator import LogisticCalibrator
```

3. Extend `ModelPackage` dataclass with `calibrator` field (default `None`):

```python
@dataclass
class ModelPackage:
    classifier: Any
    yolo_model: Any
    config: dict[str, Any]
    calibrator: LogisticCalibrator | None = None

    # ... (existing properties unchanged)
```

4. Extend `build_model_package` signature and body:

```python
def build_model_package(
    *,
    yolo_weights_path: Path,
    classifier_ckpt_path: Path,
    config: dict[str, Any],
    variant: str,
    output_path: Path,
    calibrator: LogisticCalibrator | None = None,
) -> Path:
    # ... existing validation ...
    manifest = {
        "format_version": FORMAT_VERSION,
        "variant": variant,
        "yolo_weights": YOLO_WEIGHTS_FILENAME,
        "classifier_checkpoint": CLASSIFIER_CKPT_FILENAME,
        "config": CONFIG_FILENAME,
    }
    if calibrator is not None:
        manifest["logistic_calibrator"] = LOGISTIC_CALIBRATOR_FILENAME

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest, default_flow_style=False))
        zf.write(yolo_weights_path, YOLO_WEIGHTS_FILENAME)
        zf.write(classifier_ckpt_path, CLASSIFIER_CKPT_FILENAME)
        zf.writestr(CONFIG_FILENAME, yaml.dump(config, default_flow_style=False))
        if calibrator is not None:
            payload = {
                "features": list(calibrator.features),
                "coefficients": [float(c) for c in calibrator.coefficients],
                "intercept": float(calibrator.intercept),
                "sanity_checks": list(calibrator.sanity_checks),
            }
            zf.writestr(
                LOGISTIC_CALIBRATOR_FILENAME,
                json.dumps(payload, indent=2),
            )
    return output_path.resolve()
```

(Add `import json` at top if not already present.)

5. Extend `load_model_package` to parse the calibrator and verify it:

```python
def load_model_package(
    package_path: Path,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
) -> ModelPackage:
    if not package_path.exists():
        raise FileNotFoundError(f"Archive not found: {package_path}")

    with zipfile.ZipFile(package_path, "r") as zf:
        names = zf.namelist()
        # ... existing manifest / config extraction unchanged ...

        calibrator: LogisticCalibrator | None = None
        calibrator_name = manifest.get("logistic_calibrator")
        if calibrator_name is not None:
            if calibrator_name not in names:
                raise KeyError(f"Archive missing {calibrator_name}")
            payload = json.loads(zf.read(calibrator_name))
            calibrator = LogisticCalibrator(
                features=list(payload["features"]),
                coefficients=np.asarray(
                    payload["coefficients"], dtype=float
                ),
                intercept=float(payload["intercept"]),
                sanity_checks=list(payload.get("sanity_checks", [])),
            )
            calibrator.verify_sanity_checks()

    yolo_model = _load_yolo(extract_dir / yolo_name)
    classifier = _load_classifier(extract_dir / ckpt_name, config["classifier"])
    return ModelPackage(
        classifier=classifier,
        yolo_model=yolo_model,
        config=config,
        calibrator=calibrator,
    )
```

(Add `import numpy as np` at top if not already present.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_package.py -v`
Expected: PASS for all previously-passing tests + the two new ones.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/package.py tests/test_package.py
git commit -m "feat(bbox-tube-temporal): bundle logistic calibrator into model package"
```

---

### Task 8: package_predict.collect_pipeline_records

**Files:**
- Create: `src/bbox_tube_temporal/package_predict.py`
- Create: `tests/test_package_predict.py`

**Context:** `BboxTubeTemporalModel.__init__` (model.py:54-65) already accepts `(yolo_model, classifier, config, device)` as components — we construct one in-memory using the config we're about to write into `package.zip`. Its `predict()` already produces a `details["kept_tubes"]` payload of the exact shape `analyze_variant` consumes. The helper wraps that: iterate sequences in `raw_dir`, run `predict`, collect `{"label", "kept_tubes"}` tuples.

Raw dataset layout (from `evaluate_packaged` deps): `raw_dir/{smoke,fp}/<sequence_name>/*.jpg`. A helper already exists to load these — re-use it rather than re-rolling.

- [ ] **Step 1: Locate the existing sequence loader**

Run: `uv run grep -Rn "def load_sequence\|def iter_sequences\|def load_sequences" src/bbox_tube_temporal scripts`
Expected: identify the function `scripts/evaluate_packaged.py` uses to iterate raw dataset sequences and convert them to `list[Frame]`. Re-use it directly (import path determined by the grep output — most likely `bbox_tube_temporal.data` based on the `evaluate_packaged` deps).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_package_predict.py
"""Tests for the package-time full-pipeline inference helper."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from bbox_tube_temporal.package_predict import collect_pipeline_records


class _FakeOutput:
    def __init__(self, is_positive: bool, kept_tubes: list[dict]) -> None:
        self.is_positive = is_positive
        self.trigger_frame_index = None
        self.details = {"kept_tubes": kept_tubes}


class _FakeModel:
    """Stands in for BboxTubeTemporalModel; returns canned kept_tubes."""

    def __init__(self, per_seq_tubes: dict[str, list[dict]]) -> None:
        self._per_seq_tubes = per_seq_tubes
        self.calls: list[str] = []

    def predict(self, frames, _seq_name_inject: str = "") -> _FakeOutput:
        # The real model ignores sequence names; we reach them via the
        # loader (see collect_pipeline_records internals). For this
        # test we just pop per-call tubes from a list.
        name = self._per_seq_tubes_keys.pop(0)  # see setUp
        self.calls.append(name)
        return _FakeOutput(
            is_positive=False,
            kept_tubes=self._per_seq_tubes[name],
        )


def test_collect_pipeline_records_produces_expected_schema(
    tmp_path: Path, monkeypatch
):
    # Stub the sequence loader so we don't need real jpgs on disk.
    # (Replace "bbox_tube_temporal.<loader_module>" with the module
    # identified in Step 1.)
    from bbox_tube_temporal import package_predict

    fake_sequences = [
        ("smoke", "seq_1", [MagicMock()]),
        ("smoke", "seq_2", [MagicMock()]),
        ("fp", "seq_3", [MagicMock()]),
    ]
    monkeypatch.setattr(
        package_predict,
        "_iter_labelled_sequences",
        lambda raw_dir: iter(fake_sequences),
    )

    fake_model = MagicMock()
    fake_model.predict.side_effect = [
        _FakeOutput(False, [{"logit": 1.5, "start_frame": 0, "end_frame": 4, "entries": []}]),
        _FakeOutput(False, []),
        _FakeOutput(False, [{"logit": -0.2, "start_frame": 0, "end_frame": 2, "entries": []}]),
    ]

    records = collect_pipeline_records(model=fake_model, raw_dir=tmp_path)

    assert [r["label"] for r in records] == ["smoke", "smoke", "fp"]
    assert records[0]["kept_tubes"][0]["logit"] == 1.5
    assert records[1]["kept_tubes"] == []
    assert records[2]["kept_tubes"][0]["logit"] == -0.2
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_package_predict.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bbox_tube_temporal.package_predict'`.

- [ ] **Step 4: Implement the helper**

```python
# src/bbox_tube_temporal/package_predict.py
"""Run the full YOLO + tracking + classifier pipeline at package time.

Used by ``scripts/package_model.py`` to produce the training data for
``logistic_calibrator_fit.fit`` and the val data for probability-threshold
calibration. Bypasses the ``.zip`` entirely — we already have the YOLO
model, the classifier, and the config in memory at packaging time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from pyrocore import Frame

from .data import iter_labelled_sequences  # adjust import if Step 1 found a different path


def _iter_labelled_sequences(
    raw_dir: Path,
) -> Iterator[tuple[str, str, list[Frame]]]:
    """Yield ``(label, sequence_name, frames)`` for every sequence under
    ``raw_dir/{smoke,fp}/*``.

    Thin wrapper so tests can monkey-patch one symbol.
    """
    yield from iter_labelled_sequences(raw_dir)


def collect_pipeline_records(
    *,
    model: Any,
    raw_dir: Path,
) -> list[dict]:
    """Run ``model.predict`` on every labelled sequence under ``raw_dir``.

    Returns a list of ``{"label", "sequence", "kept_tubes"}`` dicts with
    the same per-tube schema (``logit``, ``start_frame``, ``end_frame``,
    ``entries``) that ``scripts/analyze_variant.py`` and
    :mod:`bbox_tube_temporal.logistic_calibrator` expect.

    Args:
        model: A ``BboxTubeTemporalModel`` (or duck-type compatible) with
            a ``.predict(frames) -> TemporalModelOutput`` method whose
            ``.details["kept_tubes"]`` carries the tube structure.
        raw_dir: ``data/01_raw/datasets/{train,val}/`` with
            ``smoke/<seq>/*.jpg`` and ``fp/<seq>/*.jpg`` sub-trees.
    """
    records: list[dict] = []
    for label, seq_name, frames in _iter_labelled_sequences(raw_dir):
        out = model.predict(frames)
        records.append(
            {
                "label": label,
                "sequence": seq_name,
                "kept_tubes": out.details.get("kept_tubes", []),
            }
        )
    return records
```

**If Step 1 did not find an existing `iter_labelled_sequences`:** reuse whatever `scripts/evaluate_packaged.py` uses, inlining the smallest possible adapter. The important invariant is: walk `raw_dir/{smoke,fp}/<sequence>/` in a deterministic order, yield the label, name, and the loaded frames list.

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_package_predict.py -v`
Expected: PASS.

Adjust the test's `monkeypatch` target if the real loader symbol name differs from `iter_labelled_sequences`.

- [ ] **Step 6: Commit**

```bash
git add src/bbox_tube_temporal/package_predict.py tests/test_package_predict.py
git commit -m "feat(bbox-tube-temporal): add package-time full-pipeline inference helper"
```

---

### Task 9: inference.pick_winner_and_trigger — aggregation branch

**Files:**
- Modify: `src/bbox_tube_temporal/inference.py`
- Modify: `tests/test_inference.py`

**Context:** `pick_winner_and_trigger` (inference.py:241-264) today takes `(tubes, logits, threshold)` and does `logits[winner] >= threshold`. We keep that signature *and* add two optional keyword-only parameters used only when the caller wants the logistic branch. Keeping both branches inside the same function preserves the single decision point.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_inference.py`:

```python
import numpy as np
import torch

from bbox_tube_temporal.inference import pick_winner_and_trigger
from bbox_tube_temporal.logistic_calibrator import LogisticCalibrator


def _make_tube(tube_id: int, *, start: int, end: int, entries_conf: list[float]):
    # Minimal tube stub: matches what inference.py reads.
    from bbox_tube_temporal.tubes import Tube, TubeEntry, Detection  # adapt to real types

    entries = [
        TubeEntry(
            frame_idx=start + i,
            detection=Detection(cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=c),
            is_gap=False,
        )
        for i, c in enumerate(entries_conf)
    ]
    return Tube(tube_id=tube_id, start_frame=start, end_frame=end, entries=entries)


def test_pick_winner_max_logit_unchanged(monkeypatch):
    tubes = [
        _make_tube(0, start=0, end=4, entries_conf=[0.3, 0.4, 0.5, 0.6, 0.7]),
        _make_tube(1, start=0, end=4, entries_conf=[0.1, 0.1, 0.1, 0.1, 0.1]),
    ]
    logits = torch.tensor([2.5, -1.0])

    is_positive, trigger, winner_id = pick_winner_and_trigger(
        tubes=tubes, logits=logits, threshold=1.0
    )
    assert is_positive is True
    assert winner_id == 0


def test_pick_winner_logistic_fires_when_prob_exceeds_threshold():
    tubes = [_make_tube(0, start=0, end=4, entries_conf=[0.5] * 5)]
    logits = torch.tensor([3.0])

    # Calibrator tuned so prob of the only tube is ~0.99.
    cal = LogisticCalibrator(
        features=["logit", "log_len", "mean_conf", "n_tubes"],
        coefficients=np.array([2.0, 0.0, 0.0, 0.0]),
        intercept=0.0,
        sanity_checks=[],
    )

    is_positive, trigger, winner_id = pick_winner_and_trigger(
        tubes=tubes,
        logits=logits,
        threshold=0.0,  # ignored
        aggregation="logistic",
        calibrator=cal,
        logistic_threshold=0.5,
    )
    assert is_positive is True
    assert winner_id == 0


def test_pick_winner_logistic_does_not_fire_below_threshold():
    tubes = [_make_tube(0, start=0, end=4, entries_conf=[0.5] * 5)]
    logits = torch.tensor([-3.0])

    cal = LogisticCalibrator(
        features=["logit", "log_len", "mean_conf", "n_tubes"],
        coefficients=np.array([2.0, 0.0, 0.0, 0.0]),
        intercept=0.0,
        sanity_checks=[],
    )

    is_positive, _, _ = pick_winner_and_trigger(
        tubes=tubes,
        logits=logits,
        threshold=0.0,
        aggregation="logistic",
        calibrator=cal,
        logistic_threshold=0.5,
    )
    assert is_positive is False


def test_pick_winner_logistic_requires_calibrator():
    tubes = [_make_tube(0, start=0, end=4, entries_conf=[0.5] * 5)]
    logits = torch.tensor([1.0])
    with pytest.raises(ValueError, match="calibrator"):
        pick_winner_and_trigger(
            tubes=tubes,
            logits=logits,
            threshold=0.0,
            aggregation="logistic",
            calibrator=None,
            logistic_threshold=0.5,
        )
```

**Note:** import paths for `Tube`, `TubeEntry`, `Detection` inside `_make_tube` must match the real types in `src/bbox_tube_temporal/tubes.py`. Adjust the import line if the class names differ (check `src/bbox_tube_temporal/tubes.py` before running the test).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_inference.py -v -k logistic`
Expected: FAIL — unrecognized kwargs `aggregation`, `calibrator`, `logistic_threshold`.

- [ ] **Step 3: Update `pick_winner_and_trigger`**

In `src/bbox_tube_temporal/inference.py`, replace the current `pick_winner_and_trigger` (lines 241-264) with:

```python
def pick_winner_and_trigger(
    *,
    tubes: list,  # list[Tube]
    logits: torch.Tensor,
    threshold: float,
    aggregation: str = "max_logit",
    calibrator: "LogisticCalibrator | None" = None,
    logistic_threshold: float = 0.5,
) -> tuple[bool, int | None, int | None]:
    """Pick the winner tube and decide whether to fire a trigger.

    Args:
        tubes: Kept tubes, aligned with ``logits``.
        logits: 1-D tensor of classifier logits, one per tube.
        threshold: Logit threshold; used only when
            ``aggregation == "max_logit"``.
        aggregation: ``"max_logit"`` (default) or ``"logistic"``.
        calibrator: Required when ``aggregation == "logistic"``; a fitted
            :class:`LogisticCalibrator`.
        logistic_threshold: Probability threshold; used only when
            ``aggregation == "logistic"``.

    Returns:
        ``(is_positive, trigger_frame_idx, winner_tube_id)``. When
        ``tubes`` is empty, returns ``(False, None, None)``.
    """
    if not tubes:
        return False, None, None

    winner_idx = int(torch.argmax(logits).item())
    winner = tubes[winner_idx]

    if aggregation == "max_logit":
        fires = bool(logits[winner_idx].item() >= threshold)
    elif aggregation == "logistic":
        if calibrator is None:
            raise ValueError(
                "aggregation='logistic' requires a fitted calibrator"
            )
        tube_dict = {
            "logit": float(logits[winner_idx].item()),
            "start_frame": winner.start_frame,
            "end_frame": winner.end_frame,
            "entries": [
                {
                    "confidence": (
                        e.detection.confidence if e.detection is not None else None
                    )
                }
                for e in winner.entries
            ],
        }
        features = extract_features(tube_dict, n_tubes=len(tubes))
        prob = calibrator.predict_proba(features)
        fires = bool(prob >= logistic_threshold)
    else:
        raise ValueError(f"unknown aggregation: {aggregation!r}")

    trigger = winner.end_frame if fires else None
    return fires, trigger, winner.tube_id
```

Add imports at top of `inference.py`:

```python
from .logistic_calibrator import LogisticCalibrator, extract_features
```

**Important:** The existing call site at `model.py:206-208` uses positional args. Keep compatibility by checking what its invocation looks like — if it passes positional `tubes, logits, threshold`, you'll need to update that call site to use keyword args (done in Task 10).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_inference.py -v`
Expected: PASS — all four new tests plus existing inference tests.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/inference.py tests/test_inference.py
git commit -m "feat(bbox-tube-temporal): add logistic aggregation branch to pick_winner_and_trigger"
```

---

### Task 10: Thread calibrator through BboxTubeTemporalModel.predict

**Files:**
- Modify: `src/bbox_tube_temporal/model.py`

**Context:** `BboxTubeTemporalModel.from_package` (model.py:71-84) today constructs the model from `ModelPackage.{yolo_model, classifier, config}` only. Now that `ModelPackage` carries an optional `calibrator`, we thread it through into `__init__` and into `pick_winner_and_trigger`.

- [ ] **Step 1: Extend __init__ and from_package**

In `src/bbox_tube_temporal/model.py`:

1. Update `__init__` signature (line 54-65):

```python
def __init__(
    self,
    *,
    yolo_model: Any,
    classifier: Any,
    config: dict[str, Any],
    device: str | torch.device | None = None,
    calibrator: LogisticCalibrator | None = None,
) -> None:
    self._yolo = yolo_model
    self._device = _select_device(device)
    self._classifier = classifier.to(self._device).eval()
    self._cfg = config
    self._calibrator = calibrator
```

2. Add import near existing ones:

```python
from .logistic_calibrator import LogisticCalibrator
```

3. Update `from_package` (model.py:71-84):

```python
@classmethod
def from_package(
    cls,
    package_path: Path,
    *,
    device: str | torch.device | None = None,
) -> Self:
    pkg: ModelPackage = load_model_package(package_path)
    return cls(
        yolo_model=pkg.yolo_model,
        classifier=pkg.classifier,
        config=pkg.config,
        device=device,
        calibrator=pkg.calibrator,
    )
```

4. Update the `pick_winner_and_trigger` call at model.py:206-208 to pass the new kwargs:

```python
dec = self._cfg["decision"]
aggregation = dec.get("aggregation", "max_logit")
is_positive, trigger, winner_id = pick_winner_and_trigger(
    tubes=kept,
    logits=logits,
    threshold=float(dec["threshold"]),
    aggregation=aggregation,
    calibrator=self._calibrator,
    logistic_threshold=float(dec.get("logistic_threshold", 0.5)),
)
```

5. Update the `"threshold"` entry in the `TemporalModelOutput.details` dict (currently `model.py:259`) so downstream consumers (notebooks, reports) can see which threshold actually fired:

```python
            "threshold": (
                float(dec["logistic_threshold"])
                if aggregation == "logistic"
                else float(dec["threshold"])
            ),
            "aggregation": aggregation,
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: PASS — including `tests/test_package.py` (round-trip) and `tests/test_inference.py` (new logistic branch tests).

- [ ] **Step 3: Commit**

```bash
git add src/bbox_tube_temporal/model.py
git commit -m "feat(bbox-tube-temporal): thread logistic calibrator through BboxTubeTemporalModel.predict"
```

---

### Task 11: Add `aggregation: "logistic"` branch to scripts/package_model.py

**Files:**
- Modify: `scripts/package_model.py`

**Context:** The packager today does: load classifier → `collect_val_probabilities` (single-tube val) → `calibrate_threshold` → write config. For the logistic branch we additionally: (a) build an in-memory `BboxTubeTemporalModel` from the classifier + YOLO + partial config, (b) run `collect_pipeline_records` on train to get fit data, (c) call `logistic_calibrator_fit.fit`, (d) run `collect_pipeline_records` on val to get threshold-calibration data, (e) call `calibrate_threshold` on the calibrated probs.

- [ ] **Step 1: Update `_build_config` to include logistic fields**

Replace `_build_config` in `scripts/package_model.py:68-96`:

```python
def _build_config(
    all_params: dict,
    variant_cfg: dict,
    package_params: dict,
    threshold: float,
    *,
    aggregation: str,
    logistic_threshold: float | None,
) -> dict:
    decision: dict = {
        "aggregation": aggregation,
        "threshold": float(threshold),
        "target_recall": package_params["target_recall"],
        "trigger_rule": "end_of_winner",
    }
    if logistic_threshold is not None:
        decision["logistic_threshold"] = float(logistic_threshold)

    return {
        "infer": package_params["infer"],
        "tubes": {
            "iou_threshold": all_params["tubes"]["iou_threshold"],
            "max_misses": all_params["tubes"]["max_misses"],
            "min_tube_length": all_params["build_tubes"]["min_tube_length"],
            "infer_min_tube_length": package_params["infer_min_tube_length"],
            "min_detected_entries": all_params["build_tubes"]["min_detected_entries"],
            "interpolate_gaps": True,
        },
        "model_input": {
            "context_factor": all_params["model_input"]["context_factor"],
            "patch_size": all_params["model_input"]["patch_size"],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "classifier": _classifier_kwargs(variant_cfg),
        "decision": decision,
    }
```

- [ ] **Step 2: Add argparse flags + raw-dataset paths**

Extend `main()` argparse block:

```python
parser.add_argument(
    "--raw-train-dir",
    type=Path,
    default=Path("data/01_raw/datasets/train"),
    help="Used only when variant aggregation is 'logistic'.",
)
parser.add_argument(
    "--raw-val-dir",
    type=Path,
    default=Path("data/01_raw/datasets/val"),
    help="Used only when variant aggregation is 'logistic'.",
)
```

- [ ] **Step 3: Add the logistic branch inside `main()`**

After the existing `threshold = calibrate_threshold(...)` line (around `package_model.py:146-148`), add:

```python
aggregation = variant_cfg.get("aggregation", "max_logit")

calibrator = None
logistic_threshold: float | None = None

if aggregation == "logistic":
    # Build a temporary config that BboxTubeTemporalModel needs in order
    # to run the full pipeline (same shape it will see inside the zip
    # once we write it). logistic_threshold is 0.5 as a placeholder —
    # we replace it immediately below with the calibrated value.
    tmp_config = _build_config(
        all_params,
        variant_cfg,
        package_params,
        threshold,
        aggregation="max_logit",  # pipeline path doesn't need calibrator yet
        logistic_threshold=None,
    )

    from ultralytics import YOLO  # noqa: PLC0415 — sanctioned inside main guard
    yolo_model = YOLO(str(args.yolo_weights_path))

    from bbox_tube_temporal.model import BboxTubeTemporalModel  # noqa: PLC0415

    fit_model = BboxTubeTemporalModel(
        yolo_model=yolo_model,
        classifier=classifier,
        config=tmp_config,
    )

    from bbox_tube_temporal.package_predict import collect_pipeline_records  # noqa: PLC0415
    from bbox_tube_temporal.logistic_calibrator_fit import fit as _fit_calibrator  # noqa: PLC0415

    train_records = collect_pipeline_records(
        model=fit_model, raw_dir=args.raw_train_dir
    )
    calibrator = _fit_calibrator(train_records)
    print(
        f"[package] logistic calibrator fit on {len(train_records)} train records; "
        f"coefs={calibrator.coefficients.tolist()} intercept={calibrator.intercept:.6f}"
    )

    # Now run on val to calibrate the logistic probability threshold.
    val_records = collect_pipeline_records(
        model=fit_model, raw_dir=args.raw_val_dir
    )
    probs = np.array([
        calibrator.predict_proba(
            extract_features(
                max(r["kept_tubes"], key=lambda t: t["logit"]),
                n_tubes=len(r["kept_tubes"]),
            )
        )
        if r["kept_tubes"]
        else 0.0
        for r in val_records
    ])
    labels = np.array(
        [1 if r["label"] == "smoke" else 0 for r in val_records]
    )
    logistic_threshold = calibrate_threshold(
        probs, labels, target_recall=package_params["target_recall"]
    )
```

**Memory rule reminder:** the project forbids function-level imports. The four `noqa: PLC0415` imports above violate it for a pragmatic reason — `ultralytics`, `BboxTubeTemporalModel`, `package_predict`, and `logistic_calibrator_fit` are heavy imports only needed on the logistic branch, and the existing codebase already sanctions this exact pattern for `ultralytics` in `package.py:113`. If the user memory says no exceptions, hoist all four to module top and accept the extra import cost on every `package_model.py` invocation.

**Simpler alternative that respects the rule strictly:** hoist these imports to module top — `numpy as np`, `ultralytics.YOLO`, `bbox_tube_temporal.model.BboxTubeTemporalModel`, `bbox_tube_temporal.package_predict.collect_pipeline_records`, `bbox_tube_temporal.logistic_calibrator_fit.fit as _fit_calibrator`, `bbox_tube_temporal.logistic_calibrator.extract_features`. Added import cost for the `max_logit` path: ~1-2s. Acceptable. Remove the `noqa: PLC0415` comments accordingly.

**This plan's recommendation: hoist.** The user's memory (`feedback_no_function_imports.md`) is explicit and the ultralytics exception is specifically documented for `package.py:_load_yolo`, not as a general license. Use the hoisted form.

- [ ] **Step 4: Update the `_build_config` call**

Replace the existing call at the bottom of `main()`:

```python
config = _build_config(
    all_params,
    variant_cfg,
    package_params,
    threshold,
    aggregation=aggregation,
    logistic_threshold=logistic_threshold,
)
build_model_package(
    yolo_weights_path=args.yolo_weights_path,
    classifier_ckpt_path=checkpoint,
    config=config,
    variant=args.variant,
    output_path=args.output,
    calibrator=calibrator,
)
```

Update the final print to report the logistic threshold too:

```python
suffix = ""
if aggregation == "logistic" and logistic_threshold is not None:
    suffix = f" logistic_threshold={logistic_threshold:.4f}"
print(
    f"[package] wrote {args.output} | variant={args.variant} "
    f"aggregation={aggregation} threshold={threshold:.4f} "
    f"target_recall={package_params['target_recall']}{suffix}"
)
```

- [ ] **Step 5: Lint + tests**

Run: `uv run ruff check scripts/package_model.py && uv run pytest tests/ -v`
Expected: clean lint, all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/package_model.py
git commit -m "feat(bbox-tube-temporal): fit + embed logistic calibrator in package stage"
```

---

### Task 12: params.yaml + dvc.yaml

**Files:**
- Modify: `params.yaml`
- Modify: `dvc.yaml`

- [ ] **Step 1: Add aggregation knob to params.yaml**

Under the `train_gru_convnext_finetune:` block in `params.yaml`, add (alongside existing fields):

```yaml
  aggregation: "max_logit"
```

Under `train_vit_dinov2_finetune:`:

```yaml
  aggregation: "logistic"
```

- [ ] **Step 2: Update `package` stage in dvc.yaml**

In `dvc.yaml` (lines ~737-761), extend the `package:` foreach stage:

```yaml
  package:
    foreach:
      - gru_convnext_finetune
      - vit_dinov2_finetune
    do:
      cmd: >-
        uv run python scripts/package_model.py
        --variant ${item}
        --output data/06_models/${item}/model.zip
      deps:
        - data/06_models/${item}/best_checkpoint.pt
        - data/01_raw/models/best.pt
        - data/05_model_input/val
        - data/01_raw/datasets/train
        - data/01_raw/datasets/val
        - scripts/package_model.py
        - src/bbox_tube_temporal/package.py
        - src/bbox_tube_temporal/calibration.py
        - src/bbox_tube_temporal/val_predict.py
        - src/bbox_tube_temporal/package_predict.py
        - src/bbox_tube_temporal/logistic_calibrator.py
        - src/bbox_tube_temporal/logistic_calibrator_fit.py
        - src/bbox_tube_temporal/inference.py
        - src/bbox_tube_temporal/model.py
      params:
        - package
        - tubes
        - build_tubes
        - model_input
        - train_${item}
      outs:
        # (unchanged — model.zip)
```

(Keep the existing `outs:` block as-is.)

- [ ] **Step 3: Sanity-check DVC graph**

Run: `uv run dvc status` and `uv run dvc dag package@vit_dinov2_finetune`
Expected: DVC recognizes the new deps; no cycle warnings.

- [ ] **Step 4: Commit**

```bash
git add params.yaml dvc.yaml
git commit -m "chore(bbox-tube-temporal): wire per-variant aggregation knob through DVC pipeline"
```

---

### Task 13: Doc updates (automated-variant-analysis spec + precision summary footnote)

**Files:**
- Modify: `docs/specs/2026-04-17-automated-variant-analysis.md`
- Modify: `data/08_reporting/precision_investigation/summary.md`

- [ ] **Step 1: Rewrite Platt → logistic in the automated-variant-analysis spec**

Open `docs/specs/2026-04-17-automated-variant-analysis.md`. Replace every occurrence of "Platt" / "platt" with "Logistic calibration" / "logistic_calibrator" (adjusting case as needed). Specifically:

- Heading `6. **Platt re-calibration**:` → `6. **Logistic calibration**:`
- File path `platt_model.json` → `logistic_calibrator.json`
- "Implementing Platt aggregation in `BboxTubeTemporalModel.predict` (separate PR)" → "Implementing logistic aggregation in `BboxTubeTemporalModel.predict` — now landed; see `docs/specs/2026-04-17-logistic-calibrator-deployment-design.md`."

Add a one-line note at the top of the **Context** section:

> Note: originally named "Platt re-calibration"; renamed to "logistic calibration" for accuracy — the model is multivariate logistic regression, not univariate Platt scaling.

- [ ] **Step 2: Append footnote to precision_investigation/summary.md**

At the end of `data/08_reporting/precision_investigation/summary.md`, append:

```markdown

---

_Footnote (2026-04-17): the "Platt re-calibration" intervention referenced throughout this investigation is productized under the more accurate name `LogisticCalibrator` (multivariate logistic regression). See `docs/specs/2026-04-17-logistic-calibrator-deployment-design.md` for the deployment design._
```

- [ ] **Step 3: Commit**

```bash
git add docs/specs/2026-04-17-automated-variant-analysis.md \
    data/08_reporting/precision_investigation/summary.md
git commit -m "docs(bbox-tube-temporal): rename Platt → logistic calibrator in active specs"
```

---

### Task 14: End-to-end verification (no commit)

**Files:** none modified.

**Context:** Validates that the integrated pipeline reaches the spec's target metrics. This is a verification checkpoint, not a commit step. Run the commands, compare against expected metrics, and either move to the finishing step or diagnose regressions.

- [ ] **Step 1: Lint + full test suite**

Run: `make lint && make test`
Expected: 0 lint issues, all tests green.

- [ ] **Step 2: Repackage ViT with logistic aggregation**

Run: `uv run dvc repro -f package@vit_dinov2_finetune`
Expected console output includes:
```
[package] wrote data/06_models/vit_dinov2_finetune/model.zip | variant=vit_dinov2_finetune aggregation=logistic threshold=<~near raw val logit threshold> target_recall=0.95 logistic_threshold=<~0.4–0.6>
```

Verify the zip contents:
```bash
uv run python -c "import zipfile; print(zipfile.ZipFile('data/06_models/vit_dinov2_finetune/model.zip').namelist())"
```
Expected: the list includes `logistic_calibrator.json`.

- [ ] **Step 3: Re-run evaluate_packaged on val for ViT**

Run: `uv run dvc repro -f 'evaluate_packaged@{variant:vit_dinov2_finetune,split:val}'`
(Exact stage-name-matrix syntax may differ; fall back to `uv run dvc repro -f evaluate_packaged` if needed.)

Read the resulting metrics:
```bash
cat data/08_reporting/val/packaged/vit_dinov2_finetune/metrics.json
```
Expected: `precision >= 0.97`, `recall >= 0.94`, `f1 >= 0.96`. Minor drift from the investigation's 0.974 / 0.950 / 0.962 is acceptable because the fit now uses package-time full-pipeline inference rather than post-hoc `analyze_variant` predictions (same recipe, slightly different input rows).

- [ ] **Step 4: Regression check on GRU**

Run: `uv run dvc repro -f package@gru_convnext_finetune 'evaluate_packaged@{variant:gru_convnext_finetune,split:val}'`

Read metrics and verify:
```bash
cat data/08_reporting/val/packaged/gru_convnext_finetune/metrics.json
```
Expected: precision / recall / F1 within ±0.005 of the pre-change baseline (`0.9560 / 0.9560 / 0.9560` per the precision-investigation summary). Verify the zip does **not** contain `logistic_calibrator.json`:

```bash
uv run python -c "import zipfile; print(zipfile.ZipFile('data/06_models/gru_convnext_finetune/model.zip').namelist())"
```
Expected: `logistic_calibrator.json` is **absent**.

- [ ] **Step 5: Commit DVC lockfile + regenerated metrics**

```bash
git add dvc.lock \
    data/08_reporting/val/packaged/vit_dinov2_finetune/metrics.json \
    data/08_reporting/train/packaged/vit_dinov2_finetune/metrics.json \
    data/08_reporting/val/packaged/gru_convnext_finetune/metrics.json \
    data/08_reporting/train/packaged/gru_convnext_finetune/metrics.json \
    data/08_reporting/val/packaged/vit_dinov2_finetune/predictions.json \
    data/08_reporting/train/packaged/vit_dinov2_finetune/predictions.json \
    data/08_reporting/val/packaged/gru_convnext_finetune/predictions.json \
    data/08_reporting/train/packaged/gru_convnext_finetune/predictions.json
git commit -m "chore(bbox-tube-temporal): record logistic calibrator deployment run"
```

(Skip any paths that didn't change.)

---

## Self-Review Notes

**Spec coverage:** Architecture §Shared module → Tasks 1-5. Packaging flow → Task 11 with supporting Task 8 (pipeline helper), Task 7 (zip schema). Runtime flow → Tasks 9-10. Naming rename → Tasks 6 + 13. Tests (spec §Testing) → `test_logistic_calibrator.py` (Tasks 1-4), `test_logistic_calibrator_fit.py` (Task 5), `test_package.py` calibrator cases (Task 7), `test_inference.py` logistic cases (Task 9), `test_package_predict.py` (Task 8). DVC changes → Task 12. Verification → Task 14.

**Type consistency:** `LogisticCalibrator` is created in Task 1 and used consistently through Tasks 2-11. `extract_features` introduced in Task 3, used in Tasks 5, 6, 9, 11. `fit` signature `list[dict] -> LogisticCalibrator` is stable from Task 5 through Task 11. `pick_winner_and_trigger` new kwargs `aggregation`, `calibrator`, `logistic_threshold` land in Task 9 and are called consistently in Task 10.

**Known open detail:** Task 8 Step 1 asks the engineer to identify the existing sequence loader via `grep`. If no matching helper exists, the engineer is directed to reuse whatever `scripts/evaluate_packaged.py` uses, inlining the smallest adapter. This is a lookup (not a judgment call) so it's safe to defer to implementation.
