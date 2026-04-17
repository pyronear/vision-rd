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
            entries = [
                e["confidence"]
                for e in t["entries"]
                if e["confidence"] is not None
            ]
            X.append(
                [
                    t["logit"],
                    math.log1p(t["end_frame"] - t["start_frame"] + 1),
                    sum(entries) / len(entries) if entries else 0.0,
                    len(kept),
                ]
            )
        else:
            X.append([0.0, 0.0, 0.0, 0.0])
        y.append(1 if r["label"] == "smoke" else 0)
    X_arr = np.array(X)

    sklearn_model = LogisticRegression(max_iter=1000, C=1.0).fit(X_arr, y)
    sklearn_probs = sklearn_model.predict_proba(X_arr)[:, 1]
    numpy_probs = cal.predict_proba_batch(X_arr)

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
