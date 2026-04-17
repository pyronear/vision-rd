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
        "features",
        "coefficients",
        "intercept",
        "sanity_checks",
    }
    assert payload["features"] == ["logit", "log_len", "mean_conf", "n_tubes"]
    assert payload["coefficients"] == [0.1, 0.2, 0.3, 0.4]
    assert payload["intercept"] == -1.0
    assert payload["sanity_checks"] == [
        {"features": [1.0, 2.0, 3.0, 4.0], "prob": 0.5}
    ]


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
