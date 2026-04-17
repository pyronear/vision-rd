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
