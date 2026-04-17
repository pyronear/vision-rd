"""Pure-numpy runtime logistic calibrator.

Loads serialized weights fitted by ``logistic_calibrator_fit.fit`` and
applies the multivariate logistic regression at inference time. No
sklearn import: runtime keeps a small dep surface and avoids pickle
version drift.
"""

from __future__ import annotations

import json
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
