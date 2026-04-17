"""CPU (and any-device) latency benchmark for the packaged temporal model.

Library module: stats helpers, timing proxies, and the ``run_benchmark_on_model``
driver live here. ``scripts/benchmark_cpu_latency.py`` is the thin CLI wrapper.

See ``docs/specs/2026-04-17-cpu-latency-benchmark-design.md``.
"""

from __future__ import annotations

import time

import torch


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolation percentile.

    Matches ``numpy.percentile(..., method='linear')``. ``p`` is in [0, 100].
    Raises ``ValueError`` on empty input.
    """
    if not xs:
        raise ValueError("percentile() got empty input")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"p must be in [0, 100], got {p!r}")
    sorted_xs = sorted(xs)
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    rank = (len(sorted_xs) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = rank - lo
    return float(sorted_xs[lo] + frac * (sorted_xs[hi] - sorted_xs[lo]))


def summarize(xs: list[float]) -> dict[str, float]:
    """Return p50/p95/mean/min/max of ``xs``.

    Raises ``ValueError`` on empty input.
    """
    if not xs:
        raise ValueError("summarize() got empty input")
    return {
        "p50": percentile(xs, 50.0),
        "p95": percentile(xs, 95.0),
        "mean": float(sum(xs) / len(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
    }


class TimedYoloProxy:
    """Forward ``.predict`` to a wrapped YOLO, accumulating wall-clock into a bucket.

    ``bucket`` is a mutable dict shared across sequences; the driver zeros
    ``bucket["yolo_s"]`` between sequences.
    """

    def __init__(self, wrapped: object, bucket: dict[str, float]) -> None:
        self._wrapped = wrapped
        self._bucket = bucket

    def predict(self, *args: object, **kwargs: object) -> object:
        t0 = time.perf_counter()
        result = self._wrapped.predict(*args, **kwargs)
        self._bucket["yolo_s"] += time.perf_counter() - t0
        return result


class TimedClassifier(torch.nn.Module):
    """Forward-wrap the packaged classifier, accumulating wall-clock into a bucket.

    Subclasses ``nn.Module`` so ``.eval()``, ``.to()``, and parameter iteration
    keep working through the proxy. Captures every forward pass — including the
    prefix-scoring calls inside ``find_first_crossing_trigger`` — because the
    wrapper is installed on ``model._classifier`` itself.
    """

    def __init__(self, wrapped: torch.nn.Module, bucket: dict[str, float]) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.bucket = bucket

    def forward(self, *args: object, **kwargs: object) -> object:
        t0 = time.perf_counter()
        result = self.wrapped(*args, **kwargs)
        self.bucket["classifier_s"] += time.perf_counter() - t0
        return result


def wrap_for_timing(model: object, bucket: dict[str, float]) -> None:
    """Install :class:`TimedYoloProxy` and :class:`TimedClassifier` on ``model``.

    Expected to be called exactly once per benchmark, immediately after
    ``BboxTubeTemporalModel.from_archive``.
    """
    model._yolo = TimedYoloProxy(model._yolo, bucket)
    model._classifier = TimedClassifier(model._classifier, bucket).eval()
