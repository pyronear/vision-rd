"""CPU (and any-device) latency benchmark for the packaged temporal model.

Library module: stats helpers, timing proxies, and the ``run_benchmark_on_model``
driver live here. ``scripts/benchmark_cpu_latency.py`` is the thin CLI wrapper.

See ``docs/specs/2026-04-17-cpu-latency-benchmark-design.md``.
"""

from __future__ import annotations


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
