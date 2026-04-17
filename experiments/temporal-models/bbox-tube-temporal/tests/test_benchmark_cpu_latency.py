"""Tests for bbox_tube_temporal.benchmark_latency."""

from __future__ import annotations

import pytest

from bbox_tube_temporal import benchmark_latency as bench


def test_summarize_percentile_values() -> None:
    stats = bench.summarize([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats == {
        "p50": 30.0,
        "p95": 48.0,
        "mean": 30.0,
        "min": 10.0,
        "max": 50.0,
    }


def test_summarize_raises_on_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        bench.summarize([])
