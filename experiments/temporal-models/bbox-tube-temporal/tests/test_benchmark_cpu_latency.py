"""Tests for bbox_tube_temporal.benchmark_latency."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

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


# ── timing proxies ─────────────────────────────────────────────────


def test_timed_yolo_proxy_accumulates_bucket() -> None:
    bucket = {"yolo_s": 0.0, "classifier_s": 0.0}
    inner = MagicMock()
    inner.predict.return_value = []
    proxy = bench.TimedYoloProxy(inner, bucket)

    proxy.predict(["/fake/path.jpg"])

    assert inner.predict.call_count == 1
    assert bucket["yolo_s"] > 0.0
    assert bucket["classifier_s"] == 0.0


def test_timed_classifier_accumulates_bucket() -> None:
    bucket = {"yolo_s": 0.0, "classifier_s": 0.0}
    inner = torch.nn.Linear(4, 1)
    proxy = bench.TimedClassifier(inner, bucket)

    proxy(torch.zeros(1, 4))

    assert bucket["classifier_s"] > 0.0
    assert bucket["yolo_s"] == 0.0


def test_wrap_for_timing_installs_both_proxies() -> None:
    model = MagicMock()
    model._yolo = MagicMock()
    model._classifier = torch.nn.Linear(4, 1)
    bucket = {"yolo_s": 0.0, "classifier_s": 0.0}

    bench.wrap_for_timing(model, bucket)

    assert isinstance(model._yolo, bench.TimedYoloProxy)
    assert isinstance(model._classifier, bench.TimedClassifier)
