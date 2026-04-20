"""Tests for bbox_tube_temporal.benchmark_latency."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from bbox_tube_temporal import benchmark_latency as bench
from bbox_tube_temporal.model import BboxTubeTemporalModel
from bbox_tube_temporal.temporal_classifier import TemporalSmokeClassifier

BENCHMARK_TEST_CONFIG: dict = {
    "infer": {
        "confidence_threshold": 0.01,
        "iou_nms": 0.2,
        "image_size": 1024,
        "pad_to_min_frames": 0,
        "pad_strategy": "symmetric",
    },
    "tubes": {
        "iou_threshold": 0.2,
        "max_misses": 2,
        "min_tube_length": 4,
        "infer_min_tube_length": 2,
        "min_detected_entries": 2,
        "interpolate_gaps": True,
    },
    "model_input": {
        "context_factor": 1.5,
        "patch_size": 8,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "classifier": {
        "backbone": "resnet18",
        "arch": "gru",
        "hidden_dim": 32,
        "num_layers": 1,
        "bidirectional": False,
        "max_frames": 6,
        "pretrained": False,
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.0,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


def _fake_yolo_factory(per_frame_xywhn):
    """Return a mock YOLO whose ``.predict`` yields fixed detections per frame.

    Mirrors the helper in ``tests/test_model_edge_cases.py``.
    """

    def fake_predict(paths, **_kwargs):
        assert len(paths) == len(per_frame_xywhn)
        results = []
        for boxes in per_frame_xywhn:
            r = MagicMock()
            if not boxes:
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self: 0
                r.boxes.xywhn = torch.zeros(0, 4)
                r.boxes.conf = torch.zeros(0)
                r.boxes.cls = torch.zeros(0)
            else:
                n = len(boxes)
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self, n=n: n
                r.boxes.xywhn = torch.tensor(
                    [[c, cy, w, h] for (c, cy, w, h, _) in boxes]
                )
                r.boxes.conf = torch.tensor([conf for (_, _, _, _, conf) in boxes])
                r.boxes.cls = torch.zeros(n)
            results.append(r)
        return results

    m = MagicMock()
    m.predict.side_effect = fake_predict
    return m


def _make_seq(seq_dir: Path, n_frames: int) -> Path:
    img_dir = seq_dir / "images"
    img_dir.mkdir(parents=True)
    for j in range(n_frames):
        arr = np.full((64, 64, 3), fill_value=[180, 30, 30], dtype=np.uint8)
        Image.fromarray(arr).save(
            img_dir / f"cam_2026-04-17T10-00-{j:02d}.jpg", format="JPEG"
        )
    return seq_dir


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


# ── run_benchmark_on_model integration ─────────────────────────────


def test_run_benchmark_on_model_happy_path(tmp_path: Path) -> None:
    # 5 sequences × 6 frames, split 3 smoke + 2 fp (the benchmark is
    # label-agnostic; mixing just proves it doesn't care).
    seq_dirs: list[Path] = []
    for i in range(5):
        label = "wildfire" if i < 3 else "fp"
        seq_dir = tmp_path / "sequences" / label / f"seq_{i}"
        _make_seq(seq_dir, n_frames=6)
        seq_dirs.append(seq_dir)

    # Fake YOLO emits one detection per frame so tubes are built and the
    # classifier is exercised.
    per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in range(6)]
    yolo = _fake_yolo_factory(per_frame)
    classifier = TemporalSmokeClassifier(
        backbone="resnet18", arch="gru", hidden_dim=32, pretrained=False
    )
    model = BboxTubeTemporalModel(
        yolo_model=yolo,
        classifier=classifier,
        config=BENCHMARK_TEST_CONFIG,
        device="cpu",
    )

    result = bench.run_benchmark_on_model(model, seq_dirs, warmup=2)

    assert len(result["records"]) == 5
    assert [r["is_warmup"] for r in result["records"]] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert result["summary"]["num_sequences"] == 3
    assert result["summary"]["num_warmup_skipped"] == 2

    for r in result["records"]:
        assert r["total_s"] >= 0.0
        assert r["yolo_s"] >= 0.0
        assert r["classifier_s"] >= 0.0
        # Timings are subsets of total — the sign is what we assert.
        assert r["total_s"] + 1e-6 >= r["yolo_s"] + r["classifier_s"]

    for r in result["records"][2:]:  # non-warmup
        assert r["yolo_s"] > 0.0, "YOLO proxy must fire on every sequence"
        assert r["classifier_s"] > 0.0, (
            "classifier proxy must fire when fake YOLO emits detections"
        )

    # Summary sub-dicts exist with expected shape.
    for key in (
        "total_ms",
        "yolo_ms",
        "classifier_ms",
        "other_ms",
        "per_frame_total_ms",
    ):
        assert set(result["summary"][key].keys()) == {
            "p50",
            "p95",
            "mean",
            "min",
            "max",
        }
