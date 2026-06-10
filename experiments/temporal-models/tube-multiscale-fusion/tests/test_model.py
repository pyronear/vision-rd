"""Tests for TubeMultiscaleFusionModel.predict().

The detect -> tube -> crop -> score -> trigger pipeline is exercised with a
mocked YOLO and a stub classifier that returns a fixed logit, so the wiring is
verified deterministically without weights. The real two-branch classifier is
covered by ``test_classifier.py``; the lib stages by the bbox-tube-temporal
suite.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from pyrocore.types import Frame
from torch import Tensor, nn

from tube_multiscale_fusion.model import TubeMultiscaleFusionModel

TEST_CONFIG: dict = {
    "infer": {
        "confidence_threshold": 0.01,
        "iou_nms": 0.2,
        "image_size": 64,
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
    "classifier": {"max_frames": 6},
    "decision": {"aggregation": "max_logit", "threshold": 0.0},
}


class _StubClassifier(nn.Module):
    """Returns a constant logit per tube, regardless of input."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self._logit = logit

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:  # noqa: ARG002
        return torch.full((patches.shape[0],), self._logit, dtype=torch.float32)


def _fake_yolo_factory(
    per_frame_xywhn: list[list[tuple[float, float, float, float, float]]],
):
    """Return a mock YOLO whose ``.predict`` yields fixed detections per frame."""

    def fake_predict(paths: list[str], **_kwargs):
        assert len(paths) == len(per_frame_xywhn)
        results = []
        for boxes in per_frame_xywhn:
            r = MagicMock()
            r.boxes = MagicMock()
            if not boxes:
                r.boxes.__len__ = lambda self: 0
                r.boxes.xywhn = torch.zeros(0, 4)
                r.boxes.conf = torch.zeros(0)
                r.boxes.cls = torch.zeros(0)
            else:
                n = len(boxes)
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


@pytest.fixture()
def red_frames(tmp_path: Path) -> list[Frame]:
    frames = []
    for i in range(6):
        arr = np.full((64, 64, 3), fill_value=[180, 30, 30], dtype=np.uint8)
        p = tmp_path / f"f_{i:02d}.jpg"
        Image.fromarray(arr).save(p, format="JPEG")
        frames.append(Frame(frame_id=p.stem, image_path=p, timestamp=None))
    return frames


def _model(yolo, logit: float) -> TubeMultiscaleFusionModel:
    return TubeMultiscaleFusionModel(
        yolo_model=yolo,
        classifier=_StubClassifier(logit),
        config=TEST_CONFIG,
        device="cpu",
    )


def test_empty_frames_returns_negative() -> None:
    yolo = MagicMock()
    out = _model(yolo, logit=10.0).predict(frames=[])
    assert out.is_positive is False
    assert out.trigger_frame_index is None
    assert out.details["preprocessing"]["num_frames_input"] == 0
    assert out.details["tubes"]["num_candidates"] == 0
    yolo.predict.assert_not_called()


def test_zero_detections_means_negative(red_frames: list[Frame]) -> None:
    yolo = _fake_yolo_factory([[] for _ in red_frames])
    out = _model(yolo, logit=10.0).predict(frames=red_frames)
    assert out.is_positive is False
    assert out.details["tubes"]["num_candidates"] == 0
    assert out.details["tubes"]["kept"] == []


def test_short_tube_below_infer_floor_discarded(red_frames: list[Frame]) -> None:
    # Only frame 0 has a detection -> tube length 1, below min_detected_entries=2.
    per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)]] + [[] for _ in red_frames[1:]]
    yolo = _fake_yolo_factory(per_frame)
    out = _model(yolo, logit=10.0).predict(frames=red_frames)
    assert out.is_positive is False
    assert out.details["tubes"]["num_candidates"] == 1
    assert out.details["tubes"]["kept"] == []


def test_full_tube_positive_fires_with_trigger(red_frames: list[Frame]) -> None:
    per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in red_frames]
    yolo = _fake_yolo_factory(per_frame)
    out = _model(yolo, logit=10.0).predict(frames=red_frames)
    assert out.is_positive is True
    assert out.trigger_frame_index is not None
    assert out.details["decision"]["trigger_tube_id"] is not None
    assert len(out.details["tubes"]["kept"]) == 1
    # logit 10 >= threshold 0 at the shortest scored prefix (min_prefix_length=2)
    assert out.trigger_frame_index == 1


def test_full_tube_negative_when_below_threshold(red_frames: list[Frame]) -> None:
    per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in red_frames]
    yolo = _fake_yolo_factory(per_frame)
    out = _model(yolo, logit=-10.0).predict(frames=red_frames)
    assert out.is_positive is False
    assert out.trigger_frame_index is None
    assert len(out.details["tubes"]["kept"]) == 1


def test_truncation_to_max_frames(red_frames: list[Frame]) -> None:
    # 6 real + 3 extra = 9; max_frames=6 so YOLO only sees the first 6.
    extra = red_frames + red_frames[:3]
    per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in range(6)]
    yolo = _fake_yolo_factory(per_frame)
    out = _model(yolo, logit=10.0).predict(frames=extra)
    assert out.details["preprocessing"]["num_frames_input"] == 9
    assert out.details["preprocessing"]["num_truncated"] == 3
