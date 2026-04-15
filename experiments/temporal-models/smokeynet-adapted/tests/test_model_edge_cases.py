"""Edge-case tests for SmokeynetTemporalModel.predict()."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from pyrocore.types import Frame

from smokeynet_adapted.model import SmokeynetTemporalModel
from smokeynet_adapted.temporal_classifier import TemporalSmokeClassifier

TEST_CONFIG: dict = {
    "infer": {"confidence_threshold": 0.01, "iou_nms": 0.2, "image_size": 1024},
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


def _fake_yolo_factory(
    per_frame_xywhn: list[list[tuple[float, float, float, float, float]]],
):
    """Return a mock YOLO whose ``.predict`` yields fixed detections per frame."""

    def fake_predict(paths: list[str], **_kwargs):
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


@pytest.fixture()
def tiny_classifier() -> TemporalSmokeClassifier:
    model = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        pretrained=False,
    )
    model.eval()
    return model


@pytest.fixture()
def red_frames(tmp_path: Path) -> list[Frame]:
    frames = []
    for i in range(6):
        arr = np.full((64, 64, 3), fill_value=[180, 30, 30], dtype=np.uint8)
        p = tmp_path / f"f_{i:02d}.jpg"
        Image.fromarray(arr).save(p, format="JPEG")
        frames.append(Frame(frame_id=p.stem, image_path=p, timestamp=None))
    return frames


class TestEmptyFrames:
    def test_returns_negative(self, tiny_classifier: TemporalSmokeClassifier) -> None:
        yolo = MagicMock()
        model = SmokeynetTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=[])
        assert out.is_positive is False
        assert out.trigger_frame_index is None
        assert out.details["num_frames"] == 0
        yolo.predict.assert_not_called()


class TestZeroDetections:
    def test_no_tubes_means_negative(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        yolo = _fake_yolo_factory([[] for _ in red_frames])
        model = SmokeynetTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=red_frames)
        assert out.is_positive is False
        assert out.trigger_frame_index is None
        assert out.details["num_tubes_total"] == 0
        assert out.details["num_tubes_kept"] == 0


class TestShortTubeBelowInferFloor:
    def test_single_frame_detection_discarded(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        # Only frame 0 has a detection — tube length 1, below infer_min_tube_length=2.
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)]] + [[] for _ in red_frames[1:]]
        yolo = _fake_yolo_factory(per_frame)
        model = SmokeynetTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=red_frames)
        assert out.is_positive is False
        assert out.details["num_tubes_total"] == 1
        assert out.details["num_tubes_kept"] == 0


class TestTruncation:
    def test_sequence_longer_than_max_frames(
        self, tiny_classifier: TemporalSmokeClassifier, red_frames: list[Frame]
    ) -> None:
        # red_frames has 6; TEST_CONFIG max_frames=6; extend to 9.
        extra = red_frames + red_frames[:3]
        # YOLO will only see the first 6 (truncated).
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in range(6)]
        yolo = _fake_yolo_factory(per_frame)
        model = SmokeynetTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=extra)
        assert out.details["num_frames"] == 9
        assert out.details["num_truncated"] == 3
