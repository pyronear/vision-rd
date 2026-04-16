"""Pure-function unit tests for inference helpers."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from pyrocore.types import Frame

from bbox_tube_temporal.inference import (
    crop_tube_patches,
    filter_and_interpolate_tubes,
    pick_winner_and_trigger,
    run_yolo_on_frames,
    score_tubes,
)
from bbox_tube_temporal.types import Detection, FrameDetections, Tube, TubeEntry


def _fake_yolo_result(
    xywhn: list[tuple[float, float, float, float, float]],
) -> MagicMock:
    """Build a MagicMock shaped like ultralytics Results for one image.

    xywhn: list of (cx, cy, w, h, conf) tuples.
    """
    boxes = MagicMock()
    boxes.__len__ = lambda self: len(xywhn)
    boxes.xywhn = torch.tensor([[c, cy, w, h] for (c, cy, w, h, _) in xywhn])
    boxes.conf = torch.tensor([conf for (_, _, _, _, conf) in xywhn])
    boxes.cls = torch.zeros(len(xywhn))
    result = MagicMock()
    result.boxes = boxes
    return result


class TestRunYoloOnFrames:
    def test_batches_all_frames_in_single_call(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = [_fake_yolo_result([]), _fake_yolo_result([])]
        frames = [
            Frame(frame_id="f0", image_path=Path("/x/f0.jpg"), timestamp=None),
            Frame(frame_id="f1", image_path=Path("/x/f1.jpg"), timestamp=None),
        ]

        run_yolo_on_frames(
            yolo, frames, confidence_threshold=0.01, iou_nms=0.2, image_size=1024
        )

        assert yolo.predict.call_count == 1
        args, kwargs = yolo.predict.call_args
        assert args[0] == ["/x/f0.jpg", "/x/f1.jpg"]
        assert kwargs["conf"] == 0.01
        assert kwargs["iou"] == 0.2
        assert kwargs["imgsz"] == 1024
        assert kwargs["verbose"] is False

    def test_converts_detections(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = [
            _fake_yolo_result([(0.5, 0.4, 0.1, 0.2, 0.9)]),
            _fake_yolo_result([]),
        ]
        ts = datetime(2024, 1, 1, 12, 0, 0)
        frames = [
            Frame(frame_id="f0", image_path=Path("/x/f0.jpg"), timestamp=ts),
            Frame(frame_id="f1", image_path=Path("/x/f1.jpg"), timestamp=None),
        ]

        result = run_yolo_on_frames(
            yolo, frames, confidence_threshold=0.01, iou_nms=0.2, image_size=1024
        )

        assert len(result) == 2

        # Frame 0: one detection, check structural + numeric approximately.
        fd0 = result[0]
        assert fd0.frame_idx == 0
        assert fd0.frame_id == "f0"
        assert fd0.timestamp == ts
        assert len(fd0.detections) == 1
        d0 = fd0.detections[0]
        assert d0.class_id == 0
        assert d0.cx == pytest.approx(0.5, rel=1e-6)
        assert d0.cy == pytest.approx(0.4, rel=1e-6)
        assert d0.w == pytest.approx(0.1, rel=1e-6)
        assert d0.h == pytest.approx(0.2, rel=1e-6)
        assert d0.confidence == pytest.approx(0.9, rel=1e-6)

        # Frame 1: no detections.
        assert result[1] == FrameDetections(
            frame_idx=1, frame_id="f1", timestamp=None, detections=[]
        )

    def test_empty_frames_returns_empty(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = []
        result = run_yolo_on_frames(
            yolo, [], confidence_threshold=0.01, iou_nms=0.2, image_size=1024
        )
        assert result == []
        yolo.predict.assert_not_called()


def _tube(tid: int, entries: list[tuple[int, Detection | None]]) -> Tube:
    return Tube(
        tube_id=tid,
        entries=[TubeEntry(frame_idx=i, detection=d) for (i, d) in entries],
        start_frame=entries[0][0],
        end_frame=entries[-1][0],
    )


def _det(cx: float = 0.5, cy: float = 0.5, w: float = 0.1, h: float = 0.1) -> Detection:
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=0.9)


class TestFilterAndInterpolate:
    def test_drops_tubes_shorter_than_min_length(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, _det())]),  # length 2 - keep
            _tube(1, [(3, _det())]),  # length 1 - drop
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=1, interpolate_gaps=False
        )
        assert [t.tube_id for t in out] == [0]

    def test_drops_tubes_with_too_few_observed(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, None), (2, None), (3, None)]),
            _tube(1, [(0, _det()), (1, _det()), (2, None), (3, None)]),
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=False
        )
        assert [t.tube_id for t in out] == [1]

    def test_interpolation_applied_when_enabled(self) -> None:
        tubes = [
            _tube(
                0,
                [
                    (0, _det(cx=0.2)),
                    (1, None),
                    (2, _det(cx=0.4)),
                ],
            ),
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=True
        )
        assert len(out) == 1
        mid = out[0].entries[1]
        assert mid.is_gap is True
        assert mid.detection is not None
        assert mid.detection.cx == pytest.approx(0.3)

    def test_interpolation_skipped_when_disabled(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, None), (2, _det())]),
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=False
        )
        assert out[0].entries[1].detection is None

    def test_empty_input(self) -> None:
        assert (
            filter_and_interpolate_tubes(
                [], min_tube_length=2, min_detected_entries=1, interpolate_gaps=True
            )
            == []
        )


@pytest.fixture()
def red_image_sequence(tmp_path: Path) -> list[Path]:
    """Three 128x128 solid-red JPGs."""
    paths = []
    for i in range(3):
        img = np.full((128, 128, 3), fill_value=[200, 30, 30], dtype=np.uint8)
        p = tmp_path / f"frame_{i:02d}.jpg"
        Image.fromarray(img).save(p, format="JPEG", quality=95)
        paths.append(p)
    return paths


class TestCropTubePatches:
    def test_output_shape(self, red_image_sequence: list[Path]) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in red_image_sequence
        ]
        tube = _tube(
            0,
            [
                (0, _det(cx=0.5, cy=0.5, w=0.2, h=0.2)),
                (1, _det(cx=0.5, cy=0.5, w=0.2, h=0.2)),
                (2, _det(cx=0.5, cy=0.5, w=0.2, h=0.2)),
            ],
        )
        patches, mask = crop_tube_patches(
            tube,
            frames,
            context_factor=1.5,
            patch_size=224,
            max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        assert patches.shape == (5, 3, 224, 224)
        assert patches.dtype == torch.float32
        assert mask.shape == (5,)
        assert mask.tolist() == [True, True, True, False, False]

    def test_padding_slots_are_zero(self, red_image_sequence: list[Path]) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in red_image_sequence
        ]
        tube = _tube(0, [(0, _det()), (1, _det())])
        patches, mask = crop_tube_patches(
            tube,
            frames,
            context_factor=1.5,
            patch_size=224,
            max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        assert torch.all(patches[2:] == 0.0)

    def test_truncates_tubes_longer_than_max_frames(
        self, red_image_sequence: list[Path]
    ) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in red_image_sequence
        ]
        tube = _tube(0, [(0, _det()), (1, _det()), (2, _det())])
        patches, mask = crop_tube_patches(
            tube,
            frames,
            context_factor=1.5,
            patch_size=224,
            max_frames=2,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        assert patches.shape == (2, 3, 224, 224)
        assert mask.tolist() == [True, True]


class TestScoreTubes:
    def test_empty_input_returns_empty(self) -> None:
        classifier = MagicMock()
        logits = score_tubes(classifier, patches_per_tube=[], masks_per_tube=[])
        assert logits.shape == (0,)
        classifier.assert_not_called()

    def test_single_batched_forward(self) -> None:
        classifier = MagicMock(return_value=torch.tensor([1.2, -0.3]))
        patches = [torch.zeros(4, 3, 8, 8), torch.zeros(4, 3, 8, 8)]
        masks = [
            torch.tensor([True, True, True, True]),
            torch.tensor([True, True, False, False]),
        ]

        logits = score_tubes(classifier, patches_per_tube=patches, masks_per_tube=masks)

        assert classifier.call_count == 1
        args, _ = classifier.call_args
        assert args[0].shape == (2, 4, 3, 8, 8)
        assert args[1].shape == (2, 4)
        assert logits.tolist() == pytest.approx([1.2, -0.3], rel=1e-5)


class TestPickWinnerAndTrigger:
    def test_no_tubes_returns_negative(self) -> None:
        res = pick_winner_and_trigger(tubes=[], logits=torch.zeros(0), threshold=0.0)
        assert res == (False, None, None)

    def test_argmax_and_threshold_crossed(self) -> None:
        tubes = [
            _tube(10, [(0, _det()), (1, _det())]),  # end_frame = 1
            _tube(20, [(2, _det()), (3, _det()), (4, _det())]),  # end_frame = 4
        ]
        logits = torch.tensor([-1.0, 0.5])
        res = pick_winner_and_trigger(tubes=tubes, logits=logits, threshold=0.0)
        assert res == (True, 4, 20)

    def test_argmax_below_threshold(self) -> None:
        tubes = [
            _tube(1, [(0, _det()), (1, _det())]),
            _tube(2, [(2, _det()), (3, _det())]),
        ]
        logits = torch.tensor([-2.0, -0.5])
        is_positive, trigger, winner = pick_winner_and_trigger(
            tubes=tubes, logits=logits, threshold=0.0
        )
        assert is_positive is False
        assert trigger is None
        assert winner == 2
