"""Tests for smokeynet_adapted.net."""

import torch

from smokeynet_adapted.net import SmokeyNetAdapted
from smokeynet_adapted.types import Detection, Tube, TubeEntry


def _det(cx=0.5, cy=0.5):
    return Detection(class_id=0, cx=cx, cy=cy, w=0.1, h=0.1, confidence=0.8)


def _make_model(d_model=32):
    return SmokeyNetAdapted(
        d_model=d_model,
        lstm_layers=1,
        spatial_layers=1,
        spatial_heads=4,
        dropout=0.0,
    )


class TestSmokeyNetAdapted:
    def test_forward_shape(self):
        model = _make_model(d_model=32)
        # 3 detections across 2 frames, 1 tube
        roi_features = torch.randn(3, 32)
        frame_indices = torch.tensor([0, 0, 1])
        bbox_coords = torch.randn(3, 4)
        tubes = [
            Tube(
                tube_id=0,
                entries=[
                    TubeEntry(frame_idx=0, detection=_det()),
                    TubeEntry(frame_idx=1, detection=_det()),
                ],
                start_frame=0,
                end_frame=1,
            )
        ]

        seq_logit, intermediates = model(
            roi_features, frame_indices, bbox_coords, tubes
        )
        assert seq_logit.shape == (1,)
        assert intermediates["cnn"].shape == (3, 1)

    def test_no_detections(self):
        model = _make_model(d_model=32)
        roi_features = torch.zeros(0, 32)
        frame_indices = torch.zeros(0, dtype=torch.long)
        bbox_coords = torch.zeros(0, 4)
        tubes = []

        seq_logit, intermediates = model(
            roi_features, frame_indices, bbox_coords, tubes
        )
        assert seq_logit.shape == (1,)
        assert seq_logit.item() < 0  # negative logit
        assert intermediates["cnn"].shape == (0, 1)

    def test_multiple_tubes(self):
        model = _make_model(d_model=32)
        # 4 detections, 2 tubes
        roi_features = torch.randn(4, 32)
        frame_indices = torch.tensor([0, 0, 1, 1])
        bbox_coords = torch.randn(4, 4)
        tubes = [
            Tube(
                tube_id=0,
                entries=[
                    TubeEntry(frame_idx=0, detection=_det(0.2, 0.2)),
                    TubeEntry(frame_idx=1, detection=_det(0.2, 0.2)),
                ],
                start_frame=0,
                end_frame=1,
            ),
            Tube(
                tube_id=1,
                entries=[
                    TubeEntry(frame_idx=0, detection=_det(0.8, 0.8)),
                    TubeEntry(frame_idx=1, detection=_det(0.8, 0.8)),
                ],
                start_frame=0,
                end_frame=1,
            ),
        ]

        seq_logit, intermediates = model(
            roi_features, frame_indices, bbox_coords, tubes
        )
        assert seq_logit.shape == (1,)
        assert intermediates["cnn"].shape == (4, 1)

    def test_gradient_flows(self):
        """Ensure gradients flow through the full model."""
        model = _make_model(d_model=16)
        roi_features = torch.randn(2, 16, requires_grad=True)
        frame_indices = torch.tensor([0, 1])
        bbox_coords = torch.randn(2, 4)
        tubes = [
            Tube(
                tube_id=0,
                entries=[
                    TubeEntry(frame_idx=0, detection=_det()),
                    TubeEntry(frame_idx=1, detection=_det()),
                ],
                start_frame=0,
                end_frame=1,
            )
        ]

        seq_logit, _ = model(roi_features, frame_indices, bbox_coords, tubes)
        seq_logit.backward()
        assert roi_features.grad is not None
        assert roi_features.grad.abs().sum() > 0
