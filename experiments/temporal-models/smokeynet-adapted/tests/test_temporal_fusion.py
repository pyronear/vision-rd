"""Tests for smokeynet_adapted.temporal_fusion."""

import torch

from smokeynet_adapted.temporal_fusion import TemporalLSTM, interpolate_tube_features
from smokeynet_adapted.types import Detection, Tube, TubeEntry


def _det(cx=0.5, cy=0.5):
    return Detection(class_id=0, cx=cx, cy=cy, w=0.1, h=0.1, confidence=0.8)


class TestInterpolateTubeFeatures:
    def test_no_gaps(self):
        """All entries have detections -- no interpolation needed."""
        tube = Tube(
            tube_id=0,
            entries=[
                TubeEntry(frame_idx=0, detection=_det()),
                TubeEntry(frame_idx=1, detection=_det()),
                TubeEntry(frame_idx=2, detection=_det()),
            ],
            start_frame=0,
            end_frame=2,
        )
        features = {
            (0, 0): torch.tensor([1.0, 0.0]),
            (1, 0): torch.tensor([2.0, 0.0]),
            (2, 0): torch.tensor([3.0, 0.0]),
        }
        result = interpolate_tube_features(features, tube)
        assert result.shape == (3, 2)
        assert torch.allclose(result[0], torch.tensor([1.0, 0.0]))
        assert torch.allclose(result[2], torch.tensor([3.0, 0.0]))

    def test_interior_gap_interpolated(self):
        """Gap between two observed features should be linearly interpolated."""
        tube = Tube(
            tube_id=0,
            entries=[
                TubeEntry(frame_idx=0, detection=_det()),
                TubeEntry(frame_idx=1, detection=None),  # gap
                TubeEntry(frame_idx=2, detection=_det()),
            ],
            start_frame=0,
            end_frame=2,
        )
        features = {
            (0, 0): torch.tensor([0.0, 0.0]),
            (2, 0): torch.tensor([2.0, 4.0]),
        }
        result = interpolate_tube_features(features, tube)
        assert result.shape == (3, 2)
        # Gap at index 1: lerp(0, 2, 0.5) = 1.0; lerp(0, 4, 0.5) = 2.0
        assert torch.allclose(result[1], torch.tensor([1.0, 2.0]))

    def test_boundary_gap_start_repeats_nearest(self):
        """Gap at the start of a tube should repeat the first observed feat."""
        tube = Tube(
            tube_id=0,
            entries=[
                TubeEntry(frame_idx=0, detection=None),  # gap
                TubeEntry(frame_idx=1, detection=_det()),
            ],
            start_frame=0,
            end_frame=1,
        )
        features = {(1, 0): torch.tensor([5.0, 3.0])}
        result = interpolate_tube_features(features, tube)
        assert torch.allclose(result[0], torch.tensor([5.0, 3.0]))

    def test_boundary_gap_end_repeats_nearest(self):
        """Gap at the end of a tube should repeat the last observed feat."""
        tube = Tube(
            tube_id=0,
            entries=[
                TubeEntry(frame_idx=0, detection=_det()),
                TubeEntry(frame_idx=1, detection=None),  # gap
            ],
            start_frame=0,
            end_frame=1,
        )
        features = {(0, 0): torch.tensor([5.0, 3.0])}
        result = interpolate_tube_features(features, tube)
        assert torch.allclose(result[1], torch.tensor([5.0, 3.0]))

    def test_multiple_interior_gaps(self):
        """Multiple consecutive gaps should be properly interpolated."""
        tube = Tube(
            tube_id=0,
            entries=[
                TubeEntry(frame_idx=0, detection=_det()),
                TubeEntry(frame_idx=1, detection=None),
                TubeEntry(frame_idx=2, detection=None),
                TubeEntry(frame_idx=3, detection=_det()),
            ],
            start_frame=0,
            end_frame=3,
        )
        features = {
            (0, 0): torch.tensor([0.0]),
            (3, 0): torch.tensor([3.0]),
        }
        result = interpolate_tube_features(features, tube)
        assert torch.allclose(result[1], torch.tensor([1.0]))
        assert torch.allclose(result[2], torch.tensor([2.0]))


class TestTemporalLSTM:
    def test_output_shape(self):
        lstm = TemporalLSTM(d_model=32, num_layers=1)
        tube_feats = [
            torch.randn(5, 32),
            torch.randn(3, 32),
        ]
        out = lstm(tube_feats)
        assert out.shape == (2, 32)

    def test_single_tube(self):
        lstm = TemporalLSTM(d_model=16, num_layers=1)
        tube_feats = [torch.randn(4, 16)]
        out = lstm(tube_feats)
        assert out.shape == (1, 16)

    def test_empty_tubes(self):
        lstm = TemporalLSTM(d_model=16, num_layers=1)
        out = lstm([])
        assert out.shape == (0, 16)

    def test_single_frame_tube(self):
        lstm = TemporalLSTM(d_model=16, num_layers=1)
        tube_feats = [torch.randn(1, 16)]
        out = lstm(tube_feats)
        assert out.shape == (1, 16)
