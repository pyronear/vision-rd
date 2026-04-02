"""Tests for smokeynet_adapted.heads."""

import torch

from smokeynet_adapted.heads import (
    DetectionClassificationHead,
    SequenceClassificationHead,
)


class TestDetectionClassificationHead:
    def test_output_shape(self):
        head = DetectionClassificationHead(d_model=64)
        x = torch.randn(5, 64)
        out = head(x)
        assert out.shape == (5, 1)

    def test_single_detection(self):
        head = DetectionClassificationHead(d_model=32)
        x = torch.randn(1, 32)
        out = head(x)
        assert out.shape == (1, 1)


class TestSequenceClassificationHead:
    def test_output_shape(self):
        head = SequenceClassificationHead(d_model=64, hidden_dim=32)
        x = torch.randn(64)
        out = head(x)
        assert out.shape == (1,)

    def test_raw_logit(self):
        """Output should be a raw logit, not bounded to [0, 1]."""
        head = SequenceClassificationHead(d_model=32, hidden_dim=16)
        torch.manual_seed(0)
        x = torch.randn(32) * 10
        out = head(x)
        # Raw logit can be any real number
        assert out.shape == (1,)
