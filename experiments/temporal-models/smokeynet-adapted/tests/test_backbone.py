"""Tests for smokeynet_adapted.backbone."""

import torch

from smokeynet_adapted.backbone import YoloRoiExtractor, _norm_cxcywh_to_abs_xyxy


class TestNormCxcywhToAbsXyxy:
    def test_center_box(self):
        bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        result = _norm_cxcywh_to_abs_xyxy(bboxes, img_w=100, img_h=100)
        # cx=50, cy=50, w=20, h=20 -> x1=40, y1=40, x2=60, y2=60
        assert torch.allclose(result, torch.tensor([[40.0, 40.0, 60.0, 60.0]]))

    def test_context_factor(self):
        bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        result = _norm_cxcywh_to_abs_xyxy(
            bboxes, img_w=100, img_h=100, context_factor=1.5
        )
        # w=20*1.5=30, h=20*1.5=30 -> x1=35, y1=35, x2=65, y2=65
        assert torch.allclose(result, torch.tensor([[35.0, 35.0, 65.0, 65.0]]))

    def test_clamp_to_image_bounds(self):
        bboxes = torch.tensor([[0.0, 0.0, 0.4, 0.4]])
        result = _norm_cxcywh_to_abs_xyxy(bboxes, img_w=100, img_h=100)
        # cx=0, cy=0, w=40, h=40 -> x1=-20, y1=-20 -> clamped to 0
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0

    def test_no_bboxes(self):
        bboxes = torch.zeros(0, 4)
        result = _norm_cxcywh_to_abs_xyxy(bboxes, img_w=100, img_h=100)
        assert result.shape == (0, 4)

    def test_multiple_bboxes(self):
        bboxes = torch.tensor(
            [
                [0.2, 0.2, 0.1, 0.1],
                [0.8, 0.8, 0.1, 0.1],
            ]
        )
        result = _norm_cxcywh_to_abs_xyxy(bboxes, img_w=200, img_h=200)
        assert result.shape == (2, 4)


class TestYoloRoiExtractor:
    @staticmethod
    def _make_fake_yolo():
        """Create a minimal fake YOLO model with a sequential backbone."""

        class FakeConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        class FakeConcat(torch.nn.Module):
            """Marker for backbone boundary detection."""

            def forward(self, x):
                return x

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Sequential(
                    FakeConv(),
                    FakeConcat(),
                )

        class FakeYolo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeModel()

        return FakeYolo()

    def test_output_shape(self):
        yolo = self._make_fake_yolo()
        extractor = YoloRoiExtractor(yolo, d_model=64, roi_size=7)

        image = torch.randn(1, 3, 64, 64)
        bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])

        features = extractor(image, bboxes, image_size=(64, 64))
        assert features.shape == (2, 64)

    def test_no_detections(self):
        yolo = self._make_fake_yolo()
        extractor = YoloRoiExtractor(yolo, d_model=64, roi_size=7)

        image = torch.randn(1, 3, 64, 64)
        bboxes = torch.zeros(0, 4)

        features = extractor(image, bboxes, image_size=(64, 64))
        assert features.shape == (0, 64)

    def test_different_d_model(self):
        yolo = self._make_fake_yolo()
        extractor = YoloRoiExtractor(yolo, d_model=256, roi_size=7)

        image = torch.randn(1, 3, 64, 64)
        bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])

        features = extractor(image, bboxes, image_size=(64, 64))
        assert features.shape == (1, 256)
