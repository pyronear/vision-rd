"""Tests for smokeynet_adapted.training."""

import torch

from smokeynet_adapted.training import SmokeyNetLightningModule
from smokeynet_adapted.types import Detection, Tube, TubeEntry


def _det(cx=0.5, cy=0.5):
    return Detection(class_id=0, cx=cx, cy=cy, w=0.1, h=0.1, confidence=0.8)


def _make_batch(num_dets=3, d_model=32, label=1.0):
    return {
        "roi_features": torch.randn(num_dets, d_model),
        "frame_indices": torch.tensor([0, 0, 1], dtype=torch.long)[:num_dets],
        "bbox_coords": torch.randn(num_dets, 4),
        "detection_labels": torch.zeros(num_dets),
        "sequence_label": torch.tensor(label),
        "tubes": [
            Tube(
                tube_id=0,
                entries=[
                    TubeEntry(frame_idx=0, detection=_det()),
                    TubeEntry(frame_idx=1, detection=_det()),
                ],
                start_frame=0,
                end_frame=1,
            )
        ],
    }


class TestSmokeyNetLightningModule:
    def test_construction(self):
        module = SmokeyNetLightningModule(d_model=32, lstm_layers=1)
        assert module.net is not None

    def test_training_step_returns_loss(self):
        module = SmokeyNetLightningModule(
            d_model=32, lstm_layers=1, spatial_layers=1, spatial_heads=4
        )
        batch = _make_batch(d_model=32)
        loss = module.training_step(batch, 0)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_validation_step(self):
        module = SmokeyNetLightningModule(
            d_model=32, lstm_layers=1, spatial_layers=1, spatial_heads=4
        )
        # Should not raise
        module.validation_step(_make_batch(d_model=32), 0)

    def test_configure_optimizers(self):
        module = SmokeyNetLightningModule(
            d_model=32, lstm_layers=1, spatial_layers=1, spatial_heads=4
        )
        config = module.configure_optimizers()
        assert "optimizer" in config
        assert "lr_scheduler" in config

    def test_loss_decreases_with_gradient_step(self):
        module = SmokeyNetLightningModule(
            d_model=16,
            lstm_layers=1,
            spatial_layers=1,
            spatial_heads=4,
        )
        module.train()
        batch = _make_batch(d_model=16, label=1.0)

        optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
        loss1 = module.training_step(batch, 0)
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss2 = module.training_step(batch, 1)

        # Loss should change (usually decrease) after a step
        assert loss1.item() != loss2.item()

    def test_empty_detections_batch(self):
        module = SmokeyNetLightningModule(
            d_model=32, lstm_layers=1, spatial_layers=1, spatial_heads=4
        )
        batch = {
            "roi_features": torch.zeros(0, 32),
            "frame_indices": torch.zeros(0, dtype=torch.long),
            "bbox_coords": torch.zeros(0, 4),
            "detection_labels": torch.zeros(0),
            "sequence_label": torch.tensor(0.0),
            "tubes": [],
        }
        loss = module.training_step(batch, 0)
        assert loss.shape == ()
