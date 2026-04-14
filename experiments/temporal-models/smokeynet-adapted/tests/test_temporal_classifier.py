"""Tests for TemporalSmokeClassifier and its components."""

import torch

from smokeynet_adapted.temporal_classifier import FrozenTimmBackbone


def test_frozen_timm_backbone_outputs_features_per_frame():
    bb = FrozenTimmBackbone(name="resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = bb(x)
    assert out.shape == (2, bb.feat_dim)
    assert bb.feat_dim == 512


def test_frozen_timm_backbone_has_no_trainable_params():
    bb = FrozenTimmBackbone(name="resnet18", pretrained=False)
    trainable = [p for p in bb.parameters() if p.requires_grad]
    assert trainable == []


def test_frozen_timm_backbone_stays_in_eval_mode_after_train_call():
    bb = FrozenTimmBackbone(name="resnet18", pretrained=False)
    bb.train()
    assert not bb.backbone.training
