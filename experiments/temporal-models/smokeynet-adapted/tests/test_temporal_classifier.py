"""Tests for TemporalSmokeClassifier and its components."""

import torch

from smokeynet_adapted.temporal_classifier import (
    FrozenTimmBackbone,
    GRUHead,
    MeanPoolHead,
)


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


def test_mean_pool_head_returns_logits_per_batch():
    head = MeanPoolHead(feat_dim=512, hidden_dim=128)
    feats = torch.randn(4, 20, 512)
    mask = torch.ones(4, 20, dtype=torch.bool)
    logits = head(feats, mask)
    assert logits.shape == (4,)


def test_mean_pool_head_respects_mask():
    head = MeanPoolHead(feat_dim=4, hidden_dim=4)
    base = torch.zeros(20, 4)
    base[0] = 1.0
    base[1] = 2.0
    a = base.clone()
    b = base.clone()
    b[2:] = 999.0
    feats = torch.stack([a, b])
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[:, :2] = True
    logits = head(feats, mask)
    assert torch.allclose(logits[0], logits[1], atol=1e-5)


def test_gru_head_returns_logits_per_batch():
    head = GRUHead(feat_dim=512, hidden_dim=128, num_layers=1, bidirectional=False)
    feats = torch.randn(3, 20, 512)
    mask = torch.ones(3, 20, dtype=torch.bool)
    logits = head(feats, mask)
    assert logits.shape == (3,)


def test_gru_head_respects_mask_via_packed_sequences():
    head = GRUHead(feat_dim=4, hidden_dim=4, num_layers=1, bidirectional=False)
    real = torch.randn(2, 4)
    a = torch.zeros(20, 4)
    a[:2] = real
    b = a.clone()
    b[2:] = 1e6
    feats = torch.stack([a, b])
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[:, :2] = True
    logits = head(feats, mask)
    assert torch.allclose(logits[0], logits[1], atol=1e-4)


def test_gru_head_bidirectional_doubles_hidden_then_projects():
    head = GRUHead(feat_dim=8, hidden_dim=4, num_layers=1, bidirectional=True)
    feats = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    logits = head(feats, mask)
    assert logits.shape == (2,)
