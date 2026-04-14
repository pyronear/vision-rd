"""Tests for TemporalSmokeClassifier and its components."""

import pytest
import torch

from smokeynet_adapted.temporal_classifier import (
    GRUHead,
    MeanPoolHead,
    TemporalSmokeClassifier,
    TimmBackbone,
)


def test_timm_backbone_outputs_features_per_frame():
    bb = TimmBackbone(name="resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = bb(x)
    assert out.shape == (2, bb.feat_dim)
    assert bb.feat_dim == 512


def test_timm_backbone_has_no_trainable_params():
    bb = TimmBackbone(name="resnet18", pretrained=False)
    trainable = [p for p in bb.parameters() if p.requires_grad]
    assert trainable == []


def test_timm_backbone_stays_in_eval_mode_after_train_call():
    bb = TimmBackbone(name="resnet18", pretrained=False)
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


def test_classifier_mean_pool_forward_shape():
    clf = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="mean_pool",
        hidden_dim=64,
        pretrained=False,
    )
    patches = torch.randn(2, 5, 3, 224, 224)
    mask = torch.ones(2, 5, dtype=torch.bool)
    logits = clf(patches, mask)
    assert logits.shape == (2,)


def test_classifier_gru_forward_shape():
    clf = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=64,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
    )
    patches = torch.randn(2, 5, 3, 224, 224)
    mask = torch.ones(2, 5, dtype=torch.bool)
    logits = clf(patches, mask)
    assert logits.shape == (2,)


def test_classifier_only_head_params_are_trainable():
    clf = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=64,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
    )
    trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
    assert all(n.startswith("head.") for n in trainable)
    assert any(n.startswith("head.gru") for n in trainable)


def test_classifier_unknown_arch_raises():
    with pytest.raises(ValueError, match="arch"):
        TemporalSmokeClassifier(
            backbone="resnet18",
            arch="lstm",
            hidden_dim=64,
            pretrained=False,
        )


def test_classifier_handles_padded_batches():
    clf = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
    )
    patches = torch.randn(3, 20, 3, 224, 224)
    mask = torch.zeros(3, 20, dtype=torch.bool)
    mask[0, :20] = True
    mask[1, :10] = True
    mask[2, :3] = True
    logits = clf(patches, mask)
    assert logits.shape == (3,)


def test_timm_backbone_frozen_default_has_no_trainable_params():
    bb = TimmBackbone(name="resnet18", pretrained=False)
    assert [p for p in bb.parameters() if p.requires_grad] == []


def test_timm_backbone_finetune_resnet18_unfreezes_only_layer4():
    bb = TimmBackbone(
        name="resnet18",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=1,
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names, "expected some trainable params"
    assert all(".layer4." in n for n in trainable_names), trainable_names


def test_timm_backbone_finetune_resnet18_n2_unfreezes_layer3_and_layer4():
    bb = TimmBackbone(
        name="resnet18",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=2,
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all((".layer3." in n) or (".layer4." in n) for n in trainable_names)


def test_timm_backbone_finetune_train_mode_propagates_resnet18():
    bb = TimmBackbone(
        name="resnet18",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=1,
    )
    bb.train()
    assert bb.backbone.layer4.training is True
    assert bb.backbone.layer1.training is False


def test_timm_backbone_finetune_convnext_tiny_unfreezes_only_last_stage():
    bb = TimmBackbone(
        name="convnext_tiny",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=1,
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names
    # timm's convnext wraps blocks under `stages.<i>.*`; last stage is index 3
    assert all(".stages.3." in n for n in trainable_names), trainable_names


def test_timm_backbone_finetune_unsupported_family_raises():
    with pytest.raises(NotImplementedError) as exc:
        TimmBackbone(
            name="vit_small_patch16_224",
            pretrained=False,
            finetune=True,
            finetune_last_n_blocks=1,
        )
    msg = str(exc.value)
    assert "vit_small_patch16_224" in msg
    # Error message should mention the backbone's top-level children so the
    # operator knows which stage names are available.
    assert "children" in msg.lower()


def test_timm_backbone_frozen_forward_matches_no_grad_eval():
    torch.manual_seed(0)
    bb = TimmBackbone(name="resnet18", pretrained=False, finetune=False)
    x = torch.randn(2, 3, 224, 224)
    # Put the module into train mode; frozen path must still force eval().
    bb.train()
    assert bb.backbone.training is False
    out_a = bb(x)
    # Independent reference: call the underlying timm model directly in eval/no_grad.
    bb.backbone.eval()
    with torch.no_grad():
        out_b = bb.backbone(x)
    assert torch.allclose(out_a, out_b, atol=1e-6)
