"""End-to-end tests for TubeMultiscaleClassifier.

These tests substitute a tiny backbone via monkey-patching when possible to
avoid downloading large DINOv2 weights. The full DINOv2 path lives in
``test_global_branch.py`` behind ``@pytest.mark.slow``.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from tube_multiscale_fusion import classifier as classifier_mod
from tube_multiscale_fusion.classifier import TubeMultiscaleClassifier


class _TinyBackbone(nn.Module):
    """Drop-in replacement for TimmBackbone with the same interface."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
        global_pool: str = "token",
        img_size: int | None = 224,
    ) -> None:
        super().__init__()
        self.feat_dim = 32
        self.finetune = finetune
        self.name = name
        self.conv = nn.Conv2d(3, self.feat_dim, kernel_size=img_size or 224)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x).flatten(2).mean(dim=-1)


@pytest.fixture
def tiny_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap TimmBackbone for a tiny conv-based stand-in for speed."""
    monkeypatch.setattr(
        "tube_multiscale_fusion.global_branch.TimmBackbone", _TinyBackbone
    )


def _build_classifier() -> TubeMultiscaleClassifier:
    return TubeMultiscaleClassifier(
        backbone="tiny",
        finetune=False,
        finetune_last_n_blocks=0,
        max_frames=16,
        global_aggregator_num_layers=1,
        global_aggregator_num_heads=4,
        global_aggregator_ffn_dim=64,
        global_aggregator_dropout=0.0,
        grid_size=4,
        cell_size=56,
        tube_length=4,
        temporal_stride=2,
        local_t_kernel=2,
        local_h_patch=14,
        local_w_patch=14,
        local_embed_dim=48,
        local_num_layers=1,
        local_num_heads=4,
        local_ffn_dim=96,
        local_dropout=0.0,
        d_fusion=64,
        fusion_num_layers=1,
        fusion_num_heads=4,
        fusion_ffn_dim=128,
        fusion_dropout=0.0,
        head_hidden_dim=32,
        head_dropout=0.0,
        img_size=224,
        pretrained=False,
    )


def test_classifier_forward_shape(tiny_backbone: None) -> None:
    del tiny_backbone
    model = _build_classifier()
    patches = torch.randn(2, 16, 3, 224, 224)
    mask = torch.ones(2, 16, dtype=torch.bool)
    logits = model(patches, mask)
    assert logits.shape == (2,)


def test_classifier_handles_padding(tiny_backbone: None) -> None:
    del tiny_backbone
    model = _build_classifier()
    patches = torch.randn(1, 16, 3, 224, 224)
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[:, :8] = True
    logits = model(patches, mask)
    assert logits.shape == (1,)
    assert torch.isfinite(logits).all()


def test_classifier_gradient_flows_through_both_branches(
    tiny_backbone: None,
) -> None:
    del tiny_backbone
    model = _build_classifier()
    patches = torch.randn(1, 16, 3, 224, 224, requires_grad=False)
    mask = torch.ones(1, 16, dtype=torch.bool)
    target = torch.tensor([1.0])
    logits = model(patches, mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    # Local branch encoder params must have grads.
    encoder_grads = [
        p.grad for p in model.local_branch.encoder.parameters() if p.requires_grad
    ]
    assert encoder_grads and all(g is not None for g in encoder_grads)

    # Fusion params must have grads.
    fusion_grads = [
        p.grad for p in model.fusion.parameters() if p.requires_grad
    ]
    assert fusion_grads and all(g is not None for g in fusion_grads)

    # Global aggregator params must have grads.
    agg_grads = [
        p.grad
        for p in model.global_branch.aggregator.parameters()
        if p.requires_grad
    ]
    assert agg_grads and all(g is not None for g in agg_grads)


def test_classifier_module_referenced(tiny_backbone: None) -> None:
    """Sanity: the classifier module is what we expect."""
    del tiny_backbone
    assert hasattr(classifier_mod, "TubeMultiscaleClassifier")
