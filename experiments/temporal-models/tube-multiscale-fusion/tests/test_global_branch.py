"""Unit tests for the global branch: TimmBackbone wrapper + GlobalAggregator."""

from __future__ import annotations

import pytest
import torch

from tube_multiscale_fusion.global_branch import (
    GlobalAggregator,
    GlobalBranch,
)


def test_global_aggregator_shape() -> None:
    agg = GlobalAggregator(
        feat_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        dropout=0.0,
        max_frames=16,
    )
    feats = torch.randn(2, 16, 64)
    mask = torch.ones(2, 16, dtype=torch.bool)
    out = agg(feats, mask)
    assert out.shape == (2, 64)


def test_global_aggregator_masking_changes_output() -> None:
    """Masking out frames should change the [CLS] readout."""
    agg = GlobalAggregator(
        feat_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        max_frames=8,
    )
    feats = torch.randn(1, 8, 32)
    full_mask = torch.ones(1, 8, dtype=torch.bool)
    partial = torch.zeros(1, 8, dtype=torch.bool)
    partial[:, :4] = True
    with torch.no_grad():
        out_full = agg(feats, full_mask)
        out_partial = agg(feats, partial)
    assert not torch.allclose(out_full, out_partial)


def test_global_aggregator_rejects_too_many_frames() -> None:
    agg = GlobalAggregator(
        feat_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        max_frames=8,
    )
    feats = torch.randn(1, 16, 32)
    mask = torch.ones(1, 16, dtype=torch.bool)
    with pytest.raises(ValueError, match="max_frames"):
        agg(feats, mask)


@pytest.mark.slow
def test_global_branch_with_dinov2_frozen() -> None:
    """Smoke test the full branch with the real DINOv2 backbone (downloads weights)."""
    branch = GlobalBranch(
        backbone="vit_small_patch14_dinov2.lvd142m",
        finetune=False,
        finetune_last_n_blocks=0,
        max_frames=16,
        aggregator_num_layers=1,
        aggregator_num_heads=6,
        aggregator_ffn_dim=384,
        aggregator_dropout=0.0,
    )
    assert branch.feat_dim == 384
    # All backbone params should be frozen.
    for p in branch.backbone.backbone.parameters():
        assert not p.requires_grad
    patches = torch.randn(1, 16, 3, 224, 224)
    mask = torch.ones(1, 16, dtype=torch.bool)
    branch.eval()
    with torch.no_grad():
        out = branch(patches, mask)
    assert out.shape == (1, 384)


@pytest.mark.slow
def test_global_branch_finetune_last_block_unfrozen() -> None:
    branch = GlobalBranch(
        backbone="vit_small_patch14_dinov2.lvd142m",
        finetune=True,
        finetune_last_n_blocks=1,
        max_frames=16,
        aggregator_num_layers=1,
        aggregator_num_heads=6,
        aggregator_ffn_dim=384,
        aggregator_dropout=0.0,
    )
    blocks = list(branch.backbone.backbone.blocks)
    # Last block trainable, second-to-last block frozen.
    assert all(p.requires_grad for p in blocks[-1].parameters())
    assert all(not p.requires_grad for p in blocks[-2].parameters())
    # Aggregator always trainable.
    assert all(p.requires_grad for p in branch.aggregator.parameters())
