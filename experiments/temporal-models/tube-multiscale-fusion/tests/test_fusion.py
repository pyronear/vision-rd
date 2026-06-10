"""Unit tests for FusionModule."""

from __future__ import annotations

import torch

from tube_multiscale_fusion.fusion import FusionModule


def _build(d_fusion: int = 32) -> FusionModule:
    return FusionModule(
        global_dim=64,
        local_dim=48,
        d_fusion=d_fusion,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
    )


def test_fusion_output_shape() -> None:
    fusion = _build()
    g = torch.randn(2, 64)
    locals_ = torch.randn(2, 7, 48)
    mask = torch.ones(2, 7, dtype=torch.bool)
    out = fusion(g, locals_, mask)
    assert out.shape == (2, 32)


def test_fusion_handles_all_masked_row_without_nan() -> None:
    fusion = _build()
    g = torch.randn(2, 64)
    locals_ = torch.randn(2, 7, 48)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1] = False  # second sample has no valid locals.
    fusion.eval()
    with torch.no_grad():
        out = fusion(g, locals_, mask)
    assert torch.isfinite(out).all()


def test_fusion_masking_changes_output() -> None:
    fusion = _build()
    g = torch.randn(1, 64)
    locals_ = torch.randn(1, 8, 48)
    full = torch.ones(1, 8, dtype=torch.bool)
    half = torch.zeros(1, 8, dtype=torch.bool)
    half[:, :4] = True
    fusion.eval()
    with torch.no_grad():
        out_full = fusion(g, locals_, full)
        out_half = fusion(g, locals_, half)
    assert not torch.allclose(out_full, out_half)


def test_fusion_identity_projection_when_dims_match() -> None:
    fusion = FusionModule(
        global_dim=32,
        local_dim=32,
        d_fusion=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
    )
    assert isinstance(fusion.global_proj, torch.nn.Identity)
    assert isinstance(fusion.local_proj, torch.nn.Identity)
