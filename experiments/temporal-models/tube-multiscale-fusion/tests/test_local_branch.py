"""Unit tests for the local branch: tube extraction + encoder."""

from __future__ import annotations

import pytest
import torch

from tube_multiscale_fusion.local_branch import (
    LocalBranch,
    LocalTubeEncoder,
    extract_tubes,
    n_temporal_windows,
    tube_validity_mask,
)


def test_n_temporal_windows_stride2() -> None:
    assert n_temporal_windows(16, 4, 2) == 7


def test_n_temporal_windows_stride4_no_overlap() -> None:
    assert n_temporal_windows(16, 4, 4) == 4


def test_extract_tubes_shape_default() -> None:
    b, t, c, h, w = 2, 16, 3, 224, 224
    patches = torch.randn(b, t, c, h, w)
    tubes = extract_tubes(
        patches, grid_size=4, cell_size=56, tube_length=4, temporal_stride=2
    )
    # n_windows=7, n_cells=16 -> 112 tubes
    assert tubes.shape == (b, 7 * 16, c, 4, 56, 56)


def test_extract_tubes_shape_no_overlap() -> None:
    patches = torch.randn(1, 16, 3, 224, 224)
    tubes = extract_tubes(
        patches, grid_size=4, cell_size=56, tube_length=4, temporal_stride=4
    )
    assert tubes.shape == (1, 4 * 16, 3, 4, 56, 56)


def test_extract_tubes_content_first_cell_first_window() -> None:
    """Tube (window=0, cell=(0,0)) must equal patches[:, 0:4, :, :56, :56]."""
    patches = torch.randn(1, 16, 3, 224, 224)
    tubes = extract_tubes(
        patches, grid_size=4, cell_size=56, tube_length=4, temporal_stride=2
    )
    expected = patches[:, 0:4, :, :56, :56].permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    # Tube index 0 = window 0, cell (0, 0).
    assert torch.allclose(tubes[:, 0], expected)


def test_extract_tubes_content_second_window_first_cell() -> None:
    """Window 1 with stride=2 covers frames 2..5."""
    patches = torch.randn(1, 16, 3, 224, 224)
    tubes = extract_tubes(
        patches, grid_size=4, cell_size=56, tube_length=4, temporal_stride=2
    )
    # n_cells_per_window = 16, so window 1 starts at tube index 16.
    expected = patches[:, 2:6, :, :56, :56].permute(0, 2, 1, 3, 4)
    assert torch.allclose(tubes[:, 16], expected)


def test_extract_tubes_rejects_mismatched_resolution() -> None:
    patches = torch.randn(1, 16, 3, 224, 224)
    with pytest.raises(ValueError, match="grid_size"):
        extract_tubes(
            patches, grid_size=3, cell_size=56, tube_length=4, temporal_stride=2
        )


def test_tube_validity_mask_all_valid() -> None:
    mask = torch.ones(2, 16, dtype=torch.bool)
    tube_mask = tube_validity_mask(mask, grid_size=4, tube_length=4, temporal_stride=2)
    assert tube_mask.shape == (2, 7 * 16)
    assert tube_mask.all()


def test_tube_validity_mask_partial_padding() -> None:
    """Frames 8..15 padded -> windows covering only padded frames must be invalid."""
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[:, :8] = True
    tube_mask = tube_validity_mask(mask, grid_size=4, tube_length=4, temporal_stride=2)
    # Windows: [0:4], [2:6], [4:8], [6:10], [8:12], [10:14], [12:16]
    # Valid windows: 0..3 (touch frames < 8); invalid: 4..6 (only padded).
    per_window = tube_mask.reshape(1, 7, 16).any(dim=-1)
    assert per_window[0].tolist() == [True, True, True, True, False, False, False]


def test_tube_validity_mask_zero_valid_frames() -> None:
    mask = torch.zeros(1, 16, dtype=torch.bool)
    tube_mask = tube_validity_mask(mask, grid_size=4, tube_length=4, temporal_stride=2)
    assert tube_mask.shape == (1, 112)
    assert not tube_mask.any()


def test_local_tube_encoder_shape() -> None:
    encoder = LocalTubeEncoder(
        tube_length=4,
        cell_size=56,
        t_kernel=2,
        h_patch=14,
        w_patch=14,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
        dropout=0.0,
    )
    tubes = torch.randn(2, 5, 3, 4, 56, 56)
    out = encoder(tubes)
    assert out.shape == (2, 5, 128)


def test_local_tube_encoder_token_count() -> None:
    """Tubelet conv 2x14x14 over 4x56x56 -> 2 * 4 * 4 = 32 tokens (+1 CLS)."""
    encoder = LocalTubeEncoder(
        tube_length=4,
        cell_size=56,
        t_kernel=2,
        h_patch=14,
        w_patch=14,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        ffn_dim=128,
        dropout=0.0,
    )
    assert encoder.n_tokens == 32


def test_local_branch_end_to_end_shape_and_mask() -> None:
    branch = LocalBranch(
        grid_size=4,
        cell_size=56,
        tube_length=4,
        temporal_stride=2,
        t_kernel=2,
        h_patch=14,
        w_patch=14,
        embed_dim=64,
        num_layers=1,
        num_heads=4,
        ffn_dim=128,
        dropout=0.0,
    )
    patches = torch.randn(2, 16, 3, 224, 224)
    mask = torch.ones(2, 16, dtype=torch.bool)
    tube_feats, tube_mask = branch(patches, mask)
    assert tube_feats.shape == (2, 112, 64)
    assert tube_mask.shape == (2, 112)
    assert tube_mask.all()


def test_local_branch_rejects_invalid_tube_length() -> None:
    with pytest.raises(ValueError, match="tube_length"):
        LocalTubeEncoder(
            tube_length=4,
            cell_size=56,
            t_kernel=3,  # 4 % 3 != 0
            h_patch=14,
            w_patch=14,
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            ffn_dim=128,
            dropout=0.0,
        )
