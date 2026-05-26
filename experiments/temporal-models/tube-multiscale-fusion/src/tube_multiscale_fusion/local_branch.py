"""Local branch: spatio-temporal tube extraction + per-tube video transformer.

A 16-frame sequence of ``(3, H, W)`` patches is decomposed into a grid of
``grid_size x grid_size`` spatial cells of side ``cell_size``, then sliced
along time into overlapping windows of length ``tube_length`` with stride
``temporal_stride``. Each resulting ``(3, tube_length, cell_size, cell_size)``
tube is embedded into a single vector by a small video transformer
(tubelet patch embed -> learnable [CLS] + pos embed -> self-attention).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def n_temporal_windows(num_frames: int, tube_length: int, temporal_stride: int) -> int:
    """Number of fully-contained sliding windows over ``num_frames``."""
    if num_frames < tube_length:
        return 0
    return (num_frames - tube_length) // temporal_stride + 1


def extract_tubes(
    patches: Tensor,
    grid_size: int,
    cell_size: int,
    tube_length: int,
    temporal_stride: int,
) -> Tensor:
    """Decompose ``(B, T, 3, H, W)`` patches into spatio-temporal tubes.

    Args:
        patches: ``(B, T, 3, H, W)`` RGB tensors. Requires
            ``H == W == grid_size * cell_size``.
        grid_size: Number of spatial cells per axis (e.g. 4).
        cell_size: Side length of each spatial cell in pixels.
        tube_length: Number of frames per temporal window.
        temporal_stride: Frame stride between consecutive windows.

    Returns:
        ``(B, N_tubes, 3, tube_length, cell_size, cell_size)`` where
        ``N_tubes = n_windows * grid_size**2``. Tubes are laid out as
        ``[(window_0, cell_(0,0)), (window_0, cell_(0,1)), ...]``.
    """
    b, t, c, h, w = patches.shape
    expected = grid_size * cell_size
    if h != expected or w != expected:
        raise ValueError(
            f"extract_tubes expects H = W = grid_size * cell_size = {expected}, "
            f"got H={h}, W={w}"
        )
    n_windows = n_temporal_windows(t, tube_length, temporal_stride)
    if n_windows == 0:
        raise ValueError(
            f"extract_tubes: T={t} is shorter than tube_length={tube_length}"
        )

    # Temporal sliding windows: (B, n_windows, 3, H, W, tube_length)
    temporal = patches.unfold(1, tube_length, temporal_stride)
    # Move the unfold-appended tube_length axis next to the channel dim:
    # -> (B, n_windows, 3, tube_length, H, W)
    temporal = temporal.permute(0, 1, 2, 5, 3, 4).contiguous()

    # Spatial unfold along H and W:
    # -> (B, n_windows, 3, tube_length, grid_size, grid_size, cell_size, cell_size)
    spatial = temporal.unfold(4, cell_size, cell_size).unfold(5, cell_size, cell_size)

    # Collapse spatial grid -> tube axis. Reorder so each (window, cell) is one tube:
    # (B, n_windows, grid_size, grid_size, 3, tube_length, cell_size, cell_size)
    spatial = spatial.permute(0, 1, 4, 5, 2, 3, 6, 7).contiguous()
    n_tubes = n_windows * grid_size * grid_size
    return spatial.view(b, n_tubes, c, tube_length, cell_size, cell_size)


def tube_validity_mask(
    mask: Tensor,
    grid_size: int,
    tube_length: int,
    temporal_stride: int,
) -> Tensor:
    """Derive ``(B, N_tubes)`` boolean validity mask from a ``(B, T)`` frame mask.

    A tube is valid iff at least one of its constituent frames is real.
    Cells inside the same temporal window share validity (purely temporal
    derivation).
    """
    b, t = mask.shape
    n_windows = n_temporal_windows(t, tube_length, temporal_stride)
    # (B, n_windows, tube_length)
    win_mask = mask.unfold(1, tube_length, temporal_stride)
    per_window = win_mask.any(dim=-1)  # (B, n_windows)
    # Broadcast to all spatial cells in the window: (B, n_windows, grid_size**2)
    per_tube = per_window.unsqueeze(-1).expand(b, n_windows, grid_size * grid_size)
    return per_tube.reshape(b, n_windows * grid_size * grid_size).contiguous()


class _SelfAttnBlock(nn.Module):
    """Pre-norm self-attention + FFN block (batch_first)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class LocalTubeEncoder(nn.Module):
    """Encode each ``(3, T_t, h, w)`` tube into one embedding via a video transformer.

    Architecture per tube:
        - Tubelet patch embed: ``Conv3d`` with stride matching its kernel,
          producing a grid of spatio-temporal tokens.
        - Prepend a learnable ``[CLS]`` token; add learned positional embeddings.
        - ``num_layers`` self-attention blocks.
        - Return the final ``[CLS]`` state as the tube embedding.
    """

    def __init__(
        self,
        tube_length: int,
        cell_size: int,
        t_kernel: int,
        h_patch: int,
        w_patch: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if tube_length % t_kernel != 0:
            raise ValueError(
                f"tube_length ({tube_length}) must be divisible "
                f"by t_kernel ({t_kernel})"
            )
        if cell_size % h_patch != 0 or cell_size % w_patch != 0:
            raise ValueError(
                f"cell_size ({cell_size}) must be divisible by h_patch ({h_patch}) "
                f"and w_patch ({w_patch})"
            )
        self.tubelet_embed = nn.Conv3d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=(t_kernel, h_patch, w_patch),
            stride=(t_kernel, h_patch, w_patch),
        )
        n_t = tube_length // t_kernel
        n_h = cell_size // h_patch
        n_w = cell_size // w_patch
        self.n_tokens = n_t * n_h * n_w
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                _SelfAttnBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tubes: Tensor) -> Tensor:
        """``tubes: (B, N_tubes, 3, T_t, h, w) -> (B, N_tubes, embed_dim)``."""
        b, n, c, t, h, w = tubes.shape
        x = tubes.reshape(b * n, c, t, h, w)
        x = self.tubelet_embed(x)  # (B*N, D, n_t, n_h, n_w)
        x = x.flatten(2).transpose(1, 2)  # (B*N, n_tokens, D)
        cls = self.cls_token.expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        cls_out = self.norm(x[:, 0, :])
        return cls_out.reshape(b, n, self.embed_dim)


class LocalBranch(nn.Module):
    """Full local branch: tube extraction + per-tube encoder + validity mask."""

    def __init__(
        self,
        grid_size: int,
        cell_size: int,
        tube_length: int,
        temporal_stride: int,
        t_kernel: int,
        h_patch: int,
        w_patch: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.tube_length = tube_length
        self.temporal_stride = temporal_stride
        self.encoder = LocalTubeEncoder(
            tube_length=tube_length,
            cell_size=cell_size,
            t_kernel=t_kernel,
            h_patch=h_patch,
            w_patch=w_patch,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.embed_dim = embed_dim

    def forward(self, patches: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        tubes = extract_tubes(
            patches,
            grid_size=self.grid_size,
            cell_size=self.cell_size,
            tube_length=self.tube_length,
            temporal_stride=self.temporal_stride,
        )
        tube_feats = self.encoder(tubes)
        tube_mask = tube_validity_mask(
            mask,
            grid_size=self.grid_size,
            tube_length=self.tube_length,
            temporal_stride=self.temporal_stride,
        )
        return tube_feats, tube_mask
