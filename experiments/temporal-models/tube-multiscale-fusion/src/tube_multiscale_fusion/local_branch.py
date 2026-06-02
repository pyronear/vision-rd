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
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.video import R3D_18_Weights, r3d_18

_KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
_KINETICS_STD = (0.22803, 0.22145, 0.216989)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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


class _TubeEncoderBase(nn.Module):
    """Flattens the ``(B, N, …)`` tube axis, calls ``_encode``, reshapes back."""

    embed_dim: int

    def forward(self, tubes: Tensor) -> Tensor:
        b, n, c, t, h, w = tubes.shape
        feat = self._encode(tubes.reshape(b * n, c, t, h, w))  # (B*N, D)
        return feat.reshape(b, n, self.embed_dim)

    def _encode(self, x: Tensor) -> Tensor:  # (B', 3, T, h, w) -> (B', D)
        raise NotImplementedError


class Resnet3DTubeEncoder(_TubeEncoderBase):
    """Kinetics-400 pretrained r3d_18 applied to each tube."""

    def __init__(
        self, embed_dim: int, pretrained: bool = True, clip_spatial_size: int = 112
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.clip_spatial_size = clip_spatial_size
        self.backbone = r3d_18(
            weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None
        )
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj: nn.Module = (
            nn.Identity() if embed_dim == feat_dim else nn.Linear(feat_dim, embed_dim)
        )
        im_mean, im_std = torch.tensor(_IMAGENET_MEAN), torch.tensor(_IMAGENET_STD)
        ki_mean, ki_std = torch.tensor(_KINETICS_MEAN), torch.tensor(_KINETICS_STD)
        self.register_buffer(
            "_scale", (im_std / ki_std).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "_bias",
            ((im_mean - ki_mean) / ki_std).view(1, 3, 1, 1, 1),
            persistent=False,
        )

    def _encode(self, x: Tensor) -> Tensor:
        x = x * self._scale + self._bias  # ImageNet -> Kinetics
        s = self.clip_spatial_size
        if x.shape[-1] != s or x.shape[-2] != s:
            x = F.interpolate(
                x, size=(x.shape[2], s, s), mode="trilinear", align_corners=False
            )
        return self.proj(self.backbone(x))


class ViViTTubeEncoder(_TubeEncoderBase):
    """ViViT-style tubelet embedding + factorised spatial-then-temporal attention."""

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
        self.embed_dim = embed_dim
        self.tubelet = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(t_kernel, h_patch, w_patch),
            stride=(t_kernel, h_patch, w_patch),
        )
        n_t = tube_length // t_kernel
        n_s = (cell_size // h_patch) * (cell_size // w_patch)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, n_s, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, n_t, 1, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.spatial_blocks = nn.ModuleList(
            [
                _SelfAttnBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.temporal_blocks = nn.ModuleList(
            [
                _SelfAttnBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def _encode(self, x: Tensor) -> Tensor:
        z = self.tubelet(x)  # (B', D, T', Hp, Wp)
        b2, d, tp, hp, wp = z.shape
        z = z.flatten(3).permute(0, 2, 3, 1)  # (B', T', S, D)
        z = z + self.spatial_pos + self.temporal_pos
        s = hp * wp
        for sblk, tblk in zip(self.spatial_blocks, self.temporal_blocks, strict=True):
            z = sblk(z.reshape(b2 * tp, s, d)).reshape(b2, tp, s, d)
            z = (
                tblk(z.permute(0, 2, 1, 3).reshape(b2 * s, tp, d))
                .reshape(b2, s, tp, d)
                .permute(0, 2, 1, 3)
            )
        return self.norm(z.reshape(b2, tp * s, d).mean(dim=1))


class _ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=kernel // 2)

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMTubeEncoder(_TubeEncoderBase):
    """Per-frame 2D CNN stem + ConvLSTM recurrence over the tube's frames."""

    def __init__(
        self, embed_dim: int, base_channels: int = 64, hidden_channels: int = 128
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_channels = hidden_channels
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 3, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.GELU(),
            nn.Conv2d(c * 2, hidden_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.cell = _ConvLSTMCell(hidden_channels, hidden_channels)
        self.proj = nn.Linear(hidden_channels, embed_dim)

    def _encode(self, x: Tensor) -> Tensor:
        b2, c, t, h, w = x.shape
        feat = self.stem(x.transpose(1, 2).reshape(b2 * t, c, h, w))
        _, cc, hh, ww = feat.shape
        feat = feat.reshape(b2, t, cc, hh, ww)
        h_t = feat.new_zeros(b2, self.hidden_channels, hh, ww)
        c_t = feat.new_zeros(b2, self.hidden_channels, hh, ww)
        for ti in range(t):
            h_t, c_t = self.cell(feat[:, ti], h_t, c_t)
        return self.proj(F.adaptive_avg_pool2d(h_t, 1).flatten(1))


def _temporal_shift(x: Tensor, n_frames: int, fold_div: int = 4) -> Tensor:
    """Temporal shift on ``(B'*T, C, H, W)`` (TSM): part of channels shift in time."""
    bt, c, h, w = x.shape
    b = bt // n_frames
    x = x.view(b, n_frames, c, h, w)
    fold = c // fold_div
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]  # shift toward future
    out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # shift toward past
    out[:, :, 2 * fold :] = x[:, :, 2 * fold :]  # unchanged
    return out.view(bt, c, h, w)


class _TSMBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_frames: int) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(_temporal_shift(x, self.n_frames))


class TSMTubeEncoder(_TubeEncoderBase):
    """Temporal Shift Module 2D CNN + temporal mean pooling."""

    def __init__(
        self, tube_length: int, embed_dim: int, base_channels: int = 64
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.tube_length = tube_length
        c = base_channels
        self.blocks = nn.Sequential(
            _TSMBlock(3, c, tube_length),
            _TSMBlock(c, c * 2, tube_length),
            _TSMBlock(c * 2, c * 4, tube_length),
        )
        self.proj = nn.Linear(c * 4, embed_dim)

    def _encode(self, x: Tensor) -> Tensor:
        b2, c, t, h, w = x.shape
        feat = self.blocks(x.transpose(1, 2).reshape(b2 * t, c, h, w))
        pooled = (
            F.adaptive_avg_pool2d(feat, 1).flatten(1).reshape(b2, t, -1).mean(dim=1)
        )
        return self.proj(pooled)


def build_tube_encoder(
    kind: str,
    *,
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
    base_channels: int = 64,
    resnet_pretrained: bool = True,
) -> nn.Module:
    """Construct a per-tube encoder by name (see SPATIAL_COMPARISON.md)."""
    if kind == "tubelet_transformer":
        return LocalTubeEncoder(
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
    if kind == "resnet3d":
        return Resnet3DTubeEncoder(
            embed_dim=embed_dim,
            pretrained=resnet_pretrained,
            clip_spatial_size=cell_size,
        )
    if kind == "vivit":
        return ViViTTubeEncoder(
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
    if kind == "convlstm":
        return ConvLSTMTubeEncoder(embed_dim=embed_dim, base_channels=base_channels)
    if kind == "tsm":
        return TSMTubeEncoder(
            tube_length=tube_length, embed_dim=embed_dim, base_channels=base_channels
        )
    raise ValueError(f"unknown tube encoder kind {kind!r}")


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
        encoder_kind: str = "tubelet_transformer",
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.tube_length = tube_length
        self.temporal_stride = temporal_stride
        self.encoder = build_tube_encoder(
            encoder_kind,
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
