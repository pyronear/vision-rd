"""Global branch: per-frame DINOv2 + sequence transformer aggregator.

Embeds each of ``T`` frames independently with a timm backbone (default
DINOv2 ViT-S/14), then aggregates the resulting ``(B, T, D)`` sequence into
a single ``(B, D)`` global context vector via a small Transformer encoder
with a learnable ``[CLS]`` token.
"""

from __future__ import annotations

import timm
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class TimmBackbone(nn.Module):
    """Per-frame feature extractor backed by a pretrained timm model.

    Ported from ``bbox-tube-motion-fusion/temporal_classifier.py``.
    Supports either fully-frozen mode or finetuning the last ``N`` blocks
    (resolved per backbone family).
    """

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
        extra = {"img_size": img_size} if img_size is not None else {}
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=global_pool,
            **extra,
        )
        self.feat_dim: int = self.backbone.num_features
        self.finetune = finetune
        self.name = name

        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            return

        for p in self.backbone.parameters():
            p.requires_grad = False
        self._unfreeze_last_n_blocks(finetune_last_n_blocks)

    def _unfreeze_last_n_blocks(self, n: int) -> None:
        if n <= 0:
            return
        name = self.name
        if name.startswith("resnet"):
            stages = [getattr(self.backbone, f"layer{i}") for i in range(1, 5)]
        elif name.startswith("convnext"):
            stages = list(self.backbone.stages)
        elif name.startswith("vit_"):
            stages = list(self.backbone.blocks)
        else:
            stage_names = [n_ for n_, _ in self.backbone.named_children()]
            raise NotImplementedError(
                f"finetune=True is not implemented for backbone family "
                f"{name!r}. Top-level children: {stage_names}."
            )
        for stage in stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True) -> TimmBackbone:
        super().train(mode)
        if not self.finetune:
            self.backbone.eval()
        else:
            for module in self.backbone.modules():
                if not any(p.requires_grad for p in module.parameters(recurse=False)):
                    has_trainable_descendant = any(
                        p.requires_grad for p in module.parameters()
                    )
                    if not has_trainable_descendant:
                        module.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if not self.finetune:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)


class GlobalAggregator(nn.Module):
    """Aggregates a ``(B, T, D)`` per-frame sequence into a single ``(B, D)`` vector.

    Architecture: learnable ``[CLS]`` token + learned positional embeddings +
    Transformer encoder. The CLS token's final state is returned.
    """

    def __init__(
        self,
        feat_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        max_frames: int,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames + 1, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_frames = max_frames

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        b, t, _ = feats.shape
        if t > self.max_frames:
            raise ValueError(
                f"GlobalAggregator received T={t} > max_frames={self.max_frames}"
            )
        cls = self.cls_token.expand(b, 1, -1)
        x = torch.cat([cls, feats], dim=1)
        x = x + self.pos_embed[:, : t + 1, :]
        cls_real = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
        real_mask = torch.cat([cls_real, mask], dim=1)
        key_padding_mask = ~real_mask
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return out[:, 0, :]


class RNNAggregator(nn.Module):
    """LSTM/GRU over the per-frame sequence; returns the final hidden state.

    Uses packed sequences so padded frames are ignored. ``hidden_size`` is set
    to ``feat_dim`` so the output matches the rest of the architecture.
    """

    def __init__(
        self,
        feat_dim: int,
        kind: str,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.kind = kind
        self.rnn = rnn_cls(
            input_size=feat_dim,
            hidden_size=feat_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            feats, lengths, batch_first=True, enforce_sorted=False
        )
        if self.kind == "lstm":
            _, (h_n, _) = self.rnn(packed)
        else:
            _, h_n = self.rnn(packed)
        return h_n[-1]  # (B, feat_dim)


class MLPAggregator(nn.Module):
    """Flatten the (masked) ``T×D`` sequence and run an MLP -> ``(B, D)``.

    Padded frames are zeroed before flattening. Sees the whole sequence at once
    but has no inductive bias for temporal order beyond position-in-vector.
    """

    def __init__(
        self,
        feat_dim: int,
        max_frames: int,
        hidden: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.net = nn.Sequential(
            nn.Linear(feat_dim * max_frames, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        feats = feats * mask.unsqueeze(-1).to(feats.dtype)
        return self.net(feats.reshape(feats.shape[0], -1))


class Conv1DAggregator(nn.Module):
    """1D temporal convolutions over the (B, D, T) sequence + masked mean pool."""

    def __init__(
        self,
        feat_dim: int,
        num_layers: int = 2,
        kernel: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(feat_dim, feat_dim, kernel, padding=kernel // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.conv = nn.Sequential(*layers)

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        x = self.conv(feats.transpose(1, 2)).transpose(1, 2)  # (B, T, D)
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class LinearWeightedAvgAggregator(nn.Module):
    """Per-frame Linear projection + learned weighted average over time.

    A lightweight attention-pool with no token-token interaction: each frame is
    projected, scored to a scalar, softmax-normalized over the valid frames, and
    combined by weighted mean.
    """

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(feat_dim, feat_dim)
        self.score = nn.Linear(feat_dim, 1)

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        h = self.proj(feats)  # (B, T, D)
        s = self.score(h).squeeze(-1)  # (B, T)
        s = s.masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=1)
        w = torch.nan_to_num(w, nan=0.0)
        return (w.unsqueeze(-1) * h).sum(dim=1)  # (B, D)


def build_aggregator(
    kind: str,
    feat_dim: int,
    max_frames: int,
    *,
    num_layers: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    rnn_layers: int = 1,
    mlp_hidden: int = 1024,
    conv_layers: int = 2,
    conv_kernel: int = 3,
) -> nn.Module:
    """Construct a temporal aggregator ``(B, T, D), mask -> (B, D)`` by name."""
    if kind == "transformer":
        return GlobalAggregator(
            feat_dim=feat_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_frames=max_frames,
        )
    if kind in ("lstm", "gru"):
        return RNNAggregator(
            feat_dim, kind=kind, num_layers=rnn_layers, dropout=dropout
        )
    if kind == "mlp":
        return MLPAggregator(feat_dim, max_frames, hidden=mlp_hidden, dropout=dropout)
    if kind == "conv1d":
        return Conv1DAggregator(
            feat_dim, num_layers=conv_layers, kernel=conv_kernel, dropout=dropout
        )
    if kind == "linear_wavg":
        return LinearWeightedAvgAggregator(feat_dim)
    raise ValueError(f"unknown aggregator kind {kind!r}")


class GlobalBranch(nn.Module):
    """DINOv2 per-frame embeddings + a temporal aggregator -> ``(B, D)``."""

    def __init__(
        self,
        backbone: str,
        finetune: bool,
        finetune_last_n_blocks: int,
        max_frames: int,
        aggregator_num_layers: int,
        aggregator_num_heads: int,
        aggregator_ffn_dim: int,
        aggregator_dropout: float,
        aggregator_kind: str = "transformer",
        img_size: int = 224,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = TimmBackbone(
            name=backbone,
            pretrained=pretrained,
            finetune=finetune,
            finetune_last_n_blocks=finetune_last_n_blocks,
            img_size=img_size,
        )
        self.feat_dim = self.backbone.feat_dim
        self.aggregator = build_aggregator(
            aggregator_kind,
            feat_dim=self.feat_dim,
            max_frames=max_frames,
            num_layers=aggregator_num_layers,
            num_heads=aggregator_num_heads,
            ffn_dim=aggregator_ffn_dim,
            dropout=aggregator_dropout,
        )

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        b, t, c, h, w = patches.shape
        feats = self.backbone(patches.reshape(b * t, c, h, w))
        feats = feats.reshape(b, t, self.feat_dim)
        return self.aggregator(feats, mask)
