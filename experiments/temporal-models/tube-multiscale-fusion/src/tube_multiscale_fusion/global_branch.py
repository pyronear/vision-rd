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


class GlobalBranch(nn.Module):
    """DINOv2 per-frame embeddings + transformer aggregator -> ``(B, D)``."""

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
        self.aggregator = GlobalAggregator(
            feat_dim=self.feat_dim,
            num_layers=aggregator_num_layers,
            num_heads=aggregator_num_heads,
            ffn_dim=aggregator_ffn_dim,
            dropout=aggregator_dropout,
            max_frames=max_frames,
        )

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        b, t, c, h, w = patches.shape
        feats = self.backbone(patches.reshape(b * t, c, h, w))
        feats = feats.reshape(b, t, self.feat_dim)
        return self.aggregator(feats, mask)
