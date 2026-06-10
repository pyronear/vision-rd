"""Ablation: the full model with the temporal (global) module removed.

This is a *faithful* ablation of :class:`TubeMultiscaleClassifier`: the local
branch and the fusion module are the **exact same modules** with the **same
hyperparameters**; the only thing removed is the global DINOv2 sequence branch
(the "temporal module" that embeds all 16 frames and aggregates them into a
context vector).

Because the fusion module expects a query vector (in the full model: the global
context vector), we replace that single input with a **learnable query token**
— a constant, data-independent parameter. Everything downstream (self-attention
over local tubes, cross-attention query→locals, FFN, MLP head) is identical.

Removing the global branch removes the 16 per-frame DINOv2 forward passes, which
dominate the full model's compute — so this ablation has strictly *fewer*
parameters and FLOPs than the full model, isolating the contribution of the
global temporal context.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .fusion import FusionModule, WeightedMeanFusion
from .global_branch import GlobalBranch
from .local_branch import LocalBranch


class AblationNoTemporalClassifier(nn.Module):
    """Full model minus the global/temporal branch (learnable query in its place)."""

    def __init__(
        self,
        # Local branch — identical to the full model
        grid_size: int,
        cell_size: int,
        tube_length: int,
        temporal_stride: int,
        local_t_kernel: int,
        local_h_patch: int,
        local_w_patch: int,
        local_embed_dim: int,
        local_num_layers: int,
        local_num_heads: int,
        local_ffn_dim: int,
        local_dropout: float,
        # Fusion — identical to the full model
        d_fusion: int,
        fusion_num_layers: int,
        fusion_num_heads: int,
        fusion_ffn_dim: int,
        fusion_dropout: float,
        # Head — identical to the full model
        head_hidden_dim: int,
        head_dropout: float = 0.0,
        # Dim of the learnable query that stands in for the global vector.
        query_dim: int = 384,
        # Per-tube spatial encoder (the module under test in the spatial study).
        encoder_kind: str = "tubelet_transformer",
    ) -> None:
        super().__init__()
        self.local_branch = LocalBranch(
            grid_size=grid_size,
            cell_size=cell_size,
            tube_length=tube_length,
            temporal_stride=temporal_stride,
            t_kernel=local_t_kernel,
            h_patch=local_h_patch,
            w_patch=local_w_patch,
            embed_dim=local_embed_dim,
            num_layers=local_num_layers,
            num_heads=local_num_heads,
            ffn_dim=local_ffn_dim,
            dropout=local_dropout,
            encoder_kind=encoder_kind,
        )
        # Learnable stand-in for the (removed) global context vector.
        self.query_token = nn.Parameter(torch.zeros(1, query_dim))
        nn.init.trunc_normal_(self.query_token, std=0.02)
        self.fusion = FusionModule(
            global_dim=query_dim,
            local_dim=local_embed_dim,
            d_fusion=d_fusion,
            num_layers=fusion_num_layers,
            num_heads=fusion_num_heads,
            ffn_dim=fusion_ffn_dim,
            dropout=fusion_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(d_fusion, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        local_tokens, local_mask = self.local_branch(patches, mask)
        query = self.query_token.expand(patches.shape[0], -1)
        fused = self.fusion(query, local_tokens, local_mask)
        return self.head(fused).squeeze(-1)


class AblationNoSpatialClassifier(nn.Module):
    """Full model minus the spatial (local tube) branch — global branch only.

    Keeps the global DINOv2 sequence branch exactly as in the full model and
    classifies its context vector directly with the same MLP head. No local
    branch, no fusion. Isolates the standalone power of the global branch.
    """

    def __init__(
        self,
        backbone: str,
        finetune: bool,
        finetune_last_n_blocks: int,
        max_frames: int,
        global_aggregator_num_layers: int,
        global_aggregator_num_heads: int,
        global_aggregator_ffn_dim: int,
        global_aggregator_dropout: float,
        head_hidden_dim: int,
        head_dropout: float = 0.0,
        aggregator_kind: str = "transformer",
        img_size: int = 224,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.global_branch = GlobalBranch(
            backbone=backbone,
            finetune=finetune,
            finetune_last_n_blocks=finetune_last_n_blocks,
            max_frames=max_frames,
            aggregator_num_layers=global_aggregator_num_layers,
            aggregator_num_heads=global_aggregator_num_heads,
            aggregator_ffn_dim=global_aggregator_ffn_dim,
            aggregator_dropout=global_aggregator_dropout,
            aggregator_kind=aggregator_kind,
            img_size=img_size,
            pretrained=pretrained,
        )
        self.head = nn.Sequential(
            nn.Linear(self.global_branch.feat_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        global_vec = self.global_branch(patches, mask)
        return self.head(global_vec).squeeze(-1)


class AblationWeightedMeanClassifier(nn.Module):
    """Full model with cross-attention fusion replaced by weighted means.

    Both branches are kept exactly as in the full model; only the fusion module
    is swapped from :class:`FusionModule` (self-attn over tubes + global→locals
    cross-attention) to :class:`WeightedMeanFusion` (learned weighted mean over
    tubes + learned gate between the two branches). Isolates the contribution of
    the attention mechanism in the fusion step.
    """

    def __init__(
        self,
        # Global branch — identical to the full model
        backbone: str,
        finetune: bool,
        finetune_last_n_blocks: int,
        max_frames: int,
        global_aggregator_num_layers: int,
        global_aggregator_num_heads: int,
        global_aggregator_ffn_dim: int,
        global_aggregator_dropout: float,
        # Local branch — identical to the full model
        grid_size: int,
        cell_size: int,
        tube_length: int,
        temporal_stride: int,
        local_t_kernel: int,
        local_h_patch: int,
        local_w_patch: int,
        local_embed_dim: int,
        local_num_layers: int,
        local_num_heads: int,
        local_ffn_dim: int,
        local_dropout: float,
        # Fusion / head
        d_fusion: int,
        head_hidden_dim: int,
        head_dropout: float = 0.0,
        img_size: int = 224,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.global_branch = GlobalBranch(
            backbone=backbone,
            finetune=finetune,
            finetune_last_n_blocks=finetune_last_n_blocks,
            max_frames=max_frames,
            aggregator_num_layers=global_aggregator_num_layers,
            aggregator_num_heads=global_aggregator_num_heads,
            aggregator_ffn_dim=global_aggregator_ffn_dim,
            aggregator_dropout=global_aggregator_dropout,
            img_size=img_size,
            pretrained=pretrained,
        )
        self.local_branch = LocalBranch(
            grid_size=grid_size,
            cell_size=cell_size,
            tube_length=tube_length,
            temporal_stride=temporal_stride,
            t_kernel=local_t_kernel,
            h_patch=local_h_patch,
            w_patch=local_w_patch,
            embed_dim=local_embed_dim,
            num_layers=local_num_layers,
            num_heads=local_num_heads,
            ffn_dim=local_ffn_dim,
            dropout=local_dropout,
        )
        self.fusion = WeightedMeanFusion(
            global_dim=self.global_branch.feat_dim,
            local_dim=local_embed_dim,
            d_fusion=d_fusion,
        )
        self.head = nn.Sequential(
            nn.Linear(d_fusion, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        global_vec = self.global_branch(patches, mask)
        local_tokens, local_mask = self.local_branch(patches, mask)
        fused = self.fusion(global_vec, local_tokens, local_mask)
        return self.head(fused).squeeze(-1)
