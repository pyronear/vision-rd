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

from .fusion import FusionModule
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
