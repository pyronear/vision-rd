"""Fusion module: self-attention over local tubes + cross-attention with global.

Takes the global context vector ``(B, D_global)`` and local tube embeddings
``(B, N_tubes, D_local)`` plus a ``(B, N_tubes)`` validity mask, projects both
to a shared ``d_fusion`` width, and runs ``num_layers`` decoder-style blocks:

    1. Self-attention over local tube tokens (with key-padding mask).
    2. Cross-attention: global token = query, local tubes = key/value.
    3. FFN on the global token.

Returns the updated ``(B, d_fusion)`` global representation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _FusionBlock(nn.Module):
    """One layer: self-attn on locals + cross-attn (global=Q, locals=KV) + FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Self-attention over local tube tokens.
        self.norm_local = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_local_ffn = nn.LayerNorm(d_model)
        self.local_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        # Cross-attention: global queries locals.
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_g_ffn = nn.LayerNorm(d_model)
        self.g_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        global_tok: Tensor,
        local_tokens: Tensor,
        local_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """global_tok ``(B, 1, D)``, local_tokens ``(B, N, D)``, mask ``(B, N)``.

        Returns updated ``(global_tok, local_tokens)``.
        """
        key_padding = ~local_mask  # True = ignore.

        # Self-attention on local tube tokens.
        h = self.norm_local(local_tokens)
        attn_out, _ = self.self_attn(
            h, h, h, key_padding_mask=key_padding, need_weights=False
        )
        local_tokens = local_tokens + attn_out
        local_tokens = local_tokens + self.local_ffn(self.norm_local_ffn(local_tokens))

        # Cross-attention: global queries locals.
        q = self.norm_q(global_tok)
        kv = self.norm_kv(local_tokens)
        cross_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding, need_weights=False
        )
        global_tok = global_tok + cross_out
        global_tok = global_tok + self.g_ffn(self.norm_g_ffn(global_tok))
        return global_tok, local_tokens


class FusionModule(nn.Module):
    """Project global + local to ``d_fusion``, then stack ``num_layers`` blocks."""

    def __init__(
        self,
        global_dim: int,
        local_dim: int,
        d_fusion: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.global_proj: nn.Module = (
            nn.Identity() if global_dim == d_fusion else nn.Linear(global_dim, d_fusion)
        )
        self.local_proj: nn.Module = (
            nn.Identity() if local_dim == d_fusion else nn.Linear(local_dim, d_fusion)
        )
        self.blocks = nn.ModuleList(
            [
                _FusionBlock(
                    d_model=d_fusion,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_fusion)
        self.d_fusion = d_fusion

    def forward(
        self,
        global_vec: Tensor,
        local_tokens: Tensor,
        local_mask: Tensor,
    ) -> Tensor:
        """``global_vec: (B, D_global)``, ``local_tokens: (B, N, D_local)``,
        ``local_mask: (B, N)`` -> ``(B, d_fusion)``.

        If a sample has *no* valid local tubes (all masked), key-padding would
        produce NaNs in attention; we guard against that by treating empty
        masks as fully visible (the global readout is then just self-mixed).
        """
        g = self.global_proj(global_vec).unsqueeze(1)  # (B, 1, D)
        ltok = self.local_proj(local_tokens)
        # Guard: if any row has zero valid locals, keep it visible to avoid NaN.
        safe_mask = local_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if torch.any(empty_rows):
            safe_mask[empty_rows] = True
        for block in self.blocks:
            g, ltok = block(g, ltok, safe_mask)
        return self.final_norm(g.squeeze(1))
