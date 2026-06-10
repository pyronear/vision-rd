"""Top-level tube multiscale fusion classifier.

Combines:
    - :class:`GlobalBranch`: per-frame DINOv2 + transformer aggregator -> ``(B, D_g)``
    - :class:`LocalBranch`: spatio-temporal tube extraction + video transformer
      -> ``(B, N_tubes, D_l)`` + validity mask ``(B, N_tubes)``
    - :class:`FusionModule`: self-attn on locals + cross-attn (global=Q, locals=KV)
      -> ``(B, d_fusion)``
    - 2-layer MLP head -> logit ``(B,)``.
"""

from __future__ import annotations

from torch import Tensor, nn

from .fusion import FusionModule
from .global_branch import GlobalBranch
from .local_branch import LocalBranch


class TubeMultiscaleClassifier(nn.Module):
    def __init__(
        self,
        # Global branch
        backbone: str,
        finetune: bool,
        finetune_last_n_blocks: int,
        max_frames: int,
        global_aggregator_num_layers: int,
        global_aggregator_num_heads: int,
        global_aggregator_ffn_dim: int,
        global_aggregator_dropout: float,
        # Local branch
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
        # Fusion
        d_fusion: int,
        fusion_num_layers: int,
        fusion_num_heads: int,
        fusion_ffn_dim: int,
        fusion_dropout: float,
        # Head
        head_hidden_dim: int,
        head_dropout: float = 0.0,
        # Module variants (default = production architecture)
        aggregator_kind: str = "transformer",
        encoder_kind: str = "tubelet_transformer",
        # Misc
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
        self.fusion = FusionModule(
            global_dim=self.global_branch.feat_dim,
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
        global_vec = self.global_branch(patches, mask)
        local_tokens, local_mask = self.local_branch(patches, mask)
        fused = self.fusion(global_vec, local_tokens, local_mask)
        return self.head(fused).squeeze(-1)
