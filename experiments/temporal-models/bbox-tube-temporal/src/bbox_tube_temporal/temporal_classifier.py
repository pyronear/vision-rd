"""Basic temporal smoke classifier: frozen timm backbone + temporal head."""

import timm
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class TimmBackbone(nn.Module):
    """Wraps a pretrained timm model as a per-frame feature extractor.

    When ``finetune=False`` (default), all params are frozen and the
    inner model is forced into ``eval()`` mode regardless of the parent
    module's training flag; forward is wrapped in ``torch.no_grad()``.

    When ``finetune=True``, the last ``finetune_last_n_blocks`` blocks
    are unfrozen (family-specific resolution); everything else stays
    frozen, and ``.train()`` propagates normally so BatchNorm on
    unfrozen blocks updates.
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
        global_pool: str = "avg",
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool=global_pool
        )
        self.feat_dim: int = self.backbone.num_features
        self.finetune = finetune
        self.name = name

        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            return

        # Finetune path: freeze everything first, then unfreeze last N blocks.
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
            # timm's convnext exposes a `stages` ModuleList
            stages = list(self.backbone.stages)
        elif name.startswith("vit_"):
            # timm's ViT exposes a `blocks` ModuleList of transformer blocks.
            stages = list(self.backbone.blocks)
        else:
            stage_names = [n_ for n_, _ in self.backbone.named_children()]
            raise NotImplementedError(
                f"finetune=True is not implemented for backbone family "
                f"{name!r}. Top-level children: {stage_names}. Add an "
                f"explicit unfreeze rule in TimmBackbone._unfreeze_last_n_blocks."
            )
        for stage in stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True) -> "TimmBackbone":
        super().train(mode)
        if not self.finetune:
            self.backbone.eval()
        else:
            # Keep frozen sub-modules in eval mode so their BatchNorm
            # running stats stay deterministic.
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


class MeanPoolHead(nn.Module):
    """Masked mean over time + 2-layer MLP yielding a single logit."""

    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        m = mask.unsqueeze(-1).to(feats.dtype)
        summed = (feats * m).sum(dim=1)
        counts = m.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.mlp(pooled).squeeze(-1)


class GRUHead(nn.Module):
    """1+ layer GRU over time + MLP yielding a single logit. Uses packed sequences."""

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            feats, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        if self.bidirectional:
            last_fwd = h_n[-2]
            last_bwd = h_n[-1]
            last = torch.cat([last_fwd, last_bwd], dim=-1)
        else:
            last = h_n[-1]
        return self.mlp(last).squeeze(-1)


class TransformerHead(nn.Module):
    """Learnable [CLS] + learned positional embeddings + Transformer encoder.

    Returns one logit per tube. Padded positions are masked out of
    attention via ``src_key_padding_mask``.
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
        # +1 for the prepended CLS position.
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
        self.classifier = nn.Linear(feat_dim, 1)
        self.max_frames = max_frames

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        b, t, _ = feats.shape
        if t > self.max_frames:
            raise ValueError(
                f"TransformerHead received T={t} frames but was configured "
                f"with max_frames={self.max_frames}"
            )
        cls = self.cls_token.expand(b, 1, -1)
        x = torch.cat([cls, feats], dim=1)  # (B, T+1, D)
        x = x + self.pos_embed[:, : t + 1, :]
        # True = pad (ignored) per torch convention; our input mask is True = real.
        cls_real = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
        real_mask = torch.cat([cls_real, mask], dim=1)
        key_padding_mask = ~real_mask  # (B, T+1), True = pad
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        cls_out = out[:, 0, :]
        return self.classifier(cls_out).squeeze(-1)


class TemporalSmokeClassifier(nn.Module):
    """Frozen backbone applied per-frame plus a temporal head.

    Produces a single binary logit per tube.
    """

    def __init__(
        self,
        backbone: str,
        arch: str,
        hidden_dim: int,
        pretrained: bool = True,
        num_layers: int = 1,
        bidirectional: bool = False,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 6,
        transformer_ffn_dim: int = 1536,
        transformer_dropout: float = 0.1,
        max_frames: int = 20,
        global_pool: str = "avg",
    ) -> None:
        super().__init__()
        self.backbone = TimmBackbone(
            name=backbone,
            pretrained=pretrained,
            finetune=finetune,
            finetune_last_n_blocks=finetune_last_n_blocks,
            global_pool=global_pool,
        )
        feat_dim = self.backbone.feat_dim
        if arch == "mean_pool":
            self.head: nn.Module = MeanPoolHead(
                feat_dim=feat_dim, hidden_dim=hidden_dim
            )
        elif arch == "gru":
            self.head = GRUHead(
                feat_dim=feat_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
        elif arch == "transformer":
            self.head = TransformerHead(
                feat_dim=feat_dim,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                ffn_dim=transformer_ffn_dim,
                dropout=transformer_dropout,
                max_frames=max_frames,
            )
        else:
            raise ValueError(
                f"unknown arch: {arch!r} "
                f"(expected 'mean_pool', 'gru', or 'transformer')"
            )
        self.arch = arch

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        b, t, c, h, w = patches.shape
        flat = patches.reshape(b * t, c, h, w)
        feats = self.backbone(flat).reshape(b, t, -1)
        return self.head(feats, mask)
