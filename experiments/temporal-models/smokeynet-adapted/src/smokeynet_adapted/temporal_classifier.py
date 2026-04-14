"""Basic temporal smoke classifier: frozen timm backbone + temporal head."""

import timm
import torch
from torch import Tensor, nn


class FrozenTimmBackbone(nn.Module):
    """Wraps a pretrained timm model as a per-frame feature extractor.

    Always frozen: parameters have ``requires_grad=False`` and the inner
    model is forced to ``eval()`` mode regardless of the parent module's
    training flag (so BatchNorm/Dropout stay deterministic).
    """

    def __init__(self, name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.feat_dim: int = self.backbone.num_features

    def train(self, mode: bool = True) -> "FrozenTimmBackbone":
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
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
