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
