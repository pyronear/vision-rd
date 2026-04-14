"""Basic temporal smoke classifier: frozen timm backbone + temporal head."""

import timm
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


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
