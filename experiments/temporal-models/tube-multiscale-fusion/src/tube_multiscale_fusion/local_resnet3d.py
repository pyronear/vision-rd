"""Local-only ablation: per-tube 3D ResNet, no global branch, no fusion.

This is the ablation counterpart to the full two-branch model. It keeps the
local tube decomposition (``extract_tubes``) but:

- replaces the tubelet-transformer ``LocalTubeEncoder`` with a Kinetics-400
  pretrained 3D ResNet (``r3d_18`` by default), and
- drops the global DINOv2 sequence branch *and* the cross-attention fusion
  ("temporal module"), aggregating the per-tube embeddings with a simple
  mask-aware mean pool before a small MLP head.

Comparing this to ``TubeMultiscaleClassifier`` isolates the contribution of the
global context branch + fusion attention over a pure local 3D-CNN model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.video import (
    MC3_18_Weights,
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    mc3_18,
    r2plus1d_18,
    r3d_18,
)

from .local_branch import extract_tubes, tube_validity_mask

_FACTORIES = {
    "r3d_18": (r3d_18, R3D_18_Weights.KINETICS400_V1),
    "mc3_18": (mc3_18, MC3_18_Weights.KINETICS400_V1),
    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.KINETICS400_V1),
}

_KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
_KINETICS_STD = (0.22803, 0.22145, 0.216989)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Resnet3DEncoder(nn.Module):
    """Pretrained 3D ResNet wrapped as a clip-level feature extractor.

    Ported from ``bbox-tube-motion-fusion/motion_3d.py``. Forward expects a
    clip ``(B, T, 3, H, W)`` in ImageNet normalization (as produced by
    ``TubePatchDataset``); returns ``(B, out_dim)``.
    """

    def __init__(
        self,
        model_name: str = "r3d_18",
        pretrained: bool = True,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
        out_dim: int | None = None,
        clip_spatial_size: int = 112,
    ) -> None:
        super().__init__()
        if model_name not in _FACTORIES:
            raise ValueError(
                f"unknown 3D ResNet {model_name!r}; expected one of "
                f"{tuple(_FACTORIES)}"
            )
        factory, weights = _FACTORIES[model_name]
        self.backbone = factory(weights=weights if pretrained else None)
        feat_dim: int = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.model_name = model_name
        self.feat_dim = feat_dim
        self.clip_spatial_size = clip_spatial_size
        self.finetune = finetune

        im_mean = torch.tensor(_IMAGENET_MEAN)
        im_std = torch.tensor(_IMAGENET_STD)
        ki_mean = torch.tensor(_KINETICS_MEAN)
        ki_std = torch.tensor(_KINETICS_STD)
        scale = (im_std / ki_std).view(1, 3, 1, 1, 1)
        bias = ((im_mean - ki_mean) / ki_std).view(1, 3, 1, 1, 1)
        self.register_buffer("_renorm_scale", scale, persistent=False)
        self.register_buffer("_renorm_bias", bias, persistent=False)

        for p in self.backbone.parameters():
            p.requires_grad = False
        if finetune:
            stages = [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            ]
            n = max(0, min(finetune_last_n_blocks, len(stages)))
            for stage in stages[-n:] if n > 0 else []:
                for p in stage.parameters():
                    p.requires_grad = True

        if out_dim is not None and out_dim != feat_dim:
            self.proj: nn.Module = nn.Linear(feat_dim, out_dim)
            self.out_dim = out_dim
        else:
            self.proj = nn.Identity()
            self.out_dim = feat_dim

    def train(self, mode: bool = True) -> Resnet3DEncoder:
        super().train(mode)
        if not self.finetune:
            self.backbone.eval()
        else:
            for module in self.backbone.modules():
                shallow_frozen = not any(
                    p.requires_grad for p in module.parameters(recurse=False)
                )
                fully_frozen = not any(
                    p.requires_grad for p in module.parameters()
                )
                if shallow_frozen and fully_frozen:
                    module.eval()
        return self

    def forward(self, clip: Tensor) -> Tensor:
        # (B, T, 3, H, W) -> (B, 3, T, H, W) for conv3d.
        x = clip.permute(0, 2, 1, 3, 4)
        x = x * self._renorm_scale + self._renorm_bias
        b, c, t, h, w = x.shape
        if h != self.clip_spatial_size or w != self.clip_spatial_size:
            x = x.reshape(b * c, t, h, w)
            x = F.interpolate(
                x.unsqueeze(0),
                size=(t, self.clip_spatial_size, self.clip_spatial_size),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            x = x.reshape(b, c, t, self.clip_spatial_size, self.clip_spatial_size)
        if not self.finetune:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        return self.proj(feat)


class AblationLocalResnet3DClassifier(nn.Module):
    """Local-only ablation classifier.

    Pipeline: ``extract_tubes`` -> per-tube 3D ResNet -> mask-aware mean pool
    over valid tubes -> 2-layer MLP head -> single logit. No global branch,
    no cross-attention fusion.
    """

    def __init__(
        self,
        grid_size: int,
        cell_size: int,
        tube_length: int,
        temporal_stride: int,
        resnet_model: str = "r3d_18",
        resnet_pretrained: bool = True,
        resnet_finetune: bool = True,
        resnet_finetune_last_n_blocks: int = 1,
        resnet_clip_spatial_size: int = 112,
        embed_dim: int | None = None,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.tube_length = tube_length
        self.temporal_stride = temporal_stride
        self.encoder = Resnet3DEncoder(
            model_name=resnet_model,
            pretrained=resnet_pretrained,
            finetune=resnet_finetune,
            finetune_last_n_blocks=resnet_finetune_last_n_blocks,
            out_dim=embed_dim,
            clip_spatial_size=resnet_clip_spatial_size,
        )
        self.embed_dim = self.encoder.out_dim
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        tubes = extract_tubes(
            patches,
            grid_size=self.grid_size,
            cell_size=self.cell_size,
            tube_length=self.tube_length,
            temporal_stride=self.temporal_stride,
        )  # (B, N, 3, T_t, h, w)
        b, n, c, t, h, w = tubes.shape
        # 3D ResNet expects a clip (B', T, 3, H, W); flatten the tube axis.
        clips = tubes.permute(0, 1, 3, 2, 4, 5).reshape(b * n, t, c, h, w)
        feats = self.encoder(clips).reshape(b, n, self.embed_dim)

        tube_mask = tube_validity_mask(
            mask,
            grid_size=self.grid_size,
            tube_length=self.tube_length,
            temporal_stride=self.temporal_stride,
        )  # (B, N) bool
        m = tube_mask.unsqueeze(-1).to(feats.dtype)
        summed = (feats * m).sum(dim=1)
        counts = m.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts  # (B, embed_dim)
        return self.head(pooled).squeeze(-1)
