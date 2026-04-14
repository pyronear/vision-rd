"""Per-tube-consistent augmentation transforms for the temporal classifier.

See ``docs/specs/2026-04-14-training-augmentation-design.md`` for design.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.transforms.v2 import functional as TVF


class SpatialTubeTransform:
    """Per-tube-consistent spatial aug (flip + rotation + scale + translate).

    A single set of params is sampled per call and applied identically to
    every frame in the tube so motion direction and inter-frame geometry
    are preserved.
    """

    def __init__(
        self,
        flip_prob: float,
        rotation_deg: float,
        scale_range: tuple[float, float],
        translate_frac: float,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.translate_frac = translate_frac

    def __call__(self, item: dict) -> dict:
        patches: Tensor = item["patches"]  # [T, 3, H, W]
        _, _, h, w = patches.shape

        # Sample once per tube
        do_flip = torch.rand(()).item() < self.flip_prob
        angle = float((torch.rand(()).item() * 2 - 1) * self.rotation_deg)
        s_lo, s_hi = self.scale_range
        scale = float(s_lo + torch.rand(()).item() * (s_hi - s_lo))
        tx = float((torch.rand(()).item() * 2 - 1) * self.translate_frac * w)
        ty = float((torch.rand(()).item() * 2 - 1) * self.translate_frac * h)

        if do_flip:
            patches = TVF.horizontal_flip(patches)

        # Skip affine call entirely when parameters are no-ops (identity).
        is_identity_affine = angle == 0.0 and scale == 1.0 and tx == 0.0 and ty == 0.0
        if not is_identity_affine:
            patches = TVF.affine(
                patches,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
            )

        item["patches"] = patches
        return item
