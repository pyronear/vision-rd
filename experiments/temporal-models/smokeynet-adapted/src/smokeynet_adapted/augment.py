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


class PhotometricTubeTransform:
    """Per-tube-consistent brightness / contrast / saturation.

    Operates on ``[0, 1]`` tensors (pre-normalization). One set of factors
    sampled per call, applied in fixed order (brightness -> contrast ->
    saturation) to every frame.
    """

    def __init__(
        self,
        brightness_range: tuple[float, float],
        contrast_range: tuple[float, float],
        saturation_range: tuple[float, float],
    ) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range

    @staticmethod
    def _sample(r: tuple[float, float]) -> float:
        lo, hi = r
        return float(lo + torch.rand(()).item() * (hi - lo))

    def __call__(self, item: dict) -> dict:
        patches: Tensor = item["patches"]  # [T, 3, H, W], values in [0, 1]

        b = self._sample(self.brightness_range)
        c = self._sample(self.contrast_range)
        s = self._sample(self.saturation_range)

        if b != 1.0:
            patches = TVF.adjust_brightness(patches, brightness_factor=b)
        if c != 1.0:
            patches = TVF.adjust_contrast(patches, contrast_factor=c)
        if s != 1.0:
            patches = TVF.adjust_saturation(patches, saturation_factor=s)

        # Clamp to valid photometric range; adjust_contrast can push slightly
        # below 0 / above 1 and the downstream ImageNet normalize expects [0, 1].
        item["patches"] = patches.clamp_(0.0, 1.0)
        return item


class TemporalTubeTransform:
    """Sub-sequence sampling + random stride + per-frame drop with re-compaction.

    Operates on ``(patches: [T, 3, H, W], mask: [T])``. Output valid frames
    always occupy positions ``[0..k-1]`` so ``pack_padded_sequence`` (used by
    the GRU head) sees a contiguous valid prefix.
    """

    def __init__(
        self,
        subseq_prob: float,
        subseq_min_len: int,
        stride_prob: float,
        frame_drop_prob: float,
        min_valid_after_drop: int,
    ) -> None:
        self.subseq_prob = subseq_prob
        self.subseq_min_len = subseq_min_len
        self.stride_prob = stride_prob
        self.frame_drop_prob = frame_drop_prob
        self.min_valid_after_drop = min_valid_after_drop

    def __call__(self, item: dict) -> dict:
        patches: Tensor = item["patches"]  # [T, 3, H, W]
        mask: Tensor = item["mask"]  # [T] bool

        valid_idx = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        n = len(valid_idx)
        if n == 0:
            return item

        # 1. Sub-sequence sampling
        if torch.rand(()).item() < self.subseq_prob and n > self.subseq_min_len:
            k = int(
                torch.randint(self.subseq_min_len, n + 1, (1,)).item()
            )
            start = int(torch.randint(0, n - k + 1, (1,)).item())
            valid_idx = valid_idx[start : start + k]

        # 2. Random stride
        if torch.rand(()).item() < self.stride_prob and len(valid_idx) > 2:
            valid_idx = valid_idx[::2]

        # 3. Per-frame drop, clamped to a floor
        if self.frame_drop_prob > 0.0 and len(valid_idx) > self.min_valid_after_drop:
            keeps = torch.rand(len(valid_idx)) >= self.frame_drop_prob
            # Enforce floor: if we dropped too many, randomly restore indices.
            n_kept = int(keeps.sum())
            if n_kept < self.min_valid_after_drop:
                dropped_positions = torch.nonzero(~keeps, as_tuple=False).flatten()
                n_to_restore = self.min_valid_after_drop - n_kept
                perm = torch.randperm(len(dropped_positions))[:n_to_restore]
                for pos in dropped_positions[perm].tolist():
                    keeps[pos] = True
            valid_idx = [
                idx
                for idx, keep in zip(valid_idx, keeps.tolist(), strict=False)
                if keep
            ]

        # Re-compact to a fresh padded tensor.
        out_patches = torch.zeros_like(patches)
        out_mask = torch.zeros_like(mask)
        for new_pos, src_pos in enumerate(valid_idx):
            out_patches[new_pos] = patches[src_pos]
            out_mask[new_pos] = True

        item["patches"] = out_patches
        item["mask"] = out_mask
        return item
