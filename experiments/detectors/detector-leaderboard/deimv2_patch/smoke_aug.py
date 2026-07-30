"""Custom augmentations for the smoke tiled-training experiment (detector-leaderboard).

Added on top of DEIMv2's transform registry (kept as a tracked patch in the
experiment). Two ops, both operating at the PIL stage on the (image, target,
dataset) sample tuple used across the pipeline:

- ``RandomGaussianBlur``: torchvision v2 GaussianBlur gated by probability
  ``p`` (mirrors the repo's ``RandomIoUCrop`` p-wrapper). Image-only.
- ``FPInject``: pastes random hard-negative "false-positive" patches (crops of
  smoke-like non-smoke scenes) at random locations onto the image and leaves the
  target UNCHANGED — distractors the model must learn NOT to detect.
"""

import random
from pathlib import Path

import PIL.Image
import torch
import torch.multiprocessing
import torch.nn as nn
import torchvision.transforms.v2 as T

from ...core import register

# DataLoader workers on the large tiled set hit SIGBUS ("out of shared memory")
# with the default file_descriptor sharing strategy; file_system avoids the
# /dev/shm fd pressure. Set at import (main process, before workers fork). This
# module is imported when engine.data.transforms loads, i.e. before training.
torch.multiprocessing.set_sharing_strategy("file_system")


@register()
class RandomGaussianBlur(T.GaussianBlur):
    """GaussianBlur applied with probability ``p`` (image-only)."""

    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p: float = 0.5):
        super().__init__(kernel_size=kernel_size, sigma=sigma)
        self.p = p

    def __call__(self, *inputs):
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


@register()
class FPInject(nn.Module):
    """Paste random false-positive (hard-negative) patches; add NO boxes.

    Args:
        crop_dir: directory of distractor patch ``.jpg`` files (built by
            ``scripts/build_fp_crop_pool.py``).
        prob: probability of applying injection to a sample.
        max_objects: up to this many patches pasted (uniform 1..max_objects).
        min_scale/max_scale: random rescale applied to each patch before paste.
    """

    def __init__(self, crop_dir, prob=0.5, max_objects=3, min_scale=0.6, max_scale=1.2):
        super().__init__()
        self.paths = sorted(str(p) for p in Path(crop_dir).glob("*.jpg"))
        self.prob = prob
        self.max_objects = max_objects
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, *inputs):
        sample = inputs[0] if len(inputs) == 1 else inputs
        image = sample[0]
        if (
            not self.paths
            or random.random() >= self.prob
            or not isinstance(image, PIL.Image.Image)
        ):
            return sample

        image = image.copy()  # don't mutate cached/shared source (Mosaic caches)
        width, height = image.size
        for _ in range(random.randint(1, self.max_objects)):
            patch = PIL.Image.open(random.choice(self.paths)).convert("RGB")
            scale = random.uniform(self.min_scale, self.max_scale)
            pw = min(max(8, int(patch.width * scale)), width)
            ph = min(max(8, int(patch.height * scale)), height)
            patch = patch.resize((pw, ph))
            x = random.randint(0, width - pw)
            y = random.randint(0, height - ph)
            image.paste(patch, (x, y))
        return (image, *sample[1:])
