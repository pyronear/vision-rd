"""Render side-by-side grids of augmented tube variants for sanity checking.

Reads a few tubes from the training split, generates N augmented copies
of each, and saves one PNG per tube to
``data/08_reporting/augment_samples/`` showing original vs. augmented
frames side by side.
"""

from __future__ import annotations

import os

# Force a headless backend before any matplotlib import, matching
# scripts/render_tubes.py so this runs on CI / servers without a display.
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torchvision.utils import make_grid

from smokeynet_adapted.augment import build_tube_augment
from smokeynet_adapted.dataset import TubePatchDataset


def _denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tubes", type=int, default=3)
    parser.add_argument("--num-variants", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = yaml.safe_load(args.params_path.read_text())
    augment_cfg = params.get("augment", {"enabled": True})

    torch.manual_seed(args.seed)

    transform = build_tube_augment(augment_cfg, train=True)
    ds = TubePatchDataset(
        args.split_dir, max_frames=args.max_frames, transform=transform
    )
    raw_ds = TubePatchDataset(
        args.split_dir, max_frames=args.max_frames, transform=None
    )

    for tube_idx in range(min(args.num_tubes, len(ds))):
        seq_id = ds.index[tube_idx]["sequence_id"]

        raw = raw_ds[tube_idx]
        raw_patches = _denormalize(raw["patches"][: int(raw["mask"].sum())])

        rows = [raw_patches]
        for _ in range(args.num_variants):
            aug = ds[tube_idx]
            patches = _denormalize(aug["patches"][: int(aug["mask"].sum())])
            rows.append(patches)

        max_t = max(r.shape[0] for r in rows)
        padded_rows = [
            torch.cat(
                [r, torch.zeros(max_t - r.shape[0], 3, 224, 224, dtype=r.dtype)],
                dim=0,
            )
            if r.shape[0] < max_t
            else r
            for r in rows
        ]
        grid_tensor = torch.stack(padded_rows).reshape(-1, 3, 224, 224)
        grid = make_grid(grid_tensor, nrow=max_t, padding=2)

        fig, ax = plt.subplots(figsize=(max_t * 1.5, (len(rows)) * 1.5))
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
        ax.set_axis_off()
        ax.set_title(f"{seq_id} — top row: raw, below: {args.num_variants} augmented")
        fig.savefig(args.output_dir / f"{seq_id}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {min(args.num_tubes, len(ds))} grids to {args.output_dir}")


if __name__ == "__main__":
    main()
