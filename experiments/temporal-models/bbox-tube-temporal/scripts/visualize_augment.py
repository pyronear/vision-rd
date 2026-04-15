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

from bbox_tube_temporal.augment import build_tube_augment
from bbox_tube_temporal.batch_samples import render_batch_grid
from bbox_tube_temporal.dataset import TubePatchDataset


def _pack_rows(rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length tube rows into a [B, T, 3, H, W] tensor + mask.

    Each input row is ``[k_i, 3, H, W]`` (the valid frames of one variant).
    Shorter rows are right-padded with zeros; the mask tracks valid frames.
    """
    max_t = max(r.shape[0] for r in rows)
    padded = torch.stack(
        [
            (
                torch.cat(
                    [r, torch.zeros(max_t - r.shape[0], *r.shape[1:], dtype=r.dtype)],
                    dim=0,
                )
                if r.shape[0] < max_t
                else r
            )
            for r in rows
        ],
        dim=0,
    )
    mask = torch.zeros(len(rows), max_t, dtype=torch.bool)
    for i, r in enumerate(rows):
        mask[i, : r.shape[0]] = True
    return padded, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tubes", type=int, default=3)
    parser.add_argument("--num-variants", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=120)
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
        rows = [raw["patches"][: int(raw["mask"].sum())]]
        labels = ["raw"]
        for i in range(args.num_variants):
            aug = ds[tube_idx]
            rows.append(aug["patches"][: int(aug["mask"].sum())])
            labels.append(f"aug #{i}")

        patches_packed, mask = _pack_rows(rows)
        fig = render_batch_grid(
            patches_packed,
            mask,
            title=seq_id,
            denormalize=True,
            row_labels=labels,
        )
        fig.savefig(
            args.output_dir / f"{seq_id}.png", dpi=args.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    print(f"Wrote {min(args.num_tubes, len(ds))} grids to {args.output_dir}")


if __name__ == "__main__":
    main()
