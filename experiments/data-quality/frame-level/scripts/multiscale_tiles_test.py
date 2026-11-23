"""Quick test: global pass vs global pass + native-resolution tiles on a band.

Global pass  = whole image letterboxed to ``--net`` (what production does).
Tiles        = ``--net`` x ``--net`` crops at native resolution, >=30 % overlap,
               in one horizontal row centred on ``--band-cy`` (fraction of the
               image height — a stand-in for the horizon line).
Fusion       = boxes shifted back to image coordinates, then one NMS.

Reports recall per GT size bucket (size measured in global-pass pixels) and
false positives per image, for a few confidence thresholds.

    uv run python scripts/multiscale_tiles_test.py \
        --model <best.pt> --split-dir data/01_raw/datasets/test --net 352
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm
from ultralytics import YOLO

from data_quality_frame_level.audit_app.matching import evaluate_frame
from data_quality_frame_level.audit_app.types import Prediction
from data_quality_frame_level.dataset import iter_frames

CONFS = (0.05, 0.15, 0.25)
BUCKETS = ((0, 8), (8, 16), (16, 32), (32, math.inf))  # sqrt(area), global-pass px
MIN_OVERLAP = 0.3


def tile_boxes(
    w: int, h: int, t: int, band_cy: float
) -> list[tuple[int, int, int, int]]:
    """One row of t-wide crops covering the full width, centred on the band."""
    tw, th = min(t, w), min(t, h)
    n = math.ceil((w - tw) / (tw * (1 - MIN_OVERLAP))) + 1 if w > tw else 1
    y0 = int(min(max(band_cy * h - th / 2, 0), h - th))
    xs = [round(i * (w - tw) / (n - 1)) for i in range(n)] if n > 1 else [0]
    return [(x, y0, x + tw, y0 + th) for x in xs]


def _check() -> None:
    tiles = tile_boxes(3840, 2160, 1024, 0.5)
    assert len(tiles) == 5 and tiles[0][0] == 0 and tiles[-1][2] == 3840
    assert all(
        b[0] - a[0] <= 1024 * (1 - MIN_OVERLAP) + 1
        for a, b in zip(tiles, tiles[1:], strict=False)
    )
    assert tiles[0][1] == 568 and tiles[0][3] == 1592
    assert tile_boxes(1280, 720, 1024, 0.5)[0][3] == 720  # tile taller than image


def run(model: YOLO, imgs: list, imgsz: int, device: str) -> list[torch.Tensor]:
    """Per image: tensor [n, 5] = x1, y1, x2, y2, conf in that image's pixels."""
    res = model.predict(
        imgs, imgsz=imgsz, conf=min(CONFS), device=device, verbose=False
    )
    return [torch.cat([r.boxes.xyxy, r.boxes.conf[:, None]], 1).cpu() for r in res]


def fuse(parts: list[torch.Tensor]) -> torch.Tensor:
    # ponytail: plain NMS; a smoke cut by a tile edge keeps its truncated box
    # unless the global box overlaps it. Use merge-by-containment if that hurts.
    d = torch.cat(parts)
    return d[nms(d[:, :4], d[:, 4], 0.5)]


def to_preds(d: torch.Tensor, w: int, h: int, conf: float) -> list[Prediction]:
    return [
        Prediction(
            0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h, c
        )
        for x1, y1, x2, y2, c in d.tolist()
        if c >= conf
    ]


def main() -> None:
    _check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--net", type=int, default=1024, help="network input = tile size")
    ap.add_argument(
        "--ref", type=int, default=0, help="extra full-image pass at this size"
    )
    ap.add_argument("--band-cy", type=float, default=0.5)
    ap.add_argument("--min-width", type=int, default=0)
    ap.add_argument("--max-width", type=int, default=10**9)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()

    model = YOLO(str(a.model))
    variants = ["global", "tiles", "global+tiles"] + (
        [f"full@{a.ref}"] if a.ref else []
    )
    # stats[variant][conf] -> counters
    tp = {v: {c: defaultdict(int) for c in CONFS} for v in variants}
    fp = {v: {c: 0 for c in CONFS} for v in variants}
    n_gt = defaultdict(int)
    n_img = n_tiles = gt_in_band = 0

    frames = [f for d in a.split_dir for f in iter_frames(d)]
    for f in tqdm(frames):
        im = Image.open(f.image_path).convert("RGB")
        w, h = im.size
        if not a.min_width <= w <= a.max_width:
            continue
        n_img += 1
        tiles = tile_boxes(w, h, a.net, a.band_cy)
        n_tiles += len(tiles)
        dets = {"global": run(model, [im], a.net, a.device)[0]}
        shifted = []
        for (x0, y0, _, _), d in zip(
            tiles,
            run(model, [im.crop(t) for t in tiles], a.net, a.device),
            strict=True,
        ):
            shifted.append(d + torch.tensor([x0, y0, x0, y0, 0.0]))
        dets["tiles"] = fuse(shifted)
        dets["global+tiles"] = fuse([dets["global"], *shifted])
        if a.ref:
            dets[f"full@{a.ref}"] = run(model, [im], a.ref, a.device)[0]

        scale = a.net / max(w, h)
        sizes = [math.sqrt(g.w * w * g.h * h) * scale for g in f.gt_bboxes]
        bks = [
            next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= s < hi) for s in sizes
        ]
        for g, b in zip(f.gt_bboxes, bks, strict=True):
            n_gt[b] += 1
            gt_in_band += tiles[0][1] <= g.cy * h <= tiles[0][3]
        for v in variants:
            for c in CONFS:
                ev = evaluate_frame(
                    gt=f.gt_bboxes,
                    predictions=to_preds(dets[v], w, h, c),
                    iou_thresh=0.3,
                    containment_thresh=0.7,
                )
                fp[v][c] += ev.pred_status.count("fp")
                for b, st in zip(bks, ev.gt_status, strict=True):
                    tp[v][c][b] += st == "tp"

    total = sum(n_gt.values())
    print(
        f"\n{n_img} images, {total} GT boxes, "
        f"{n_tiles / max(n_img, 1):.1f} tiles/image, "
        f"{gt_in_band / max(total, 1):.0%} of GT centres inside the tile band"
    )
    names = [f"{lo}-{hi}px (n={n_gt[i]})" for i, (lo, hi) in enumerate(BUCKETS)]
    print("| conf | variant | " + " | ".join(names) + " | recall | FP/img |")
    print("|" + "---|" * (len(BUCKETS) + 4))
    out = []
    for c in CONFS:
        for v in variants:
            rec = [
                tp[v][c][i] / n_gt[i] if n_gt[i] else float("nan")
                for i in range(len(BUCKETS))
            ]
            allr = sum(tp[v][c].values()) / max(total, 1)
            print(
                f"| {c} | {v} | "
                + " | ".join(f"{r:.1%}" for r in rec)
                + f" | {allr:.1%} | {fp[v][c] / max(n_img, 1):.3f} |"
            )
            out.append(
                {
                    "conf": c,
                    "variant": v,
                    "recall_by_bucket": rec,
                    "recall": allr,
                    "tp": sum(tp[v][c].values()),
                    "fp": fp[v][c],
                }
            )
    if a.output:
        a.output.write_text(
            json.dumps(
                {
                    "args": {k: str(v) for k, v in vars(a).items()},
                    "n_images": n_img,
                    "n_gt": dict(n_gt),
                    "rows": out,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
