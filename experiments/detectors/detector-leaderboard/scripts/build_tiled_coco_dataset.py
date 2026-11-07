"""Tile a flat YOLO split into overlapping 640x640 tiles -> COCO dataset.

Motivation: DEIMv2-S runs at 640px, so a ~1280px frame is downscaled ~2x and
small smoke detail is lost. Training on native-resolution tiles keeps that detail
and produces many background/negative tiles (positives are ~1 sparse box/image).

Each image is sliced into ``tile``x``tile`` tiles with ``overlap`` (edge tiles are
shifted back so every tile is exactly ``tile``x``tile`` — no resize/pad). A GT box
is kept for a tile if its intersection with the tile covers >= ``min_visibility``
of the original box area; it is clipped to the tile and written in tile-local COCO
pixel ``xywh`` (category id 1 = smoke). Tiles with >=1 kept box ("positive") are
oversampled ``pos_oversample`` times to balance the negative-heavy distribution.

Output mirrors the rfdetr_coco layout: ``<out>/{train,valid}/<tiles>.jpg`` +
``_annotations.coco.json`` (so the DEIMv2 CocoDetection reads it directly).

    uv run python scripts/build_tiled_coco_dataset.py \
        --train-dir data/01_raw/datasets/train --val-dir data/01_raw/datasets/val \
        --output-dir data/05_model_input/tiled_coco \
        --tile 640 --overlap 0.2 --min-visibility 0.3 --pos-oversample 2
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from detector_leaderboard.data import list_frame_images, parse_yolo_label

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = [{"id": 1, "name": "smoke", "supercategory": "none"}]


def tile_starts(dim: int, tile: int, overlap: float) -> list[int]:
    """Tile-origin coordinates along one axis; edge tile shifted to stay full-size."""
    if dim <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    starts = list(range(0, dim - tile + 1, stride))
    if starts[-1] != dim - tile:
        starts.append(dim - tile)
    return starts


def _build_split(
    src_split: Path,
    dst_split: Path,
    tile: int,
    overlap: float,
    min_visibility: float,
    pos_oversample: int,
    max_neg_per_image: int,
) -> tuple[int, int, int]:
    """Slice one split into tiles. Returns (n_tile_entries, n_pos_tiles, n_boxes)."""
    dst_split.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    img_id, ann_id = 0, 1
    n_pos = 0

    frames = list_frame_images(src_split)
    for frame_path in tqdm(frames, desc=f"tiling {src_split.name}"):
        with Image.open(frame_path) as im:
            im = im.convert("RGB")
            width, height = im.size
            # GT boxes -> full-image pixel xyxy
            gt = []
            for b in parse_yolo_label(src_split / "labels" / f"{frame_path.stem}.txt"):
                bw, bh = b.w * width, b.h * height
                x1, y1 = (b.cx - b.w / 2) * width, (b.cy - b.h / 2) * height
                gt.append((x1, y1, x1 + bw, y1 + bh, bw * bh))

            neg_written = 0
            for y0 in tile_starts(height, tile, overlap):
                for x0 in tile_starts(width, tile, overlap):
                    kept = []
                    for bx1, by1, bx2, by2, area in gt:
                        ix1, iy1 = max(bx1, x0), max(by1, y0)
                        ix2, iy2 = min(bx2, x0 + tile), min(by2, y0 + tile)
                        iw, ih = ix2 - ix1, iy2 - iy1
                        if iw <= 0 or ih <= 0 or area <= 0:
                            continue
                        if (iw * ih) / area < min_visibility:
                            continue
                        kept.append([ix1 - x0, iy1 - y0, iw, ih])  # tile-local xywh
                    is_pos = len(kept) > 0
                    if not is_pos and neg_written >= max_neg_per_image:
                        continue

                    fname = f"{frame_path.stem}_x{x0}_y{y0}.jpg"
                    crop = im.crop((x0, y0, x0 + tile, y0 + tile))
                    crop.save(dst_split / fname, quality=95)
                    if not is_pos:
                        neg_written += 1
                    else:
                        n_pos += 1

                    # Positive tiles get `pos_oversample` COCO entries (same file).
                    copies = pos_oversample if is_pos else 1
                    for _ in range(copies):
                        img_id += 1
                        images.append(
                            {
                                "id": img_id,
                                "file_name": fname,
                                "width": tile,
                                "height": tile,
                            }
                        )
                        for x, y, bw, bh in kept:
                            annotations.append(
                                {
                                    "id": ann_id,
                                    "image_id": img_id,
                                    "category_id": 1,
                                    "bbox": [
                                        round(x, 2),
                                        round(y, 2),
                                        round(bw, 2),
                                        round(bh, 2),
                                    ],
                                    "area": round(bw * bh, 2),
                                    "iscrowd": 0,
                                }
                            )
                            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    (dst_split / "_annotations.coco.json").write_text(json.dumps(coco))
    return len(images), n_pos, len(annotations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile YOLO splits into COCO tiles.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--min-visibility", type=float, default=0.3)
    parser.add_argument("--pos-oversample", type=int, default=2)
    parser.add_argument("--max-neg-per-image", type=int, default=10**9)
    args = parser.parse_args()

    for src, name in [(args.train_dir, "train"), (args.val_dir, "valid")]:
        n_entries, n_pos, n_boxes = _build_split(
            src,
            args.output_dir / name,
            args.tile,
            args.overlap,
            args.min_visibility,
            args.pos_oversample,
            args.max_neg_per_image,
        )
        logger.info(
            "%s -> %s: %d tile-entries (%d positive incl. x%d oversample), %d boxes",
            src,
            name,
            n_entries,
            n_pos,
            args.pos_oversample,
            n_boxes,
        )


if __name__ == "__main__":
    main()
