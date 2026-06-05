"""Pure viz helpers (timeline shaping, bbox geometry, colours) + PIL wrappers.

Operates directly on lib ``Tube`` objects. The pure helpers are unit-tested;
the PIL/crop wrappers are exercised by the app and manual verification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from bbox_tube_temporal.model_input import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)
from bbox_tube_temporal.types import Tube
from PIL import Image, ImageDraw, ImageFont

CROP_CONTEXT = 1.6
CROP_SIZE = 224

TUBE_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

try:
    _BBOX_FONT = ImageFont.load_default(size=18)
except TypeError:  # older Pillow without the size kwarg
    _BBOX_FONT = ImageFont.load_default()


def tube_color(tube_id: int) -> str:
    """Stable colour for a tube id (cycles through a 10-colour palette)."""
    return TUBE_PALETTE[tube_id % len(TUBE_PALETTE)]


def tube_count(tubes: list[Tube]) -> int:
    return len(tubes)


def norm_bbox_to_pixel(
    bbox: tuple[float, float, float, float], w: int, h: int
) -> tuple[float, float, float, float]:
    """(cx,cy,w,h) normalized -> (x0,y0,x1,y1) in pixels."""
    cx, cy, bw, bh = bbox
    return (
        (cx - bw / 2) * w,
        (cy - bh / 2) * h,
        (cx + bw / 2) * w,
        (cy + bh / 2) * h,
    )


def bboxes_at_frame(
    tubes: list[Tube], frame_idx: int
) -> list[tuple[tuple[float, float, float, float], float, int, bool]]:
    """For each tube active at ``frame_idx``: (bbox, confidence, tube_id, is_gap)."""
    out = []
    for tube in tubes:
        for e in tube.entries:
            if e.frame_idx == frame_idx and e.detection is not None:
                d = e.detection
                out.append(
                    ((d.cx, d.cy, d.w, d.h), d.confidence, tube.tube_id, e.is_gap)
                )
                break
    return out


def tube_timeline_df(tubes: list[Tube]) -> pd.DataFrame:
    """Long frame for the Altair timeline: one row per present tube entry."""
    records = [
        {
            "tube": f"T{tube.tube_id}",
            "frame": e.frame_idx,
            "frame_end": e.frame_idx + 1,
            "confidence": e.detection.confidence,
            "is_gap": e.is_gap,
        }
        for tube in tubes
        for e in tube.entries
        if e.detection is not None
    ]
    return pd.DataFrame(
        records, columns=["tube", "frame", "frame_end", "confidence", "is_gap"]
    )


def draw_tube_bboxes(
    image_path: Path, tubes: list[Tube], frame_idx: int, width: int = 4
):
    """Frame image with each active tube's bbox drawn in its tube colour."""
    img = Image.open(image_path).convert("RGB")
    w_img, h_img = img.size
    draw = ImageDraw.Draw(img)
    for bbox, conf, tid, is_gap in bboxes_at_frame(tubes, frame_idx):
        x0, y0, x1, y1 = norm_bbox_to_pixel(bbox, w_img, h_img)
        color = tube_color(tid)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        label = f"T{tid}" + (" (gap)" if is_gap else f" {conf:.2f}")
        draw.text((x0, max(0, y0 - 20)), label, fill=color, font=_BBOX_FONT)
    return img


def crop_box_px(
    bbox: tuple[float, float, float, float], img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Square pixel crop box for a normalized bbox, expanded by ``CROP_CONTEXT``.

    The single source of truth for the crop region: ``crop_tube_at_frame``
    extracts it and ``draw_tube_windows`` draws it, so the box you see equals the
    crop you get — for both per-frame boxes and stabilized union windows."""
    cx, cy, w, h = expand_bbox(*bbox, CROP_CONTEXT)
    return norm_bbox_to_pixel_square(cx, cy, w, h, img_w, img_h)


def crop_tube_at_frame(image_path: Path, bbox: tuple[float, float, float, float]):
    """Square context crop centred on a normalized bbox (matches the explorer).

    Pass a per-frame detection box for the non-stabilized crop, or a tube's union
    window (``stabilize.tube_window``) for the stabilized crop — same code path,
    same ``CROP_CONTEXT``."""
    img = np.array(Image.open(image_path).convert("RGB"))
    img_h, img_w = img.shape[:2]
    return Image.fromarray(
        crop_and_resize(img, crop_box_px(bbox, img_w, img_h), CROP_SIZE)
    )


def draw_tube_windows(
    image_path: Path,
    windows: list[tuple[tuple[float, float, float, float], str]],
    width: int = 5,
):
    """Frame image with each ``(window, colour)`` drawn as its FIXED stabilized
    crop region (the exact region ``crop_tube_at_frame`` extracts for that union
    window). The stabilized-side mirror of ``draw_tube_bboxes``: each box is in
    its tube colour and identical on every frame the tube is active."""
    img = Image.open(image_path).convert("RGB")
    w_img, h_img = img.size
    draw = ImageDraw.Draw(img)
    for window, color in windows:
        draw.rectangle(
            list(crop_box_px(window, w_img, h_img)),
            outline=color,
            width=width,
        )
    return img
