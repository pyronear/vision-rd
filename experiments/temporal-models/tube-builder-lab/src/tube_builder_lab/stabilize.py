"""Derive one fixed crop window per tube (pure, no Streamlit, no I/O).

Stabilization is a crop-window decision, not a tube-building step: it does not
change which detections link into which tube, so it never mutates the Tube. It
returns a single normalized (cx, cy, w, h) window — the enclosing box of all the
tube's observed detections, expanded by a context margin — so the same crop can
be taken from every frame and the temporal head sees a static background with
the smoke moving inside it.

Returned coordinates are normalized and may exceed [0, 1] once the margin is
applied; clamping to the image happens at crop time in
``norm_bbox_to_pixel_square``.
"""

from __future__ import annotations

from bbox_tube_temporal.types import Tube

# Context margin applied to the union box. Tune in-file (saved edits reload live
# in the lab, same workflow as candidate.py).
MARGIN = 1.3


def tube_window(
    tube: Tube, margin: float = MARGIN
) -> tuple[float, float, float, float]:
    """Enclosing box of the tube's observed detections, expanded by ``margin``.

    Returns normalized ``(cx, cy, w, h)``. Entries without a detection (gaps) are
    ignored: interpolated gap boxes are lerps of observed boxes, so the union of
    observed boxes already encloses them.
    """
    dets = [e.detection for e in tube.entries if e.detection is not None]
    if not dets:
        raise ValueError("tube_window requires at least one observed detection")
    x0 = min(d.cx - d.w / 2 for d in dets)
    y0 = min(d.cy - d.h / 2 for d in dets)
    x1 = max(d.cx + d.w / 2 for d in dets)
    y1 = max(d.cy + d.h / 2 for d in dets)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = (x1 - x0) * margin
    h = (y1 - y0) * margin
    return cx, cy, w, h
