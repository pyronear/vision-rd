"""Derive one fixed crop window per tube (pure, no Streamlit, no I/O).

Stabilization is a crop-window decision, not a tube-building step: it does not
change which detections link into which tube, so it never mutates the Tube. It
returns the union (enclosing) box of all the tube's observed detections, so the
same region can be cropped from every frame and the temporal head sees a static
background with the smoke moving inside it.

No context margin is applied here: the crop step expands this box by the same
``context_factor`` (``viz.CROP_CONTEXT``) it uses for per-frame crops, so the
stabilized crop stays in the model's training distribution. Returned coordinates
are normalized; clamping to the image happens at crop time in
``norm_bbox_to_pixel_square``.
"""

from __future__ import annotations

from bbox_tube_temporal.types import Tube


def tube_window(tube: Tube) -> tuple[float, float, float, float]:
    """Union (enclosing) box of the tube's observed detections, normalized.

    Entries without a detection (gaps) are ignored: interpolated gap boxes are
    lerps of observed boxes, so the union of observed boxes already encloses them.
    """
    dets = [e.detection for e in tube.entries if e.detection is not None]
    if not dets:
        raise ValueError("tube_window requires at least one observed detection")
    x0 = min(d.cx - d.w / 2 for d in dets)
    y0 = min(d.cy - d.h / 2 for d in dets)
    x1 = max(d.cx + d.w / 2 for d in dets)
    y1 = max(d.cy + d.h / 2 for d in dets)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
