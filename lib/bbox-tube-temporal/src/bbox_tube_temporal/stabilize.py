"""Per-tube fixed crop window (pure geometry, no I/O).

Stabilization is a crop-window decision, not a tube-building step. ``union_window``
returns the enclosing box of a tube's observed detection boxes so the same region
can be cropped from every frame — a static background with the smoke moving inside
it. No context margin here: the crop step adds context via ``context_factor``.
"""

from __future__ import annotations


def union_window(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Enclosing box of ``boxes`` (each normalized ``(cx, cy, w, h)``).

    Returns normalized ``(cx, cy, w, h)``. Raises ``ValueError`` if ``boxes`` is
    empty. Width and height are computed independently from the x/y extents.
    """
    if not boxes:
        raise ValueError("union_window requires at least one box")
    x0 = min(cx - w / 2 for cx, _, w, _ in boxes)
    y0 = min(cy - h / 2 for _, cy, _, h in boxes)
    x1 = max(cx + w / 2 for cx, _, w, _ in boxes)
    y1 = max(cy + h / 2 for _, cy, _, h in boxes)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def tube_union_window(
    entries,
) -> tuple[float, float, float, float] | None:
    """Stabilized crop window for a tube: union of its observed detection boxes.

    ``entries`` are tube entries exposing ``.detection`` (with ``cx, cy, w, h``)
    or ``None``, and ``.is_gap``. Prefers real (non-gap) detections; falls back
    to any detection when every entry is a gap. Returns ``None`` if the tube has
    no detections at all. The single source of truth for both the crop step and
    the window reported in ``predict()`` details.
    """
    observed = [
        (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
        for e in entries
        if e.detection is not None and not e.is_gap
    ]
    if not observed:
        observed = [
            (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
            for e in entries
            if e.detection is not None
        ]
    if not observed:
        return None
    return union_window(observed)
