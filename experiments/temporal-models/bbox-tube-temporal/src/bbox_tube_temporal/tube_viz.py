"""Visualisation helpers for smoke tubes.

Split from :mod:`bbox_tube_temporal.tubes` to keep construction logic
free of matplotlib / opencv dependencies.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .types import Detection, Tube

# Distinct colours for up to 20 tubes (stored as RGB)
_TUBE_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
]


def draw_tubes_on_frames(
    image_paths: list[Path],
    tubes: list[Tube],
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> list[np.ndarray]:
    """Draw tube bounding boxes on each frame image.

    Each tube gets a unique colour.  Solid rectangles mark detected
    frames; dashed rectangles mark gap frames (using the last known
    bbox position from that tube).

    Args:
        image_paths: Ordered list of frame image paths.
        tubes: Tubes to draw.
        line_thickness: Rectangle line width in pixels.
        font_scale: Font scale for tube ID labels.

    Returns:
        List of annotated BGR images (same order as *image_paths*).
    """
    images = [cv2.imread(str(p)) for p in image_paths]

    # Build per-frame lookup: frame_idx -> list[(tube_id, det, is_gap)]
    frame_annotations: dict[int, list[tuple[int, Detection, bool]]] = {}
    for tube in tubes:
        last_det: Detection | None = None
        for entry in tube.entries:
            if entry.detection is not None:
                last_det = entry.detection
                frame_annotations.setdefault(entry.frame_idx, []).append(
                    (tube.tube_id, entry.detection, False)
                )
            elif last_det is not None:
                frame_annotations.setdefault(entry.frame_idx, []).append(
                    (tube.tube_id, last_det, True)
                )

    for frame_idx, img in enumerate(images):
        if img is None or frame_idx not in frame_annotations:
            continue
        h, w = img.shape[:2]
        for tube_id, det, is_gap in frame_annotations[frame_idx]:
            rgb = _TUBE_COLORS[tube_id % len(_TUBE_COLORS)]
            color = (rgb[2], rgb[1], rgb[0])  # RGB -> BGR for cv2
            x1 = int((det.cx - det.w / 2) * w)
            y1 = int((det.cy - det.h / 2) * h)
            x2 = int((det.cx + det.w / 2) * w)
            y2 = int((det.cy + det.h / 2) * h)

            if is_gap:
                _draw_dashed_rect(img, (x1, y1), (x2, y2), color, line_thickness)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

            label = f"T{tube_id}"
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                max(1, line_thickness - 1),
            )

    return images


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 8,
) -> None:
    """Draw a dashed rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2
    for edge in [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]:
        _draw_dashed_line(img, edge[0], edge[1], color, thickness, dash_length)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 8,
) -> None:
    """Draw a dashed line between two points."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = max(1, int(np.hypot(dx, dy)))
    num_dashes = max(1, length // (dash_length * 2))
    for i in range(num_dashes):
        t_start = i / num_dashes
        t_end = (i + 0.5) / num_dashes
        start = (
            int(pt1[0] + dx * t_start),
            int(pt1[1] + dy * t_start),
        )
        end = (
            int(pt1[0] + dx * t_end),
            int(pt1[1] + dy * t_end),
        )
        cv2.line(img, start, end, color, thickness)


def plot_tube_grid(
    annotated_frames: list[np.ndarray],
    frame_ids: list[str] | None = None,
    cols: int = 5,
    figsize_per_cell: tuple[float, float] = (3.0, 3.0),
) -> plt.Figure:
    """Arrange annotated frames in a grid.

    Args:
        annotated_frames: BGR images from :func:`draw_tubes_on_frames`.
        frame_ids: Optional labels for each frame.
        cols: Number of columns in the grid.
        figsize_per_cell: ``(width, height)`` per cell in inches.

    Returns:
        A matplotlib :class:`Figure`.
    """
    n = len(annotated_frames)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_cell[0], rows * figsize_per_cell[1]),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, ax in enumerate(axes_flat):
        if i < n:
            img_rgb = cv2.cvtColor(annotated_frames[i], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            title = frame_ids[i] if frame_ids else f"Frame {i}"
            ax.set_title(title, fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_tube_timeline(
    tubes: list[Tube],
    num_frames: int,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot a timeline diagram of tubes.

    Each row is a tube, time on x-axis.  Filled cells = detection,
    hatched cells = gap.

    Args:
        tubes: Tubes to visualise.
        num_frames: Total number of frames in the sequence.
        figsize: Optional figure size.

    Returns:
        A matplotlib :class:`Figure`.
    """
    if not tubes:
        fig, ax = plt.subplots(figsize=figsize or (8, 1))
        ax.text(0.5, 0.5, "No tubes", ha="center", va="center")
        ax.axis("off")
        return fig

    n_tubes = len(tubes)
    if figsize is None:
        figsize = (max(8, num_frames * 0.5), max(2, n_tubes * 0.6))

    fig, ax = plt.subplots(figsize=figsize)

    for row, tube in enumerate(tubes):
        idx = tube.tube_id % len(_TUBE_COLORS)
        color_rgb = tuple(c / 255 for c in _TUBE_COLORS[idx])
        entry_map = {e.frame_idx: e for e in tube.entries}

        for fi in range(tube.start_frame, tube.end_frame + 1):
            entry = entry_map.get(fi)
            is_gap = entry is not None and entry.detection is None
            is_det = entry is not None and entry.detection is not None

            if is_det:
                ax.barh(
                    row,
                    1,
                    left=fi,
                    height=0.6,
                    color=color_rgb,
                    edgecolor="black",
                    linewidth=0.5,
                )
            elif is_gap:
                ax.barh(
                    row,
                    1,
                    left=fi,
                    height=0.6,
                    color="white",
                    edgecolor=color_rgb,
                    linewidth=1.0,
                    linestyle="--",
                )

    ax.set_yticks(range(n_tubes))
    ax.set_yticklabels([f"Tube {t.tube_id}" for t in tubes], fontsize=8)
    ax.set_xlabel("Frame index")
    ax.set_xlim(-0.5, num_frames + 0.5)
    ax.set_xticks(range(num_frames))
    ax.set_ylim(-0.5, n_tubes - 0.5)
    ax.invert_yaxis()
    ax.set_title("Tube Timeline")
    fig.tight_layout()
    return fig


def _crop_detection(
    image: np.ndarray,
    det: Detection,
    context_factor: float = 1.2,
    crop_size: int = 112,
) -> np.ndarray:
    """Crop a detection region from an image with context expansion.

    Args:
        image: BGR image array.
        det: Detection with normalised coordinates.
        context_factor: Bbox expansion factor.
        crop_size: Output size ``(crop_size, crop_size)``.

    Returns:
        Resized BGR crop.
    """
    h_img, w_img = image.shape[:2]
    w = det.w * w_img * context_factor
    h = det.h * h_img * context_factor
    cx = det.cx * w_img
    cy = det.cy * h_img

    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(w_img, int(cx + w / 2))
    y2 = min(h_img, int(cy + h / 2))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (crop_size, crop_size))


def _make_gap_cell(crop_size: int = 112) -> np.ndarray:
    """Create a grey placeholder image for gap frames."""
    cell = np.full((crop_size, crop_size, 3), 180, dtype=np.uint8)
    cv2.putText(
        cell,
        "gap",
        (crop_size // 4, crop_size // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (80, 80, 80),
        2,
    )
    return cell


def plot_tube_filmstrips(
    image_paths: list[Path],
    tubes: list[Tube],
    tube_labels: list[bool] | None = None,
    context_factor: float = 1.2,
    crop_size: int = 112,
    max_tubes: int = 10,
) -> plt.Figure:
    """Plot a filmstrip for each tube showing cropped detection regions.

    Each row is a tube, each column a time step.  Detected frames show
    the image crop at the bbox (with context expansion).  Gap frames
    show a grey placeholder.

    When *tube_labels* is provided, rows are colour-coded: green
    background for smoke (``True``), red for false positive (``False``).

    Args:
        image_paths: Ordered list of frame image paths.
        tubes: Tubes to visualise.
        tube_labels: Per-tube ground-truth flag (``True`` = smoke).
            Must have the same length as *tubes*.  ``None`` to skip.
        context_factor: Bbox expansion factor for crops.
        crop_size: Size to resize each crop to.
        max_tubes: Maximum number of tubes to display.

    Returns:
        A matplotlib :class:`Figure`.
    """
    if not tubes:
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.text(0.5, 0.5, "No tubes", ha="center", va="center")
        ax.axis("off")
        return fig

    display_tubes = tubes[:max_tubes]
    display_labels = tube_labels[:max_tubes] if tube_labels is not None else None

    # Load images lazily (cache to avoid re-reading)
    image_cache: dict[int, np.ndarray] = {}

    def _get_image(frame_idx: int) -> np.ndarray:
        if frame_idx not in image_cache:
            image_cache[frame_idx] = cv2.imread(str(image_paths[frame_idx]))
        return image_cache[frame_idx]

    n_tubes = len(display_tubes)
    max_cols = max(len(t.entries) for t in display_tubes)

    cell_inches = crop_size / 80
    fig, axes = plt.subplots(
        n_tubes,
        max_cols,
        figsize=(max_cols * cell_inches, n_tubes * cell_inches),
        squeeze=False,
    )

    # GT border colour (BGR for cv2): green = smoke, red = FP
    gt_border_colors = {
        True: (0, 180, 0),
        False: (0, 0, 220),
    }
    border_px = 4

    for row, tube in enumerate(display_tubes):
        idx = tube.tube_id % len(_TUBE_COLORS)
        color_rgb = tuple(c / 255 for c in _TUBE_COLORS[idx])

        # Determine GT border for this row
        gt_border = None
        if display_labels is not None:
            gt_border = gt_border_colors[display_labels[row]]

        for col in range(max_cols):
            ax = axes[row, col]
            ax.axis("off")

            if col >= len(tube.entries):
                continue

            entry = tube.entries[col]

            if entry.detection is not None:
                img = _get_image(entry.frame_idx)
                crop = _crop_detection(
                    img,
                    entry.detection,
                    context_factor,
                    crop_size,
                )
                if gt_border is not None:
                    _draw_border(crop, gt_border, border_px)
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ax.imshow(crop_rgb)
                for spine in ax.spines.values():
                    spine.set_edgecolor(color_rgb)
                    spine.set_linewidth(2)
                    spine.set_visible(True)
            else:
                gap = _make_gap_cell(crop_size)
                if gt_border is not None:
                    _draw_border(gap, gt_border, border_px)
                gap_rgb = cv2.cvtColor(gap, cv2.COLOR_BGR2RGB)
                ax.imshow(gap_rgb)
                for spine in ax.spines.values():
                    spine.set_edgecolor(color_rgb)
                    spine.set_linewidth(1)
                    spine.set_linestyle((0, (4, 3)))
                    spine.set_visible(True)

            if row == 0:
                ax.set_title(f"F{entry.frame_idx}", fontsize=7)

        # Row label
        gt_tag = ""
        label_color = "black"
        if display_labels is not None:
            gt_tag = " SMOKE" if display_labels[row] else " FP"
            label_color = "green" if display_labels[row] else "red"
        axes[row, 0].set_ylabel(
            f"T{tube.tube_id}{gt_tag}",
            fontsize=8,
            rotation=0,
            labelpad=30,
            va="center",
            color=label_color,
        )

    fig.tight_layout()
    return fig


def _draw_border(
    img: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 4,
) -> None:
    """Draw a solid coloured border around an image (in-place)."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness)


def plot_tube_summary(
    image_paths: list[Path],
    tubes: list[Tube],
    num_frames: int,
    tube_labels: list[bool] | None = None,
    context_factor: float = 1.2,
    crop_size: int = 112,
    max_tubes: int = 10,
    title: str | None = None,
) -> plt.Figure:
    """Combined view: tube timeline on top, filmstrips below.

    Args:
        image_paths: Ordered list of frame image paths.
        tubes: Tubes to visualise.
        num_frames: Total number of frames in the sequence.
        tube_labels: Per-tube ground-truth (``True`` = smoke).
        context_factor: Bbox expansion for crops.
        crop_size: Crop resize target.
        max_tubes: Maximum tubes to display.
        title: Optional figure title.

    Returns:
        A matplotlib :class:`Figure`.
    """
    if not tubes:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No tubes", ha="center", va="center")
        ax.axis("off")
        return fig

    display_tubes = tubes[:max_tubes]
    display_labels = tube_labels[:max_tubes] if tube_labels is not None else None
    n_tubes = len(display_tubes)
    max_cols = max(len(t.entries) for t in display_tubes)
    cell_inches = crop_size / 80

    fig_w = max(8, max_cols * cell_inches)
    timeline_h = max(1.5, n_tubes * 0.5)
    filmstrip_h = n_tubes * cell_inches
    fig_h = timeline_h + filmstrip_h + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[timeline_h, filmstrip_h],
        hspace=0.35,
    )

    # ── Top: timeline ────────────────────────────────────────────
    ax_tl = fig.add_subplot(gs[0])
    for row, tube in enumerate(display_tubes):
        idx = tube.tube_id % len(_TUBE_COLORS)
        color_rgb = tuple(c / 255 for c in _TUBE_COLORS[idx])
        entry_map = {e.frame_idx: e for e in tube.entries}

        for fi in range(tube.start_frame, tube.end_frame + 1):
            entry = entry_map.get(fi)
            is_gap = entry is not None and entry.detection is None
            is_det = entry is not None and entry.detection is not None

            if is_det:
                ax_tl.barh(
                    row,
                    1,
                    left=fi,
                    height=0.6,
                    color=color_rgb,
                    edgecolor="black",
                    linewidth=0.5,
                )
            elif is_gap:
                ax_tl.barh(
                    row,
                    1,
                    left=fi,
                    height=0.6,
                    color="white",
                    edgecolor=color_rgb,
                    linewidth=1.0,
                    linestyle="--",
                )

    ax_tl.set_yticks(range(n_tubes))
    ax_tl.set_yticklabels([f"Tube {t.tube_id}" for t in display_tubes], fontsize=8)
    ax_tl.set_xlabel("Frame index")
    ax_tl.set_xlim(-0.5, num_frames + 0.5)
    ax_tl.set_xticks(range(num_frames))
    ax_tl.set_ylim(-0.5, n_tubes - 0.5)
    ax_tl.invert_yaxis()
    ax_tl.set_title("Tube Timeline", fontsize=9)

    # ── Bottom: filmstrips ───────────────────────────────────────
    gs_strips = gs[1].subgridspec(n_tubes, max_cols, hspace=0.1)

    image_cache: dict[int, np.ndarray] = {}

    def _get_image(frame_idx: int) -> np.ndarray:
        if frame_idx not in image_cache:
            image_cache[frame_idx] = cv2.imread(str(image_paths[frame_idx]))
        return image_cache[frame_idx]

    gt_border_colors = {True: (0, 180, 0), False: (0, 0, 220)}
    border_px = 4

    for row, tube in enumerate(display_tubes):
        idx = tube.tube_id % len(_TUBE_COLORS)
        color_rgb = tuple(c / 255 for c in _TUBE_COLORS[idx])

        gt_border = None
        if display_labels is not None:
            gt_border = gt_border_colors[display_labels[row]]

        for col in range(max_cols):
            ax = fig.add_subplot(gs_strips[row, col])
            ax.axis("off")

            if col >= len(tube.entries):
                continue

            entry = tube.entries[col]

            if entry.detection is not None:
                img = _get_image(entry.frame_idx)
                crop = _crop_detection(
                    img,
                    entry.detection,
                    context_factor,
                    crop_size,
                )
                if gt_border is not None:
                    _draw_border(crop, gt_border, border_px)
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                ax.imshow(crop_rgb)
                for spine in ax.spines.values():
                    spine.set_edgecolor(color_rgb)
                    spine.set_linewidth(2)
                    spine.set_visible(True)
            else:
                gap = _make_gap_cell(crop_size)
                if gt_border is not None:
                    _draw_border(gap, gt_border, border_px)
                gap_rgb = cv2.cvtColor(gap, cv2.COLOR_BGR2RGB)
                ax.imshow(gap_rgb)
                for spine in ax.spines.values():
                    spine.set_edgecolor(color_rgb)
                    spine.set_linewidth(1)
                    spine.set_linestyle((0, (4, 3)))
                    spine.set_visible(True)

        # Row label for first column
        first_ax = fig.add_subplot(gs_strips[row, 0])
        first_ax.axis("off")
        gt_tag = ""
        label_color = "black"
        if display_labels is not None:
            gt_tag = " SMOKE" if display_labels[row] else " FP"
            label_color = "green" if display_labels[row] else "red"
        first_ax.set_ylabel(
            f"T{tube.tube_id}{gt_tag}",
            fontsize=7,
            rotation=0,
            labelpad=30,
            va="center",
            color=label_color,
        )

    if title:
        fig.suptitle(title, fontsize=10, y=1.01)

    return fig
