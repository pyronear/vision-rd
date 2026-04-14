"""Smoke tube construction via greedy IoU matching across frames.

A *tube* is a chain of YOLO detections across consecutive frames that
correspond to the same spatial smoke region.  Tubes bridge per-frame
detections and the LSTM's need for temporally-ordered features of the
same entity.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .data import load_gt_labels
from .types import Detection, FrameDetections, Tube, TubeEntry

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


def compute_iou(det_a: Detection, det_b: Detection) -> float:
    """Compute Intersection-over-Union between two detections.

    Both detections use normalised center-based coordinates
    (cx, cy, w, h) in [0, 1].

    Returns:
        IoU value in [0, 1].  Returns 0.0 when the union area is zero.
    """
    a_x1 = det_a.cx - det_a.w / 2
    a_y1 = det_a.cy - det_a.h / 2
    a_x2 = det_a.cx + det_a.w / 2
    a_y2 = det_a.cy + det_a.h / 2

    b_x1 = det_b.cx - det_b.w / 2
    b_y1 = det_b.cy - det_b.h / 2
    b_x2 = det_b.cx + det_b.w / 2
    b_y2 = det_b.cy + det_b.h / 2

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = det_a.w * det_a.h
    b_area = det_b.w * det_b.h
    union_area = a_area + b_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_detections(
    prev_dets: list[Detection],
    curr_dets: list[Detection],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching of detections between consecutive frames.

    Args:
        prev_dets: Detections from the previous frame.
        curr_dets: Detections from the current frame.
        iou_threshold: Minimum IoU required to consider a match.

    Returns:
        List of ``(prev_idx, curr_idx, iou)`` tuples.
    """
    if not prev_dets or not curr_dets:
        return []

    pairs: list[tuple[float, int, int]] = []
    for i, prev in enumerate(prev_dets):
        for j, curr in enumerate(curr_dets):
            iou = compute_iou(prev, curr)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))

    pairs.sort(key=lambda x: -x[0])
    matched_prev: set[int] = set()
    matched_curr: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou_val, i, j in pairs:
        if i not in matched_prev and j not in matched_curr:
            matches.append((i, j, iou_val))
            matched_prev.add(i)
            matched_curr.add(j)

    return matches


def build_tubes(
    frame_detections: list[FrameDetections],
    iou_threshold: float = 0.2,
    max_misses: int = 2,
) -> list[Tube]:
    """Build smoke tubes from per-frame detections.

    Tubes are constructed by greedy IoU matching: each detection in
    frame *t* is matched to the closest (by IoU) active tube from
    frame *t-1*.  Unmatched detections start new tubes; unmatched
    tubes accumulate misses and are terminated after *max_misses*
    consecutive gaps.

    Args:
        frame_detections: Temporally ordered per-frame detection lists.
        iou_threshold: Minimum IoU for matching a detection to a tube.
        max_misses: Maximum consecutive gap frames before a tube is
            terminated.

    Returns:
        List of :class:`Tube` objects covering the full sequence.
    """
    if not frame_detections:
        return []

    active_tubes: list[Tube] = []
    finished_tubes: list[Tube] = []
    next_tube_id = 0
    # Track consecutive misses per active tube
    consecutive_misses: list[int] = []

    for frame in frame_detections:
        frame_idx = frame.frame_idx
        curr_dets = frame.detections

        if active_tubes and curr_dets:
            # Get last known detection for each active tube
            last_dets = [_last_detection(tube) for tube in active_tubes]
            matches = match_detections(last_dets, curr_dets, iou_threshold)
            matched_tube_idxs = {m[0] for m in matches}
            matched_det_idxs = {m[1] for m in matches}

            # Extend matched tubes
            for tube_idx, det_idx, _ in matches:
                tube = active_tubes[tube_idx]
                # Fill gap entries for any missed frames
                _fill_gaps(tube, frame_idx)
                tube.entries.append(
                    TubeEntry(frame_idx=frame_idx, detection=curr_dets[det_idx])
                )
                tube.end_frame = frame_idx
                consecutive_misses[tube_idx] = 0

            # Increment misses for unmatched tubes
            for i in range(len(active_tubes)):
                if i not in matched_tube_idxs:
                    consecutive_misses[i] += 1

            # Start new tubes for unmatched detections
            for j, det in enumerate(curr_dets):
                if j not in matched_det_idxs:
                    new_tube = Tube(
                        tube_id=next_tube_id,
                        entries=[TubeEntry(frame_idx=frame_idx, detection=det)],
                        start_frame=frame_idx,
                        end_frame=frame_idx,
                    )
                    next_tube_id += 1
                    active_tubes.append(new_tube)
                    consecutive_misses.append(0)
        else:
            # Increment misses for all active tubes (no detections this frame)
            for i in range(len(active_tubes)):
                consecutive_misses[i] += 1

            # Start new tubes for all detections
            for det in curr_dets:
                new_tube = Tube(
                    tube_id=next_tube_id,
                    entries=[TubeEntry(frame_idx=frame_idx, detection=det)],
                    start_frame=frame_idx,
                    end_frame=frame_idx,
                )
                next_tube_id += 1
                active_tubes.append(new_tube)
                consecutive_misses.append(0)

        # Prune tubes that exceeded max_misses
        surviving_tubes: list[Tube] = []
        surviving_misses: list[int] = []
        for tube, misses in zip(active_tubes, consecutive_misses, strict=True):
            if misses > max_misses:
                finished_tubes.append(tube)
            else:
                surviving_tubes.append(tube)
                surviving_misses.append(misses)
        active_tubes = surviving_tubes
        consecutive_misses = surviving_misses

    # All remaining active tubes are finished
    finished_tubes.extend(active_tubes)

    # Sort by tube_id for deterministic output
    finished_tubes.sort(key=lambda t: t.tube_id)
    return finished_tubes


def compute_containment(det_a: Detection, det_b: Detection) -> float:
    """Compute how much of the smaller box is inside the larger one.

    Returns ``intersection_area / min(area_a, area_b)``.  This is more
    robust than IoU when one box is much larger than the other (e.g.,
    a large YOLO detection fully containing a small GT annotation).

    Returns:
        Containment ratio in [0, 1].  Returns 0.0 when either area is zero.
    """
    a_x1 = det_a.cx - det_a.w / 2
    a_y1 = det_a.cy - det_a.h / 2
    a_x2 = det_a.cx + det_a.w / 2
    a_y2 = det_a.cy + det_a.h / 2

    b_x1 = det_b.cx - det_b.w / 2
    b_y1 = det_b.cy - det_b.h / 2
    b_x2 = det_b.cx + det_b.w / 2
    b_y2 = det_b.cy + det_b.h / 2

    inter_w = max(0.0, min(a_x2, b_x2) - max(a_x1, b_x1))
    inter_h = max(0.0, min(a_y2, b_y2) - max(a_y1, b_y1))
    inter_area = inter_w * inter_h

    min_area = min(det_a.w * det_a.h, det_b.w * det_b.h)
    if min_area <= 0:
        return 0.0
    return inter_area / min_area


def classify_tube_gt(
    tube: Tube,
    sequence_dir: Path,
    frame_ids: list[str],
    containment_threshold: float = 0.5,
) -> bool:
    """Determine whether a tube corresponds to real smoke.

    Uses **containment** (intersection / smaller box area) rather than
    IoU.  This handles the common case where YOLO produces a larger
    bounding box that fully contains a small GT annotation -- IoU
    would be low due to the size mismatch, but containment is high.

    For FP sequences (6-column labels), :func:`load_gt_labels` returns
    nothing, so all tubes will be classified as negative.

    Args:
        tube: The tube to classify.
        sequence_dir: Path to the sequence directory (contains ``labels/``).
        frame_ids: Ordered list of frame ID strings for the sequence.
        containment_threshold: Minimum containment ratio to count as a match.

    Returns:
        ``True`` if the tube overlaps with ground-truth smoke.
    """
    for entry in tube.entries:
        if entry.detection is None:
            continue
        gt_boxes = load_gt_labels(sequence_dir, frame_ids[entry.frame_idx])
        for gt_cx, gt_cy, gt_w, gt_h in gt_boxes:
            gt_det = Detection(
                class_id=0,
                cx=gt_cx,
                cy=gt_cy,
                w=gt_w,
                h=gt_h,
                confidence=1.0,
            )
            if compute_containment(entry.detection, gt_det) >= containment_threshold:
                return True
    return False


def _last_detection(tube: Tube) -> Detection:
    """Return the most recent non-gap detection in a tube."""
    for entry in reversed(tube.entries):
        if entry.detection is not None:
            return entry.detection
    raise ValueError(f"Tube {tube.tube_id} has no detections")


def _fill_gaps(tube: Tube, target_frame_idx: int) -> None:
    """Insert gap entries (detection=None) for missed frames."""
    if not tube.entries:
        return
    last_frame = tube.entries[-1].frame_idx
    for gap_idx in range(last_frame + 1, target_frame_idx):
        tube.entries.append(TubeEntry(frame_idx=gap_idx, detection=None))


# ── Visualisation ────────────────────────────────────────────────────────


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


def interpolate_gaps(tube: Tube) -> Tube:
    """Fill gap entries with a geometrically-interpolated bbox.

    For each entry whose ``detection`` is ``None``:

    * **Interior gap** (observed dets on both sides): linearly interpolate
      ``(cx, cy, w, h)`` between the nearest observed detection before and
      after, using the entry's index as the position parameter.
    * **Boundary gap** (no observation on one side): repeat the nearest
      observed detection on the other side.

    Synthesized detections always carry ``confidence=0.0``. The returned
    tube has ``is_gap=True`` flags on every previously-empty entry.

    Observed entries are left untouched.

    Args:
        tube: Tube whose gap entries (``detection=None``) need filling.

    Returns:
        The same tube object, mutated in place. Returned for chaining.
    """
    observed = [
        (i, e.detection) for i, e in enumerate(tube.entries) if e.detection is not None
    ]
    if not observed:
        return tube

    for i, entry in enumerate(tube.entries):
        if entry.detection is not None:
            continue

        before = next(
            ((j, d) for j, d in reversed(observed) if j < i),
            None,
        )
        after = next(
            ((j, d) for j, d in observed if j > i),
            None,
        )

        if before is not None and after is not None:
            j_b, d_b = before
            j_a, d_a = after
            t = (i - j_b) / (j_a - j_b)
            cx = d_b.cx + t * (d_a.cx - d_b.cx)
            cy = d_b.cy + t * (d_a.cy - d_b.cy)
            w = d_b.w + t * (d_a.w - d_b.w)
            h = d_b.h + t * (d_a.h - d_b.h)
            class_id = d_b.class_id
        elif before is not None:
            d = before[1]
            cx, cy, w, h, class_id = d.cx, d.cy, d.w, d.h, d.class_id
        else:
            assert after is not None
            d = after[1]
            cx, cy, w, h, class_id = d.cx, d.cy, d.w, d.h, d.class_id

        entry.detection = Detection(
            class_id=class_id,
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            confidence=0.0,
        )
        entry.is_gap = True

    return tube


def select_longest_tube(tubes: list[Tube]) -> Tube | None:
    """Pick the single longest tube from a list.

    Length is measured as ``end_frame - start_frame + 1`` (so gaps count
    toward length). Ties are broken by the number of non-gap entries --
    the tube with more real observations wins. If still tied, the first
    in the input order wins.

    Args:
        tubes: Candidate tubes.

    Returns:
        The selected tube, or ``None`` if ``tubes`` is empty.
    """
    if not tubes:
        return None

    def _key(tube: Tube) -> tuple[int, int]:
        length = tube.end_frame - tube.start_frame + 1
        n_observed = sum(1 for e in tube.entries if e.detection is not None)
        return (length, n_observed)

    return max(tubes, key=_key)


def tube_from_record(record: dict) -> Tube:
    """Rebuild a :class:`Tube` from a tube JSON record.

    Inverse of ``_serialize_tube`` in ``scripts/build_tubes.py``. Pure
    function; no I/O.

    Entries with ``bbox=None`` are reconstructed with ``detection=None``
    (pre-interpolation gap shape). Otherwise a :class:`Detection` is
    built from the bbox + confidence; ``confidence=None`` falls back to
    ``0.0``.

    Args:
        record: Parsed tube record. Only the ``tube`` sub-object is
            consulted; other top-level keys are ignored.

    Returns:
        A :class:`Tube` with ``tube_id=0`` (the on-disk dataset is
        single-tube-per-sequence so the id is informational only).
    """
    t = record["tube"]
    entries: list[TubeEntry] = []
    for e in t["entries"]:
        bbox = e["bbox"]
        if bbox is None:
            det: Detection | None = None
        else:
            det = Detection(
                class_id=0,
                cx=bbox[0],
                cy=bbox[1],
                w=bbox[2],
                h=bbox[3],
                confidence=e["confidence"] if e["confidence"] is not None else 0.0,
            )
        entries.append(
            TubeEntry(
                frame_idx=e["frame_idx"],
                detection=det,
                is_gap=e["is_gap"],
            )
        )
    return Tube(
        tube_id=0,
        entries=entries,
        start_frame=t["start_frame"],
        end_frame=t["end_frame"],
    )
