"""Visualization utilities for prediction analysis.

Renders annotated frame strips showing ground truth boxes, prior YOLO
predictions (from label files), and current model predictions for
tracking result sequences.
"""

from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.types import Detection


def load_label_boxes(label_path: Path) -> tuple[list[Detection], bool]:
    """Load bounding boxes from a YOLO-format label file.

    Returns a tuple of (detections, is_human_annotation).
    WF labels have 5 columns (human annotations), FP labels have 6
    columns (prior YOLO predictions with confidence).
    """
    if not label_path.is_file():
        return [], True
    content = label_path.read_text().strip()
    if not content:
        return [], True
    boxes = []
    is_human = True
    for line in content.split("\n"):
        parts = line.strip().split()
        if len(parts) == 5:
            boxes.append(
                Detection(
                    class_id=int(parts[0]),
                    cx=float(parts[1]),
                    cy=float(parts[2]),
                    w=float(parts[3]),
                    h=float(parts[4]),
                    confidence=1.0,
                )
            )
        elif len(parts) == 6:
            is_human = False
            boxes.append(
                Detection(
                    class_id=int(parts[0]),
                    cx=float(parts[1]),
                    cy=float(parts[2]),
                    w=float(parts[3]),
                    h=float(parts[4]),
                    confidence=float(parts[5]),
                )
            )
    return boxes, is_human


def draw_boxes_on_frame(
    image: Image.Image,
    detections: list[Detection],
    color: tuple[int, int, int],
    line_width: int,
    show_confidence: bool = False,
) -> Image.Image:
    """Draw bounding boxes on a PIL image.

    Converts normalized (cx, cy, w, h) to pixel coordinates and draws
    rectangles. Optionally labels each box with its confidence score.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()

    for det in detections:
        x1 = (det.cx - det.w / 2) * img_w
        y1 = (det.cy - det.h / 2) * img_h
        x2 = (det.cx + det.w / 2) * img_w
        y2 = (det.cy + det.h / 2) * img_h
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        if show_confidence:
            label = f"{det.confidence:.2f}"
            draw.text((x1, max(y1 - 14, 0)), label, fill=color, font=font)

    return img


# Box colors
GT_COLOR = (0, 180, 0)  # Green — human annotations
PRIOR_COLOR = (140, 0, 200)  # Purple — prior YOLO predictions from labels
PRED_COLOR = (220, 0, 0)  # Red — current model predictions


def render_sequence_strip(
    frames_data: list[dict],
    metadata: dict,
    output_path: Path,
    thumb_width: int = 320,
    max_cols: int = 10,
) -> None:
    """Render an annotated frame strip for a single sequence.

    Parameters
    ----------
    frames_data
        List of dicts with keys: image_path, gt_boxes, prior_boxes,
        pred_detections, frame_index, timestamp_str, num_preds.
        gt_boxes: human-annotated GT (5-col labels, WF sequences).
        prior_boxes: prior YOLO predictions (6-col labels, FP sequences).
    metadata
        Dict with keys: sequence_id, error_type, is_positive_gt,
        is_positive_pred, num_frames, num_detections_total, num_tracks,
        confirmed_frame_index, confirmed_timestamp.
    output_path
        Path to save the output PNG.
    thumb_width
        Width of each frame thumbnail in pixels.
    max_cols
        Maximum number of frames per row.
    """
    if not frames_data:
        return

    # Load first image to determine aspect ratio
    sample = Image.open(frames_data[0]["image_path"])
    aspect = sample.height / sample.width
    thumb_height = int(thumb_width * aspect)
    sample.close()

    n_frames = len(frames_data)
    n_cols = min(n_frames, max_cols)
    n_rows = ceil(n_frames / max_cols)

    header_height = 144
    label_height = 40
    canvas_w = n_cols * thumb_width
    canvas_h = header_height + n_rows * (thumb_height + label_height)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default(size=13)
        font_small = ImageFont.load_default(size=11)
    except TypeError:
        font = ImageFont.load_default()
        font_small = font

    # Draw header
    error_label = metadata["error_type"].replace("_", " ").upper()
    gt_label = "positive" if metadata["is_positive_gt"] else "negative"
    pred_label = "positive" if metadata["is_positive_pred"] else "negative"

    n_dets = metadata["num_detections_total"]
    n_tracks = metadata["num_tracks"]
    conf_frame = metadata["confirmed_frame_index"] or "None"
    conf_ts = metadata["confirmed_timestamp"] or "N/A"

    lines = [
        f"Sequence: {metadata['sequence_id']}",
        f"Type: {error_label}  |  GT: {gt_label}  |  Pred: {pred_label}",
        f"Frames: {metadata['num_frames']}  |  Dets: {n_dets}  |  Tracks: {n_tracks}",
        f"Confirmed at frame: {conf_frame}  ({conf_ts})",
    ]
    header_colors = {
        "false_positive": (180, 0, 0),
        "false_negative": (0, 0, 180),
        "true_positive": (0, 130, 0),
        "true_negative": (80, 80, 80),
    }
    header_color = header_colors.get(metadata["error_type"], (0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((10, 8 + i * 22), line, fill=header_color, font=font)

    # Draw legend
    legend_y = 8 + len(lines) * 22
    x = 10

    draw.rectangle([x, legend_y, x + 12, legend_y + 12], outline=GT_COLOR, width=2)
    draw.text((x + 16, legend_y - 1), "Ground Truth", fill=(60, 60, 60), font=font)
    x += 150

    draw.rectangle([x, legend_y, x + 12, legend_y + 12], outline=PRIOR_COLOR, width=2)
    draw.text(
        (x + 16, legend_y - 1),
        "Prior YOLO Prediction",
        fill=(60, 60, 60),
        font=font,
    )
    x += 200

    draw.rectangle([x, legend_y, x + 12, legend_y + 12], outline=PRED_COLOR, width=2)
    draw.text(
        (x + 16, legend_y - 1),
        "Current Prediction (confidence)",
        fill=(60, 60, 60),
        font=font,
    )

    # Draw frames
    for idx, frame in enumerate(frames_data):
        row = idx // max_cols
        col = idx % max_cols
        x_off = col * thumb_width
        y_off = header_height + row * (thumb_height + label_height)

        img = Image.open(frame["image_path"]).resize(
            (thumb_width, thumb_height), Image.LANCZOS
        )

        # Draw GT boxes (green, thick) — human annotations
        if frame["gt_boxes"]:
            img = draw_boxes_on_frame(
                img, frame["gt_boxes"], color=GT_COLOR, line_width=3
            )

        # Draw prior YOLO boxes (purple) — from label files
        if frame["prior_boxes"]:
            img = draw_boxes_on_frame(
                img,
                frame["prior_boxes"],
                color=PRIOR_COLOR,
                line_width=2,
                show_confidence=True,
            )

        # Draw current prediction boxes (red, with confidence)
        if frame["pred_detections"]:
            img = draw_boxes_on_frame(
                img,
                frame["pred_detections"],
                color=PRED_COLOR,
                line_width=2,
                show_confidence=True,
            )

        canvas.paste(img, (x_off, y_off))

        # Frame label below
        fi = frame["frame_index"]
        ts = frame["timestamp_str"]
        nd = frame["num_preds"]
        label = f"Frame {fi}  |  {ts}  |  {nd} detections"
        draw.text(
            (x_off + 4, y_off + thumb_height + 4),
            label,
            fill=(60, 60, 60),
            font=font_small,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
