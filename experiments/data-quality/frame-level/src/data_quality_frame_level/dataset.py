"""Frame discovery + YOLO label parsing for a flat YOLO split.

Walks a split directory with the layout::

    <split>/
      images/*.jpg
      labels/*.txt

and emits :class:`FrameRef` records carrying the parsed ground-truth bboxes
for each frame (empty list for frames with empty or missing label files).
Also provides the YOLO-center -> FiftyOne-top-left bbox conversion used by
:mod:`data_quality_frame_level.fiftyone_build`.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BBox:
    """One YOLO-format bounding box in normalized center form.

    All coordinates are in ``[0, 1]`` relative to the image.
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class FrameRef:
    """One image + its ground-truth bboxes.

    Attributes:
        stem: Filename without extension (e.g. ``"adf_avinyonet_..."``).
        image_path: Absolute path to the ``.jpg`` file.
        label_path: Absolute path to the matching ``.txt`` file (may not
            exist on disk — empty label files and missing label files are
            treated identically).
        gt_bboxes: Parsed YOLO bboxes; empty list if the label is empty
            or missing.
    """

    stem: str
    image_path: Path
    label_path: Path
    gt_bboxes: list[BBox]


def parse_yolo_label(label_path: Path) -> list[BBox]:
    """Parse a YOLO ``.txt`` file into a list of :class:`BBox`.

    Returns an empty list if the file doesn't exist or is empty. Blank
    lines are skipped. Each non-blank line must be
    ``class cx cy w h`` with floats in ``[0, 1]``.
    """
    if not label_path.is_file():
        return []
    bboxes: list[BBox] = []
    for raw_line in label_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        class_id = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:5])
        bboxes.append(BBox(class_id=class_id, cx=cx, cy=cy, w=w, h=h))
    return bboxes


def iter_frames(split_dir: Path) -> Iterator[FrameRef]:
    """Yield one :class:`FrameRef` per ``.jpg`` under ``split_dir/images/``.

    Frames are emitted in filename-sorted order. A missing ``labels/``
    directory is tolerated (all frames get empty ``gt_bboxes``).
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    for image_path in sorted(images_dir.glob("*.jpg")):
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"
        yield FrameRef(
            stem=stem,
            image_path=image_path,
            label_path=label_path,
            gt_bboxes=parse_yolo_label(label_path),
        )


def yolo_to_fiftyone_xywh(bbox: BBox) -> tuple[float, float, float, float]:
    """Convert a YOLO normalized center bbox to FiftyOne top-left xywh.

    FiftyOne ``Detection.bounding_box`` expects ``[x, y, w, h]`` with the
    top-left corner relative to the image and ``(0, 0)`` at the top-left.
    YOLO stores the same relative coordinates but uses the bbox center as
    the reference point.
    """
    return (bbox.cx - bbox.w / 2, bbox.cy - bbox.h / 2, bbox.w, bbox.h)
