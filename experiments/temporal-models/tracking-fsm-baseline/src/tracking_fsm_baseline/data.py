"""File I/O and data utilities for sequence datasets.

Provides helpers to discover sequences on disk, parse timestamps from
filenames, and determine ground-truth labels from the pyro-dataset label
convention.
"""

import re
from datetime import datetime
from pathlib import Path

from .types import Detection, FrameResult


def pad_sequence(frames: list[FrameResult], min_length: int) -> list[FrameResult]:
    """Pad a short sequence symmetrically by repeating boundary frames.

    If *frames* already has at least *min_length* entries (or is empty),
    it is returned unchanged.  Otherwise the first frame is prepended and
    the last frame is appended in alternation until the list reaches
    *min_length*.

    Returns a new list; the input list is not modified.
    """
    if not frames or len(frames) >= min_length:
        return list(frames)
    result = list(frames)
    prepend = True
    while len(result) < min_length:
        src = frames[0] if prepend else frames[-1]
        result.insert(
            0 if prepend else len(result),
            FrameResult(
                frame_id=src.frame_id,
                timestamp=src.timestamp,
                detections=list(src.detections),
            ),
        )
        prepend = not prepend
    return result


_CATEGORY_DIRS = {"wildfire", "fp"}


def list_sequences(split_dir: Path) -> list[Path]:
    """List all sequence directories in a split, sorted by name.

    Supports two layouts:

    * **Nested** (pyro-dataset v2.2.0): ``split_dir/{wildfire,fp}/seq/``
    * **Flat** (legacy): ``split_dir/seq/``

    In the nested layout the ``wildfire/`` and ``fp/`` category directories
    are transparently expanded so callers always receive a flat list of
    individual sequence paths.
    """
    subdirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    child_names = {d.name for d in subdirs}

    # Detect nested layout: top-level contains only the category dirs
    if child_names <= _CATEGORY_DIRS:
        sequences: list[Path] = []
        for cat_dir in subdirs:
            sequences.extend(sorted(d for d in cat_dir.iterdir() if d.is_dir()))
        return sorted(sequences, key=lambda p: p.name)

    return subdirs


def parse_timestamp(filename: str) -> datetime:
    """Extract timestamp from an image filename.

    Expected pattern: ..._YYYY-MM-DDTHH-MM-SS.jpg
    """
    stem = Path(filename).stem
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$", stem)
    if not match:
        raise ValueError(f"Cannot parse timestamp from: {filename}")
    return datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S")


def get_sorted_frames(sequence_dir: Path) -> list[Path]:
    """Return image paths from ``sequence_dir/images/`` sorted by timestamp.

    Args:
        sequence_dir: Path to a sequence directory containing an ``images/``
            subdirectory with ``.jpg`` files.

    Returns:
        Sorted list of image paths, or an empty list if ``images/`` does not
        exist.
    """
    images_dir = sequence_dir / "images"
    if not images_dir.is_dir():
        return []
    images = sorted(images_dir.glob("*.jpg"), key=lambda p: parse_timestamp(p.name))
    return images


def find_sequence_dir(data_dir: Path, seq_id: str) -> Path | None:
    """Find a sequence directory by ID within the nested layout.

    Searches ``data_dir/{wildfire,fp}/seq_id`` and returns the first match,
    or ``None`` if the sequence is not found.
    """
    for category in ("wildfire", "fp"):
        candidate = data_dir / category / seq_id
        if candidate.is_dir():
            return candidate
    return None


def is_wf_sequence(sequence_dir: Path) -> bool:
    """Determine if a sequence is wildfire (positive) based on parent directory.

    In the nested layout, wildfire sequences live under a ``wildfire/``
    parent directory and false-positive sequences under ``fp/``.
    """
    return sequence_dir.parent.name == "wildfire"


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
                    class_id=int(float(parts[0])),
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
                    class_id=int(float(parts[0])),
                    cx=float(parts[1]),
                    cy=float(parts[2]),
                    w=float(parts[3]),
                    h=float(parts[4]),
                    confidence=float(parts[5]),
                )
            )
    return boxes, is_human
