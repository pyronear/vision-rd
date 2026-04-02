"""Data I/O utilities for sequence discovery and ground-truth labelling."""

import re
from datetime import datetime
from pathlib import Path

_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")


def list_sequences(split_dir: Path) -> list[Path]:
    """List all sequence directories in a split, sorted by name.

    Supports the nested pyro-dataset v2.2.0 layout::

        split_dir/{wildfire,fp}/<sequence>/

    Returns:
        Sorted list of sequence directory paths.
    """
    sequences: list[Path] = []
    for category in ("wildfire", "fp"):
        cat_dir = split_dir / category
        if cat_dir.is_dir():
            sequences.extend(d for d in sorted(cat_dir.iterdir()) if d.is_dir())
    sequences.sort(key=lambda p: p.name)
    return sequences


def find_sequence_dir(data_dir: Path, seq_id: str) -> Path | None:
    """Find a sequence directory by ID within the nested layout."""
    for category in ("wildfire", "fp"):
        candidate = data_dir / category / seq_id
        if candidate.is_dir():
            return candidate
    return None


def is_wf_sequence(sequence_dir: Path) -> bool:
    """Determine if a sequence is wildfire based on parent directory name."""
    return sequence_dir.parent.name == "wildfire"


def get_sorted_frames(sequence_dir: Path) -> list[Path]:
    """Return image paths from a sequence directory sorted by timestamp.

    Looks for ``*.jpg`` files in ``sequence_dir/images/``.
    """
    images_dir = sequence_dir / "images"
    if not images_dir.is_dir():
        return []
    return sorted(images_dir.glob("*.jpg"), key=lambda p: p.stem)


def parse_timestamp(frame_id: str) -> datetime | None:
    """Extract a timestamp from a Pyronear-style frame ID.

    Expects the frame ID to contain ``YYYY-MM-DDTHH-MM-SS``.
    """
    match = _TIMESTAMP_RE.search(frame_id)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None


def load_gt_labels(
    sequence_dir: Path, frame_id: str
) -> list[tuple[float, float, float, float]]:
    """Load ground-truth bounding boxes for a frame.

    Reads YOLO-format label files from ``sequence_dir/labels/``.
    Only returns boxes from 5-column files (human annotations in WF
    sequences).  6-column files (YOLO predictions in FP sequences)
    are ignored since they are not real smoke.

    Args:
        sequence_dir: Path to the sequence directory.
        frame_id: Frame filename stem.

    Returns:
        List of ``(cx, cy, w, h)`` normalised bboxes.
    """
    label_path = sequence_dir / "labels" / f"{frame_id}.txt"
    if not label_path.is_file():
        return []
    content = label_path.read_text().strip()
    if not content:
        return []
    boxes = []
    for line in content.split("\n"):
        parts = line.strip().split()
        if len(parts) == 5:
            boxes.append(
                (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            )
    return boxes
