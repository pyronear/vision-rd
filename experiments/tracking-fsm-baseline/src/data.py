"""File I/O and data utilities for sequence datasets.

Provides helpers to discover sequences on disk, parse timestamps from
filenames, and determine ground-truth labels from the pyro-dataset label
convention.
"""

import re
from datetime import datetime
from pathlib import Path


def list_sequences(split_dir: Path) -> list[Path]:
    """List all sequence directories in a split, sorted by name."""
    return sorted(d for d in split_dir.iterdir() if d.is_dir())


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


def is_wf_sequence(sequence_dir: Path) -> bool:
    """Determine if a sequence is wildfire (positive) based on label format.

    WF labels have 5 columns: class_id cx cy w h (human annotations).
    FP labels have 6 columns: class_id cx cy w h confidence (YOLO predictions).

    Limitation: this heuristic relies on the pyro-dataset label convention.
    Sequences with only empty label files default to FP (negative).
    """
    labels_dir = sequence_dir / "labels"
    if not labels_dir.is_dir():
        return False
    for label_file in labels_dir.iterdir():
        if label_file.suffix != ".txt":
            continue
        content = label_file.read_text().strip()
        if not content:
            continue
        first_line = content.split("\n")[0].strip()
        n_cols = len(first_line.split())
        if n_cols == 5:
            return True
        if n_cols == 6:
            return False
    # No non-empty label files — treat as FP
    return False
