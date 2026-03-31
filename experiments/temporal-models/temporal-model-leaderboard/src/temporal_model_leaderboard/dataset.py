"""Pyro-dataset test set discovery and ground-truth labeling.

Reads the sequential test set structure where ground truth is
determined by directory location (``wildfire/`` vs ``fp/``).
"""

import re
from datetime import datetime
from pathlib import Path

_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$")


def _parse_timestamp_for_sort(filename: str) -> datetime:
    """Extract timestamp from a Pyronear-style image filename for sorting.

    Raises:
        ValueError: If the timestamp cannot be parsed.
    """
    match = _TIMESTAMP_RE.search(Path(filename).stem)
    if not match:
        raise ValueError(f"Cannot parse timestamp from: {filename}")
    return datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%S")


def list_sequences(test_dir: Path) -> list[tuple[Path, bool]]:
    """List all sequences in a test split with their ground truth labels.

    Scans ``test_dir/wildfire/`` (ground truth = positive) and
    ``test_dir/fp/`` (ground truth = negative).

    Args:
        test_dir: Path to the test split root (e.g., ``.../sequential_test/test``).

    Returns:
        Sorted list of ``(sequence_path, is_wildfire)`` tuples.
    """
    sequences: list[tuple[Path, bool]] = []

    wf_dir = test_dir / "wildfire"
    if wf_dir.is_dir():
        for d in sorted(wf_dir.iterdir()):
            if d.is_dir():
                sequences.append((d, True))

    fp_dir = test_dir / "fp"
    if fp_dir.is_dir():
        for d in sorted(fp_dir.iterdir()):
            if d.is_dir():
                sequences.append((d, False))

    return sequences


def get_sorted_frames(sequence_dir: Path) -> list[Path]:
    """Return image paths from ``sequence_dir/images/`` sorted by timestamp.

    Args:
        sequence_dir: Path to a sequence directory containing an ``images/``
            subdirectory with ``.jpg`` files.

    Returns:
        Sorted list of image paths, or an empty list if ``images/`` does not
        exist or contains no images.
    """
    images_dir = sequence_dir / "images"
    if not images_dir.is_dir():
        return []
    return sorted(
        images_dir.glob("*.jpg"),
        key=lambda p: _parse_timestamp_for_sort(p.name),
    )
