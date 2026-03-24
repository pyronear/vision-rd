from __future__ import annotations

import json
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
    """Return image paths sorted by timestamp."""
    images_dir = sequence_dir / "images"
    if not images_dir.is_dir():
        return []
    images = sorted(images_dir.glob("*.jpg"), key=lambda p: parse_timestamp(p.name))
    return images


def load_wf_folders(registry_path: Path) -> set[str]:
    """Load the set of wildfire folder names from registry.json."""
    with registry_path.open() as f:
        data = json.load(f)
    return {seq["folder"] for seq in data["sequences"]}


def is_wf_sequence(folder_name: str, wf_folders: set[str]) -> bool:
    """Return True if the sequence folder is a wildfire (positive) sequence."""
    return folder_name in wf_folders
