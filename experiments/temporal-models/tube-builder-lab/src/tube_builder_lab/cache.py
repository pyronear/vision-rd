"""Cache per-frame YOLO detections for the working-set sequences."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from bbox_tube_temporal.types import FrameDetections
from pyrocore import Frame

from .detections_io import read_detections, write_detections

log = logging.getLogger(__name__)

YoloRunner = Callable[[list[Frame]], list[FrameDetections]]


def cache_one(
    *,
    out_dir: Path,
    key: str,
    frames: list[Frame],
    run_yolo: YoloRunner,
    overwrite: bool = False,
) -> Path:
    """Run (or skip) YOLO for one sequence; write detections JSON; return path."""
    path = out_dir / f"{key}.json"
    if path.exists() and not overwrite:
        log.info("cache hit for %s; skipping", key)
        return path
    fds = run_yolo(frames)
    write_detections(path, fds)
    log.info("cached %d frames for %s", len(fds), key)
    return path


def detections_present(out_dir: Path, key: str) -> bool:
    return (out_dir / f"{key}.json").exists()


def load_cached(out_dir: Path, key: str) -> list[FrameDetections]:
    return read_detections(out_dir / f"{key}.json")
