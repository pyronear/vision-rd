"""Import working-set sequences BY ID into the lab's own store (flat layout)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from . import platform_api
from .store import FrameRef, SequenceMeta, write_meta

log = logging.getLogger(__name__)


def import_one_by_id(
    *,
    store_dir: Path,
    sequence_id: int,
    detections: list[dict],
    download: Callable[[str], bytes],
) -> Path:
    """Write frames + minimal meta for one sequence; returns its dir."""
    ordered = sorted(detections, key=lambda d: d.get("created_at") or "")
    seq_dir = store_dir / f"platform_{sequence_id}"
    (seq_dir / "images").mkdir(parents=True, exist_ok=True)

    frames: list[FrameRef] = []
    for det in ordered:
        url = det.get("url")
        if not url:
            log.warning(
                "detection %s of seq %s has no url; skipping",
                det.get("id"),
                sequence_id,
            )
            continue
        fname = f"detection_{det['id']}.jpg"
        (seq_dir / "images" / fname).write_bytes(download(url))
        frames.append(
            FrameRef(
                file=f"images/{fname}",
                detection_id=det["id"],
                created_at=det.get("created_at"),
            )
        )

    write_meta(
        seq_dir,
        SequenceMeta(
            key=f"platform_{sequence_id}",
            sequence_id=str(sequence_id),
            frames=frames,
        ),
    )
    return seq_dir


def sequence_id_from_key(key: str) -> int:
    """'platform_42538' -> 42538."""
    return int(key.rsplit("_", 1)[-1])


def import_keys(
    *,
    store_dir: Path,
    keys: list[str],
    api_endpoint: str,
    token: str,
    detections_limit: int,
    download: Callable[[str], bytes] = platform_api.download_image,
) -> int:
    """Fetch + import every key from the platform. Returns #sequences imported."""
    count = 0
    for key in keys:
        sid = sequence_id_from_key(key)
        dets = platform_api.list_sequence_detections(
            api_endpoint, token, sid, limit=detections_limit
        )
        import_one_by_id(
            store_dir=store_dir, sequence_id=sid, detections=dets, download=download
        )
        count += 1
        log.info("imported %s (%d detections)", key, len(dets))
    return count
