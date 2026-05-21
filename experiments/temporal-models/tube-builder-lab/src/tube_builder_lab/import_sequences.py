"""Import working-set sequences BY ID into the lab's own store (flat layout)."""

from __future__ import annotations

import json
import logging
import shutil
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


def _explorer_seq_dir_for_key(explorer_store: Path, key: str) -> Path | None:
    """Find an explorer sequence dir (nested layout, rich meta) by its key."""
    for meta_path in explorer_store.rglob("meta.json"):
        try:
            if json.loads(meta_path.read_text()).get("key") == key:
                return meta_path.parent
        except (json.JSONDecodeError, OSError):
            continue
    return None


def copy_one_from_explorer(*, lab_store: Path, explorer_seq_dir: Path) -> Path:
    """Copy one explorer sequence into the lab's flat store with a minimal meta.

    The explorer uses a nested org/camera layout and a richer meta; the lab
    keeps a flat ``platform_<id>/`` layout with only key/sequence_id/frames.
    """
    raw = json.loads((explorer_seq_dir / "meta.json").read_text())
    key = raw["key"]
    seq_dir = lab_store / key
    (seq_dir / "images").mkdir(parents=True, exist_ok=True)
    frames: list[FrameRef] = []
    for fr in raw.get("frames", []):
        rel = fr["file"]
        dst = seq_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(explorer_seq_dir / rel, dst)
        frames.append(
            FrameRef(
                file=rel,
                detection_id=fr.get("detection_id"),
                created_at=fr.get("created_at"),
            )
        )
    write_meta(
        seq_dir,
        SequenceMeta(key=key, sequence_id=str(raw["sequence_id"]), frames=frames),
    )
    return seq_dir


def bootstrap_from_explorer(
    *, lab_store: Path, explorer_store: Path, keys: list[str]
) -> tuple[int, list[str]]:
    """Copy each key from the explorer store into the lab store (no creds).

    Returns ``(copied_count, missing_keys)``.
    """
    copied = 0
    missing: list[str] = []
    for key in keys:
        src = _explorer_seq_dir_for_key(explorer_store, key)
        if src is None:
            missing.append(key)
            log.warning("key %s not found in explorer store %s", key, explorer_store)
            continue
        copy_one_from_explorer(lab_store=lab_store, explorer_seq_dir=src)
        copied += 1
        log.info("copied %s from explorer", key)
    return copied, missing
