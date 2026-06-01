"""Import platform sequences (date range, optional camera filter) into the store.

Each detection's full-frame image is downloaded via its presigned ``url``. Frames
are ordered by detection ``created_at``. Requires only regular platform creds.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import requests

from . import platform_api
from .store import FrameRef, SequenceMeta, normalize_label, slug, write_meta

log = logging.getLogger(__name__)


def download_image(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def build_camera_index(api_endpoint: str, token: str) -> dict[int, dict]:
    return {c["id"]: c for c in platform_api.list_cameras(api_endpoint, token)}


def build_org_index(api_endpoint: str, admin_token: str) -> dict[int, str]:
    """org_id -> name, from the admin-only /organizations endpoint."""
    orgs = platform_api.list_organizations(api_endpoint, admin_token)
    return {o["id"]: o["name"] for o in orgs}


def _org_slug(org_id: int | None, org_index: dict[int, str] | None) -> str:
    """On-disk subdir for an org: its name when known, else org_<id>, else 'unknown'."""
    if org_id is None:
        return "unknown"
    name = (org_index or {}).get(org_id)
    return slug(name) if name else f"org_{org_id}"


def _camera_slug(cam: dict, camera_id: int | None) -> str:
    """On-disk subdir for a camera: name when known, else cam_<id>, else 'unknown'."""
    name = cam.get("name")
    if name:
        return slug(name)
    return f"cam_{camera_id}" if camera_id is not None else "unknown"


def _import_one(
    api_endpoint,
    token,
    store_dir,
    seq,
    camera_index,
    org_index,
    detections_limit,
    smoke_values,
    fp_values,
    download,
) -> int:
    sid = seq["id"]
    raw_label = seq.get("is_wildfire")
    cam = camera_index.get(seq.get("camera_id"), {})
    org_id = cam.get("organization_id")
    dets = platform_api.list_sequence_detections(
        api_endpoint, token, sid, limit=detections_limit, desc=False
    )
    dets = sorted(dets, key=lambda d: d.get("created_at") or "")
    out_dir = (
        store_dir
        / _org_slug(org_id, org_index)
        / _camera_slug(cam, seq.get("camera_id"))
        / f"seq_{sid}"
    )
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    frames: list[FrameRef] = []
    for det in dets:
        url = det.get("url")
        if not url:
            log.warning(
                "detection %s of seq %s has no url; skipping", det.get("id"), sid
            )
            continue
        try:
            data = download(url)
        except Exception as exc:  # noqa: BLE001 - log + skip a bad frame, keep going
            log.warning("download failed for detection %s: %s", det.get("id"), exc)
            continue
        fname = f"detection_{det['id']}.jpg"
        (out_dir / "images" / fname).write_bytes(data)
        frames.append(
            FrameRef(
                file=f"images/{fname}",
                detection_id=det["id"],
                created_at=det.get("created_at"),
            )
        )
    write_meta(
        out_dir,
        SequenceMeta(
            key=f"platform_{sid}",
            sequence_id=str(sid),
            source="platform",
            label=normalize_label(raw_label, smoke_values, fp_values),
            label_detail=raw_label,
            label_source="platform_is_wildfire",
            frames=frames,
            camera_id=seq.get("camera_id"),
            camera_name=cam.get("name"),
            organization_id=org_id,
            organization_name=(org_index or {}).get(org_id),
            started_at=seq.get("started_at"),
        ),
    )
    return 1


def import_platform(
    api_endpoint: str,
    token: str,
    store_dir: Path,
    day_from: date,
    day_to: date,
    *,
    detections_limit: int,
    smoke_values: list[str],
    fp_values: list[str],
    camera_ids: set[int] | None = None,
    camera_index: dict | None = None,
    org_index: dict[int, str] | None = None,
    download=download_image,
) -> int:
    """Import all sequences in [day_from, day_to]. Returns #sequences imported."""
    store_dir.mkdir(parents=True, exist_ok=True)
    if camera_index is None:
        camera_index = build_camera_index(api_endpoint, token)
    count = 0
    day = day_from
    while day <= day_to:
        for seq in platform_api.list_sequences_for_date(
            api_endpoint, token, day, 100, 0
        ):
            if camera_ids and seq.get("camera_id") not in camera_ids:
                continue
            count += _import_one(
                api_endpoint,
                token,
                store_dir,
                seq,
                camera_index,
                org_index,
                detections_limit,
                smoke_values,
                fp_values,
                download,
            )
        day += timedelta(days=1)
    return count
