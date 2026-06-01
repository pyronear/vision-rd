"""Import pyro-annotator sequences (human-labeled zip export) into the store.

The label comes from the folder path (``smoke/<subtype>``, ``fp/<subtype>``,
``unlabeled``). Frames already live in the zip, so images are copied (not
downloaded). Camera/org/timestamps are enriched per sequence via the admin
platform API, so these sequences sit in the same org -> camera navigation as the
alert-API source. Enrichment requires admin creds.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

from . import platform_api
from .store import FrameRef, SequenceMeta, slug, write_meta

log = logging.getLogger(__name__)


def parse_label(klass: str, subtype: str | None) -> tuple[str, str | None]:
    """Map a (class, subtype) folder pair to the tri-state label + detail."""
    if klass == "smoke":
        return "smoke", subtype
    if klass == "fp":
        return "fp", subtype
    if klass == "unlabeled":
        return "unknown", None
    raise ValueError(f"unknown class folder: {klass!r}")


def iter_zip_sequences(
    src: Path,
) -> Iterator[tuple[str, str | None, int, Path]]:
    """Yield (class, subtype, seq_id, seq_dir) for each seq_<id>/ with images/.

    Layout: ``<class>/<subtype>/seq_<id>`` (smoke, fp) or ``<class>/seq_<id>``
    (unlabeled). macOS ``__MACOSX`` entries are skipped.
    """
    for images_dir in sorted(src.rglob("images")):
        seq_dir = images_dir.parent
        rel = seq_dir.relative_to(src).parts
        if "__MACOSX" in rel or not seq_dir.name.startswith("seq_"):
            continue
        klass = rel[0]
        subtype = rel[1] if len(rel) == 3 else None
        seq_id = int(seq_dir.name[len("seq_") :])
        yield klass, subtype, seq_id, seq_dir


def _import_one(
    api_endpoint: str,
    token: str,
    out: Path,
    klass: str,
    subtype: str | None,
    seq_id: int,
    seq_dir: Path,
    camera_index: dict,
    org_index: dict[int, str] | None,
    detections_limit: int,
    list_detections,
) -> int:
    label, label_detail = parse_label(klass, subtype)

    # Enrich from the platform API. The zip's ``detection_<id>`` filenames are a
    # different id space from the platform detection ids, so timestamps are
    # matched by capture order, not id: the zip is the same frame set as the API
    # detections. Both are ordered ascending in time (zip ids increment with
    # capture; API sorted by created_at), so the i-th frames line up.
    api_times: list[str | None] = []
    camera_id: int | None = None
    try:
        dets = list_detections(
            api_endpoint, token, seq_id, limit=detections_limit, desc=False
        )
        dets = sorted(dets, key=lambda d: d.get("created_at") or "")
        api_times = [d.get("created_at") for d in dets]
        if dets:
            camera_id = dets[0].get("camera_id")
    except Exception as exc:  # noqa: BLE001 - enrichment is best-effort; log + fall back
        log.warning("enrichment failed for seq %s: %s", seq_id, exc)

    cam = camera_index.get(camera_id, {}) if camera_id is not None else {}
    org_id = cam.get("organization_id")
    camera_name = cam.get("name") or "unknown"
    org_name = (org_index or {}).get(org_id) or "unknown"

    seq_out = (
        out / "pyro-annotator" / slug(org_name) / slug(camera_name) / f"seq_{seq_id}"
    )
    (seq_out / "images").mkdir(parents=True, exist_ok=True)

    # Frames in capture order (detection_<id> increments with time). Globbing the
    # ``detection_`` prefix excludes macOS AppleDouble ``._*`` siblings; any
    # remaining odd filename is logged and skipped rather than aborting the run.
    parsed: list[tuple[int, Path]] = []
    for p in (seq_dir / "images").glob("detection_*.jpg"):
        try:
            parsed.append((int(p.stem.split("_")[-1]), p))
        except ValueError:
            log.warning("seq %s: skipping unparseable frame %s", seq_id, p.name)
    parsed.sort(key=lambda x: x[0])

    # Per-frame timestamps only when the frame sets line up; else leave them None.
    if api_times and len(api_times) != len(parsed):
        log.warning(
            "seq %s: %d frames != %d API detections; per-frame timestamps skipped",
            seq_id,
            len(parsed),
            len(api_times),
        )
    times = api_times if len(api_times) == len(parsed) else [None] * len(parsed)
    frames: list[FrameRef] = []
    for (det_id, img), ts in zip(parsed, times, strict=True):
        shutil.copyfile(img, seq_out / "images" / img.name)
        frames.append(
            FrameRef(file=f"images/{img.name}", detection_id=det_id, created_at=ts)
        )
    # Sequence start: earliest API timestamp, independent of per-frame matching.
    started_at = api_times[0] if api_times else None

    write_meta(
        seq_out,
        SequenceMeta(
            key=f"pyro_annotator_{seq_id}",
            sequence_id=str(seq_id),
            source="pyro-annotator",
            label=label,
            label_detail=label_detail,
            label_source="pyro_annotator_folder",
            frames=frames,
            camera_id=camera_id,
            camera_name=camera_name,
            organization_id=org_id,
            organization_name=org_name,
            started_at=started_at,
        ),
    )
    return 1


def import_pyro_annotator(
    src: Path,
    out: Path,
    api_endpoint: str,
    token: str,
    *,
    detections_limit: int = 100,  # platform API caps detections per call at 100
    camera_index: dict | None = None,
    org_index: dict[int, str] | None = None,
    list_detections=platform_api.list_sequence_detections,
) -> int:
    """Import every sequence under ``src`` into ``out``. Returns #sequences."""
    camera_index = camera_index or {}
    count = 0
    for klass, subtype, seq_id, seq_dir in iter_zip_sequences(src):
        try:
            count += _import_one(
                api_endpoint,
                token,
                out,
                klass,
                subtype,
                seq_id,
                seq_dir,
                camera_index,
                org_index,
                detections_limit,
                list_detections,
            )
        except Exception as exc:  # noqa: BLE001 - one bad seq shouldn't kill the run
            log.warning("failed to import seq %s: %s", seq_id, exc)
    return count
