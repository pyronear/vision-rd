"""Atomic read/write of ``review.json`` per ``(model, split)``.

The file shape is documented in the design spec §5.1. Writes go through
a sibling ``.tmp`` + ``os.replace`` so partial writes can never be
observed. Reads of missing files return an empty :class:`ReviewState`.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from data_quality_frame_level.dataset import BBox

PAYLOAD_VERSION = 1
ALLOWED_STATUS = ("reviewed", "unclear")


@dataclass
class SampleReview:
    status: str
    bboxes: list[BBox] = field(default_factory=list)
    spurious_originals: list[BBox] = field(default_factory=list)
    reviewer: str | None = None
    note: str | None = None
    reviewed_at: str | None = None


@dataclass
class ReviewState:
    model: str
    split: str
    samples: dict[str, SampleReview] = field(default_factory=dict)


def _bbox_to_dict(b: BBox) -> dict:
    return {"class_id": b.class_id, "cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h}


def _dict_to_bbox(d: dict) -> BBox:
    return BBox(
        class_id=int(d["class_id"]),
        cx=float(d["cx"]),
        cy=float(d["cy"]),
        w=float(d["w"]),
        h=float(d["h"]),
    )


def _sample_to_dict(s: SampleReview) -> dict:
    out: dict = {
        "status": s.status,
        "bboxes": [_bbox_to_dict(b) for b in s.bboxes],
    }
    if s.spurious_originals:
        out["spurious_originals"] = [_bbox_to_dict(b) for b in s.spurious_originals]
    if s.reviewer is not None:
        out["reviewer"] = s.reviewer
    if s.note is not None:
        out["note"] = s.note
    if s.reviewed_at is not None:
        out["reviewed_at"] = s.reviewed_at
    return out


def _dict_to_sample(d: dict) -> SampleReview:
    if d["status"] not in ALLOWED_STATUS:
        raise ValueError(f"unknown status: {d['status']!r}")
    return SampleReview(
        status=d["status"],
        bboxes=[_dict_to_bbox(b) for b in d.get("bboxes", [])],
        spurious_originals=[_dict_to_bbox(b) for b in d.get("spurious_originals", [])],
        reviewer=d.get("reviewer"),
        note=d.get("note"),
        reviewed_at=d.get("reviewed_at"),
    )


def read_review_state(path: Path, *, model: str, split: str) -> ReviewState:
    if not path.is_file():
        return ReviewState(model=model, split=split)
    payload = json.loads(path.read_text())
    if payload.get("version") != PAYLOAD_VERSION:
        raise ValueError(f"unsupported review.json version: {payload.get('version')}")
    return ReviewState(
        model=payload.get("model_name", model),
        split=payload.get("split", split),
        samples={
            stem: _dict_to_sample(d)
            for stem, d in sorted(payload.get("samples", {}).items())
        },
    )


def write_review_state(path: Path, state: ReviewState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PAYLOAD_VERSION,
        "model_name": state.model,
        "split": state.split,
        "samples": {
            stem: _sample_to_dict(state.samples[stem]) for stem in sorted(state.samples)
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tracked_md5(dvc_path: Path, target_filename: str) -> str | None:
    """Read the md5 for ``target_filename`` from a single-file ``.dvc``."""
    payload = yaml.safe_load(dvc_path.read_text())
    for out in payload.get("outs", []):
        if out.get("path") == target_filename and out.get("hash", "md5") == "md5":
            return out.get("md5")
    return None


def dvc_warning_for_review(review_path: Path) -> dict | None:
    """Compare local review.json md5 with the .dvc-tracked md5.

    Returns a warning dict if the local file is stale or missing relative
    to the tracked version. Returns None when:

    - There is no sibling ``.dvc`` file (untracked / first session).
    - The local file matches the tracked md5.
    """
    dvc_path = review_path.with_suffix(review_path.suffix + ".dvc")
    if not dvc_path.is_file():
        return None
    tracked = _tracked_md5(dvc_path, review_path.name)
    if tracked is None:
        return None
    if not review_path.is_file():
        return {
            "kind": "missing_local",
            "tracked_md5": tracked,
            "local_md5": None,
            "message": (
                f"DVC tracks {review_path.name} but the local file is missing. "
                "Run `make review-pull` before reviewing."
            ),
        }
    local = _md5_of_file(review_path)
    if local == tracked:
        return None
    return {
        "kind": "stale_local",
        "tracked_md5": tracked,
        "local_md5": local,
        "message": (
            f"Local {review_path.name} differs from the DVC-tracked version. "
            "Run `make review-pull` before reviewing — your saves may "
            "overwrite a peer's work."
        ),
    }
