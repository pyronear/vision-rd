"""Atomic read/write of ``review.json`` per ``(model, split)``.

The file shape is documented in the design spec §5.1. Writes go through
a sibling ``.tmp`` + ``os.replace`` so partial writes can never be
observed. Reads of missing files return an empty :class:`ReviewState`.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from data_quality_frame_level.dataset import BBox

PAYLOAD_VERSION = 1
ALLOWED_STATUS = ("reviewed", "unclear")


@dataclass
class SampleReview:
    status: str
    bboxes: list[BBox] = field(default_factory=list)
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
            stem: _sample_to_dict(state.samples[stem])
            for stem in sorted(state.samples)
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
