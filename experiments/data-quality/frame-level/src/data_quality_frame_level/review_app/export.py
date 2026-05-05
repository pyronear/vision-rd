"""Export corrected GT to a YOLO-format patch + manifest.

Only stems whose corrected bboxes differ from the on-disk original are
written. Emits a flat ``labels/<stem>.txt`` tree (no split subdir) and
a ``manifest.json`` summarizing what changed. ``unclear`` samples are
excluded — they are open questions, not decisions.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.matching import iou
from data_quality_frame_level.review_app.persistence import ReviewState

UNCHANGED_IOU = 0.95


@dataclass(frozen=True)
class DiffCounts:
    added: int
    removed: int
    modified: int

    @property
    def is_change(self) -> bool:
        return self.added + self.removed + self.modified > 0


def compute_diff(*, original: list[BBox], corrected: list[BBox]) -> DiffCounts:
    matched_orig: set[int] = set()
    matched_corr: set[int] = set()
    modified = 0
    candidates = sorted(
        (
            (i, j, iou(o, c))
            for i, o in enumerate(original)
            for j, c in enumerate(corrected)
        ),
        key=lambda x: x[2],
        reverse=True,
    )
    for oi, cj, score in candidates:
        if score == 0.0:
            break
        if oi in matched_orig or cj in matched_corr:
            continue
        matched_orig.add(oi)
        matched_corr.add(cj)
        if score < UNCHANGED_IOU:
            modified += 1
    removed = len(original) - len(matched_orig)
    added = len(corrected) - len(matched_corr)
    return DiffCounts(added=added, removed=removed, modified=modified)


def _write_yolo_txt(path: Path, bboxes: list[BBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{b.class_id} {b.cx} {b.cy} {b.w} {b.h}" for b in bboxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def export_corrections(
    *,
    review: ReviewState,
    originals: dict[str, list[BBox]],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels"
    changed: list[dict] = []
    totals = {"changed": 0, "added": 0, "removed": 0, "modified": 0}
    for stem in sorted(review.samples):
        sample = review.samples[stem]
        if sample.status != "reviewed":
            continue
        original = originals.get(stem, [])
        diff = compute_diff(original=original, corrected=sample.bboxes)
        if not diff.is_change:
            continue
        _write_yolo_txt(labels_dir / f"{stem}.txt", sample.bboxes)
        changed.append(
            {
                "stem": stem,
                "added": diff.added,
                "removed": diff.removed,
                "modified": diff.modified,
                "reviewer": sample.reviewer,
                "note": sample.note,
            }
        )
        totals["changed"] += 1
        totals["added"] += diff.added
        totals["removed"] += diff.removed
        totals["modified"] += diff.modified
    manifest = {
        "version": 1,
        "model_name": review.model,
        "split": review.split,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "changed": changed,
        "totals": totals,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
