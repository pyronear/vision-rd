"""Export corrected GT to a YOLO-format patch + manifest + provenance.

The export directory contains four siblings:

  labels/<stem>.txt     # corrected YOLO labels (only-changed frames)
  manifest.json         # apply contract — pyro-dataset reads this
  pending.json          # unclear-status frames for second-opinion review
  provenance.json       # audit-side context for reproducibility

Each writer here is a pure function over its inputs; the CLI
(``scripts/export_review_app.py``) gathers the git/DVC/params context
and feeds it in.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
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


@dataclass(frozen=True)
class ProvenanceInput:
    audit_repo: str
    audit_commit: str
    audit_branch: str
    experiment: str
    thresholds: dict[str, float]
    predictions_path: str
    predictions_md5: str


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


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest_and_labels(
    *,
    review: ReviewState,
    originals: dict[str, list[BBox]],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels"
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    changed: list[dict] = []
    totals = {"changed": 0, "added": 0, "removed": 0, "modified": 0}
    for stem in sorted(review.samples):
        sample = review.samples[stem]
        if sample.status != "reviewed":
            continue
        original = originals.get(stem, [])
        if sample.bboxes:
            effective = sample.bboxes
        else:
            effective = [
                o
                for o in original
                if not any(
                    iou(o, s) >= UNCHANGED_IOU for s in sample.spurious_originals
                )
            ]
        diff = compute_diff(original=original, corrected=effective)
        if not diff.is_change:
            continue
        _write_yolo_txt(labels_dir / f"{stem}.txt", effective)
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
    contributors = sorted({c["reviewer"] for c in changed if c["reviewer"]})
    manifest = {
        "version": 1,
        "model": review.model,
        "split": review.split,
        "exported_at": _now_iso(),
        "contributors": contributors,
        "changed": changed,
        "totals": totals,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_pending(*, review: ReviewState, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        {
            "stem": stem,
            "reviewer": sample.reviewer,
            "note": sample.note,
        }
        for stem, sample in sorted(review.samples.items())
        if sample.status == "unclear"
    ]
    payload = {
        "version": 1,
        "model": review.model,
        "split": review.split,
        "exported_at": _now_iso(),
        "pending": pending,
    }
    (out_dir / "pending.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_provenance(
    *,
    prov: ProvenanceInput,
    model: str,
    split: str,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "audit_repo": prov.audit_repo,
        "audit_commit": prov.audit_commit,
        "audit_branch": prov.audit_branch,
        "experiment": prov.experiment,
        "model": model,
        "split": split,
        "thresholds": prov.thresholds,
        "predictions_path": prov.predictions_path,
        "predictions_md5": prov.predictions_md5,
        "exported_at": _now_iso(),
    }
    (out_dir / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def export_corrections(
    *,
    review: ReviewState,
    originals: dict[str, list[BBox]],
    out_dir: Path,
    provenance: ProvenanceInput | None = None,
) -> dict:
    """Orchestrator: write all four files. Returns the manifest payload."""
    manifest = write_manifest_and_labels(
        review=review, originals=originals, out_dir=out_dir
    )
    write_pending(review=review, out_dir=out_dir)
    if provenance is not None:
        write_provenance(
            prov=provenance, model=review.model, split=review.split, out_dir=out_dir
        )
    return manifest
