import json
from pathlib import Path

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.export import (
    DiffCounts,
    compute_diff,
    export_corrections,
    write_manifest_and_labels,
)
from data_quality_frame_level.review_app.persistence import (
    ReviewState,
    SampleReview,
)


def _bb(cx, cy, w=0.1, h=0.1):
    return BBox(class_id=0, cx=cx, cy=cy, w=w, h=h)


def test_compute_diff_added_removed_modified():
    original = [_bb(0.1, 0.1), _bb(0.5, 0.5)]
    corrected = [_bb(0.5, 0.5), _bb(0.9, 0.9)]
    counts = compute_diff(original=original, corrected=corrected)
    assert counts == DiffCounts(added=1, removed=1, modified=0)


def test_compute_diff_modified():
    original = [_bb(0.5, 0.5, w=0.1, h=0.1)]
    corrected = [_bb(0.55, 0.55, w=0.1, h=0.1)]
    counts = compute_diff(original=original, corrected=corrected)
    assert counts.added == 0 and counts.removed == 0 and counts.modified == 1


def test_export_writes_only_changed(tmp_path: Path):
    originals = {
        "stem_a": [_bb(0.5, 0.5)],
        "stem_b": [_bb(0.5, 0.5)],
    }
    review = ReviewState(
        model="m",
        split="val",
        samples={
            "stem_a": SampleReview(status="reviewed", bboxes=[_bb(0.5, 0.5)]),
            "stem_b": SampleReview(status="reviewed", bboxes=[_bb(0.6, 0.6)]),
            "stem_c": SampleReview(status="unclear", bboxes=[_bb(0.5, 0.5)]),
        },
    )
    out = tmp_path / "10_export" / "m" / "val"
    export_corrections(review=review, originals=originals, out_dir=out)
    assert (out / "labels" / "stem_b.txt").exists()
    assert not (out / "labels" / "stem_a.txt").exists()
    assert not (out / "labels" / "stem_c.txt").exists()
    text = (out / "labels" / "stem_b.txt").read_text()
    assert text.strip().split() == ["0", "0.6", "0.6", "0.1", "0.1"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["totals"]["changed"] == 1
    assert [c["stem"] for c in manifest["changed"]] == ["stem_b"]


def test_export_manifest_contributors_is_sorted_unique_reviewers(tmp_path: Path):
    review = ReviewState(
        model="m",
        split="val",
        samples={
            "stem_a": SampleReview(
                status="reviewed", bboxes=[_bb(0.6, 0.6)], reviewer="mateo"
            ),
            "stem_b": SampleReview(
                status="reviewed", bboxes=[_bb(0.7, 0.7)], reviewer="arthur"
            ),
            "stem_c": SampleReview(
                status="reviewed", bboxes=[_bb(0.8, 0.8)], reviewer="arthur"
            ),
            "stem_d": SampleReview(status="reviewed", bboxes=[_bb(0.9, 0.9)]),
        },
    )
    originals = {st: [] for st in review.samples}
    out = tmp_path / "10_export" / "m" / "val"
    write_manifest_and_labels(review=review, originals=originals, out_dir=out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["contributors"] == ["arthur", "mateo"]
