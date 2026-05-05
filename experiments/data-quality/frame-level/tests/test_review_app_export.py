import json
from pathlib import Path

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.export import (
    DiffCounts,
    ProvenanceInput,
    compute_diff,
    export_corrections,
    write_manifest_and_labels,
    write_pending,
    write_provenance,
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


def test_export_pending_includes_only_unclear(tmp_path: Path):
    review = ReviewState(
        model="m",
        split="val",
        samples={
            "stem_a": SampleReview(status="reviewed", bboxes=[_bb(0.5, 0.5)]),
            "stem_b": SampleReview(
                status="unclear", bboxes=[], reviewer="arthur", note="check this"
            ),
            "stem_c": SampleReview(status="unclear", bboxes=[], reviewer="mateo"),
        },
    )
    out = tmp_path / "10_export" / "m" / "val"
    write_pending(review=review, out_dir=out)
    pending = json.loads((out / "pending.json").read_text())
    assert pending["version"] == 1
    assert pending["model"] == "m"
    assert pending["split"] == "val"
    assert [p["stem"] for p in pending["pending"]] == ["stem_b", "stem_c"]
    by_stem = {p["stem"]: p for p in pending["pending"]}
    assert by_stem["stem_b"]["reviewer"] == "arthur"
    assert by_stem["stem_b"]["note"] == "check this"
    assert by_stem["stem_c"]["reviewer"] == "mateo"


def test_export_pending_empty_when_no_unclear(tmp_path: Path):
    review = ReviewState(
        model="m",
        split="val",
        samples={
            "stem_a": SampleReview(status="reviewed", bboxes=[_bb(0.5, 0.5)]),
        },
    )
    out = tmp_path / "10_export" / "m" / "val"
    write_pending(review=review, out_dir=out)
    pending = json.loads((out / "pending.json").read_text())
    assert pending["pending"] == []


def test_export_provenance_writes_all_fields(tmp_path: Path):
    prov = ProvenanceInput(
        audit_repo="pyronear/vision-rd",
        audit_commit="abc1234",
        audit_branch="arthur/feature",
        experiment="experiments/data-quality/frame-level",
        thresholds={"conf": 0.05, "iou": 0.05, "review_conf": 0.35},
        predictions_path="data/07_model_output/m/val/predictions.json",
        predictions_md5="deadbeefcafe",
    )
    out = tmp_path / "10_export" / "m" / "val"
    write_provenance(prov=prov, model="m", split="val", out_dir=out)
    payload = json.loads((out / "provenance.json").read_text())
    assert payload["version"] == 1
    assert payload["audit_repo"] == "pyronear/vision-rd"
    assert payload["audit_commit"] == "abc1234"
    assert payload["audit_branch"] == "arthur/feature"
    assert payload["experiment"] == "experiments/data-quality/frame-level"
    assert payload["model"] == "m"
    assert payload["split"] == "val"
    assert payload["thresholds"] == {"conf": 0.05, "iou": 0.05, "review_conf": 0.35}
    assert payload["predictions_path"] == "data/07_model_output/m/val/predictions.json"
    assert payload["predictions_md5"] == "deadbeefcafe"
    assert "exported_at" in payload
