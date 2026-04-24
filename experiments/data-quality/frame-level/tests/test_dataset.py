"""Tests for data_quality_frame_level.dataset."""

from pathlib import Path

import pytest

from data_quality_frame_level.dataset import (
    BBox,
    FrameRef,
    iter_frames,
    parse_yolo_label,
    yolo_to_fiftyone_xywh,
)


@pytest.fixture()
def split_dir(tmp_path: Path) -> Path:
    """A synthetic split with three frames: multi-bbox, empty, missing-label."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    # Frame A: two bboxes.
    (images / "a.jpg").write_bytes(b"fake")
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.05 0.05\n")

    # Frame B: empty label file (fp-origin frame).
    (images / "b.jpg").write_bytes(b"fake")
    (labels / "b.txt").write_text("")

    # Frame C: label file missing entirely.
    (images / "c.jpg").write_bytes(b"fake")

    return tmp_path


def test_parse_yolo_label_multi_line(tmp_path: Path):
    label = tmp_path / "lbl.txt"
    label.write_text("0 0.25 0.75 0.1 0.2\n0 0.5 0.5 0.4 0.4\n")

    bboxes = parse_yolo_label(label)

    assert bboxes == [
        BBox(class_id=0, cx=0.25, cy=0.75, w=0.1, h=0.2),
        BBox(class_id=0, cx=0.5, cy=0.5, w=0.4, h=0.4),
    ]


def test_parse_yolo_label_empty(tmp_path: Path):
    label = tmp_path / "empty.txt"
    label.write_text("")

    assert parse_yolo_label(label) == []


def test_parse_yolo_label_missing(tmp_path: Path):
    assert parse_yolo_label(tmp_path / "does_not_exist.txt") == []


def test_parse_yolo_label_ignores_blank_lines(tmp_path: Path):
    label = tmp_path / "lbl.txt"
    label.write_text("\n0 0.5 0.5 0.2 0.2\n\n")

    assert parse_yolo_label(label) == [BBox(0, 0.5, 0.5, 0.2, 0.2)]


def test_iter_frames_yields_all_images_in_sorted_order(split_dir: Path):
    frames = list(iter_frames(split_dir))

    assert [f.stem for f in frames] == ["a", "b", "c"]


def test_iter_frames_attaches_parsed_bboxes(split_dir: Path):
    by_stem: dict[str, FrameRef] = {f.stem: f for f in iter_frames(split_dir)}

    assert by_stem["a"].gt_bboxes == [
        BBox(0, 0.5, 0.5, 0.2, 0.2),
        BBox(0, 0.1, 0.1, 0.05, 0.05),
    ]
    assert by_stem["b"].gt_bboxes == []
    assert by_stem["c"].gt_bboxes == []


def test_iter_frames_paths_point_at_images(split_dir: Path):
    frames = list(iter_frames(split_dir))
    for frame in frames:
        assert frame.image_path.parent.name == "images"
        assert frame.image_path.suffix == ".jpg"


def test_yolo_to_fiftyone_xywh_centered():
    # cx=cy=0.5, w=h=0.2  ->  top-left corner at (0.4, 0.4)
    assert yolo_to_fiftyone_xywh(BBox(0, 0.5, 0.5, 0.2, 0.2)) == (0.4, 0.4, 0.2, 0.2)


def test_yolo_to_fiftyone_xywh_corner():
    # cx=0.1, cy=0.1, w=0.2, h=0.2  ->  top-left at (0.0, 0.0)
    assert yolo_to_fiftyone_xywh(BBox(0, 0.1, 0.1, 0.2, 0.2)) == (
        pytest.approx(0.0),
        pytest.approx(0.0),
        0.2,
        0.2,
    )


def test_yolo_to_fiftyone_xywh_wide_bottom():
    assert yolo_to_fiftyone_xywh(BBox(0, 0.5, 0.9, 1.0, 0.2)) == (
        pytest.approx(0.0),
        pytest.approx(0.8),
        1.0,
        0.2,
    )
