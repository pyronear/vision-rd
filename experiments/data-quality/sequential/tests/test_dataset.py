"""Tests for data_quality_sequential.dataset."""

from pathlib import Path

from data_quality_sequential.dataset import SequenceRef, iter_sequences


def _make_split(tmp_path: Path, split: str, wildfire: list[str], fp: list[str]) -> Path:
    split_dir = tmp_path / split
    for name in wildfire:
        seq = split_dir / "wildfire" / name / "images"
        seq.mkdir(parents=True)
        (seq / f"{name}_2023-05-23T17-18-31.jpg").touch()
        (seq / f"{name}_2023-05-23T17-18-01.jpg").touch()
    for name in fp:
        seq = split_dir / "fp" / name / "images"
        seq.mkdir(parents=True)
        (seq / f"{name}_2023-05-23T18-00-00.jpg").touch()
    return split_dir


def test_iter_sequences_emits_ground_truth_from_parent_dir(tmp_path: Path) -> None:
    split_dir = _make_split(tmp_path, "val", wildfire=["wf_a", "wf_b"], fp=["fp_a"])

    refs = list(iter_sequences(split_dir, split="val"))

    by_name = {r.name: r for r in refs}
    assert set(by_name) == {"wf_a", "wf_b", "fp_a"}
    assert by_name["wf_a"].ground_truth is True
    assert by_name["wf_b"].ground_truth is True
    assert by_name["fp_a"].ground_truth is False
    assert all(r.split == "val" for r in refs)


def test_iter_sequences_returns_frames_sorted_by_filename(tmp_path: Path) -> None:
    split_dir = _make_split(tmp_path, "val", wildfire=["wf_a"], fp=[])

    [ref] = list(iter_sequences(split_dir, split="val"))

    assert len(ref.frame_paths) == 2
    # Filename-sort puts 17-18-01 before 17-18-31.
    assert ref.frame_paths[0].name.endswith("17-18-01.jpg")
    assert ref.frame_paths[1].name.endswith("17-18-31.jpg")


def test_iter_sequences_skips_sequences_with_no_images(tmp_path: Path) -> None:
    split_dir = tmp_path / "val"
    empty = split_dir / "wildfire" / "empty_seq" / "images"
    empty.mkdir(parents=True)
    populated = split_dir / "wildfire" / "ok_seq" / "images"
    populated.mkdir(parents=True)
    (populated / "ok_seq_2023-05-23T17-18-01.jpg").touch()

    refs = list(iter_sequences(split_dir, split="val"))
    names = {r.name for r in refs}

    assert names == {"ok_seq"}


def test_iter_sequences_handles_missing_wildfire_or_fp_dir(tmp_path: Path) -> None:
    split_dir = _make_split(tmp_path, "test", wildfire=[], fp=["fp_only"])

    refs = list(iter_sequences(split_dir, split="test"))

    assert [r.name for r in refs] == ["fp_only"]
    assert refs[0].ground_truth is False


def test_sequence_ref_is_a_dataclass_with_expected_fields() -> None:
    ref = SequenceRef(
        name="x",
        split="val",
        ground_truth=True,
        frame_paths=[Path("/tmp/x_2023-01-01T00-00-00.jpg")],
    )
    assert ref.name == "x"
    assert ref.split == "val"
    assert ref.ground_truth is True
    assert ref.frame_paths[0].name.endswith(".jpg")
