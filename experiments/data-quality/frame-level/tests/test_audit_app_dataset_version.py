from pathlib import Path

from data_quality_frame_level.audit_app.dataset_version import read_dataset_version

_DVC_TEMPLATE = """\
md5: abc
frozen: true
deps:
- path: data/processed/yolo_train_val/images/{split}
  repo:
    url: https://github.com/pyronear/pyro-dataset
    rev: {rev}
    rev_lock: deadbeef
outs:
- md5: cafebabe.dir
  path: {kind}
"""


def _write(root: Path, split: str, kind: str, rev: str) -> None:
    d = root / split
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{kind}.dvc").write_text(
        _DVC_TEMPLATE.format(split=split, rev=rev, kind=kind)
    )


def test_returns_none_when_root_missing(tmp_path: Path):
    assert read_dataset_version(tmp_path / "nope") is None


def test_returns_none_when_no_dvc_files(tmp_path: Path):
    (tmp_path / "train").mkdir()
    assert read_dataset_version(tmp_path) is None


def test_returns_single_rev_when_all_agree(tmp_path: Path):
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            _write(tmp_path, split, kind, "v4.0.0")
    assert read_dataset_version(tmp_path) == "v4.0.0"


def test_returns_mixed_when_revs_disagree(tmp_path: Path):
    _write(tmp_path, "train", "images", "v4.0.0")
    _write(tmp_path, "train", "labels", "v4.0.0")
    _write(tmp_path, "val", "images", "v3.0.0")
    _write(tmp_path, "val", "labels", "v4.0.0")
    out = read_dataset_version(tmp_path)
    assert out == "mixed: v3.0.0, v4.0.0"


def test_ignores_dvc_without_repo_section(tmp_path: Path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "images.dvc").write_text(
        "outs:\n- md5: foo.dir\n  path: images\n"
    )
    _write(tmp_path, "val", "images", "v4.0.0")
    assert read_dataset_version(tmp_path) == "v4.0.0"
