import zipfile
from pathlib import Path

from temporal_model_explorer.import_local_zip import import_zip, label_from_parts
from temporal_model_explorer.store import read_meta


def test_label_from_parts():
    assert label_from_parts(("smoke", "wildfire")) == ("smoke", "wildfire")
    assert label_from_parts(("fp", "tree")) == ("fp", "tree")
    assert label_from_parts(("unlabeled",)) == ("unknown", None)
    assert label_from_parts(("smoke",)) == ("smoke", None)


def _make_zip(path: Path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("root/smoke/wildfire/seq_10/images/detection_2.jpg", b"b")
        zf.writestr("root/smoke/wildfire/seq_10/images/detection_1.jpg", b"a")
        zf.writestr("root/smoke/wildfire/seq_10/labels/detection_1.txt", b"0 0 0 0 0")
        zf.writestr("root/fp/tree/seq_20/images/detection_3.jpg", b"c")
        zf.writestr("__MACOSX/root/._x", b"junk")
        zf.writestr("root/.DS_Store", b"junk")


def test_import_zip_writes_store(tmp_path):
    z = tmp_path / "data.zip"
    _make_zip(z)
    store = tmp_path / "store"
    n = import_zip(z, store)
    assert n == 2

    smoke = read_meta(store / "zip_10")
    assert smoke.label == "smoke" and smoke.label_detail == "wildfire"
    assert smoke.source == "local_zip" and smoke.sequence_id == "10"
    # images copied, ordered by filename (detection_1 before detection_2)
    assert [f.file for f in smoke.frames] == [
        "images/detection_1.jpg",
        "images/detection_2.jpg",
    ]
    assert (store / "zip_10" / "images" / "detection_1.jpg").read_bytes() == b"a"

    fp = read_meta(store / "zip_20")
    assert fp.label == "fp" and fp.label_detail == "tree"
