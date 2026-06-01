import pytest

import temporal_model_explorer.import_pyro_annotator as ipa
from temporal_model_explorer.store import read_meta


def test_parse_label_smoke_keeps_subtype():
    assert ipa.parse_label("smoke", "wildfire") == ("smoke", "wildfire")
    assert ipa.parse_label("smoke", "industrial") == ("smoke", "industrial")


def test_parse_label_fp_keeps_subtype():
    assert ipa.parse_label("fp", "low_cloud") == ("fp", "low_cloud")


def test_parse_label_unlabeled_is_unknown_with_no_detail():
    assert ipa.parse_label("unlabeled", None) == ("unknown", None)


def test_parse_label_rejects_unknown_class():
    with pytest.raises(ValueError):
        ipa.parse_label("bogus", None)


def _make_seq(root, *parts, det_ids):
    """Create <root>/<parts...>/images/detection_<id>.jpg files."""
    seq_dir = root.joinpath(*parts)
    (seq_dir / "images").mkdir(parents=True)
    for d in det_ids:
        (seq_dir / "images" / f"detection_{d}.jpg").write_bytes(b"img")
    return seq_dir


def test_iter_zip_sequences_finds_seqs_with_class_and_subtype(tmp_path):
    src = tmp_path / "seq_annotation_done_by_label"
    _make_seq(src, "smoke", "wildfire", "seq_40972", det_ids=[1, 2])
    _make_seq(src, "fp", "low_cloud", "seq_40720", det_ids=[5])
    _make_seq(src, "unlabeled", "seq_40438", det_ids=[9])
    # macOS junk must be ignored
    (src / "__MACOSX" / "smoke").mkdir(parents=True)
    (src / "__MACOSX" / "smoke" / "._wildfire").write_bytes(b"junk")

    found = sorted(
        (klass, subtype, seq_id)
        for klass, subtype, seq_id, _ in ipa.iter_zip_sequences(src)
    )
    assert found == [
        ("fp", "low_cloud", 40720),
        ("smoke", "wildfire", 40972),
        ("unlabeled", None, 40438),
    ]


def test_import_enriches_and_matches_timestamps_by_capture_order(tmp_path):
    src = tmp_path / "seq_annotation_done_by_label"
    # zip frame ids (14549, 14551) are a different id space from API ids below.
    _make_seq(src, "smoke", "wildfire", "seq_40972", det_ids=[14551, 14549])
    out = tmp_path / "store"

    def fake_detections(ep, tok, sid, limit, desc):
        assert sid == 40972
        # returned out of order; matched to frames by capture order, not id
        return [
            {"id": 9002, "camera_id": 65, "created_at": "2026-05-09T15:04:50"},
            {"id": 9001, "camera_id": 65, "created_at": "2026-05-09T15:03:49"},
        ]

    camera_index = {65: {"id": 65, "name": "nemours-02", "organization_id": 7}}
    n = ipa.import_pyro_annotator(
        src,
        out,
        "https://x",
        "admintok",
        camera_index=camera_index,
        org_index={7: "sdis-77"},
        list_detections=fake_detections,
    )

    assert n == 1
    seq_dir = out / "pyro-annotator" / "sdis-77" / "nemours-02" / "seq_40972"
    meta = read_meta(seq_dir)
    assert meta.key == "pyro_annotator_40972"
    assert meta.source == "pyro-annotator"
    assert meta.label == "smoke" and meta.label_detail == "wildfire"
    assert meta.label_source == "pyro_annotator_folder"
    assert meta.camera_id == 65 and meta.camera_name == "nemours-02"
    assert meta.organization_id == 7 and meta.organization_name == "sdis-77"
    # frames in capture order (by zip detection id): 14549 then 14551
    assert [f.detection_id for f in meta.frames] == [14549, 14551]
    # timestamps assigned positionally from API times sorted ascending
    assert meta.frames[0].created_at == "2026-05-09T15:03:49"
    assert meta.frames[1].created_at == "2026-05-09T15:04:50"
    assert meta.started_at == "2026-05-09T15:03:49"
    # image copied from the zip
    assert (seq_dir / "images" / "detection_14549.jpg").read_bytes() == b"img"


def test_import_skips_per_frame_times_on_count_mismatch_but_keeps_start(tmp_path):
    src = tmp_path / "seq_annotation_done_by_label"
    _make_seq(src, "smoke", "wildfire", "seq_40973", det_ids=[100])  # 1 frame
    out = tmp_path / "store"

    def fake_detections(ep, tok, sid, limit, desc):
        return [  # 2 detections != 1 frame
            {"id": 1, "camera_id": 65, "created_at": "2026-05-09T15:03:49"},
            {"id": 2, "camera_id": 65, "created_at": "2026-05-09T15:04:50"},
        ]

    ipa.import_pyro_annotator(
        src,
        out,
        "https://x",
        "admintok",
        camera_index={65: {"id": 65, "name": "nemours-02", "organization_id": 7}},
        org_index={7: "sdis-77"},
        list_detections=fake_detections,
    )

    meta = read_meta(out / "pyro-annotator" / "sdis-77" / "nemours-02" / "seq_40973")
    # camera/org still enriched, sequence start still set from earliest API time
    assert meta.camera_name == "nemours-02" and meta.organization_name == "sdis-77"
    assert meta.started_at == "2026-05-09T15:03:49"
    # but per-frame timestamps are left None (frame sets don't line up)
    assert meta.frames[0].created_at is None


def test_import_falls_back_to_unknown_when_no_detections(tmp_path):
    src = tmp_path / "seq_annotation_done_by_label"
    _make_seq(src, "fp", "low_cloud", "seq_99999", det_ids=[3])
    out = tmp_path / "store"

    n = ipa.import_pyro_annotator(
        src,
        out,
        "https://x",
        "admintok",
        camera_index={},
        org_index={},
        list_detections=lambda ep, tok, sid, limit, desc: [],
    )

    assert n == 1
    seq_dir = out / "pyro-annotator" / "unknown" / "unknown" / "seq_99999"
    meta = read_meta(seq_dir)
    assert meta.label == "fp" and meta.label_detail == "low_cloud"
    assert meta.camera_name == "unknown" and meta.organization_name == "unknown"
    assert meta.camera_id is None and meta.organization_id is None
    assert meta.frames[0].created_at is None
    assert meta.started_at is None
