import temporal_model_explorer.import_pyro_annotator as ipa


def test_parse_label_smoke_keeps_subtype():
    assert ipa.parse_label("smoke", "wildfire") == ("smoke", "wildfire")
    assert ipa.parse_label("smoke", "industrial") == ("smoke", "industrial")


def test_parse_label_fp_keeps_subtype():
    assert ipa.parse_label("fp", "low_cloud") == ("fp", "low_cloud")


def test_parse_label_unlabeled_is_unknown_with_no_detail():
    assert ipa.parse_label("unlabeled", None) == ("unknown", None)


def test_parse_label_rejects_unknown_class():
    import pytest

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
