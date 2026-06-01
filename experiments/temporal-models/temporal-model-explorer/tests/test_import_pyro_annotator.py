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
