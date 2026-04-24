"""Tests for the pure helpers in data_quality_frame_level.review."""

from data_quality_frame_level.review import (
    REVIEW_VOCAB,
    VOCAB_SEED_TAG,
    format_invalid_report,
    invalid_tags,
    is_valid_tag,
    is_vocab_seed,
    merge_tags,
    payload_from_stem_tags,
    scan_invalid,
    stem_tags_from_payload,
    suggest_tag,
)


def test_payload_roundtrip_preserves_stems_and_tags():
    stem_tags = {
        "img_a": ["label:add-smoke", "reviewer:arthur"],
        "img_b": ["status:unclear"],
    }

    payload = payload_from_stem_tags("dq-frame_foo_val", stem_tags)
    assert payload["dataset_name"] == "dq-frame_foo_val"
    assert payload["tags_by_stem"] == stem_tags

    # Going back to the map strips metadata.
    recovered = stem_tags_from_payload(payload)
    assert recovered == stem_tags


def test_payload_skips_stems_with_empty_tag_lists():
    stem_tags = {
        "img_a": ["label:ok"],
        "img_b": [],
        "img_c": [],
    }

    payload = payload_from_stem_tags("dq-frame_foo_val", stem_tags)

    # No reason to persist empty tag lists — they bloat the file without info.
    assert payload["tags_by_stem"] == {"img_a": ["label:ok"]}


def test_payload_sorts_tags_deterministically():
    stem_tags = {"img_a": ["status:unclear", "label:add-smoke", "reviewer:arthur"]}

    payload = payload_from_stem_tags("dq-frame_foo_val", stem_tags)

    # Stable order makes diffs in git review-friendly.
    assert payload["tags_by_stem"]["img_a"] == [
        "label:add-smoke",
        "reviewer:arthur",
        "status:unclear",
    ]


def test_merge_tags_unions_without_duplicates():
    existing = ["label:add-smoke"]
    incoming = ["label:add-smoke", "reviewer:arthur"]

    assert merge_tags(existing, incoming) == ["label:add-smoke", "reviewer:arthur"]


def test_merge_tags_preserves_existing_when_incoming_is_empty():
    existing = ["label:ok"]

    assert merge_tags(existing, []) == ["label:ok"]


def test_merge_tags_is_sorted():
    assert merge_tags(["z"], ["a", "m"]) == ["a", "m", "z"]


def test_is_vocab_seed_detects_marker():
    assert is_vocab_seed([VOCAB_SEED_TAG, *REVIEW_VOCAB]) is True
    assert is_vocab_seed(["label:add-smoke"]) is False
    assert is_vocab_seed([]) is False


def test_is_valid_tag_accepts_vocab_and_seed_and_reviewer_prefix():
    for vocab_entry in REVIEW_VOCAB:
        assert is_valid_tag(vocab_entry) is True
    assert is_valid_tag(VOCAB_SEED_TAG) is True
    assert is_valid_tag("reviewer:arthur") is True
    assert is_valid_tag("reviewer:") is True  # empty handle is still prefix-valid


def test_is_valid_tag_rejects_typos_and_unknown_prefixes():
    assert is_valid_tag("lable:add-smoke") is False
    assert is_valid_tag("label:add-smoke ") is False  # trailing space
    assert is_valid_tag("LABEL:add-smoke") is False  # case-sensitive
    assert is_valid_tag("notes:foo") is False  # unknown prefix
    assert is_valid_tag("") is False


def test_invalid_tags_returns_only_rejected_entries():
    tags = ["label:add-smoke", "lable:add-smoke", "reviewer:arthur", "typo"]
    assert invalid_tags(tags) == ["lable:add-smoke", "typo"]


def test_invalid_tags_empty_in_empty_out():
    assert invalid_tags([]) == []
    assert invalid_tags(["label:ok"]) == []


def test_suggest_tag_returns_closest_vocab_entry():
    assert suggest_tag("lable:add-smoke") == "label:add-smoke"
    assert suggest_tag("label:add-smok") == "label:add-smoke"
    assert suggest_tag("status:unclearr") == "status:unclear"


def test_suggest_tag_returns_none_for_unrelated_input():
    assert suggest_tag("") is None
    assert suggest_tag("totally-different-tag-name") is None


def test_scan_invalid_returns_only_stems_with_bad_tags():
    stem_tags = {
        "a": ["label:ok"],
        "b": ["lable:add-smoke", "reviewer:arthur"],
        "c": ["unknown-prefix:foo"],
    }

    assert scan_invalid(stem_tags) == {
        "b": ["lable:add-smoke"],
        "c": ["unknown-prefix:foo"],
    }


def test_scan_invalid_clean_input_returns_empty():
    assert scan_invalid({"a": ["label:ok"], "b": []}) == {}


def test_format_invalid_report_includes_suggestions():
    report = format_invalid_report("ds-foo", {"stem1": ["lable:add-smoke", "wat"]})

    assert "ds-foo" in report
    assert "stem1" in report
    assert "lable:add-smoke" in report
    assert "label:add-smoke" in report  # suggestion
    assert "wat" in report  # no suggestion (too far); must still show


def test_format_invalid_report_lists_every_bad_tag():
    stem_to_invalid = {
        "x": ["foo", "bar"],
        "y": ["baz"],
    }

    report = format_invalid_report("ds", stem_to_invalid)
    for bad in ("foo", "bar", "baz"):
        assert bad in report
    # "3 invalid tag(s)" summary across "2 sample(s)"
    assert "3 invalid" in report
    assert "2 sample" in report
