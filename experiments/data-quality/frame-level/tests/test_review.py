"""Tests for the pure helpers in data_quality_frame_level.review."""

from data_quality_frame_level.review import (
    REVIEW_VOCAB,
    VOCAB_SEED_TAG,
    is_vocab_seed,
    merge_tags,
    payload_from_stem_tags,
    stem_tags_from_payload,
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
