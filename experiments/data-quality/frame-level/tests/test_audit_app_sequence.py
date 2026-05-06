from data_quality_frame_level.audit_app.sequence import (
    assign_temporal_sequences,
    parse_stem,
)


def test_parse_stem_pyronear():
    assert parse_stem("pyronear-force-06_courmettes_275_2024-02-17T17-36-57") == (
        "pyronear-force-06_courmettes_275",
        "2024-02-17T17-36-57",
    )


def test_parse_stem_hyphenated_source():
    assert parse_stem("awf-axis_baldca_999_2023-06-04T07-35-26") == (
        "awf-axis_baldca_999",
        "2023-06-04T07-35-26",
    )


def test_parse_stem_no_hyphen_in_source():
    assert parse_stem("adf_avinyonet_999_2023-05-23T17-21-00") == (
        "adf_avinyonet_999",
        "2023-05-23T17-21-00",
    )


def test_temporal_sequences_close_frames_same_cluster():
    out = assign_temporal_sequences(
        [
            "p_2024-01-01T00-00-00",
            "p_2024-01-01T00-00-30",
            "p_2024-01-01T00-01-00",
        ]
    )
    assert out == {
        "p_2024-01-01T00-00-00": "p#0",
        "p_2024-01-01T00-00-30": "p#0",
        "p_2024-01-01T00-01-00": "p#0",
    }


def test_temporal_sequences_split_on_gap():
    out = assign_temporal_sequences(
        [
            "p_2024-01-01T00-00-00",
            "p_2024-01-01T00-00-30",
            "p_2024-01-15T00-00-00",
            "p_2024-01-15T00-00-30",
        ],
        max_gap_seconds=180,
    )
    assert out["p_2024-01-01T00-00-00"] == "p#0"
    assert out["p_2024-01-01T00-00-30"] == "p#0"
    assert out["p_2024-01-15T00-00-00"] == "p#1"
    assert out["p_2024-01-15T00-00-30"] == "p#1"


def test_temporal_sequences_disjoint_prefixes_independent():
    out = assign_temporal_sequences(
        [
            "a_2024-01-01T00-00-00",
            "b_2024-01-01T00-00-00",
        ]
    )
    assert out == {
        "a_2024-01-01T00-00-00": "a#0",
        "b_2024-01-01T00-00-00": "b#0",
    }


def test_temporal_sequences_threshold_boundary():
    # Exactly at the boundary stays in the same cluster (gap is not > threshold).
    out = assign_temporal_sequences(
        [
            "p_2024-01-01T00-00-00",
            "p_2024-01-01T00-03-00",
        ],
        max_gap_seconds=180,
    )
    assert out["p_2024-01-01T00-00-00"] == out["p_2024-01-01T00-03-00"]
