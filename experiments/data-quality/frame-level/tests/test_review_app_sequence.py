from data_quality_frame_level.review_app.sequence import parse_stem


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
