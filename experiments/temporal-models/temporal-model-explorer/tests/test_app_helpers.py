from PIL import Image

from temporal_model_explorer.app import (
    day_of,
    draw_bboxes,
    frame_bboxes_by_input_index,
    processed_to_input_index,
    tube_input_boxes,
)


def test_day_of():
    assert day_of("2026-05-19T14:10:01.123") == "2026-05-19"
    assert day_of(None) == "unknown"


def test_processed_to_input_index_no_padding_is_identity():
    assert processed_to_input_index(0, []) == 0
    assert processed_to_input_index(5, []) == 5


def test_processed_to_input_index_with_padding():
    # 2 real frames padded to 4 -> synthetic slots [0, 3], real at 1,2 -> input 0,1
    padded = [0, 3]
    assert processed_to_input_index(1, padded) == 0
    assert processed_to_input_index(2, padded) == 1
    assert processed_to_input_index(0, padded) is None
    assert processed_to_input_index(3, padded) is None


def test_frame_bboxes_by_input_index_skips_none_bbox():
    details = {
        "preprocessing": {"padded_frame_indices": []},
        "tubes": {
            "kept": [
                {
                    "entries": [
                        {"frame_idx": 0, "bbox": [0.5, 0.5, 0.1, 0.1]},
                        {"frame_idx": 1, "bbox": None},
                        {"frame_idx": 2, "bbox": [0.2, 0.2, 0.05, 0.05]},
                    ]
                }
            ]
        },
    }
    out = frame_bboxes_by_input_index(details)
    assert out == {0: [(0.5, 0.5, 0.1, 0.1)], 2: [(0.2, 0.2, 0.05, 0.05)]}


def test_tube_input_boxes():
    tube = {
        "entries": [
            {"frame_idx": 0, "bbox": [0.5, 0.5, 0.1, 0.1]},
            {"frame_idx": 1, "bbox": None},
        ]
    }
    assert tube_input_boxes(tube, []) == [(0, (0.5, 0.5, 0.1, 0.1))]


def test_draw_bboxes_preserves_size(tmp_path):
    p = tmp_path / "f.jpg"
    Image.new("RGB", (100, 80), "white").save(p)
    out = draw_bboxes(p, [(0.5, 0.5, 0.2, 0.2)])
    assert out.size == (100, 80)
