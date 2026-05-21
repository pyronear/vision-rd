from bbox_tube_temporal.types import Detection, FrameDetections
from tube_builder_lab.pipeline import (
    PipelineConfig,
    current_builder,
    detections_to_display_tubes,
    extract_pipeline_config,
)


def _fd(idx, dets):
    return FrameDetections(
        frame_idx=idx, frame_id=str(idx), timestamp=None, detections=dets
    )


def _d(cx, cy):
    return Detection(class_id=0, cx=cx, cy=cy, w=0.1, h=0.1, confidence=0.9)


CFG = PipelineConfig(
    max_frames=20,
    iou_threshold=0.2,
    max_misses=2,
    infer_min_tube_length=2,
    min_detected_entries=2,
    interpolate_gaps=True,
    confidence_threshold=0.1,
    iou_nms=0.2,
    image_size=1024,
)


def test_extract_pipeline_config_from_model_config():
    raw = {
        "infer": {"confidence_threshold": 0.1, "iou_nms": 0.2, "image_size": 1024},
        "tubes": {
            "iou_threshold": 0.2,
            "max_misses": 2,
            "infer_min_tube_length": 2,
            "min_detected_entries": 2,
            "interpolate_gaps": True,
        },
        "classifier": {"max_frames": 20},
    }
    assert extract_pipeline_config(raw) == CFG


def test_current_builder_links_a_steady_box():
    # Same box across 3 frames -> a single kept tube (length 3 >= 2).
    fds = [_fd(i, [_d(0.5, 0.5)]) for i in range(3)]
    tubes = detections_to_display_tubes(fds, current_builder(CFG), CFG, truncate=True)
    assert len(tubes) == 1
    assert tubes[0].start_frame == 0
    assert tubes[0].end_frame == 2


def test_truncation_limits_frames():
    # 25 frames of a steady box; truncate to max_frames=20 -> tube ends at 19.
    fds = [_fd(i, [_d(0.5, 0.5)]) for i in range(25)]
    tubes = detections_to_display_tubes(fds, current_builder(CFG), CFG, truncate=True)
    assert len(tubes) == 1
    assert tubes[0].end_frame == 19
    # Untruncated -> ends at 24.
    tubes_full = detections_to_display_tubes(
        fds, current_builder(CFG), CFG, truncate=False
    )
    assert tubes_full[0].end_frame == 24


def test_filter_drops_singleton_tube():
    # A box present in only one frame -> length 1 < infer_min_tube_length -> dropped.
    fds = [_fd(0, [_d(0.5, 0.5)]), _fd(1, []), _fd(2, []), _fd(3, [])]
    tubes = detections_to_display_tubes(fds, current_builder(CFG), CFG, truncate=True)
    assert tubes == []
