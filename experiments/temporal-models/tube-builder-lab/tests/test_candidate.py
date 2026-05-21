from bbox_tube_temporal.tubes import build_tubes
from bbox_tube_temporal.types import Detection, FrameDetections

from tube_builder_lab.candidate import build_tubes_candidate


def _fd(idx, *boxes):
    return FrameDetections(
        frame_idx=idx,
        frame_id=str(idx),
        timestamp=None,
        detections=[
            Detection(class_id=0, cx=cx, cy=cy, w=0.1, h=0.1, confidence=0.9)
            for cx, cy in boxes
        ],
    )


def test_candidate_seed_matches_current_builder():
    fds = [_fd(0, (0.5, 0.5)), _fd(1, (0.52, 0.5)), _fd(2, (0.8, 0.2))]
    expected = build_tubes(fds, iou_threshold=0.2, max_misses=2)
    got = build_tubes_candidate(fds)
    assert [(t.start_frame, t.end_frame, len(t.entries)) for t in got] == [
        (t.start_frame, t.end_frame, len(t.entries)) for t in expected
    ]
