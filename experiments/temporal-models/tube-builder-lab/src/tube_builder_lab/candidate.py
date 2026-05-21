"""The candidate tube builder — EDIT THIS to improve linking.

Seeded identical to the current lib builder so the lab's diff starts empty.
Iterate freely here (containment / IoMin association, larger max_misses, a
post-hoc merge pass, ...) and click "Re-run candidate" in the app to see the
result against the working-set sequences. Build on the lib primitives
(`compute_iou`, `match_detections`, types) so propagating the winner to
lib/bbox-tube-temporal later is mechanical.
"""

from __future__ import annotations

from bbox_tube_temporal.tubes import build_tubes
from bbox_tube_temporal.types import FrameDetections, Tube

# Current defaults (kept in sync with the model config); change as you iterate.
IOU_THRESHOLD = 0.2
MAX_MISSES = 2


def build_tubes_candidate(frame_detections: list[FrameDetections]) -> list[Tube]:
    return build_tubes(
        frame_detections,
        iou_threshold=IOU_THRESHOLD,
        max_misses=MAX_MISSES,
    )
