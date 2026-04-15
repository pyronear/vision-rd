"""Pure-function helpers used by :class:`SmokeynetTemporalModel.predict`.

Each helper corresponds to one stage of the six-stage pipeline described in
``docs/specs/2026-04-15-temporal-model-protocol-design.md``. Kept separate so
``predict()`` is thin and each stage is unit-testable in isolation.
"""

from typing import Any

import numpy as np
import torch
from PIL import Image
from pyrocore.types import Frame
from torchvision.transforms.functional import to_tensor

from .model_input import crop_and_resize, expand_bbox, norm_bbox_to_pixel_square
from .tubes import interpolate_gaps as _interpolate_gaps
from .types import Detection, FrameDetections, Tube


def run_yolo_on_frames(
    yolo_model: Any,
    frames: list[Frame],
    *,
    confidence_threshold: float,
    iou_nms: float,
    image_size: int,
) -> list[FrameDetections]:
    """Run YOLO once over all frames in a single batched call.

    Args:
        yolo_model: An ultralytics ``YOLO`` instance (or any object exposing
            ``predict(list_of_paths, ...)`` with the same return shape).
        frames: Temporally ordered Pyronear :class:`Frame` objects.
        confidence_threshold: Minimum detection confidence.
        iou_nms: IoU threshold for YOLO's internal NMS.
        image_size: Inference resolution passed to YOLO.

    Returns:
        One :class:`FrameDetections` per input frame (possibly with zero
        detections), in the same order, with ``frame_idx`` = position.
    """
    if not frames:
        return []

    paths = [str(f.image_path) for f in frames]
    results = yolo_model.predict(
        paths,
        conf=confidence_threshold,
        iou=iou_nms,
        imgsz=image_size,
        verbose=False,
    )

    out: list[FrameDetections] = []
    for idx, (frame, pred) in enumerate(zip(frames, results, strict=True)):
        detections: list[Detection] = []
        boxes = pred.boxes
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn
            confs = boxes.conf
            cls = boxes.cls
            for i in range(len(boxes)):
                row = xywhn[i].tolist()
                detections.append(
                    Detection(
                        class_id=int(cls[i].item()),
                        cx=row[0],
                        cy=row[1],
                        w=row[2],
                        h=row[3],
                        confidence=float(confs[i].item()),
                    )
                )
        out.append(
            FrameDetections(
                frame_idx=idx,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                detections=detections,
            )
        )
    return out


def filter_and_interpolate_tubes(
    tubes: list[Tube],
    *,
    min_tube_length: int,
    min_detected_entries: int,
    interpolate_gaps: bool,
) -> list[Tube]:
    """Filter tubes by length / observation count, then optionally interpolate gaps.

    Args:
        tubes: Candidate tubes (output of :func:`~smokeynet_adapted.tubes.build_tubes`).
        min_tube_length: Keep tubes where
            ``end_frame - start_frame + 1 >= min_tube_length``.
        min_detected_entries: Keep tubes with at least this many non-gap entries.
        interpolate_gaps: If True, fill gap entries in surviving tubes via
            :func:`~smokeynet_adapted.tubes.interpolate_gaps`.

    Returns:
        Surviving tubes in original order.
    """
    survivors: list[Tube] = []
    for t in tubes:
        length = t.end_frame - t.start_frame + 1
        if length < min_tube_length:
            continue
        n_obs = sum(1 for e in t.entries if e.detection is not None)
        if n_obs < min_detected_entries:
            continue
        if interpolate_gaps:
            _interpolate_gaps(t)
        survivors.append(t)
    return survivors


def crop_tube_patches(
    tube: Tube,
    frames: list[Frame],
    *,
    context_factor: float,
    patch_size: int,
    max_frames: int,
    normalization_mean: list[float],
    normalization_std: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop patches for a single tube, padded/truncated to ``max_frames``.

    Matches ``TubePatchDataset.__getitem__`` exactly: PIL→uint8 array,
    ``expand_bbox → norm_bbox_to_pixel_square → crop_and_resize``, then
    ``to_tensor`` (CHW, float32, [0,1]), then mean/std normalization.
    """
    frame_by_idx = {i: f for i, f in enumerate(frames)}

    n = min(len(tube.entries), max_frames)
    patches = torch.zeros(max_frames, 3, patch_size, patch_size, dtype=torch.float32)
    mask = torch.zeros(max_frames, dtype=torch.bool)
    mean_t = torch.tensor(normalization_mean).view(3, 1, 1)
    std_t = torch.tensor(normalization_std).view(3, 1, 1)

    for slot, entry in enumerate(tube.entries[:n]):
        det = entry.detection
        if det is None:
            # Shouldn't happen post-interpolation; leave zero-padded + mask=False.
            continue
        frame = frame_by_idx[entry.frame_idx]
        image = np.array(Image.open(frame.image_path).convert("RGB"))
        img_h, img_w, _ = image.shape

        cx, cy, w, h = expand_bbox(det.cx, det.cy, det.w, det.h, context_factor)
        box = norm_bbox_to_pixel_square(cx, cy, w, h, img_w, img_h)
        patch_np = crop_and_resize(image, box, patch_size)
        patch_t = to_tensor(Image.fromarray(patch_np))  # CHW float32 [0,1]
        patches[slot] = (patch_t - mean_t) / std_t
        mask[slot] = True

    return patches, mask


def score_tubes(
    classifier: Any,
    *,
    patches_per_tube: list[torch.Tensor],
    masks_per_tube: list[torch.Tensor],
) -> torch.Tensor:
    """Run one batched classifier forward over all tubes.

    Args:
        classifier: A callable ``(patches[N,T,3,H,W], mask[N,T]) -> logits[N]``.
        patches_per_tube: One ``[T, 3, H, W]`` tensor per tube.
        masks_per_tube: One ``[T]`` bool tensor per tube.

    Returns:
        ``Tensor[N]`` of logits (empty tensor if no tubes).
    """
    if not patches_per_tube:
        return torch.zeros(0)
    patches = torch.stack(patches_per_tube, dim=0)
    mask = torch.stack(masks_per_tube, dim=0)
    with torch.no_grad():
        return classifier(patches, mask)
