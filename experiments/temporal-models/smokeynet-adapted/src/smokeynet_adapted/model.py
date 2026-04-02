"""TemporalModel implementation for the adapted SmokeyNet."""

from pathlib import Path
from typing import Any, Self

import cv2
import torch
from pyrocore import Frame, TemporalModel, TemporalModelOutput
from torch import Tensor

from .backbone import YoloRoiExtractor
from .detector import run_yolo_on_frame
from .net import SmokeyNetAdapted
from .package import ModelPackage, load_model_package
from .tubes import build_tubes
from .types import FrameDetections


class SmokeyNetModel(TemporalModel):
    """Adapted SmokeyNet: YOLO + LSTM + ViT temporal smoke detection.

    Implements the pyrocore :class:`TemporalModel` ABC.  The full inference
    pipeline is: YOLO detection -> RoI Align features -> build tubes online
    -> LSTM temporal fusion -> ViT spatial attention -> classify.

    Construct from a packaged archive via :meth:`from_package`.
    """

    def __init__(
        self,
        yolo_model: Any,
        net: SmokeyNetAdapted,
        roi_extractor: YoloRoiExtractor,
        config: dict[str, Any],
    ) -> None:
        self._yolo_model = yolo_model
        self._net = net
        self._net.eval()
        self._roi_extractor = roi_extractor
        self._roi_extractor.eval()
        self._config = config

    @classmethod
    def from_package(cls, package_path: Path) -> Self:
        """Load from a packaged model archive.

        Args:
            package_path: Path to a ``.zip`` archive created by
                :func:`~smokeynet_adapted.package.build_model_package`.
        """
        pkg: ModelPackage = load_model_package(package_path)
        roi_extractor = YoloRoiExtractor(
            yolo_model=pkg.yolo_model,
            d_model=pkg.config["train"]["d_model"],
            roi_size=pkg.extract_params.get("roi_size", 7),
            context_factor=pkg.extract_params.get("context_factor", 1.2),
        )
        return cls(
            yolo_model=pkg.yolo_model,
            net=pkg.net,
            roi_extractor=roi_extractor,
            config=pkg.config,
        )

    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        """Run the full inference pipeline on a loaded sequence.

        Args:
            frames: Temporally ordered :class:`Frame` objects.

        Returns:
            :class:`TemporalModelOutput` with classification and metadata.
        """
        infer = self._config["infer"]
        tubes_cfg = self._config["tubes"]
        threshold = self._config.get("classification_threshold", 0.5)
        max_dets = self._config["extract"].get("max_detections_per_frame", 10)

        # 1. YOLO detection on each frame
        frame_dets = self._run_detections(frames, infer, max_dets)

        # 2. Build tubes online
        tubes = build_tubes(
            frame_dets,
            iou_threshold=tubes_cfg["iou_threshold"],
            max_misses=tubes_cfg["max_misses"],
        )

        # 3. Extract RoI features per frame
        roi_features, frame_indices, bbox_coords = self._extract_roi_features(
            frames, frame_dets
        )

        # 4. Forward through net
        with torch.no_grad():
            seq_logit, _ = self._net(roi_features, frame_indices, bbox_coords, tubes)

        # 5. Classify
        prob = torch.sigmoid(seq_logit).item()
        is_positive = prob > threshold

        # 6. Determine trigger_frame_index
        trigger_idx = self._find_trigger_frame(frame_dets, is_positive)

        return TemporalModelOutput(
            is_positive=is_positive,
            trigger_frame_index=trigger_idx,
            details={
                "probability": prob,
                "num_tubes": len(tubes),
                "num_detections_total": int(roi_features.shape[0]),
                "num_frames": len(frames),
            },
        )

    def _run_detections(
        self,
        frames: list[Frame],
        infer: dict[str, Any],
        max_dets: int,
    ) -> list[FrameDetections]:
        """Run YOLO on each frame."""
        results = []
        for idx, frame in enumerate(frames):
            dets = run_yolo_on_frame(
                self._yolo_model,
                frame.image_path,
                conf=infer["confidence_threshold"],
                iou_nms=infer["iou_nms"],
                img_size=infer["image_size"],
            )
            if len(dets) > max_dets:
                dets.sort(key=lambda d: d.confidence, reverse=True)
                dets = dets[:max_dets]
            results.append(
                FrameDetections(
                    frame_idx=idx,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    detections=dets,
                )
            )
        return results

    def _extract_roi_features(
        self,
        frames: list[Frame],
        frame_dets: list[FrameDetections],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Extract RoI-aligned features for all detections."""
        all_features = []
        all_frame_indices = []
        all_bbox_coords = []
        d_model = self._net.d_model

        for fd in frame_dets:
            if not fd.detections:
                continue

            # Load image
            img = cv2.imread(str(frames[fd.frame_idx].image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img = img.shape[:2]

            # Prepare image tensor
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            # Prepare bbox tensor
            bboxes = torch.tensor([[d.cx, d.cy, d.w, d.h] for d in fd.detections])

            # Extract features
            with torch.no_grad():
                feats = self._roi_extractor(img_t, bboxes, image_size=(h_img, w_img))

            all_features.append(feats)
            all_frame_indices.extend([fd.frame_idx] * len(fd.detections))
            all_bbox_coords.append(bboxes)

        if all_features:
            roi_features = torch.cat(all_features, dim=0)
            frame_indices = torch.tensor(all_frame_indices, dtype=torch.long)
            bbox_coords = torch.cat(all_bbox_coords, dim=0)
        else:
            roi_features = torch.zeros(0, d_model)
            frame_indices = torch.zeros(0, dtype=torch.long)
            bbox_coords = torch.zeros(0, 4)

        return roi_features, frame_indices, bbox_coords

    @staticmethod
    def _find_trigger_frame(
        frame_dets: list[FrameDetections],
        is_positive: bool,
    ) -> int | None:
        """Find the first frame with any detection as trigger frame."""
        if not is_positive:
            return None
        for fd in frame_dets:
            if fd.detections:
                return fd.frame_idx
        return None
