"""TemporalModel implementation for bbox-tube-temporal.

Wires the YOLO companion + tube building + patch cropping + the trained
temporal classifier into the pyrocore :class:`TemporalModel` contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from pyrocore import Frame, TemporalModel, TemporalModelOutput

from .inference import (
    crop_tube_patches,
    filter_and_interpolate_tubes,
    find_first_crossing_trigger,
    pad_frames_symmetrically,
    pad_frames_uniform,
    run_yolo_on_frames,
    score_tubes,
)
from .logistic_calibrator import LogisticCalibrator
from .package import ModelPackage, load_model_package
from .tubes import build_tubes

_PAD_STRATEGIES = {
    "symmetric": pad_frames_symmetrically,
    "uniform": pad_frames_uniform,
}


def _select_device(device: str | torch.device | None) -> torch.device:
    """Resolve the requested device, auto-picking the best available when None.

    Preference order: CUDA > MPS (Apple Silicon) > CPU.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BboxTubeTemporalModel(TemporalModel):
    """YOLO companion + tube classifier.

    See ``docs/specs/2026-04-15-temporal-model-protocol-design.md`` for the
    full pipeline description.
    """

    def __init__(
        self,
        *,
        yolo_model: Any,
        classifier: Any,
        config: dict[str, Any],
        device: str | torch.device | None = None,
        calibrator: LogisticCalibrator | None = None,
    ) -> None:
        self._yolo = yolo_model
        self._device = _select_device(device)
        self._classifier = classifier.to(self._device).eval()
        self._cfg = config
        self._calibrator = calibrator

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def from_package(
        cls,
        package_path: Path,
        *,
        device: str | torch.device | None = None,
    ) -> Self:
        pkg: ModelPackage = load_model_package(package_path)
        return cls(
            yolo_model=pkg.yolo_model,
            classifier=pkg.classifier,
            config=pkg.config,
            device=device,
            calibrator=pkg.calibrator,
        )

    @classmethod
    def from_archive(
        cls,
        archive_path: Path,
        *,
        device: str | torch.device | None = None,
    ) -> Self:
        """Alias for :meth:`from_package`.

        Convenience name used by the evaluation driver so callers can
        refer to the archive by a generic name independent of the internal
        packaging terminology.
        """
        return cls.from_package(archive_path, device=device)

    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        infer = self._cfg["infer"]
        tubes_cfg = self._cfg["tubes"]
        mi = self._cfg["model_input"]
        clf_cfg = self._cfg["classifier"]
        dec = self._cfg["decision"]

        original_len = len(frames)
        if original_len == 0:
            return TemporalModelOutput(
                is_positive=False,
                trigger_frame_index=None,
                details={
                    "num_frames": 0,
                    "num_truncated": 0,
                    "num_padded": 0,
                    "num_detections_per_frame": [],
                    "num_tubes_total": 0,
                    "num_tubes_kept": 0,
                    "tube_logits": [],
                    "winner_tube_id": None,
                    "winner_tube_entries": [],
                    "kept_tubes": [],
                    "per_tube_first_crossing": {},
                    "threshold": float(dec["threshold"]),
                },
            )

        truncated = frames[: clf_cfg["max_frames"]]
        n_truncated = original_len - len(truncated)

        pad_min = int(infer.get("pad_to_min_frames", 0))
        if pad_min > 0 and len(truncated) < pad_min:
            strategy = infer.get("pad_strategy", "symmetric")
            try:
                pad_fn = _PAD_STRATEGIES[strategy]
            except KeyError as e:
                raise ValueError(
                    f"unknown pad_strategy {strategy!r}; "
                    f"expected one of {sorted(_PAD_STRATEGIES)}"
                ) from e
            truncated = pad_fn(truncated, min_length=pad_min)
        n_padded = len(truncated) - (original_len - n_truncated)

        frame_dets = run_yolo_on_frames(
            self._yolo,
            truncated,
            confidence_threshold=infer["confidence_threshold"],
            iou_nms=infer["iou_nms"],
            image_size=infer["image_size"],
            device=self._device,
        )
        num_dets_per_frame = [len(fd.detections) for fd in frame_dets]

        candidate_tubes = build_tubes(
            frame_dets,
            iou_threshold=tubes_cfg["iou_threshold"],
            max_misses=tubes_cfg["max_misses"],
        )
        kept = filter_and_interpolate_tubes(
            candidate_tubes,
            min_tube_length=tubes_cfg["infer_min_tube_length"],
            min_detected_entries=tubes_cfg["min_detected_entries"],
            interpolate_gaps=tubes_cfg["interpolate_gaps"],
        )

        if not kept:
            return TemporalModelOutput(
                is_positive=False,
                trigger_frame_index=None,
                details={
                    "num_frames": original_len,
                    "num_truncated": n_truncated,
                    "num_padded": n_padded,
                    "num_detections_per_frame": num_dets_per_frame,
                    "num_tubes_total": len(candidate_tubes),
                    "num_tubes_kept": 0,
                    "tube_logits": [],
                    "winner_tube_id": None,
                    "winner_tube_entries": [],
                    "kept_tubes": [],
                    "per_tube_first_crossing": {},
                    "threshold": float(dec["threshold"]),
                },
            )

        patches_per_tube: list[torch.Tensor] = []
        masks_per_tube: list[torch.Tensor] = []
        for t in kept:
            p, m = crop_tube_patches(
                t,
                truncated,
                context_factor=mi["context_factor"],
                patch_size=mi["patch_size"],
                max_frames=clf_cfg["max_frames"],
                normalization_mean=mi["normalization"]["mean"],
                normalization_std=mi["normalization"]["std"],
            )
            patches_per_tube.append(p.to(self._device))
            masks_per_tube.append(m.to(self._device))

        logits = score_tubes(
            self._classifier,
            patches_per_tube=patches_per_tube,
            masks_per_tube=masks_per_tube,
        )

        aggregation = dec.get("aggregation", "max_logit")
        is_positive, trigger, winner_id, per_tube_first_crossing = (
            find_first_crossing_trigger(
                classifier=self._classifier,
                tubes=kept,
                patches_per_tube=patches_per_tube,
                masks_per_tube=masks_per_tube,
                full_logits=logits,
                aggregation=aggregation,
                threshold=float(dec["threshold"]),
                calibrator=self._calibrator,
                logistic_threshold=float(dec.get("logistic_threshold", 0.5)),
                min_prefix_length=tubes_cfg["infer_min_tube_length"],
            )
        )

        logits_list: list[float] = logits.tolist()
        kept_tubes: list[dict] = []
        for tube_idx, tube in enumerate(kept):
            entries = [
                {
                    "frame_idx": e.frame_idx,
                    "bbox": (
                        [e.detection.cx, e.detection.cy, e.detection.w, e.detection.h]
                        if e.detection is not None
                        else None
                    ),
                    "is_gap": e.is_gap,
                    "confidence": (
                        e.detection.confidence if e.detection is not None else None
                    ),
                }
                for e in tube.entries
            ]
            kept_tubes.append(
                {
                    "tube_id": tube.tube_id,
                    "start_frame": tube.start_frame,
                    "end_frame": tube.end_frame,
                    "logit": logits_list[tube_idx],
                    "is_winner": tube.tube_id == winner_id,
                    "entries": entries,
                }
            )

        winner_entries: list[dict] = (
            next(t["entries"] for t in kept_tubes if t["is_winner"])
            if winner_id is not None
            else []
        )

        return TemporalModelOutput(
            is_positive=is_positive,
            trigger_frame_index=trigger,
            details={
                "num_frames": original_len,
                "num_truncated": n_truncated,
                "num_padded": n_padded,
                "num_detections_per_frame": num_dets_per_frame,
                "num_tubes_total": len(candidate_tubes),
                "num_tubes_kept": len(kept),
                "tube_logits": logits_list,
                "winner_tube_id": winner_id,
                "winner_tube_entries": winner_entries,
                "kept_tubes": kept_tubes,
                "per_tube_first_crossing": per_tube_first_crossing,
                "threshold": (
                    float(dec["logistic_threshold"])
                    if aggregation == "logistic"
                    else float(dec["threshold"])
                ),
                "aggregation": aggregation,
            },
        )
