"""TemporalModel implementation for tube-multiscale-fusion.

Wires the YOLO companion + tube building + patch cropping + the trained
two-branch classifier into the pyrocore :class:`TemporalModel` contract, so
the model is comparable on the shared leaderboard alongside the other
temporal models.

The detect -> build_tubes -> filter/interpolate -> crop -> score -> trigger
pipeline is reused verbatim from ``bbox_tube_temporal`` (the shared
``bbox-tube-temporal-core`` library); only the classifier differs: here it is
the two-branch :class:`LitTubeMultiscaleClassifier` instead of the
single-branch ``TemporalSmokeClassifier``. Both expose the same callable
contract ``(patches[N, T, 3, H, W], mask[N, T]) -> logits[N]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from bbox_tube_temporal.inference import (
    crop_tube_patches,
    filter_and_interpolate_tubes,
    find_first_crossing_trigger,
    pad_frames_symmetrically,
    pad_frames_uniform,
    run_yolo_on_frames,
    score_tubes,
)
from bbox_tube_temporal.tubes import build_tubes
from pyrocore import Frame, TemporalModel, TemporalModelOutput

from .package import ModelPackage, load_model_package

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


class TubeMultiscaleFusionModel(TemporalModel):
    """YOLO companion + two-branch tube/global classifier.

    Mirrors :class:`bbox_tube_temporal.model.BboxTubeTemporalModel`; see that
    module and ``docs/specs/2026-04-15-temporal-model-protocol-design.md`` for
    the full pipeline description.
    """

    def __init__(
        self,
        *,
        yolo_model: Any,
        classifier: Any,
        config: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> None:
        self._yolo = yolo_model
        self._device = _select_device(device)
        self._classifier = classifier.to(self._device).eval()
        self._cfg = config

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
        )

    @classmethod
    def from_archive(
        cls,
        archive_path: Path,
        *,
        device: str | torch.device | None = None,
    ) -> Self:
        """Alias for :meth:`from_package` (generic name used by eval drivers)."""
        return cls.from_package(archive_path, device=device)

    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        infer = self._cfg["infer"]
        tubes_cfg = self._cfg["tubes"]
        mi = self._cfg["model_input"]
        clf_cfg = self._cfg["classifier"]
        dec = self._cfg["decision"]

        threshold = float(dec["threshold"])
        max_frames = int(clf_cfg["max_frames"])

        def _make_details(
            *,
            num_frames_input: int,
            num_truncated: int,
            padded_indices: list[int],
            num_candidates: int,
            kept_tubes: list[dict],
            trigger_tube_id: int | None,
        ) -> dict:
            return {
                "preprocessing": {
                    "num_frames_input": num_frames_input,
                    "num_truncated": num_truncated,
                    "padded_frame_indices": padded_indices,
                },
                "tubes": {
                    "num_candidates": num_candidates,
                    "kept": kept_tubes,
                },
                "decision": {
                    "aggregation": "max_logit",
                    "threshold": threshold,
                    "trigger_tube_id": trigger_tube_id,
                },
            }

        original_len = len(frames)
        if original_len == 0:
            return TemporalModelOutput(
                is_positive=False,
                trigger_frame_index=None,
                details=_make_details(
                    num_frames_input=0,
                    num_truncated=0,
                    padded_indices=[],
                    num_candidates=0,
                    kept_tubes=[],
                    trigger_tube_id=None,
                ),
            )

        truncated = frames[:max_frames]
        n_truncated = original_len - len(truncated)

        padded_indices: list[int] = []
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
            truncated, padded_indices = pad_fn(truncated, min_length=pad_min)

        frame_dets = run_yolo_on_frames(
            self._yolo,
            truncated,
            confidence_threshold=infer["confidence_threshold"],
            iou_nms=infer["iou_nms"],
            image_size=infer["image_size"],
            device=self._device,
        )

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
                details=_make_details(
                    num_frames_input=original_len,
                    num_truncated=n_truncated,
                    padded_indices=padded_indices,
                    num_candidates=len(candidate_tubes),
                    kept_tubes=[],
                    trigger_tube_id=None,
                ),
            )

        patches_per_tube: list[torch.Tensor] = []
        masks_per_tube: list[torch.Tensor] = []
        for t in kept:
            p, m = crop_tube_patches(
                t,
                truncated,
                context_factor=mi["context_factor"],
                patch_size=mi["patch_size"],
                max_frames=max_frames,
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

        is_positive, trigger, trigger_tube_id, per_tube_first_crossing = (
            find_first_crossing_trigger(
                classifier=self._classifier,
                tubes=kept,
                patches_per_tube=patches_per_tube,
                masks_per_tube=masks_per_tube,
                full_logits=logits,
                aggregation="max_logit",
                threshold=threshold,
                min_prefix_length=tubes_cfg["infer_min_tube_length"],
            )
        )

        logits_list: list[float] = logits.tolist()
        kept_tubes: list[dict] = []
        for tube_idx, tube in enumerate(kept):
            first_crossing = per_tube_first_crossing.get(tube.tube_id, {}).get(
                "crossing_frame"
            )
            kept_tubes.append(
                {
                    "tube_id": tube.tube_id,
                    "start_frame": tube.start_frame,
                    "end_frame": tube.end_frame,
                    "logit": logits_list[tube_idx],
                    "first_crossing_frame": first_crossing,
                }
            )

        return TemporalModelOutput(
            is_positive=is_positive,
            trigger_frame_index=trigger,
            details=_make_details(
                num_frames_input=original_len,
                num_truncated=n_truncated,
                padded_indices=padded_indices,
                num_candidates=len(candidate_tubes),
                kept_tubes=kept_tubes,
                trigger_tube_id=trigger_tube_id,
            ),
        )
