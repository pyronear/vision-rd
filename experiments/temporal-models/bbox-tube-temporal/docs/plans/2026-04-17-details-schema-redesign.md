# Details schema redesign — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat `details` dict returned by `BboxTubeTemporalModel.predict` with a slim, Pydantic-validated `BboxTubeDetails` schema organised into `preprocessing` / `tubes` / `decision` sections, and update every in-repo consumer (scripts, tests, notebooks) to the new shape in one breaking change.

**Architecture:** Introduce a local Pydantic module (`details_schema.py`) in `bbox_tube_temporal`. `predict()` constructs a `BboxTubeDetails`, validates on construction, then emits `details=model.model_dump()` so `pyrocore.TemporalModelOutput.details: dict` is untouched. Consumers that want types call `BboxTubeDetails.model_validate(output.details)`. Padding helpers are extended to also return the padded indices so they can be exposed in the schema.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, ruff. All commands run from `experiments/temporal-models/bbox-tube-temporal/`.

**Spec:** `docs/specs/2026-04-17-details-schema-redesign.md`.

**Commit style:** no Claude / Anthropic co-author trailers. Stage files explicitly, never `git add -A`.

---

## File Structure

- **Create** `src/bbox_tube_temporal/details_schema.py` — Pydantic models (`TubeEntry`, `KeptTube`, `Preprocessing`, `Tubes`, `Decision`, `BboxTubeDetails`). Single responsibility: schema definition.
- **Create** `tests/test_details_schema.py` — schema-only tests (valid round-trips, empty/no-tubes, rejection of malformed input).
- **Modify** `src/bbox_tube_temporal/inference.py` — `pad_frames_symmetrically` and `pad_frames_uniform` return `(padded_frames, padded_indices)`.
- **Modify** `src/bbox_tube_temporal/model.py` — build + validate + dump `BboxTubeDetails` in both early-return branches and the main path; thread through padded indices; remove `num_detections_per_frame` computation.
- **Modify** `src/bbox_tube_temporal/protocol_eval.py` — read `tubes.kept[*].logit` and `len(tubes.kept)` instead of `tube_logits` / `num_tubes_kept`.
- **Modify** `src/bbox_tube_temporal/aggregation_analysis.py` — read logits from `tubes.kept[*].logit` (in-memory path). Persisted-JSON path still uses a top-level `tube_logits` field in `predictions.json`, which the evaluate_packaged serialiser derives from `tubes.kept`.
- **Modify** `src/bbox_tube_temporal/logistic_calibrator_fit.py` — record contract becomes `tubes.kept` instead of `kept_tubes`.
- **Modify** `scripts/evaluate_packaged.py` — `_record_to_json` reads nested paths, derives `tube_logits` from `tubes.kept` for the persisted JSON.
- **Modify** `scripts/analyze_variant.py` — read `kept` from `tubes.kept`.
- **Modify** `notebooks/04-error-analysis.ipynb` — rename `winner_tube_id` reads to `decision.trigger_tube_id`.
- **Modify** `tests/test_model_edge_cases.py`, `tests/test_model_parity.py`, `tests/test_protocol_eval.py`, `tests/test_aggregation_analysis.py`, `tests/test_evaluate_packaged_driver.py`, `tests/test_package_predict.py`, `tests/test_logistic_calibrator_fit.py` — update assertions and fixture details dicts.

---

## Task 1: Add Pydantic dependency and scaffold the schema module (TDD)

**Files:**
- Modify: `pyproject.toml`
- Create: `src/bbox_tube_temporal/details_schema.py`
- Create: `tests/test_details_schema.py`

- [ ] **Step 1.1: Add Pydantic to project dependencies**

Open `pyproject.toml` and add `"pydantic>=2.6"` to the `dependencies` list (alphabetical order within the block keeps it near `pyyaml`):

```toml
dependencies = [
    "pyrocore",
    "lightning>=2.2",
    "matplotlib>=3.8",
    "numpy>=1.26,<2",
    "opencv-python-headless>=4.8",
    "pandas>=2.0",
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "scikit-learn>=1.4",
    "tensorboard>=2.16",
    "timm>=1.0",
    "torch>=2.2",
    "torchvision>=0.17",
    "tqdm>=4.66",
    "ultralytics>=8.3",
    "huggingface-hub>=0.24",
]
```

- [ ] **Step 1.2: Sync environment**

Run: `make install`
Expected: `uv sync` completes; `pydantic` appears in the resolved lockfile.

- [ ] **Step 1.3: Write a failing test for the happy-path schema round-trip**

Create `tests/test_details_schema.py`:

```python
"""Tests for BboxTubeDetails and sub-models."""

import pytest
from pydantic import ValidationError

from bbox_tube_temporal.details_schema import (
    BboxTubeDetails,
    Decision,
    KeptTube,
    Preprocessing,
    TubeEntry,
    Tubes,
)


def _sample_details() -> BboxTubeDetails:
    return BboxTubeDetails(
        preprocessing=Preprocessing(
            num_frames_input=6,
            num_truncated=0,
            padded_frame_indices=[],
        ),
        tubes=Tubes(
            num_candidates=1,
            kept=[
                KeptTube(
                    tube_id=0,
                    start_frame=0,
                    end_frame=5,
                    logit=1.25,
                    probability=None,
                    first_crossing_frame=3,
                    entries=[
                        TubeEntry(
                            frame_idx=0,
                            bbox=(0.5, 0.5, 0.1, 0.1),
                            is_gap=False,
                            confidence=0.9,
                        ),
                        TubeEntry(
                            frame_idx=1,
                            bbox=None,
                            is_gap=True,
                            confidence=None,
                        ),
                    ],
                )
            ],
        ),
        decision=Decision(
            aggregation="max_logit",
            threshold=0.0,
            trigger_tube_id=0,
        ),
    )


def test_round_trip_via_model_dump_and_validate() -> None:
    original = _sample_details()
    dumped = original.model_dump()
    parsed = BboxTubeDetails.model_validate(dumped)
    assert parsed == original


def test_decision_rejects_unknown_aggregation() -> None:
    with pytest.raises(ValidationError):
        Decision(aggregation="bogus", threshold=0.0, trigger_tube_id=None)  # type: ignore[arg-type]


def test_tube_entry_rejects_wrong_length_bbox() -> None:
    with pytest.raises(ValidationError):
        TubeEntry(
            frame_idx=0,
            bbox=(0.1, 0.2, 0.3),  # type: ignore[arg-type]
            is_gap=False,
            confidence=0.5,
        )


def test_empty_sequence_shape_validates() -> None:
    details = BboxTubeDetails(
        preprocessing=Preprocessing(
            num_frames_input=0, num_truncated=0, padded_frame_indices=[]
        ),
        tubes=Tubes(num_candidates=0, kept=[]),
        decision=Decision(
            aggregation="max_logit", threshold=0.0, trigger_tube_id=None
        ),
    )
    assert details.model_dump()["tubes"]["kept"] == []


def test_no_tubes_kept_shape_validates() -> None:
    details = BboxTubeDetails(
        preprocessing=Preprocessing(
            num_frames_input=5, num_truncated=0, padded_frame_indices=[]
        ),
        tubes=Tubes(num_candidates=3, kept=[]),
        decision=Decision(
            aggregation="max_logit", threshold=0.0, trigger_tube_id=None
        ),
    )
    assert details.tubes.num_candidates == 3
    assert details.decision.trigger_tube_id is None
```

- [ ] **Step 1.4: Run the test to verify it fails**

Run: `uv run pytest tests/test_details_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bbox_tube_temporal.details_schema'`.

- [ ] **Step 1.5: Implement the schema module**

Create `src/bbox_tube_temporal/details_schema.py`:

```python
"""Pydantic schema for ``BboxTubeTemporalModel.predict()`` ``details``.

Emitted as a plain ``dict`` by ``model.predict()`` (via ``model_dump()``) to
keep the ``pyrocore.TemporalModelOutput.details: dict`` contract intact.
Consumers that want typed access parse with
``BboxTubeDetails.model_validate(output.details)``.

See ``docs/specs/2026-04-17-details-schema-redesign.md``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TubeEntry(_Frozen):
    frame_idx: int
    bbox: tuple[float, float, float, float] | None
    is_gap: bool
    confidence: float | None


class KeptTube(_Frozen):
    tube_id: int
    start_frame: int
    end_frame: int
    logit: float
    probability: float | None
    first_crossing_frame: int | None
    entries: list[TubeEntry]


class Preprocessing(_Frozen):
    num_frames_input: int
    num_truncated: int
    padded_frame_indices: list[int]


class Tubes(_Frozen):
    num_candidates: int
    kept: list[KeptTube]


class Decision(_Frozen):
    aggregation: Literal["max_logit", "logistic"]
    threshold: float
    trigger_tube_id: int | None


class BboxTubeDetails(_Frozen):
    preprocessing: Preprocessing
    tubes: Tubes
    decision: Decision
```

- [ ] **Step 1.6: Run the test to verify it passes**

Run: `uv run pytest tests/test_details_schema.py -v`
Expected: PASS (5 passed).

- [ ] **Step 1.7: Lint**

Run: `make lint`
Expected: no errors.

- [ ] **Step 1.8: Commit**

```bash
git add pyproject.toml uv.lock src/bbox_tube_temporal/details_schema.py tests/test_details_schema.py
git commit -m "feat(bbox-tube-temporal): add BboxTubeDetails Pydantic schema"
```

---

## Task 2: Extend padding helpers to report padded indices (TDD)

**Files:**
- Modify: `src/bbox_tube_temporal/inference.py:22-68` (`pad_frames_symmetrically`, `pad_frames_uniform`)
- Modify: `tests/test_inference.py` (if it exists — see step 2.1; otherwise `tests/test_padding.py` is created)

- [ ] **Step 2.1: Locate or create the padding test file**

Run: `ls tests/` and open whichever of `tests/test_inference.py` or `tests/test_padding.py` already tests the pad helpers. If neither exists, create `tests/test_padding.py` with the tests below. If a file already exercises these helpers, append the new tests to it.

- [ ] **Step 2.2: Write failing tests for the new return signature**

Add to the chosen test file:

```python
from pyrocore.types import Frame

from bbox_tube_temporal.inference import (
    pad_frames_symmetrically,
    pad_frames_uniform,
)


def _mk_frames(n: int) -> list[Frame]:
    from pathlib import Path
    return [Frame(frame_id=f"f{i}", image_path=Path(f"/tmp/f{i}.jpg")) for i in range(n)]


def test_pad_symmetrically_returns_padded_indices() -> None:
    frames = _mk_frames(2)  # A, B -> [A, A, B] -> [A, A, B, B] -> [A, A, A, B, B]
    padded, indices = pad_frames_symmetrically(frames, min_length=5)
    assert len(padded) == 5
    # After three pad steps (prepend, append, prepend), real frames sit at slots 2 and 3.
    assert indices == [0, 1, 4]


def test_pad_symmetrically_noop_returns_empty_indices() -> None:
    frames = _mk_frames(6)
    padded, indices = pad_frames_symmetrically(frames, min_length=3)
    assert padded == frames
    assert indices == []


def test_pad_symmetrically_empty_input_returns_empty_indices() -> None:
    padded, indices = pad_frames_symmetrically([], min_length=3)
    assert padded == []
    assert indices == []


def test_pad_uniform_returns_padded_indices() -> None:
    frames = _mk_frames(2)
    padded, indices = pad_frames_uniform(frames, min_length=6)
    # Source mapping i*2//6 for i in 0..5 = [0,0,0,1,1,1]
    # Real frames are the first occurrences (slots 0 and 3); all others are padded.
    assert len(padded) == 6
    assert indices == [1, 2, 4, 5]


def test_pad_uniform_noop_returns_empty_indices() -> None:
    frames = _mk_frames(6)
    padded, indices = pad_frames_uniform(frames, min_length=3)
    assert padded == frames
    assert indices == []
```

- [ ] **Step 2.3: Run to verify fail**

Run: `uv run pytest tests/test_padding.py -v` (or the existing test file you chose).
Expected: FAIL — `TypeError` or `ValueError: too many values to unpack`, because helpers still return a plain list.

- [ ] **Step 2.4: Implement the new return signature**

In `src/bbox_tube_temporal/inference.py`, replace the two functions:

```python
def pad_frames_symmetrically(
    frames: list[Frame],
    *,
    min_length: int,
) -> tuple[list[Frame], list[int]]:
    """Pad ``frames`` up to ``min_length`` by alternating prepend/append.

    Returns ``(padded_frames, padded_indices)`` where ``padded_indices`` lists
    the slots in ``padded_frames`` that are synthesised duplicates. Real
    frames occupy every other slot. Empty inputs and inputs already at or
    above ``min_length`` pass through with an empty index list.
    """
    if not frames or len(frames) >= min_length:
        return list(frames), []
    result = list(frames)
    # Positions of real frames in the growing result; padded indices are the complement.
    real_positions = list(range(len(frames)))
    prepend = True
    while len(result) < min_length:
        src = frames[0] if prepend else frames[-1]
        if prepend:
            result.insert(0, src)
            real_positions = [p + 1 for p in real_positions]
        else:
            result.append(src)
        prepend = not prepend
    real_set = set(real_positions)
    padded_indices = [i for i in range(len(result)) if i not in real_set]
    return result, padded_indices


def pad_frames_uniform(
    frames: list[Frame],
    *,
    min_length: int,
) -> tuple[list[Frame], list[int]]:
    """Pad ``frames`` up to ``min_length`` by uniform nearest-neighbor upsampling.

    Returns ``(padded_frames, padded_indices)`` where ``padded_indices`` lists
    the slots whose source frame also appears earlier in the output (i.e.
    duplicates introduced by the upsample). The first occurrence of each
    real frame is treated as the "real" slot. Empty inputs and inputs
    already at or above ``min_length`` pass through with an empty index
    list.
    """
    if not frames or len(frames) >= min_length:
        return list(frames), []
    n = len(frames)
    source = [i * n // min_length for i in range(min_length)]
    padded_frames = [frames[i] for i in source]
    seen: set[int] = set()
    padded_indices: list[int] = []
    for slot, src_idx in enumerate(source):
        if src_idx in seen:
            padded_indices.append(slot)
        else:
            seen.add(src_idx)
    return padded_frames, padded_indices
```

- [ ] **Step 2.5: Run tests to verify pass**

Run: `uv run pytest tests/test_padding.py -v`
Expected: PASS.

- [ ] **Step 2.6: Also run the model edge-case tests to see which break**

Run: `uv run pytest tests/test_model_edge_cases.py -v`
Expected: FAIL — `model.py` still unpacks the padder as a single list. That's fixed in Task 3. Leave the failures for now.

- [ ] **Step 2.7: Commit**

```bash
git add src/bbox_tube_temporal/inference.py tests/test_padding.py
git commit -m "feat(bbox-tube-temporal): padding helpers return padded indices"
```

---

## Task 3: Switch `model.predict` to the new schema (TDD)

**Files:**
- Modify: `src/bbox_tube_temporal/model.py` (entire `predict()` body and both early-return branches)
- Modify: `tests/test_model_edge_cases.py` (update all `details[...]` assertions)

- [ ] **Step 3.1: Write the new expected shape into one edge-case test first**

Edit `tests/test_model_edge_cases.py::TestEmptyFrames::test_returns_negative`:

```python
class TestEmptyFrames:
    def test_returns_negative(self, tiny_classifier: TemporalSmokeClassifier) -> None:
        yolo = MagicMock()
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=[])
        assert out.is_positive is False
        assert out.trigger_frame_index is None
        assert out.details["preprocessing"]["num_frames_input"] == 0
        assert out.details["tubes"]["num_candidates"] == 0
        assert out.details["tubes"]["kept"] == []
        assert out.details["decision"]["trigger_tube_id"] is None
        yolo.predict.assert_not_called()
```

- [ ] **Step 3.2: Run to verify fail**

Run: `uv run pytest tests/test_model_edge_cases.py::TestEmptyFrames -v`
Expected: FAIL — `KeyError: 'preprocessing'`.

- [ ] **Step 3.3: Rewrite `predict()` to build the new schema**

Replace the body of `src/bbox_tube_temporal/model.py:105-285` (`predict` method) with:

```python
    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        from .details_schema import (
            BboxTubeDetails,
            Decision,
            KeptTube,
            Preprocessing,
            TubeEntry,
            Tubes,
        )

        infer = self._cfg["infer"]
        tubes_cfg = self._cfg["tubes"]
        mi = self._cfg["model_input"]
        clf_cfg = self._cfg["classifier"]
        dec = self._cfg["decision"]

        aggregation = dec.get("aggregation", "max_logit")
        effective_threshold = (
            float(dec["logistic_threshold"])
            if aggregation == "logistic"
            else float(dec["threshold"])
        )

        def _make_details(
            *,
            num_frames_input: int,
            num_truncated: int,
            padded_indices: list[int],
            num_candidates: int,
            kept_tubes_models: list[KeptTube],
            trigger_tube_id: int | None,
        ) -> dict:
            return BboxTubeDetails(
                preprocessing=Preprocessing(
                    num_frames_input=num_frames_input,
                    num_truncated=num_truncated,
                    padded_frame_indices=padded_indices,
                ),
                tubes=Tubes(
                    num_candidates=num_candidates,
                    kept=kept_tubes_models,
                ),
                decision=Decision(
                    aggregation=aggregation,
                    threshold=effective_threshold,
                    trigger_tube_id=trigger_tube_id,
                ),
            ).model_dump()

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
                    kept_tubes_models=[],
                    trigger_tube_id=None,
                ),
            )

        truncated = frames[: clf_cfg["max_frames"]]
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
                    kept_tubes_models=[],
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

        is_positive, trigger, trigger_tube_id, per_tube_first_crossing = (
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

        def _probability_for(tube_idx: int, raw_logit: float) -> float | None:
            if self._calibrator is None:
                return None
            tube = kept[tube_idx]
            tube_dict = {
                "logit": raw_logit,
                "start_frame": tube.start_frame,
                "end_frame": tube.end_frame,
                "entries": [
                    {
                        "confidence": (
                            e.detection.confidence if e.detection is not None else None
                        )
                    }
                    for e in tube.entries
                ],
            }
            from .logistic_calibrator import extract_features

            features = extract_features(tube_dict, n_tubes=len(kept))
            return float(self._calibrator.predict_proba(features))

        kept_models: list[KeptTube] = []
        for tube_idx, tube in enumerate(kept):
            entries_models = [
                TubeEntry(
                    frame_idx=e.frame_idx,
                    bbox=(
                        (e.detection.cx, e.detection.cy, e.detection.w, e.detection.h)
                        if e.detection is not None
                        else None
                    ),
                    is_gap=e.is_gap,
                    confidence=(
                        e.detection.confidence if e.detection is not None else None
                    ),
                )
                for e in tube.entries
            ]
            first_crossing = per_tube_first_crossing.get(tube.tube_id, {}).get(
                "crossing_frame"
            )
            kept_models.append(
                KeptTube(
                    tube_id=tube.tube_id,
                    start_frame=tube.start_frame,
                    end_frame=tube.end_frame,
                    logit=logits_list[tube_idx],
                    probability=_probability_for(tube_idx, logits_list[tube_idx]),
                    first_crossing_frame=first_crossing,
                    entries=entries_models,
                )
            )

        return TemporalModelOutput(
            is_positive=is_positive,
            trigger_frame_index=trigger,
            details=_make_details(
                num_frames_input=original_len,
                num_truncated=n_truncated,
                padded_indices=padded_indices,
                num_candidates=len(candidate_tubes),
                kept_tubes_models=kept_models,
                trigger_tube_id=trigger_tube_id,
            ),
        )
```

Note: the `from .details_schema import ...` sits inside `predict()` above for clarity of the diff. **After making the edit, move it to module top** so the file's imports comply with the repo rule of no mid-file imports. Do the same for the `extract_features` import — move it to the module's existing import block.

Run: `uv run ruff check src/bbox_tube_temporal/model.py`
Expected: PASS (if the imports are at module top).

- [ ] **Step 3.4: Run the edge-case test to verify pass**

Run: `uv run pytest tests/test_model_edge_cases.py::TestEmptyFrames -v`
Expected: PASS.

- [ ] **Step 3.5: Update the rest of `tests/test_model_edge_cases.py`**

Replace every `out.details[...]` access with the new paths. Concretely:

```python
# TestZeroDetections.test_no_tubes_means_negative
assert out.details["tubes"]["num_candidates"] == 0
assert out.details["tubes"]["kept"] == []

# TestShortTubeBelowInferFloor.test_single_frame_detection_discarded
assert out.details["tubes"]["num_candidates"] == 1
assert out.details["tubes"]["kept"] == []

# TestTruncation.test_sequence_longer_than_max_frames
assert out.details["preprocessing"]["num_frames_input"] == 9
assert out.details["preprocessing"]["num_truncated"] == 3

# TestShortSequencePadding.test_pad_disabled_by_default
assert out.details["preprocessing"]["padded_frame_indices"] == []
assert out.details["preprocessing"]["num_frames_input"] == 2

# TestShortSequencePadding.test_pad_extends_short_sequence_symmetrically
# Symmetric pad of [A, B] to length 5: real frames end up at slots 2 and 3
# (see test_padding.py). Padded indices are therefore [0, 1, 4].
assert out.details["preprocessing"]["padded_frame_indices"] == [0, 1, 4]
assert out.details["preprocessing"]["num_frames_input"] == 2

# TestShortSequencePadding.test_pad_noop_when_sequence_already_long_enough
assert out.details["preprocessing"]["padded_frame_indices"] == []

# TestShortSequencePadding.test_pad_strategy_uniform_spreads_duplicates
# Uniform i*2//6 for i in 0..5 -> [0,0,0,1,1,1]; duplicates are slots 1,2,4,5.
assert out.details["preprocessing"]["padded_frame_indices"] == [1, 2, 4, 5]
assert out.details["preprocessing"]["num_frames_input"] == 2

# TestDeviceSelection.test_predict_on_cpu_runs_end_to_end
assert len(out.details["tubes"]["kept"]) == 1
assert len(out.details["tubes"]["kept"][0]["entries"]) >= 1

# TestDeviceSelection.test_predict_details_include_per_tube_entries
kept = out.details["tubes"]["kept"]
assert isinstance(kept, list)
assert len(kept) == 1
tube = kept[0]
assert set(tube.keys()) == {
    "tube_id",
    "start_frame",
    "end_frame",
    "logit",
    "probability",
    "first_crossing_frame",
    "entries",
}
assert tube["logit"] == kept[0]["logit"]  # sanity only; tube_logits field dropped
entry = tube["entries"][0]
assert set(entry.keys()) == {"frame_idx", "bbox", "is_gap", "confidence"}
assert out.details["decision"]["trigger_tube_id"] == tube["tube_id"]

# TestDeviceSelection.test_predict_on_cuda_runs_end_to_end
assert len(out.details["tubes"]["kept"]) == 1

# TestFirstCrossingTrigger.test_first_crossing_trigger_never_exceeds_end_frame
trigger_tube_id = out.details["decision"]["trigger_tube_id"]
winner_tube = next(
    t for t in out.details["tubes"]["kept"] if t["tube_id"] == trigger_tube_id
)
assert out.trigger_frame_index <= winner_tube["end_frame"]
assert winner_tube["first_crossing_frame"] == out.trigger_frame_index
```

Drop the `assertions about "is_winner"` from that last test (there is no longer an `is_winner` field) and the dict lookup into `per_tube_first_crossing` (it's now per-tube).

- [ ] **Step 3.6: Run full edge-case suite**

Run: `uv run pytest tests/test_model_edge_cases.py -v`
Expected: PASS (all tests).

- [ ] **Step 3.7: Run model parity tests**

Run: `uv run pytest tests/test_model_parity.py -v`
Expected: FAIL — parity tests still read `out.details["num_tubes_kept"]` and `out.details["tube_logits"]`.

- [ ] **Step 3.8: Update parity tests**

In `tests/test_model_parity.py`, replace both occurrences of:

```python
assert out.details["num_tubes_kept"] >= 1
online = max(out.details["tube_logits"])
```

with:

```python
kept = out.details["tubes"]["kept"]
assert len(kept) >= 1
online = max(t["logit"] for t in kept)
```

- [ ] **Step 3.9: Re-run parity tests**

Run: `uv run pytest tests/test_model_parity.py -v`
Expected: PASS.

- [ ] **Step 3.10: Commit**

```bash
git add src/bbox_tube_temporal/model.py tests/test_model_edge_cases.py tests/test_model_parity.py
git commit -m "feat(bbox-tube-temporal): emit BboxTubeDetails from predict()"
```

---

## Task 4: Update `protocol_eval.build_record`

**Files:**
- Modify: `src/bbox_tube_temporal/protocol_eval.py:83-108` (`build_record`)
- Modify: `tests/test_protocol_eval.py`

- [ ] **Step 4.1: Update tests first**

In `tests/test_protocol_eval.py`, change every fixture-built `details` dict so it carries the new nested shape. Replace the three relevant blocks:

```python
# test_build_record_extracts_score_and_tube_logits_from_details
details = {
    "preprocessing": {
        "num_frames_input": 6,
        "num_truncated": 0,
        "padded_frame_indices": [],
    },
    "tubes": {
        "num_candidates": 3,
        "kept": [
            {
                "tube_id": i,
                "start_frame": 0,
                "end_frame": 5,
                "logit": logit,
                "probability": None,
                "first_crossing_frame": None,
                "entries": [],
            }
            for i, logit in enumerate([1.5, 0.3, -0.2])
        ],
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.0,
        "trigger_tube_id": 0,
    },
}
# Replaces: details={"tube_logits": [1.5, 0.3, -0.2], "num_tubes_kept": 3}

# test_build_record_empty_tube_logits_yields_minus_inf_score
# Same shape with tubes.kept=[] and tubes.num_candidates=0, trigger_tube_id=None.

# test_build_record_snapshot (... "extra": "foo" test): keep `extra` alongside
# the nested sections to confirm passthrough still works:
details = {
    "preprocessing": {"num_frames_input": 1, "num_truncated": 0, "padded_frame_indices": []},
    "tubes": {
        "num_candidates": 1,
        "kept": [
            {
                "tube_id": 0,
                "start_frame": 0,
                "end_frame": 0,
                "logit": 0.9,
                "probability": None,
                "first_crossing_frame": None,
                "entries": [],
            }
        ],
    },
    "decision": {"aggregation": "max_logit", "threshold": 0.0, "trigger_tube_id": 0},
    "extra": "foo",
}
```

- [ ] **Step 4.2: Run to verify fail**

Run: `uv run pytest tests/test_protocol_eval.py -v`
Expected: FAIL — `build_record` still reads `tube_logits` and `num_tubes_kept` at top level.

- [ ] **Step 4.3: Update `build_record`**

In `src/bbox_tube_temporal/protocol_eval.py`, replace `build_record` body:

```python
def build_record(
    *,
    sequence_dir: Path,
    label: str,
    frames: list[Frame],
    output: TemporalModelOutput,
) -> SequenceRecord:
    """Bundle a per-sequence eval record from the model's output + frames."""
    kept = output.details.get("tubes", {}).get("kept", [])
    tube_logits = [float(t["logit"]) for t in kept]
    ttd_seconds = _compute_ttd_seconds(
        ground_truth=(label == "smoke"),
        predicted=output.is_positive,
        trigger_frame_index=output.trigger_frame_index,
        frames=frames,
    )
    return SequenceRecord(
        sequence_id=sequence_dir.name,
        label=label,
        is_positive=output.is_positive,
        trigger_frame_index=output.trigger_frame_index,
        score=_score_from_tube_logits(tube_logits),
        num_tubes_kept=len(kept),
        tube_logits=tube_logits,
        ttd_seconds=ttd_seconds,
        details=dict(output.details),
    )
```

- [ ] **Step 4.4: Run to verify pass**

Run: `uv run pytest tests/test_protocol_eval.py -v`
Expected: PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/bbox_tube_temporal/protocol_eval.py tests/test_protocol_eval.py
git commit -m "feat(bbox-tube-temporal): protocol_eval reads tubes.kept logits"
```

---

## Task 5: Update `scripts/evaluate_packaged.py` and its driver test

**Files:**
- Modify: `scripts/evaluate_packaged.py:51-73` (`_record_to_json`)
- Modify: `tests/test_evaluate_packaged_driver.py`

- [ ] **Step 5.1: Update the driver test's fixture details**

In `tests/test_evaluate_packaged_driver.py`, around lines 50-70 where the fixture fake model builds its details dict, replace with the new nested shape (and drop `is_winner`, `num_tubes_total`, `winner_tube_id`, `tube_logits`, `num_tubes_kept` as top-level keys — they come from the new structure):

```python
details = {
    "preprocessing": {
        "num_frames_input": n_frames,
        "num_truncated": 0,
        "padded_frame_indices": [],
    },
    "tubes": {
        "num_candidates": 1 if is_pos else 0,
        "kept": (
            [
                {
                    "tube_id": 0,
                    "start_frame": 0,
                    "end_frame": n_frames - 1,
                    "logit": 2.5,
                    "probability": None,
                    "first_crossing_frame": 2,
                    "entries": [],
                }
            ]
            if is_pos
            else []
        ),
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.0,
        "trigger_tube_id": 0 if is_pos else None,
    },
}
```

And update the downstream assertions (lines 141-147). The persisted `predictions.json` now looks like:

```python
assert a_positive["num_tubes_total"] == 1
assert a_positive["trigger_tube_id"] == 0
assert isinstance(a_positive["kept_tubes"], list)
assert len(a_positive["kept_tubes"]) == a_positive["num_tubes_kept"]
assert a_positive["kept_tubes"][0]["logit"] == 2.5
```

Note: the persisted JSON preserves a flat-ish shape (top-level `kept_tubes`, `trigger_tube_id`, `num_tubes_total`, `num_tubes_kept`, `tube_logits`, `threshold`) for downstream tools (analyze_variant, logistic calibrator fit) — but the fields are now derived from the nested `details` by the serialiser. That is why `kept_tubes[*].is_winner` is absent from the persisted records.

- [ ] **Step 5.2: Run to verify fail**

Run: `uv run pytest tests/test_evaluate_packaged_driver.py -v`
Expected: FAIL.

- [ ] **Step 5.3: Rewrite `_record_to_json`**

In `scripts/evaluate_packaged.py`, replace `_record_to_json`:

```python
def _record_to_json(rec: SequenceRecord) -> dict:
    """Serialise a record for predictions.json.

    Flattens the nested ``details`` into the legacy predictions.json shape
    consumed by ``scripts/analyze_variant.py`` and the logistic-calibrator
    fitter: top-level ``kept_tubes``, ``trigger_tube_id``, ``tube_logits``,
    ``num_tubes_total``, ``num_tubes_kept``, and ``threshold``.
    """
    details = rec.details
    tubes = details.get("tubes", {})
    decision = details.get("decision", {})
    kept = tubes.get("kept", [])
    return {
        "sequence_id": rec.sequence_id,
        "label": rec.label,
        "is_positive": rec.is_positive,
        "trigger_frame_index": rec.trigger_frame_index,
        "score": rec.score if rec.score != float("-inf") else None,
        "num_tubes_kept": len(kept),
        "num_tubes_total": int(tubes.get("num_candidates", 0)),
        "tube_logits": [float(t["logit"]) for t in kept],
        "trigger_tube_id": decision.get("trigger_tube_id"),
        "threshold": (
            float(decision["threshold"]) if "threshold" in decision else None
        ),
        "kept_tubes": kept,
        "ttd_seconds": rec.ttd_seconds,
    }
```

- [ ] **Step 5.4: Run to verify pass**

Run: `uv run pytest tests/test_evaluate_packaged_driver.py -v`
Expected: PASS.

- [ ] **Step 5.5: Commit**

```bash
git add scripts/evaluate_packaged.py tests/test_evaluate_packaged_driver.py
git commit -m "feat(bbox-tube-temporal): evaluate_packaged serialises from nested details"
```

---

## Task 6: Update `logistic_calibrator_fit` and its tests

**Files:**
- Modify: `src/bbox_tube_temporal/logistic_calibrator_fit.py:24-29` (`_features_for_record`)
- Modify: `tests/test_logistic_calibrator_fit.py`

- [ ] **Step 6.1: Inspect the current contract**

The fitter reads `record["kept_tubes"]` from predictions.json. After Task 5, predictions.json still carries top-level `kept_tubes`, so the fitter's input shape is unchanged — **no change needed here**. Verify:

Run: `uv run pytest tests/test_logistic_calibrator_fit.py -v`
Expected: PASS without modifications.

- [ ] **Step 6.2: If anything fails, fix the fixture shape**

If any test builds a record dict that mentions `is_winner`, remove it — the new predictions.json does not carry `is_winner` on kept tubes. Otherwise no edit.

- [ ] **Step 6.3: Commit (only if edits were made)**

```bash
git add tests/test_logistic_calibrator_fit.py
git commit -m "test(bbox-tube-temporal): align calibrator-fit fixtures with new records"
```

---

## Task 7: Update `aggregation_analysis` and its tests

**Files:**
- Modify: `src/bbox_tube_temporal/aggregation_analysis.py:119-131` (`build_scores_and_labels`) — only if the on-disk contract changes
- Modify: `tests/test_aggregation_analysis.py`

- [ ] **Step 7.1: Verify on-disk contract is unchanged**

The persisted `predictions.json` still carries `tube_logits` at the top level (written by the Task-5 serialiser). `aggregation_analysis.build_scores_and_labels` reads `r["tube_logits"]`. Therefore **no code change** is needed here.

Run: `uv run pytest tests/test_aggregation_analysis.py -v`
Expected: PASS.

- [ ] **Step 7.2: If a test synthesises a record with the old top-level `num_tubes_kept`, leave it — the analysis code does not read `num_tubes_kept`; keep the fixture minimal and tidy but no semantic edit is required.**

No commit if no change.

---

## Task 8: Update `scripts/analyze_variant.py`

**Files:**
- Modify: `scripts/analyze_variant.py` (lines 63, 127, 132, 177, 351–352)

- [ ] **Step 8.1: Verify**

`analyze_variant.py` reads `r["kept_tubes"]` from `predictions.json`. That field is preserved by the Task-5 serialiser. **No code change** needed.

Run a smoke test by running the full test suite:

Run: `uv run pytest tests/ -v`
Expected: PASS (everything updated so far).

- [ ] **Step 8.2: If the script's `test_package_predict.py` mentions `is_winner` on kept tubes, remove those assertions**

Open `tests/test_package_predict.py`. Current usage is minimal:

```python
self.details = {"kept_tubes": kept_tubes}
```

Change to the new nested shape consumed by `protocol_eval.build_record`:

```python
self.details = {
    "preprocessing": {"num_frames_input": 0, "num_truncated": 0, "padded_frame_indices": []},
    "tubes": {"num_candidates": len(kept_tubes), "kept": kept_tubes},
    "decision": {"aggregation": "max_logit", "threshold": 0.0, "trigger_tube_id": None},
}
```

Then make sure each `kept_tubes` fixture dict has all required `KeptTube` fields (add `probability: None`, `first_crossing_frame: None` as needed).

- [ ] **Step 8.3: Run the affected tests**

Run: `uv run pytest tests/test_package_predict.py -v`
Expected: PASS.

- [ ] **Step 8.4: Commit**

```bash
git add tests/test_package_predict.py
git commit -m "test(bbox-tube-temporal): align package_predict fixture with new details"
```

---

## Task 9: Update the error-analysis notebook

**Files:**
- Modify: `notebooks/04-error-analysis.ipynb`

- [ ] **Step 9.1: Open the notebook and locate the `winner_tube_id` reference**

Run: `grep -n "winner_tube_id" notebooks/04-error-analysis.ipynb`

Expected: one hit around line 171.

- [ ] **Step 9.2: Rename to `trigger_tube_id`**

Edit the notebook cell so `record["winner_tube_id"]` becomes `record["trigger_tube_id"]` (the key is the persisted-records key from Task 5's serialiser).

- [ ] **Step 9.3: Lint notebooks**

Run: `make lint`
Expected: PASS.

- [ ] **Step 9.4: Commit**

```bash
git add notebooks/04-error-analysis.ipynb
git commit -m "docs(bbox-tube-temporal): notebook reads trigger_tube_id"
```

---

## Task 10: Full regression sweep

**Files:** none — verification only.

- [ ] **Step 10.1: Run the full test suite**

Run: `make test`
Expected: all tests pass.

- [ ] **Step 10.2: Run lint + format check**

Run: `make lint`
Expected: no errors.

- [ ] **Step 10.3: Sanity-check `evaluate_packaged` end-to-end if a packaged model is available locally**

Run (only if `model.zip` is on disk):
`uv run python scripts/evaluate_packaged.py --model-zip path/to/model.zip --sequences-dir path/to/seqs --output-dir /tmp/eval_out --model-name smoke_test`

Open `/tmp/eval_out/predictions.json` and confirm each record has `trigger_tube_id` (not `winner_tube_id`), `kept_tubes` entries without `is_winner`, and that `num_tubes_total`/`num_tubes_kept`/`tube_logits`/`threshold` are present and sensible.

Expected: metrics look comparable to the last committed run; schema matches the spec.

- [ ] **Step 10.4: If the DVC pipeline is configured locally, re-generate downstream artefacts**

Run: `uv run dvc repro` (or just the specific stages that depend on predictions.json).
Expected: stages complete; outputs change only in schema-shape, not in metric values (since the model itself is untouched).

- [ ] **Step 10.5: Final commit (only if any artefact changed)**

```bash
git add <paths that changed>
git commit -m "chore(bbox-tube-temporal): regenerate predictions after details schema refactor"
```

---

## Self-review checklist

- **Spec coverage.** Preprocessing section → Task 3 (padded_indices via Task 2). Tubes section (kept with logit/probability/first_crossing_frame/entries) → Task 3. Decision section (aggregation, threshold, trigger_tube_id) → Task 3. Dropped fields (tube_logits, winner_tube_entries, num_detections_per_frame, is_winner, per_tube_first_crossing dict) → Task 3. Renames (num_frames → num_frames_input, num_tubes_total → num_candidates, winner_tube_id → trigger_tube_id) → Task 3. Pydantic module → Task 1. Persisted predictions.json shape preserved where downstream requires it, else updated → Tasks 5, 6, 7, 8. Notebook rename → Task 9.
- **Placeholders.** None. Every task has concrete code.
- **Type consistency.** `BboxTubeDetails`, `KeptTube`, `TubeEntry`, `Preprocessing`, `Tubes`, `Decision` spelt identically in every task. `trigger_tube_id` used uniformly (never `winner_tube_id`). `padded_frame_indices` used uniformly (never `padded_indices` except as a local variable name in `predict()`, which is internal).
- **No Claude/Anthropic co-author trailers** in any commit message.
