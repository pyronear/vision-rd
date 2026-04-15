# TemporalModel protocol for smokeynet-adapted

Status: design (not yet implemented)
Date: 2026-04-15

## Goal

Implement `pyrocore.TemporalModel` for `smokeynet-adapted` so a trained
variant can be shipped as a single self-contained archive and run
end-to-end by the temporal-model leaderboard, which hands the model a
list of raw frame image paths and expects a binary decision.

## Context

The offline training pipeline consumes pre-cached YOLO-format label
files (GT boxes for wildfire sequences, YOLO predictions for FP
sequences), builds single-longest tubes per sequence, crops 224x224
context-expanded patches, and trains a `TemporalSmokeClassifier`
(frozen or partially-fine-tuned timm backbone + mean-pool or GRU
head). Seven variants have been trained; `gru_convnext_finetune`
currently leads on val (F1 = 0.963, FP @ recall 0.95 = 4).

No `TemporalModel` subclass exists yet. The pyrocore contract is:

```python
class TemporalModel(ABC):
    def load_sequence(self, frames: list[Path]) -> list[Frame]: ...
    @abstractmethod
    def predict(self, frames: list[Frame]) -> TemporalModelOutput: ...
    def predict_sequence(self, frames: list[Path]) -> TemporalModelOutput:
        return self.predict(self.load_sequence(frames))
```

`Frame` carries only `frame_id`, `image_path`, `timestamp` — no
detections. The leaderboard runner (`temporal-model-leaderboard`)
calls `load_sequence` then `predict` on raw frame paths with no
side-channel for cached detections.

## Key decisions

1. **Package a YOLO companion inside the archive.** Mirror the
   `tracking-fsm-baseline` pattern: one zip ships YOLO weights +
   classifier checkpoint + config. `predict()` runs YOLO live per
   sequence, builds tubes, crops patches, runs the classifier. Only
   option that matches the leaderboard contract as-is.

2. **YOLO weights source.** `data/01_raw/models/best.pt` (DVC-tracked
   inside this experiment).

3. **Single-variant packaging (for now).** Only
   `gru_convnext_finetune` is packaged — it leads the val comparison
   by a clear margin. The script `scripts/package_model.py --variant
   <name>` still accepts any variant (useful for manual runs and
   future A/B), but `dvc.yaml` wires a single, hard-coded `package`
   stage for `gru_convnext_finetune`. Additional variants can be
   added later if the leaderboard motivates it.

4. **Multi-tube aggregation at inference.** `max_logit` rule: score
   every surviving tube in a single batched classifier forward, pick
   the highest logit, compare against the calibrated threshold.
   Rationale: single-longest rule silently drops genuine smoke when
   YOLO jitters; max-logit matches the production goal of high recall
   while letting the calibrated threshold suppress clutter.

5. **Trigger frame rule.** `trigger_frame_index =
   winner_tube.end_frame`. Simplest rule that gives the leaderboard
   a usable time-to-detection proxy. Prefix scoring is a possible
   future refinement; not included here.

6. **Short-sequence handling.** No raw-frame padding (unlike
   `tracking-fsm-baseline`). The classifier is mask-aware (both
   `MeanPoolHead` and `GRUHead` handle variable lengths via masks /
   packed sequences), so repeating frames would put it out of
   distribution. Instead, loosen the tube-length filter at inference:
   `tubes.min_tube_length = 4` stays as the training data-quality
   filter, and `tubes.infer_min_tube_length = 2` is used by
   `predict()`. The calibrated threshold absorbs the added FP risk.

7. **Decision threshold calibration.** Packager reads the val
   prediction artefacts produced by `evaluate_<variant>`, finds the
   smallest threshold whose recall meets `decision.target_recall`
   (default 0.95), writes it into `config.yaml`. No retraining.

## Architecture

Six-stage pipeline inside `predict()`:

```
frames (list[Frame])
  -> 1. truncate to classifier.max_frames
  -> 2. YOLO detect (one batched call over all frames)
  -> 3. build_tubes + filter (infer_min_tube_length, min_detected_entries)
  -> 4. crop+normalize patches per tube -> stack + mask
  -> 5. classifier forward (one batched call over all tubes)
  -> 6. argmax, threshold, trigger = winner.end_frame
```

Batching summary:

- YOLO runs once for the whole sequence (`yolo.predict([p1...pT], ...)`).
- Classifier runs once for all surviving tubes
  (`[N_tubes, max_frames, 3, 224, 224]`).
- Everything else is CPU/Python and cheap.

## Components

New files under `src/smokeynet_adapted/`:

- `model.py` - `SmokeynetTemporalModel(TemporalModel)` with
  `from_package()` factory and `predict()` method.
- `package.py` - `ModelPackage` dataclass, `build_model_package()`,
  `load_model_package()`. Parallels
  `tracking_fsm_baseline/package.py`.
- `inference.py` - Pure helpers: `run_yolo_on_frames`,
  `filter_and_interpolate_tubes`, `crop_tube_patches`, `score_tubes`,
  `pick_winner_and_trigger`. Extracted so `predict()` is thin and
  each stage is unit-testable in isolation.

New script:

- `scripts/package_model.py` - CLI: `--variant <name>`. Reads
  `params.yaml`, the variant's best Lightning checkpoint, val eval
  predictions; calibrates threshold; writes
  `data/06_models/<variant>/model.zip`.

Touched files:

- `pyproject.toml` - add `ultralytics` dep.
- `params.yaml` - add a `package` block with
  `target_recall: 0.95` and `infer.*` defaults (see below).
- `dvc.yaml` - add a single `package` stage hard-wired to
  `gru_convnext_finetune`, deps: checkpoint + `params.yaml` +
  `data/01_raw/models/best.pt` + val eval predictions; outs:
  `data/06_models/gru_convnext_finetune/model.zip`.
- `README.md` - document the package artefact and the
  `SmokeynetTemporalModel` class.

## Package format

Archive: `.zip` with `ZIP_STORED` compression (weights are already
incompressible).

```
model.zip
├── manifest.yaml
├── yolo_weights.pt
├── classifier.ckpt
└── config.yaml
```

### `manifest.yaml`

```yaml
format_version: 1
variant: gru_convnext_finetune
yolo_weights: yolo_weights.pt
classifier_checkpoint: classifier.ckpt
config: config.yaml
```

### `config.yaml`

```yaml
infer:
  confidence_threshold: 0.01    # see "YOLO inference threshold" note below
  iou_nms: 0.2
  image_size: 1024

tubes:
  iou_threshold: 0.2
  max_misses: 2
  min_tube_length: 4            # training filter; kept here for parity / record
  infer_min_tube_length: 2      # actually used by predict()
  min_detected_entries: 2
  interpolate_gaps: true

model_input:
  context_factor: 1.5
  patch_size: 224
  normalization:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]

classifier:
  backbone: convnext_tiny
  arch: gru                     # "mean_pool" or "gru"
  hidden_dim: 128
  num_layers: 1
  bidirectional: false
  max_frames: 20
  pretrained: false

decision:
  aggregation: max_logit        # only aggregation strategy in this design
  threshold: 0.42               # example; populated by packager at build time
  target_recall: 0.95           # informational - what `threshold` calibrated for
  trigger_rule: end_of_winner   # fixed; winner_tube.end_frame
```

### YOLO inference threshold

`infer.confidence_threshold` is not a learned parameter, so there is
no "fine-tune on train set" step — it's a compute/recall knob for
the detector. Its job is to admit enough candidate boxes that tubes
can form; the classifier, guided by the calibrated
`decision.threshold`, is what actually decides smoke vs. no-smoke.
Default is set to `0.01` to match `tracking-fsm-baseline`'s infer
config, so both experiments see the same YOLO candidate set before
their respective temporal stages. A systematic sweep over infer
thresholds (e.g., optimising classifier val F1) is a possible future
refinement but is not part of this spec: the classifier absorbs most
of the cost of extra low-confidence boxes (they just produce more
tubes, which `max_logit` aggregation then ignores).

## Data flow inside `predict()` (concrete)

For a 20-frame sequence producing 3 surviving tubes:

| Stage | Granularity | Call |
|---|---|---|
| 1. Truncate | whole sequence | Slice to first `max_frames`. |
| 2. YOLO | one batched call, all frames | `yolo.predict([p1..p20], ...)` → list of 20 results. |
| 3. Build tubes | sequential | `build_tubes(frame_dets)` (existing fn); filter by `infer_min_tube_length`, `min_detected_entries`; `interpolate_gaps` on survivors. |
| 4. Crop | per tube, all entries | Open each image once per tube entry, `expand_bbox → norm_bbox_to_pixel_square → crop_and_resize`, `to_tensor`, ImageNet-normalize. Pad to `max_frames`; build mask. |
| 5. Classifier | one batched call, all tubes | Stack N tubes → `[N, max_frames, 3, 224, 224]`; single forward; → `Tensor[N]` logits. |
| 6. Aggregate | trivial | `winner = argmax(logits)`; `is_positive = logits[winner] >= threshold`; `trigger = tubes[winner].end_frame`. |

Two GPU passes per sequence (YOLO + classifier). Classifier memory
bounded by `N_tubes × max_frames × 3 × 224²`.

## `TemporalModelOutput.details`

Populated for debugging on the leaderboard:

```python
{
  "num_frames": int,
  "num_truncated": int,
  "num_detections_per_frame": list[int],
  "num_tubes_total": int,
  "num_tubes_kept": int,
  "tube_logits": list[float],
  "winner_tube_id": int | None,
  "winner_tube_entries": list[dict],     # frame_idx, bbox, is_gap, confidence
  "threshold": float,
}
```

We want to be able to visualize the tubes from the details dict and all the context that lead to the classifier output.

## Error handling and edge cases

Errors bubble up unchanged — the leaderboard runner catches and
reports per-sequence failures.

| Case | Behavior |
|---|---|
| Empty `frames` | `is_positive=False`, `trigger_frame_index=None`, `details={"num_frames": 0, ...}`. |
| YOLO returns 0 detections for every frame | No tubes → same as above. |
| All tubes shorter than `infer_min_tube_length` (= 2) | No tubes survive → same as above. |
| Bbox touches image border | Existing `norm_bbox_to_pixel_square` clamps + pads to square. |
| Sequence longer than `max_frames` | Truncate to first N (matches training); record `num_truncated`. |
| Tube has gaps | `interpolate_gaps` (matches training). |
| All tubes below threshold | `is_positive=False`, `trigger_frame_index=None`; still report `winner_tube_id` and logits in `details`. |
| Missing archive / bad zip / version mismatch | `FileNotFoundError` / `zipfile.BadZipFile` / `ValueError` (same as baseline). |
| YOLO load failure | `ultralytics` native exception. |
| Checkpoint/config architecture mismatch | PyTorch state-dict load error. |

## Training/inference parity checklist

These must match byte-for-byte. Enforced by the parity test.

1. Frame ordering — filename-stem sort at the caller; `load_sequence`
   preserves order.
2. Truncation — first `max_frames`, not middle or last.
3. Tube building — same `iou_threshold`, `max_misses`, same
   `build_tubes` function.
4. Tube filtering — same `min_detected_entries`; note inference uses
   `infer_min_tube_length` (looser by design; documented above).
5. Gap interpolation — same `interpolate_gaps` after filtering.
6. Patch crop — `expand_bbox → norm_bbox_to_pixel_square →
   crop_and_resize`, same `context_factor`, `patch_size`, PIL
   bilinear.
7. Normalization — `to_tensor` first (`[0, 1]` CHW float32), then
   subtract ImageNet mean, divide std.
8. Padding — zero-pad patches, `mask[i]=True` for real frames; same
   layout as `TubePatchDataset`.
9. Classifier forward — build `TemporalSmokeClassifier` with the
   exact hyperparams from `config.classifier`, `eval()`, no grad.

## Known limitations (out of scope)

- **Train/infer distribution gap on positives.** Training tubes for
  wildfire sequences were built from GT boxes; inference builds tubes
  from live YOLO predictions. Possible future mitigations: retrain on
  YOLO predictions for positives, or apply a light geometric
  perturbation to GT boxes during training. Not fixed in this spec.
- **Prefix scoring for `trigger_frame_index`.** Winner's `end_frame`
  is used. A more principled "earliest-sufficient-context" variant is
  a possible future refinement.
- **Aggregation strategies other than `max_logit`.** Mean / attention
  over tubes would be a separate experiment.
- **Leaderboard integration.** Lives in
  `experiments/temporal-models/temporal-model-leaderboard`. This spec
  only produces the class and package it consumes.

## Testing

Four layers, fastest first. `make test` stays under ~30s; YOLO is
mocked everywhere except one optional slow smoke test gated behind
`pytest -m slow`.

1. **`tests/test_inference_units.py`** - each helper in
   `inference.py` with hand-crafted fixtures. No model loading, no
   I/O.
2. **`tests/test_package.py`** - round-trip build + load with a tiny
   YOLO `.pt` stub and a small `TemporalSmokeClassifier` state_dict;
   assert weights bit-equal and config round-trips.
3. **`tests/test_model_parity.py`** - critical. For 2-3 real
   sequences from `data/01_raw/` (one wildfire, one fp):
   - Run the offline training path (`build_tubes.py` →
     `build_model_input.py` → `TubePatchDataset.__getitem__` →
     `TemporalSmokeClassifier.forward`) → reference logit.
   - Run `SmokeynetTemporalModel.predict()` on the same sequences
     with YOLO replaced by cached GT labels via dependency injection.
   - Assert logits equal to `1e-5`.
4. **`tests/test_model_edge_cases.py`** - each row in the edge-case
   table returns the documented output.

## DVC stages (new)

```yaml
# dvc.yaml (excerpt)
stages:
  package:
    cmd: uv run python scripts/package_model.py
         --variant gru_convnext_finetune
         --output data/06_models/gru_convnext_finetune/model.zip
    deps:
      - data/06_models/gru_convnext_finetune/best_checkpoint.pt
      - data/01_raw/models/best.pt
      - data/07_model_output/gru_convnext_finetune/val/predictions.json
      - scripts/package_model.py
    params:
      - package
      - tubes
      - model_input
      - train_gru_convnext_finetune
    outs:
      - data/06_models/gru_convnext_finetune/model.zip
```

The stage is single-variant by design. Additional variants can be
added as sibling stages (`package_<variant>`) later.

(Exact stage wiring may be adjusted during implementation to fit the
existing `evaluate_<variant>` output layout; the key deps are listed
above.)
