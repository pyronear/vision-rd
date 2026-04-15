# Training Augmentation Design

**Date:** 2026-04-14
**Status:** Draft, awaiting user review
**Scope:** `experiments/temporal-models/smokeynet-adapted/`

## Relationship to existing specs

- **Builds on `basic-temporal-model-design.md`** (approved, implemented). Dataset = `TubePatchDataset`, heads = `MeanPoolHead` / `GRUHead`, backbone wrapper in `temporal_classifier.py`, Lightning module in `lit_temporal.py`, training entrypoint `scripts/train.py`. Both prior approved specs (basic-temporal-model and smoke-tube-dataset) listed "data augmentation" as out of scope / deferred — this spec is the follow-up.
- **Orthogonal to `performance-improvements-design.md`** (draft). That spec changes backbone capacity (ConvNeXt swap, partial fine-tuning); this spec changes input distribution. They compose cleanly — in fact the payoff of augmentation rises once the backbone is unfrozen, because gradients flow all the way to pixel-space perturbations. No ordering dependency: either can land first.

## Context

The temporal smoke classifier (`TemporalSmokeClassifier` with frozen ResNet18 backbone + `mean_pool` or `gru` head, implemented in `src/bbox_tube_temporal/temporal_classifier.py` and wrapped by `LitTemporalClassifier` in `lit_temporal.py`) currently trains on `TubePatchDataset` without any data augmentation. Motivators:

1. **Small dataset.** 2841 train tubes, 284 val tubes — augmentation as a data multiplier is a primary goal across all variants.
2. **Identified failure modes from the FP/FN bucket analysis:**
   - **Photometric look-alikes**: sun glare, low clouds, high clouds.
   - **Static structures**: tower, building, rocks / cliffs.
   - **Train false negatives still present on the frozen-resnet18 baseline** — that model is underfitting; augmentation must not worsen this.
3. **Higher-capacity variants are already overfitting.** ConvNeXt and fine-tuning variants (introduced by the `performance-improvements` spec and partially wired in `scripts/train.py`) show the classical overfitting signature in their CSV logs.

### Two regimes, confirmed from `data/06_models/*/csv_logs/`

The eval-side picture depends on which backbone / fine-tuning regime is in use:

**Frozen ResNet18 (`gru`, `mean_pool`) — bottlenecked regime.**
Train loss collapses to ~0.0001 by epoch 7–10, but val loss stays flat (0.12–0.15) and val F1 holds 0.95–0.96. Val F1 is slightly *higher* than train F1 (GRU: 0.937 vs. 0.913; mean_pool: 0.905 vs. 0.899). The frozen backbone's fixed feature set acts as an implicit bottleneck — the head can memorize training features, but val features are bounded by the same manifold. **Aug's main job here: data multiplication** (cover more photometric / geometric / temporal variations that the frozen features still encode). Expect train loss to rise modestly and val to tick up; per-bucket FP reduction (clouds / glare / static structures) is the interpretability signal.

**ConvNeXt / fine-tuning (`gru_convnext`, `gru_finetune`, `gru_convnext_finetune`) — capacity-unlocked, overfitting regime.**
Observed on `gru_convnext` (14 epochs logged): train loss collapses by epoch 4 to 0.001–0.1; val loss drifts from 0.15 (epoch 3) up to 0.18–0.36 at epochs 10–13; val F1 oscillates, does not sustain improvement past epoch 3, and bottoms out at 0.846 at epoch 11. `gru_finetune` is less extreme but trend-similar — val F1 peaks at 0.903 around epoch 5 then regresses to 0.86–0.90 late. **Aug's main job here: regularization against overfitting** — prevent the head (and, under fine-tuning, the unfrozen block) from memorizing the train set. Expected response: train loss stays higher, val loss stops drifting up past mid-training, best-val checkpoint moves later in training, and headline val metrics improve.

### What this means for the spec

- Aug is motivated by **two different mechanisms** depending on variant (data multiplication vs. anti-overfitting regularization). The same implementation covers both; only the expected response differs.
- The verification protocol's train-F1 watchdog must be **variant-aware**: for frozen-resnet18, low train loss with flat val is fine; for ConvNeXt / fine-tune, low train loss combined with rising val loss is exactly the failure mode aug is trying to fix — so "train F1 dropped with aug" is expected and desired, not a red flag, as long as val improves.
- This also sharpens the case for composing with `performance-improvements`: extra backbone capacity without aug just lets the model memorize faster. Aug gives that capacity something productive to do.

## Goals

- **Photometric aug** to teach the model that bright gray patches are not automatically smoke (targets cloud / glare FPs).
- **Temporal aug** (frame drop, variable stride, sub-sequence sampling) to force reliance on motion / growth cues that distinguish smoke from static structures.
- **Spatial aug** (flip, small rotation, scale, translate) as a cheap data multiplier.
- **Preserve temporal coherence within a tube** — same spatial / photometric transform applied to every frame in a tube, so motion direction and relative appearance are preserved.
- **Train-only** — validation and inference stay deterministic.
- **Variant-aware watchdog on train F1.** For the frozen-resnet18 baseline, a large drop in train F1 is a red flag (the head can't absorb the noise). For ConvNeXt / fine-tune variants, a drop in train F1 with improving val is the **desired** signal — it means we're no longer overfitting. See the verification protocol for the concrete thresholds.

## Non-Goals

- No feature-space augmentation (mixup / cutmix across tubes). Deferred until backbone is unfrozen.
- No test-time augmentation.
- No augmentation during the offline patch-building stage (`scripts/build_model_input.py`) — aug runs at train time on the already-cropped PNGs.
- No new DVC stages.
- No changes to `model.py`, `package.py`, or inference.
- No curriculum schedule on tube length — uniform random length sampling instead.
- No random temporal reversal (GRU is directional; reversal teaches implausible dynamics).

## Architecture

A new module `src/bbox_tube_temporal/augment.py` exposing three composable transforms and a builder:

```python
class SpatialTubeTransform:
    """Per-tube-consistent spatial aug applied to [T, 3, H, W].

    One set of affine params sampled per tube, applied identically
    to every frame.
    """

class PhotometricTubeTransform:
    """Per-tube-consistent brightness/contrast/saturation.

    One factor set sampled per tube. Operates on [0, 1] tensors
    (pre-normalization).
    """

class TemporalTubeTransform:
    """Operates on (patches, mask); returns re-compacted (patches', mask').

    Valid frames always occupy positions [0..n-1] of the output — this
    is a hard invariant the GRU head depends on (pack_padded_sequence
    assumes contiguous valid prefix).
    """

def build_tube_augment(config: dict, train: bool) -> Callable[[dict], dict]:
    """Compose spatial -> photometric -> temporal -> normalize for train,
    or normalize-only for val."""
```

### Integration into `TubePatchDataset`

- Constructor gains `transform: Callable[[dict], dict] | None = None`.
- ImageNet normalization **moves out** of `__getitem__` into the transform. Raw loading produces `[0, 1]` tensors.
- `__getitem__` applies `self.transform(item)` if provided, else returns raw (preserving back-compat for any caller that builds a dataset without augmentation).
- `scripts/train.py` (the DVC stage entrypoint — not `src/bbox_tube_temporal/data.py`, which is I/O utilities) wires `transform=build_tube_augment(augment_cfg, train=True)` for the train `TubePatchDataset` and `transform=build_tube_augment(augment_cfg, train=False)` for val. Val transform does normalization only.
- Randomness: a `torch.Generator` seeded per worker via `worker_init_fn` in the DataLoader — ensures reproducibility at `num_workers > 0` and that workers don't all produce identical augmentations.

### Why `__getitem__` and not `collate_fn`

- Every augmentation in scope is strictly per-tube; no cross-sample operations needed.
- `__getitem__` runs in DataLoader worker processes, parallelizing the aug cost behind PNG decode.
- Default collate still handles padding / stacking — no need to reinvent it.
- Matches PyTorch idiom; test surface is small (`transform(sample)`).

If we later add mixup / cutmix, they go in `collate_fn` on top of per-tube aug; the two layers compose.

## Transforms

### Spatial (per-tube consistent)

Applied to `[T, 3, H, W]` with one sampled parameter set per tube:

| Op | Range | Notes |
|---|---|---|
| Horizontal flip | `p = 0.5` | Same decision across all frames (preserves motion direction) |
| Rotation | `±5°` | Small — patches are tight crops |
| Scale | `0.9–1.1` | Single uniform factor |
| Translate | `±5%` H / W | Keeps smoke on-screen |

Implementation: `torchvision.transforms.v2.functional.affine` applied once on the `[T, 3, H, W]` tensor using sampled-once params.

**Explicitly excluded:** shear (not representative of realistic camera / atmospheric distortion).

### Photometric (per-tube consistent)

| Op | Factor range |
|---|---|
| Brightness | `0.8–1.2` |
| Contrast | `0.8–1.2` |
| Saturation | `0.8–1.2` |

Applied in fixed order with single sampled factors per tube. Operates on `[0, 1]` tensors; normalization happens after.

**Explicitly excluded:** hue shift (smoke is near-gray; shifting hue produces pink / green smoke, which is out-of-distribution).

### Temporal (changes mask; re-compacts)

Operates on `(patches: [T, 3, H, W], mask: [T])` where the first `n = mask.sum()` positions are valid. Composes three independent stages in order:

1. **Sub-sequence sampling** (`p = 0.5`): pick a random contiguous window of length `k ∈ [subseq_min_len, n]` from the valid frames.
2. **Random stride** (`p = 0.25`): take every second frame of the current window — smoke still evolves, static structures still don't, forcing motion-cue reliance.
3. **Per-frame drop** (`p = 0.15` per frame, independent): drop each remaining frame with floor `min_valid_after_drop = 4`. Drops are **compacted away**, not turned into gaps — the valid prefix invariant is preserved for `pack_padded_sequence`.

Most tubes (~50%) receive no sub-sequence change, giving the model steady access to long sequences; a meaningful tail sees aggressive temporal perturbation.

## Config surface (`params.yaml`)

Add a new **top-level** `augment:` section (sibling to `train_mean_pool`, `train_gru`, etc.). Top-level is deliberate: it's a single source of truth, toggleable/ablatable with one `dvc exp run -S augment.temporal.frame_drop_prob=0` line instead of touching every variant's block.

```yaml
augment:
  enabled: true
  spatial:
    flip_prob: 0.5
    rotation_deg: 5.0
    scale_range: [0.9, 1.1]
    translate_frac: 0.05
  photometric:
    brightness_range: [0.8, 1.2]
    contrast_range: [0.8, 1.2]
    saturation_range: [0.8, 1.2]
  temporal:
    subseq_prob: 0.5
    subseq_min_len: 4
    stride_prob: 0.25
    frame_drop_prob: 0.15
    min_valid_after_drop: 4
```

**Config loading in `scripts/train.py`** (convention-compatible with the existing `--params-key` pattern):

```python
full = yaml.safe_load(args.params_path.read_text())
cfg = full[args.params_key]                    # existing — per-arch block
augment_cfg = full.get("augment", {"enabled": False})   # new — shared block
```

Each training DVC stage tracks both sections:

```yaml
params:
  - train_gru        # existing
  - augment          # new
```

DVC caches on the union of both sections, so changes to `augment.*` correctly invalidate the cached train stage. This pattern matches how `build_tubes` already lists two param sections (`tubes`, `build_tubes`).

**Why not duplicate `augment:` under each `train_*` block** (the duplication-is-intentional pattern from the performance-improvements spec)? Augmentation is intentionally held constant across arch variants for apples-to-apples comparison; a shared section *enforces* that. If a variant ever needs arch-specific augmentation, we can always override by adding `augment:` under that variant and teaching the loader to prefer the variant's value.

## Testing

Unit tests in `tests/test_augment.py`:

| What | Assertion |
|---|---|
| Spatial flip applied identically across frames | `out[t] == flip(inp[t])` for all `t` with the same sampled flip |
| Spatial identity (ranges collapsed) | Output equals input pixel-for-pixel |
| Photometric same factor per tube | Inter-frame brightness difference preserved from input |
| Photometric identity (`[1.0, 1.0]`) | Output equals input |
| Temporal sub-sequence | Output valid frames are a contiguous slice of input valid frames, compacted to `[0..n-1]` |
| Temporal stride (`stride_prob = 1.0`) | Valid length ≈ `ceil(n / 2)` |
| Temporal frame-drop floor | Valid length never below `min_valid_after_drop` |
| Mask invariant (always) | `mask[:k].all() and not mask[k:].any()` for some `k` |
| `build_tube_augment(train=False)` | Deterministic; returns only normalization |
| Reproducibility | Same seed, same worker → same output |
| Worker seeding | Different `worker_id` → different sampled params |

Dataset tests (`tests/test_dataset.py`):
- `TubePatchDataset(transform=None)` preserves current output shape, dtype, normalized values (back-compat).
- With a real transform: composition runs once per `__getitem__`; output satisfies shape and mask-prefix invariants.

## Visualization script

`scripts/visualize_augment.py` — dumps ~8 augmented variants of a few tubes to `data/08_reporting/augment_samples/` as side-by-side image grids. Cheap sanity check before committing GPU time.

## Verification protocol

The protocol is framed around the two regimes (frozen-bottleneck vs. capacity-unlocked). Variant lists below are the DVC stage names from `params.yaml`.

### Shared setup

- **Primary metrics: PR-AUC and `FP @ recall=0.97`** (not val F1 at threshold 0.5). F1-at-threshold conflates classifier quality with threshold choice; PR-AUC is threshold-free and `FP @ recall` is production-relevant. The `performance-improvements` spec already emits these in `compare_variants.md` — reuse the same reader.
- **Noise floor first.** Three-seed baseline runs per variant (existing `train_gru`, `train_gru_seed43`, `train_gru_seed44` already provide this for the frozen baseline; add matching seed variants for any capacity-unlocked variant we want to verify). Spread across seeds = the bar any treatment must clear.
- **Paired treatment.** Run aug enabled on the **same seeds**. Compute per-seed `Δ = treatment − baseline`. Sign-consistency across 3 seeds + mean |Δ| > baseline spread = real signal.
- **Early-stop caveat.** Aug trains slower. For any verification run, bump `early_stop_patience` from 5 to 15 (via `-S train_gru.early_stop_patience=15`), or remove early stop entirely and report last-epoch metrics alongside best-checkpoint metrics. Otherwise we risk stopping aug runs before they've converged and misattributing that to "aug didn't help."

### Frozen-bottleneck regime (`gru`, `mean_pool`)

1. **Primary signal.** PR-AUC improves by more than noise floor; `FP @ recall=0.97` drops by more than noise floor. Val F1 / PR-AUC improvement expected to be **modest** — the frozen feature set limits how much aug can actually change.
2. **Watchdog.** Train F1 must not drop below 0.85 (GRU) / 0.80 (mean_pool). If it does, the head can't absorb the input noise under the frozen backbone — back off `frame_drop_prob` and photometric ranges first.
3. **Interpretability signal.** Rerun `scripts/evaluate.py` (emits per-tube FP/FN outputs per commit `eb2e699`) and visually inspect the galleries pre vs. post. Expect cloud / glare FPs to shrink if photometric aug pulls its weight; static-structure FPs (tower, building, rocks) to shrink if temporal aug does. User-bucketed, not code-bucketed.

### Capacity-unlocked, overfitting regime (`gru_convnext`, `gru_finetune`, `gru_convnext_finetune`)

These variants already exhibit the overfitting signature in logs: train loss collapses to near-zero, val loss drifts upward past mid-training, val F1 regresses from an early peak.

1. **Primary signal.** Same as above — PR-AUC and `FP @ recall=0.97` beat noise floor, paired across seeds.
2. **Anti-overfitting signatures (specific to this regime).** All three are expected with working aug:
   - **Train loss stops collapsing.** Per-step train loss should stabilize in a healthy range (say 0.05–0.3) instead of dropping to 0.0001. Plot `train/loss` from `csv_logs/*/metrics.csv`.
   - **Val loss stops drifting upward.** The `gru_convnext` baseline showed val loss rising from 0.15 (epoch 3) to 0.18–0.36 at epochs 10–13 — with aug, val loss should remain monotonic-ish or flat past mid-training.
   - **Best-val checkpoint appears later.** Baseline best-val on `gru_convnext` was at epoch 3; with aug, expect the best checkpoint to appear at epoch ≥ 8. If it still appears by epoch 3, aug isn't regularizing enough — bump frame-drop and photometric ranges.
3. **Train-F1 interpretation inverts.** A *drop* in train F1 is the desired signal here, not a failure. No fixed watchdog threshold — instead, the watchdog fires only if val metrics *also* fail to improve (that would mean aug is simply adding noise without utility).
4. **Interpretability signal.** Same FP/FN gallery inspection as the frozen regime.

### Ablation (both regimes, optional but cheap)

Three runs — spatial+photometric only, temporal only, full — via single-line `-S` param overrides. Tells us which lever is load-bearing. Informs the next iteration of knob tuning.

### Pitfalls

- **Bootstrap confidence.** For val n=284 and PR-AUC ≈ 0.97, the 95% bootstrap CI is roughly ±0.01. So Δ < 0.005 is probably noise; Δ > 0.02 is probably real. Keep this in mind when reading `compare_variants.md`.
- **Checkpoint-by-val/F1 compounds noise** on the small val set. Report PR-AUC and `FP @ recall` at both the F1-selected checkpoint and the last epoch — divergence between the two is itself informative (usually means early-stop triggered the wrong checkpoint).
- **Don't conclude from a single run.** A single seed's Δ is within the noise floor on this val set. Paired Δ across three seeds is the minimum evidence to claim signal.

## Rollout

All changes are additive / behind `augment.enabled`. Default can ship as `enabled: true` once verification (3) and (4) pass.

## Critical files

- `src/bbox_tube_temporal/augment.py` — **new** (the three transforms + `build_tube_augment`).
- `src/bbox_tube_temporal/dataset.py` — add `transform` kwarg to `TubePatchDataset`, move ImageNet normalization out of `__getitem__`.
- `scripts/train.py` — load top-level `augment` section alongside the per-arch section, build train/val transforms, pass them into `TubePatchDataset`, wire `worker_init_fn` for per-worker seeding.
- `scripts/visualize_augment.py` — **new** (sanity-check image grids).
- `params.yaml` — add top-level `augment:` block.
- `dvc.yaml` — add `augment` to the `params:` list of `train_mean_pool` and `train_gru`; add `src/bbox_tube_temporal/augment.py` to `deps:`.
- `tests/test_augment.py` — **new**.
- `tests/test_dataset.py` — add tests for the transform wiring and the normalization refactor.

**Not touched:** `data.py` (I/O utilities only), `temporal_classifier.py`, `lit_temporal.py`, `tubes.py`, `types.py`, `model_input.py`, `evaluate.py`, `model.py`, `package.py`. Augmentation is strictly a train-time dataset concern.
