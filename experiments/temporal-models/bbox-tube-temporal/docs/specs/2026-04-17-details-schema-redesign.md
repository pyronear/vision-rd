# Details schema redesign for `BboxTubeTemporalModel`

**Status:** Proposed
**Date:** 2026-04-17
**Scope:** `experiments/temporal-models/bbox-tube-temporal/`

## Context

`BboxTubeTemporalModel.predict` currently returns a `TemporalModelOutput`
whose `details: dict` carries a large, flat bag of diagnostic fields
accumulated over successive specs (temporal-model-protocol,
first-crossing-trigger, protocol-eval, logistic-calibrator). The shape
has drift, redundancy, and no schema:

```python
details = {
    "num_frames", "num_truncated", "num_padded",
    "num_detections_per_frame",
    "num_tubes_total", "num_tubes_kept",
    "tube_logits",                  # duplicates kept_tubes[*].logit
    "winner_tube_id",
    "winner_tube_entries",          # duplicates kept_tubes[winner].entries
    "kept_tubes",                   # full per-tube data
    "per_tube_first_crossing",      # dict keyed by tube_id
    "threshold", "aggregation",
}
```

The model is now strong enough that we care about the downstream
interface. The two real consumers of `details` are:

1. **Drawing tubes** on frames for alert UI / troubleshooting.
2. **Understanding the decision** — why did the model trigger (or not)?

Everything shipped in `details` should serve one of those two uses.
Everything else is noise.

## Goals

- Slim down `details` to the minimum needed for tube rendering and
  decision understanding.
- Eliminate redundant fields (winner duplicated in multiple places,
  logits duplicated between a top-level list and per-tube records).
- Group related fields into named sections rather than one flat dict.
- Add Pydantic schema validation so the structure is typed, enforced at
  construction, and documented in code.
- Keep the `pyrocore.TemporalModelOutput.details: dict` contract
  untouched (other experiments use the same base type).

## Non-goals

- Changing the `pyrocore` base contract.
- Typed schemas for other temporal models (pyro-detector-baseline,
  mtb-change-detection, tracking-fsm-baseline).
- Surfacing dropped candidate tubes (tubes that failed
  `filter_and_interpolate_tubes`). See "Future extensions".
- Persisting backwards compatibility with artifacts produced by the old
  schema. Any consumer that reads past JSON is updated in the same PR;
  past artifacts are re-generated if needed.

## Design

### Approach

Define a Pydantic schema local to the `bbox_tube_temporal` package.
`predict()` builds a validated `BboxTubeDetails` instance, then emits
`details=model.model_dump()`. Consumers that want types parse with
`BboxTubeDetails.model_validate(output.details)`.

This keeps the blast radius inside one experiment, avoids touching
pyrocore or other experiments, and lets this model own its own schema
without imposing structure on simpler models.

### Schema

New module: `src/bbox_tube_temporal/details_schema.py`.

```python
from typing import Literal
from pydantic import BaseModel


class TubeEntry(BaseModel):
    frame_idx: int
    bbox: tuple[float, float, float, float] | None  # (cx, cy, w, h), normalized; None for gap
    is_gap: bool
    confidence: float | None                        # None for gap


class KeptTube(BaseModel):
    tube_id: int
    start_frame: int
    end_frame: int
    logit: float
    probability: float | None                       # calibrated; None when no calibrator
    first_crossing_frame: int | None                # None if this tube never crossed threshold
    entries: list[TubeEntry]


class Preprocessing(BaseModel):
    num_frames_input: int
    num_truncated: int                              # tail drop past max_frames (indices implicit)
    padded_frame_indices: list[int]                 # indices in the final sequence that are padded


class Tubes(BaseModel):
    num_candidates: int                             # pre-filter count
    kept: list[KeptTube]                            # len replaces num_tubes_kept


class Decision(BaseModel):
    aggregation: Literal["max_logit", "logistic"]
    threshold: float                                # effective threshold (logit OR probability)
    trigger_tube_id: int | None                     # tube whose prefix score first crossed


class BboxTubeDetails(BaseModel):
    preprocessing: Preprocessing
    tubes: Tubes
    decision: Decision
```

### Integration with `TemporalModelOutput`

```python
details_model = BboxTubeDetails(
    preprocessing=Preprocessing(...),
    tubes=Tubes(num_candidates=..., kept=[...]),
    decision=Decision(...),
)
return TemporalModelOutput(
    is_positive=is_positive,
    trigger_frame_index=trigger,
    details=details_model.model_dump(),
)
```

`TemporalModelOutput.details` stays `dict`. Typed consumers call
`BboxTubeDetails.model_validate(output.details)`.

### Field mapping (old → new)

| Old key                          | New path                                                   |
| -------------------------------- | ---------------------------------------------------------- |
| `num_frames`                     | `preprocessing.num_frames_input`                           |
| `num_truncated`                  | `preprocessing.num_truncated`                              |
| `num_padded` (count)             | `preprocessing.padded_frame_indices` (list)                |
| `num_detections_per_frame`       | **dropped**                                                |
| `num_tubes_total`                | `tubes.num_candidates`                                     |
| `num_tubes_kept`                 | `len(tubes.kept)`                                          |
| `tube_logits`                    | `tubes.kept[*].logit`                                      |
| `winner_tube_id`                 | `decision.trigger_tube_id`                                 |
| `winner_tube_entries`            | `tubes.kept[i].entries` where `tube_id == trigger_tube_id` |
| `per_tube_first_crossing` (dict) | `tubes.kept[*].first_crossing_frame`                       |
| `kept_tubes[*].is_winner`        | **dropped** (single source of truth on `decision`)         |
| `threshold`                      | `decision.threshold`                                       |
| `aggregation`                    | `decision.aggregation`                                     |

### Justifications for the drops

- **`num_detections_per_frame`**: Not consumed by any non-test code.
  Coarse counts without bboxes can't support tube drawing and don't
  meaningfully explain the decision. Re-add later under a debug flag if
  FN forensics needs it.
- **`tube_logits` / `winner_tube_entries`**: Pure duplication. Derivable
  from `tubes.kept`.
- **`kept_tubes[*].is_winner`**: Redundant with `decision.trigger_tube_id`.
  Keeping one source of truth prevents the two from ever disagreeing.
- **`per_tube_first_crossing` dict**: Per-tube data belongs on the tube.

### `padded_frame_indices` (replacing `num_padded`)

Today we only know *how many* frames were synthesized via the padding
strategy — not *which* slots in the final sequence are padded. The UI
case needs this: drawing a tube on a padded frame should be visually
distinct from drawing on a real capture, and "was the trigger on a
padded frame?" is a decision-understanding question that the count
can't answer.

Indices live in the final (post-pad) sequence. Producing them requires
`pad_frames_symmetrically` / `pad_frames_uniform` to return (or expose)
the padding indices, or the caller to compute them. Either is trivial.

Truncation stays as a count because truncation always drops the tail:
the truncated original-sequence indices are implied by
`range(max_frames, num_frames_input)`.

### Empty-sequence and no-tubes cases

Both early-return paths emit a well-formed `BboxTubeDetails`:

```python
# Empty input:
BboxTubeDetails(
    preprocessing=Preprocessing(num_frames_input=0, num_truncated=0, padded_frame_indices=[]),
    tubes=Tubes(num_candidates=0, kept=[]),
    decision=Decision(aggregation=..., threshold=..., trigger_tube_id=None),
)

# No tubes kept (YOLO found nothing, or all filtered):
BboxTubeDetails(
    preprocessing=Preprocessing(...),
    tubes=Tubes(num_candidates=N, kept=[]),
    decision=Decision(aggregation=..., threshold=..., trigger_tube_id=None),
)
```

## Migration

Breaking change; all in-repo consumers are updated in the same PR.
Past persisted artifacts (protocol_eval outputs, calibrator training
records) are re-generated by re-running the relevant DVC stages.

Files touched:

- **New:** `src/bbox_tube_temporal/details_schema.py`.
- `src/bbox_tube_temporal/model.py` — build and dump the Pydantic
  model; update both early-return branches.
- `src/bbox_tube_temporal/protocol_eval.py` — `build_record` snapshots
  `output.details` as-is (no change needed there), but anywhere it
  extracts `tube_logits`/`num_tubes_kept` reads from
  `tubes.kept[*].logit` and `len(tubes.kept)`.
- `src/bbox_tube_temporal/aggregation_analysis.py` — `tube_logits` →
  `tubes.kept[*].logit`.
- `src/bbox_tube_temporal/logistic_calibrator_fit.py` — `kept_tubes` →
  `tubes.kept`.
- `scripts/evaluate_packaged.py` — update all the details reads to the
  nested paths; persisted JSON schema changes accordingly.
- `scripts/analyze_variant.py` — `kept_tubes` path update.
- `notebooks/04-error-analysis.ipynb` — `winner_tube_id` →
  `decision.trigger_tube_id`.
- **Tests:** all `details[...]` assertions updated to nested paths.
  Tests that build fixture details dicts manually are updated
  accordingly. `test_model_parity` is updated to read `logit` from the
  kept-tube list.
- **Inference padding helpers:** `pad_frames_symmetrically` and
  `pad_frames_uniform` return (or expose via a side channel) the
  padded indices, so the model can populate
  `padded_frame_indices` without re-deriving.

## Testing

- Unit tests in `tests/test_details_schema.py` cover:
  - Valid construction round-trips via `model_dump` / `model_validate`.
  - Rejection of invalid states (e.g., `probability` without `logit`,
    malformed bbox).
  - Empty-sequence and no-tubes-kept shapes validate cleanly.
- Existing `tests/test_model_edge_cases.py` and
  `tests/test_model_parity.py` updated to new paths. No new behavior
  is expected; the model's decisions are unchanged, only the shape of
  `details`.
- `tests/test_protocol_eval.py` adapted to the new field paths, with
  fixture details dicts built via the Pydantic model's
  `model_dump()` to keep the test fixtures in sync with the real
  schema.

## Future extensions

- **Dropped candidate tubes (opt-in).** Add `tubes.dropped:
  list[DroppedTube]` behind `infer.include_dropped_tubes: false`.
  Each `DroppedTube` carries `tube_id`, `start_frame`, `end_frame`,
  `num_detected_entries`, a `drop_reason` literal
  (`too_short | too_few_detections`), and optionally `entries`. Useful
  for false-negative forensics; not needed for the default interface.
- **Per-frame detections (opt-in).** If richer debug is needed, add
  `preprocessing.per_frame_detections` under the same debug flag —
  full bboxes + confidences for every YOLO detection, not just counts.
- **Shared typed-details pattern in pyrocore.** If a second experiment
  grows a similarly rich schema, we can lift the pattern to pyrocore
  (generic `TemporalModelOutput[DetailsT]` with a Pydantic bound),
  but we defer that until a second concrete case exists.
