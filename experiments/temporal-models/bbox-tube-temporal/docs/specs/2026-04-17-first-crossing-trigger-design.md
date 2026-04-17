# First-crossing trigger for bbox-tube-temporal

Status: design (not yet implemented)
Date: 2026-04-17

## Goal

Fix the inflated time-to-detection (TTD) produced by
`bbox-tube-temporal`. Today `trigger_frame_index = winner_tube.end_frame`,
so TTD measures the time from sequence start to when the winning tube
*ends*, not when the model *first had enough evidence to fire*. Mean TTD
on val-packaged is ~515s as a result (see
`docs/specs/2026-04-16-bbox-tube-precision-investigation.md`, lines
19-21).

Target: replace the end-frame trigger with a first-crossing trigger
semantically equivalent to "the earliest frame at which the current
aggregation rule would have fired if the sequence had been replayed up
to that frame." Keep `is_positive` bit-identical to today — this PR
moves TTD only, not precision/recall.

## Context

Current pipeline (`src/bbox_tube_temporal/inference.py:242-310`,
`pick_winner_and_trigger`):

```
winner = argmax(full_tube_logits)                # both aggregations
# max_logit aggregation:
is_positive = full_tube_logits[winner] >= threshold
# logistic aggregation:
features = extract_features(winner_dict, n_tubes=len(tubes))
is_positive = calibrator.predict_proba(features) >= logistic_threshold
# trigger (both modes):
trigger = winner.end_frame if is_positive else None   # <-- the bug
```

The model supports two aggregation modes (`max_logit` and `logistic`,
the latter from the 2026-04-17 logistic-calibrator deployment design).
Both share the same winner-selection (`argmax(logits)`) and the same
buggy trigger rule (`winner.end_frame`). The TTD fix must work for
both.

Consumers: `protocol_eval._compute_ttd_seconds`
(`src/bbox_tube_temporal/protocol_eval.py:58-80`) and the leaderboard's
`_compute_ttd` (verbatim mirror) both do
`frames[trigger_frame_index].timestamp - frames[0].timestamp`.

## Key decisions

1. **Prefix scoring (Approach A).** For each candidate tube, score
   progressively longer prefixes and find the smallest prefix length L
   whose classifier logit crosses threshold. The frame at prefix slot
   L-1 is that tube's first-crossing frame.

2. **Earliest-crossing across all qualifying tubes (Scope C2).** Apply
   prefix scoring to every tube whose *full-length features* produce a
   positive decision under the active aggregation mode. Pick the tube
   whose first-crossing `frame_idx` is earliest in the sequence. Ties
   broken by smallest `tube_id` for determinism. Matches the
   `argmax(logits) >= threshold` (or argmax + calibrator) decision
   rule: "any tube crossing" is equivalent to "max crossing" for the
   sequence-level decision.

3. **Full-tube guard (D2).** Only tubes whose *full-length decision*
   is positive under the active aggregation mode are eligible to
   contribute a trigger. Under D2, `is_positive` is strictly preserved
   — a tube that fires on some prefix but not on its full length is
   ignored. Pure TTD relabeling, no precision/recall movement.
   Embracing prefix-only crossings (D1) is a follow-up once D2 numbers
   are in.

   The "full-length decision" is the active aggregation mode applied
   to the full tube:
   - **`max_logit`:** `classifier(tube).logit >= threshold`.
   - **`logistic`:** `calibrator.predict_proba(extract_features(
     tube, n_tubes)) >= logistic_threshold`.

4. **Serial prefix scoring (Approach 1).** For each qualifying tube,
   loop `L = infer_min_tube_length..len(tube.entries)`, run one
   classifier forward on the prefix, early-exit on first crossing.
   Simplest implementation; batched single-pass (Approach 2) is a
   performance follow-up if Approach 1 regresses p95 latency.

5. **Minimum prefix length = `infer_min_tube_length` (= 2).** Matches
   the existing inference-time tube filter. Scoring prefixes of length
   1 is meaningless; the classifier is already fed length-2 tubes at
   inference (acceptable OOD per the 2026-04-15 protocol design, lines
   75-77). No new OOD territory.

6. **No config / package-version changes.** First-crossing is the only
   trigger policy. No `trigger_rule` key in `config.yaml`. No bump to
   `manifest.yaml` `format_version`. The `package` DVC stage rebuilds
   `model.zip` in this PR so the leaderboard picks up the fix.

## Algorithm

Factor the decision rule into a single predicate
`decides_positive(prefix_logit, tube_prefix, n_tubes) -> bool` that
encapsulates the active aggregation mode:

- **`max_logit`:** `prefix_logit >= threshold`.
- **`logistic`:** `calibrator.predict_proba(extract_features(
  tube_prefix, n_tubes)) >= logistic_threshold`.

`n_tubes` is a sequence-global feature so it is constant across all
prefixes and all tubes in a single `predict()` call.

```
Inputs:
  tubes: list[Tube]
  patches_per_tube, masks_per_tube    # as prepared for score_tubes
  full_logits: Tensor[N]              # from one score_tubes(...) call
  decides_positive: (logit, tube_prefix, n_tubes) -> bool
  min_prefix_length: int              # = infer_min_tube_length

1. qualifying = { i | decides_positive(full_logits[i],
                                        tube_prefix = tubes[i] (full),
                                        n_tubes = len(tubes)) }
2. If qualifying is empty:
       return (is_positive=False, trigger=None, winner_tube_id=None,
               diag={})
3. For each i in qualifying:
       For L in [min_prefix_length .. len(tubes[i].entries)]:
           prefix_mask = masks_per_tube[i].clone()
           prefix_mask[L:] = False
           prefix_logit = classifier(patches[i].unsqueeze(0),
                                      prefix_mask.unsqueeze(0))[0]
           tube_prefix = tubes[i] restricted to entries[:L]
           if decides_positive(prefix_logit, tube_prefix, len(tubes)):
               crossing_frame_i = tubes[i].entries[L-1].frame_idx
               record (tube_id=i, crossing_frame=crossing_frame_i,
                       prefix_length=L)
               break
       # D2 guarantees that the full-length prefix (L = len(entries))
       # is itself a valid crossing, so every qualifying tube records
       # at least the trivial L = len(entries) result.
4. Winner = qualifying tube with minimal crossing_frame
            (tie -> smallest tube_id).
5. Return (is_positive=True,
           trigger=winner.crossing_frame,
           winner_tube_id=winner.tube_id,
           diag={tube_id: (crossing_frame, prefix_length), ...})
```

## Components

**Modified:**

- `src/bbox_tube_temporal/inference.py` — replace
  `pick_winner_and_trigger` with `find_first_crossing_trigger`.
  Signature:
  ```python
  def find_first_crossing_trigger(
      *,
      classifier,
      tubes: list[Tube],
      patches_per_tube: list[torch.Tensor],
      masks_per_tube: list[torch.Tensor],
      full_logits: torch.Tensor,
      aggregation: str,                                # "max_logit" | "logistic"
      threshold: float,
      calibrator: LogisticCalibrator | None = None,
      logistic_threshold: float = 0.5,
      min_prefix_length: int,
  ) -> tuple[bool, int | None, int | None, dict]:
      ...
  ```
  Internally builds `decides_positive` from the aggregation args
  (mirroring the current `pick_winner_and_trigger` branches), then
  runs the algorithm above. The trailing `dict` carries per-tube
  first-crossing diagnostics
  (`{tube_id: {"crossing_frame": int, "prefix_length": int}}`) for
  `details{}` population.

  Implementation note: when iterating `L`, at `L = len(tube.entries)`
  the prefix logit equals `full_logits[i]` — reuse it instead of
  re-running the classifier for that one value.

- `src/bbox_tube_temporal/model.py` — at the one call site
  (line 211), swap `pick_winner_and_trigger(...)` for
  `find_first_crossing_trigger(...)` with the same aggregation /
  threshold / calibrator args already in scope plus
  `min_prefix_length=tubes_cfg["infer_min_tube_length"]`.
  Extend `details{}` with the new diagnostics dict under key
  `"per_tube_first_crossing"`. Everything else in the output
  (`TemporalModelOutput` fields) keeps its current shape.

**Untouched:**

- `src/bbox_tube_temporal/tubes.py`, `temporal_classifier.py`,
  `package.py` loader, `protocol_eval.py`.
- Training pipeline (`build_tubes.py`, `build_model_input.py`,
  `lit_temporal.py`, `train.py`).
- Calibration (`scripts/package_model.py` threshold-selection logic).
- DVC stage graph (the `package` stage re-runs as a side effect of
  the code change, which is desired).
- `params.yaml` and the archive's `config.yaml` schema.

## Data flow inside `predict()` (after change)

| Stage | Granularity | Call |
|---|---|---|
| 1. Truncate | whole sequence | unchanged |
| 2. YOLO | one batched call | unchanged |
| 3. Build tubes | sequential | unchanged |
| 4. Crop | per tube | unchanged |
| 5. Classifier (full) | one batched call, all tubes | `score_tubes(...)` — unchanged |
| 6. Aggregate | per qualifying tube, serial prefix scoring | **new** `find_first_crossing_trigger` |

Added cost: one extra classifier forward per `(qualifying tube, prefix
length)` pair until each qualifying tube crosses. Zero extra cost on
sequences with no qualifying tubes. Early exit applies per tube.

## Latency expectations

Dominant cost added per sequence:

- FP sequences with no qualifying tubes: **0ms** extra.
- Typical TP: 1-2 qualifying tubes, crossing at L=3-6, ConvNext-tiny +
  GRU on GPU: **~30-150ms** extra.
- Worst case: 2 qualifying tubes, late crossings at L=15-20: **~300-600ms**
  extra.

Within the latency budget the leaderboard already accepts (YOLO +
classifier already dominate). If p95 regresses ≥ 200ms post-merge,
upgrade to Approach 2 (single batched call over all
`(tube, prefix_length)` pairs) as a drop-in replacement. Both produce
identical trigger frames.

## `TemporalModelOutput.details` additions

```python
{
  ...existing fields...
  "per_tube_first_crossing": {
      tube_id: {
          "crossing_frame": int,     # absolute frame_idx in the sequence
          "prefix_length": int,      # L at which classifier first crossed
      }
      for tube_id in qualifying_tubes
  },
}
```

`winner_tube_id` and `trigger_frame_index` keep their existing meaning;
they're now sourced from the earliest entry of
`per_tube_first_crossing` instead of from `winner.end_frame`.

## Edge cases

| Case | Behavior |
|---|---|
| Empty tubes | `is_positive=False`, `trigger=None`, `winner_tube_id=None` (unchanged) |
| No qualifying tubes (all full-tube decisions negative under the active aggregation) | same as empty tubes (unchanged) |
| One qualifying tube, crosses at L=min_prefix_length | trigger = `tube.entries[min_prefix_length-1].frame_idx` |
| One qualifying tube, no prefix crosses until full length | loop ends at L=len(entries), trigger = last entry's `frame_idx` (guaranteed by D2) |
| Two qualifying tubes with different first-crossings | winner = earliest `frame_idx` |
| Two qualifying tubes tied on `frame_idx` | winner = smallest `tube_id` |
| Qualifying tube with `len(entries) < min_prefix_length` | impossible — filtered upstream by `infer_min_tube_length`; assert the invariant |

## Testing

**New unit tests in `tests/test_inference_units.py`:**

1. `max_logit` mode, stub classifier with logit = monotone function of
   prefix length — verify correct L is found and loop early-exits at
   the crossing.
2. `max_logit` mode, stub classifier returning a fixed per-`(tube_id,
   prefix_length)` logit map — verify earliest-`frame_idx` tiebreak
   across tubes.
3. D2 guard (`max_logit`): three tubes, only one has a full-length
   logit ≥ threshold; stub returns crossing prefixes on the other two.
   Only the qualifying tube contributes a trigger.
4. Tie on `frame_idx`: verify smallest `tube_id` wins.
5. Empty qualifying set: verify `(False, None, None, {})` output.
6. `logistic` mode: stub classifier + stub calibrator whose
   `predict_proba` returns a known function of `(logit, log_len,
   mean_conf, n_tubes)` — verify the qualifying set and crossing
   frames match the predicate, including cases where
   length-dependence of `log_len` flips the decision between
   prefixes.

**Updated tests:**

- `tests/test_model_parity.py` — existing parity assertion (equal
  logits between offline path and `predict()`) stands. Add one new
  assertion: on a fixture sequence, the new trigger frame is ≤ the
  old `end_frame`-based trigger. TTD must not go up.
- `tests/test_model_edge_cases.py` — update the "trigger on winner
  end_frame" row to the new rule.

**What this PR does not verify with unit tests:**

- Numerical drift in sequence-level metrics. That's empirical; see
  validation below.

## Validation (before merge)

Part of the PR description, not a committed artefact:

1. Rebuild `model.zip` via `uv run dvc repro package` for both
   `gru_convnext_finetune` and `vit_dinov2_finetune`.
2. Re-run `evaluate_packaged` for train + val splits on both variants.
3. Report in the PR: mean + median TTD before vs. after (expect large
   drops from ~515s), precision / recall / F1 (expect bit-identical),
   and a sanity check that no TTD increased.

## Out of scope

- Calibration procedure changes (threshold stays as currently
  calibrated).
- Classifier retraining or architecture changes.
- Per-step classifier heads (emitting per-frame logits from GRU
  hidden states). More efficient but couples trigger logic to head
  architecture; separate experiment.
- D1 variant (embrace prefix-only crossings without full-tube guard).
  Follow-up after D2 numbers land.
- Approach 2 (global single-batch prefix scoring). Performance
  follow-up only if p95 regresses.
- Leaderboard code changes. `_compute_ttd` in the leaderboard is
  unchanged; semantics are unchanged from its perspective.
- TTD on FP sequences (already `None`).
- Precision / recall movement. Explicitly out of scope by D2.

## Follow-ups this PR unlocks

- Re-run the leaderboard on the rebuilt `model.zip`; update the
  2026-04-17 leaderboard results entry.
- Evaluate D1 (drop the full-tube guard) as a precision/recall
  experiment once D2 TTD numbers are established.
- Consider promoting first-crossing as the default trigger rule
  convention for future `TemporalModel` subclasses (doc in
  `pyrocore`).
