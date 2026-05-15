# Audit-app containment matching — design

**Status:** draft
**Date:** 2026-05-15
**Owner:** Arthur

## 1. Background

The frame-level audit app surfaces candidate label errors by running
the production YOLO detector against the dataset GT, classifying each
prediction as TP / FP / FN, and queuing frames that contain at least
one FP or FN.

The classifier today lives in
`src/data_quality_frame_level/audit_app/matching.py`. It is a
single-class greedy IoU matcher: all `(gt, pred)` pairs are sorted by
descending IoU and walked in order; each GT and each pred can be
matched at most once. Unmatched preds become FP, unmatched GT becomes
FN.

The per-frame severity used by the queue (`audit_app/queue.py`) is the
max confidence among preds labelled `fp` (filtered by
`review_conf_thresh`) or the max area among GTs labelled `fn`, and
sequences are ordered by their max severity.

## 2. Problem

On the train split — by far the largest of the three — the FP queue is
dominated by frames where the model emitted 2–3 overlapping boxes for
a single real smoke plume. Under greedy-1-to-1 IoU matching:

- The highest-IoU pred claims the GT and is labelled TP.
- The remaining preds for the same plume cannot match (the GT is
  already taken) and are labelled FP.
- The frame surfaces in the FP queue even though the model's detection
  is semantically correct.

A second, related shortcoming: a tight pred sitting inside a generous
GT label has low IoU (large union, small intersection) and so falls
below `iou_thresh`. It becomes FP despite being squarely on the smoke.

These two patterns inflate the train queue with frames that have no
actionable label-quality issue, increasing reviewer fatigue and time
to completion.

## 3. Goals

- Stop surfacing frames whose only FPs are duplicate or tight-inside
  predictions over a real GT.
- Keep surfacing frames with genuinely mislocated predictions (preds
  that overlap nothing real) and genuine misses (GTs with no matching
  pred).
- Make the change a tunable knob in the UI so a reviewer can dial it
  back per session if needed.
- Leave persisted review state and past exports untouched.

## 4. Non-goals

- No new diagnostic view for oversized GT labels. (Discussed; deferred.)
- No change to FN semantics. A GT is FN iff no pred agrees with it.
- No change to the export flow (`audit_app/export.py`, `data/10_export/`)
  or the review state schema (`review.json`).
- No change to predictions on disk (`data/07_model_output/`); only the
  in-memory classification of those predictions changes.

## 5. Design

### 5.1 New matcher: many-to-one + optional containment

Replace the greedy 1-to-1 matcher in
`audit_app/matching.py:evaluate_frame` with an independent-decision
matcher. For each `(gt_i, pred_j)` pair, define an `agrees` predicate:

```
agrees(gt_i, pred_j) :=
    IoU(gt_i, pred_j) >= iou_thresh
 OR (containment_thresh is not None
     AND IoP(pred_j, gt_i) >= containment_thresh)
```

where `IoP(pred, gt) = intersection_area / pred_area` — "how much of
the prediction sits inside this GT box."

Classification:

- `pred_j` is TP iff `any(agrees(gt_i, pred_j) for gt_i in gt)`,
  else FP.
- `gt_i` is TP iff `any(agrees(gt_i, pred_j) for pred_j in
  predictions)`, else FN.

The `matches` field of `EvaluatedFrame` (currently a list of
greedy-matched `(gi, pj, iou)` tuples) becomes, for each TP-GT, the
single best-IoU pred that agrees with it. This is internal — no
consumer outside `matching.py` reads it today — but keeping the field
preserves its usefulness for future visual association in the right
pane.

### 5.2 Default containment threshold

`containment_thresh = 0.7` is the default. Rationale:

- Literature-standard threshold for containment-based matching.
- Strict enough that a pred drifting half outside the GT stays an FP.
- Lenient enough to absorb the typical smoke pattern: tight pred
  inside a generous plume label.

### 5.3 Plumbing

| File | Change |
|---|---|
| `audit_app/matching.py` | Add `iop()` helper. Rewrite `evaluate_frame` per 5.1. New keyword `containment_thresh: float \| None = None`. Update module docstring. |
| `audit_app/queue.py` | Thread `containment_thresh` through `build_queue` and `_frame_severity`. Default `0.7`. |
| `audit_app/main.py` | Add `containment: float \| None = 0.7` query parameter to `GET /api/queue` and `GET /api/sample`. Pass through to `build_queue` and `evaluate_frame`. |
| `audit_app/static/index.html` | Add a fourth row in the filter panel beside `conf`, `iou`, `review`, labelled `contain ≥`, range `0–1`, step `0.05`, `value="0.7"`. |
| `audit_app/static/app.js` | Read the new slider in the queue and sample fetch builders; include it in the `filter-reset` defaults. When the slider reads `0`, send `containment=null` to the API. |
| `tests/test_matching.py` | Add focused unit tests (see 5.5). |

### 5.4 Slider semantics

Range 0.0–1.0, step 0.05, initial value 0.7. A value of `0` is the
"off" sentinel: the UI sends `containment=null` and the server matches
on IoU only. Any value `> 0` engages containment with that
threshold. Default reset returns the slider to 0.7.

### 5.5 Tests

Four focused unit tests in `tests/test_matching.py`:

1. **Duplicate preds over one GT** — 3 preds with high overlap on a
   single GT. Expect all 3 preds TP, GT TP, no FP, no FN.
2. **Tight pred inside generous GT (low IoU, high IoP)** — one pred
   that meets `containment_thresh` but not `iou_thresh`. Expect TP.
3. **Giant pred over two GTs (low IoU, low IoP)** — one large pred
   covering two small adjacent GTs, where IoU and IoP both fall below
   their thresholds against each GT. Expect pred FP, both GTs FN.
4. **`containment_thresh=None`** — pure-IoU many-to-one fallback. A
   tight pred with low IoU is FP; duplicate preds passing `iou_thresh`
   are all TP.

## 6. Risks

1. **Giant-pred edge case.** A single huge pred covering two adjacent
   GTs flips from `TP+FN` under the old matcher to `FP+FN+FN` under
   the new one. The frame is still surfaced via the FNs in both
   regimes; only the per-box labels differ. Accepted.
2. **Threshold calibration.** `0.7` is chosen from literature, not
   from data. If reviewers find too many real FPs are absorbed, the
   slider rejects the default per session; if it is systematically
   wrong, the constant lives in three places (`queue.py`, `main.py`,
   `index.html`). No data risk — only UX.
3. **Default duplicated across client and server.** `index.html`
   `value="0.7"` and the `main.py` query-param default both hardcode
   the value. This matches the existing pattern for `conf`, `iou`, and
   `review_conf`, so the convention is consistent, but it is a sharp
   edge to be aware of when changing the default later.

## 7. Migration / compatibility

None required.

- `review.json` is keyed by stem and stores reviewer decisions, not
  matching outcomes. It is unaffected by this change.
- `data/10_export/<model>/<split>/` is derived from
  `review.samples` and `originals` via
  `audit_app/export.py:write_manifest_and_labels`; the matcher is not
  in that path. Past exports remain valid.
- The audit queue is recomputed per request, so the next page load
  reflects the new matcher with no migration step.

## 8. Acceptance criteria

- Opening the train split with default thresholds shows a visibly
  smaller FP queue than today; frames whose only FPs were duplicate
  boxes on a real GT are absent.
- Moving the containment slider to `0.0` falls back to many-to-one
  IoU-only matching, isolating the containment effect.
- New unit tests in `tests/test_matching.py` cover the four scenarios
  in 5.5 and pass.
- Existing tests still pass.
- `git status` after a fresh review session shows changes only under
  `data/09_review/` (as today) — no untracked changes under
  `data/10_export/` or other data directories that this change
  was not intended to touch.
