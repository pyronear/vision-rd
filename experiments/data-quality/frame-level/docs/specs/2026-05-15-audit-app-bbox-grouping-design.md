# Audit-app bbox grouping — design

**Status:** draft
**Date:** 2026-05-15
**Owner:** Arthur

## 1. Background

The frame-level audit app renders a "Bboxes (this frame)" list in the
right pane (`static/index.html:252-253`, `static/app.js:759 renderRight`).
The list contains three row types, in this fixed order:

1. `original_gt` rows that are not already in `verified_gt` (kept or
   spurious).
2. `predictions` rows.
3. `verified_gt` rows.

Each row carries `data-layer` + `data-idx` so the hover/click handlers
(`app.js:817-873`) can find the underlying bbox in `state.sample`.

Predictions today are listed in the order the model emits them — which
is typically a confidence-descending stream, but several near-duplicate
detections at the same spatial location are interleaved with detections
from other locations once multiple smoke regions are present in a
frame. The reviewer has to scan the list and mentally re-cluster.

## 2. Problem

When the model emits many overlapping predictions for the same smoke
location, the right-pane list becomes hard to read:

- It is not obvious which predictions are duplicates of each other vs.
  distinct detections.
- The top-confidence detection for a given location is not visually
  privileged.
- The GT rows for that same location live in a separate block higher up
  the list, so the reviewer can't see "everything at this spot" in one
  place.

## 3. Goals

- Group rows in the right pane by spatial overlap so that GT bboxes and
  predictions for the same smoke region sit together.
- Within each group, order rows so the most authoritative anchor is on
  top and predictions descend by confidence.
- Keep all current row content, action buttons, and hover/selection
  behaviour intact — this is a re-ordering + visual-wrapping change,
  not a behaviour change.

## 4. Non-goals

- No new server-side data, no new API field. Clustering happens
  client-side in `renderRight()`.
- No new user-facing knob for the grouping IoU. A fixed threshold is
  good enough; tuning can come later if needed.
- The on-canvas rendering (`renderCanvas`) is unchanged — only the
  right-pane list is reorganised.

## 5. Design

### 5.1 What gets clustered

The pool of bboxes fed into clustering is exactly the set of rows that
`renderRight()` would emit today, in the same filtered form:

- `verified_gt[i]` — all of them.
- `original_gt[i]` for `i` such that `b` is not in `verified_gt`
  (existing `isInVerified(b)` filter at `app.js:788`). Spurious status
  is preserved as a per-row attribute, not as a filter.
- `predictions[i]` — all of them. Predictions are already pre-filtered
  by `state.conf` upstream (server-side at request time), so we do not
  re-filter here.

The drop-warning banner (`app.js:780-786`) is not a bbox row and stays
above the group list untouched.

### 5.2 Clustering algorithm

Union-find over the pool:

1. Assign each bbox an integer id (its index in the pooled array).
2. For each pair `(i, j)` with `i < j`, merge them if **either** of the
   following holds:
   - `bboxIou(boxes[i], boxes[j]) >= 0.3` (significant overlap), **or**
   - `bboxIoMin(boxes[i], boxes[j]) >= 0.6` (one box is mostly
     contained in the other — Intersection over Smaller).
3. Read off connected components.

The containment check is needed because IoU drops sharply when bbox
sizes differ: a small high-confidence pred sitting inside a much larger
GT will have IoU well below 0.3 even though it is, semantically, the
same detection. `IoMin = intersection / min(area_a, area_b)` is
size-invariant — at 1.0 the smaller box is fully inside the larger.

Pool size is small (typically <20 rows per frame, hard bound a few
dozen), so O(N²) pair scan is fine. The existing `bboxIou` helper at
`app.js:426-432` is reused; a new sibling helper `bboxIoMin` is added
next to it.

Module-level constants:

- `BBOX_GROUP_IOU = 0.3`
- `BBOX_GROUP_IOMIN = 0.6`

`clusterBboxes` takes a `shouldMerge(a, b)` predicate rather than a
fixed threshold, so the OR composition lives in one readable place at
the call site:

```javascript
clusterBboxes(
  pool.map(it => it.bbox),
  (a, b) => bboxIou(a, b) >= BBOX_GROUP_IOU
         || bboxIoMin(a, b) >= BBOX_GROUP_IOMIN,
);
```

### 5.3 Within-group order

Inside each group, rows are emitted in this order:

1. `verified` rows (in their natural `verified_gt` index order).
2. `original_gt` **kept** rows.
3. `original_gt` **spurious** rows.
4. `predictions` sorted by `conf` descending.

This matches the existing semantic hierarchy: locked-in ground truth
first, then originals (kept above flagged), then predictions ranked by
the model's confidence.

### 5.4 Group order

Each group gets a `groupScore`:

- `1.0` if the group contains any `verified` or any `original_gt` row
  (any GT anchors the group as confirmed-existing).
- Otherwise the max `conf` across the group's predictions.

Groups are sorted by `groupScore` descending. Ties break on group size
descending, then on the pool index of the group's first member
(insertion order) for stability.

### 5.5 Rendering

- **Singleton group (1 member):** render as a bare row, identical to
  today's markup. No frame, no header. This keeps the visual noise low
  when many isolated detections exist.
- **Multi-member group (≥2 members):** wrap the rows in a framed
  block. The header reads `Group N · K` where `N` is a 1-based group
  index over multi-member groups only, and `K` is the member count.

Markup sketch (multi-member case):

```html
<div class="bbox-group">
  <div class="bbox-group-header">Group 1 · 5</div>
  <div class="bbox-row verified" data-layer="verified" data-idx="0">…</div>
  <div class="bbox-row orig"      data-layer="orig"     data-idx="0">…</div>
  <div class="bbox-row pred"      data-layer="pred"     data-idx="3">…</div>
  …
</div>
```

The existing `.bbox-row` CSS classes (`index.html:54-64`) are untouched.
Two new classes are added to the Tailwind `@layer components` block in
`index.html`:

```css
.bbox-group        { @apply mb-2 rounded-md border border-slate-200 bg-white/40 p-1; }
.bbox-group-header { @apply mb-1 px-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500; }
```

The wrapper element gives the visual frame; rows inside keep their
existing left-border accent so the GT/pred distinction remains
recognisable.

### 5.6 Event delegation

The existing handlers at `app.js:817 / 826 / 835` attach to
`#bbox-list` and use `e.target.closest('.bbox-row')` to find the row,
then read `row.dataset.layer` and `row.dataset.idx`. Because each row
keeps the same `data-layer` / `data-idx` attributes, **no handler
changes are required**. The wrapper `<div class="bbox-group">` is
ignored by `closest('.bbox-row')`.

## 6. Edge cases

- **Empty frame** (`verified_gt`, `original_gt`, `predictions` all
  empty): list renders empty, identical to today.
- **All singletons** (no two boxes overlap at IoU ≥ 0.3): each row
  renders bare; visible difference vs. today is the reorder — verified
  rows first, then originals (kept, spurious), then predictions desc by
  conf.
- **Drop-warning banner present:** stays above the group list.
- **`verified_gt` contains a copy of an `original_gt`** (the
  `promote-orig` action): the original is filtered out of the pool by
  the existing `isInVerified(b)` check; the verified copy enters
  clustering alone.
- **Pred coincides exactly with a verified box** (`promote-pred`): both
  land in the same cluster (IoU = 1.0) — verified row sits above the
  pred row, as designed.

## 7. Out of scope / future work

- Exposing the group-IoU threshold as a slider.
- Allowing the user to manually merge / split clusters.
- Mirroring the cluster grouping on the canvas (e.g., dimming
  non-hovered groups). The canvas renderer stays one-bbox-per-row.

## 8. Files touched

- `src/data_quality_frame_level/audit_app/static/app.js`
  - New `clusterBboxes(boxes, threshold)` helper near `bboxIou`
    (around line 432).
  - Rewrite the bbox-emitting section of `renderRight()` (currently
    `app.js:780-807`) to pool rows, cluster them, and emit grouped
    markup. The drop-warning banner block and the status-button / note
    block stay as they are today.
- `src/data_quality_frame_level/audit_app/static/index.html`
  - Add `.bbox-group` and `.bbox-group-header` CSS classes inside the
    existing `@layer components` block.

No server-side or persistence changes.
