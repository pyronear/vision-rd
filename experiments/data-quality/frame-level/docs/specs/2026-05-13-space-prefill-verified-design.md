# Pre-fill `verified_gt` on transition to `reviewed` — design

**Status:** draft
**Date:** 2026-05-13
**Owner:** Arthur

## 1. Background

The frame-level audit app
([`2026-05-05-review-app-design.md`](2026-05-05-review-app-design.md))
lets reviewers approve or correct YOLO ground-truth bboxes frame by
frame. The right-hand pane holds two parallel lists of bboxes:

- `original_gt` — the bboxes shipped with `pyro-dataset`.
- `verified_gt` — the reviewer's authoritative list (initially empty).

The export contract (`src/data_quality_frame_level/audit_app/export.py`,
lines 106–115) handles the two list states differently:

```python
if sample.bboxes:            # verified_gt non-empty → use it
    effective = sample.bboxes
else:                        # legacy path: originals minus spurious
    effective = [o for o in original if not matched-spurious]
```

So today, **`status='reviewed'` + empty `verified_gt`** is an implicit
"originals are accepted as-is" path.

The `Space` keyboard shortcut (and `r` key, and "Mark reviewed" button)
sets `status='reviewed'` without touching `verified_gt`. This relies on
the implicit path above for the common case where the originals are
already correct.

## 2. Problem

The implicit path causes friction when a reviewer Space-approves a
frame, then notices a missed bbox and wants to draw it:

1. Reviewer presses `Space` → status `reviewed`, `verified_gt = []`.
2. Reviewer draws a new bbox `B` → `verified_gt = [B]`.
3. Because `verified_gt` is now non-empty, the export path switches:
   every `original_gt` bbox not also in `verified_gt` becomes a
   "dropped" original. An amber banner appears warning that
   `N` originals will be dropped on save.
4. Reviewer must click **Keep all** to also retain the originals — a
   step that is easy to forget and produces wrong exports when missed.

The fix is to make the "reviewed" transition explicit: copy the
originals into `verified_gt` so adding a missed bbox is a one-action
move.

## 3. Goals

- Reduce friction for the "approve + add a missed bbox" flow.
- Keep behavior consistent across all "mark reviewed" entry points
  (Space, `r` key, "Mark reviewed" button).
- Preserve existing spurious markings.
- No change to the export format or contract.
- No change to already-saved review state on disk.

## 4. Non-goals

- Changing `export.py` logic. The legacy `empty verified ⇒ originals`
  branch stays in place for backward compatibility with saved samples.
- Changing the keyboard map. `Space` still advances; `r` still stays.
- Changing the semantics of `unclear`, `skipped`, or other statuses.
- Auto-filling on `unclear` or any non-`reviewed` transition.

## 5. Behavior

When the sample's status transitions to `'reviewed'` from any entry
point — `Space` shortcut, `r` shortcut, or the "Mark reviewed" button:

1. **If `verified_gt` is empty:** copy each bbox from `original_gt`
   into `verified_gt` (deep copy via existing `bboxCopy`), **excluding**
   bboxes present in `spurious_originals` (matched by the existing
   `bboxClose` / `containsBbox` helpers).
2. **If `verified_gt` is non-empty:** leave it alone. The reviewer has
   already curated it and we don't second-guess that.
3. Set `status = 'reviewed'`, `markDirty()`, re-render the right pane
   and repaint the canvas.
4. Entry-point–specific behavior is preserved:
   - `Space` continues to advance to the next frame after marking.
   - `r` and the button stay on the current frame.

The same materialization runs on every transition into `'reviewed'`,
not only the first one. Going `reviewed → unclear → reviewed` with an
empty `verified_gt` at that moment re-runs the fill. This keeps the
rule simple and matches the user's mental model ("marking reviewed
means: accept the originals I see").

`spurious_originals` is never modified by this code path.

## 6. Why this is safe at export time

The export contract in `export.py:106–115` continues to work because:

- After the fill, `verified_gt = [copies of non-spurious originals]`,
  identical geometry to the originals.
- `compute_diff()` (`export.py:51`) matches `original` to `corrected`
  by IoU and treats matches with IoU ≥ `UNCHANGED_IOU = 0.95` as
  "unchanged" — so identical copies do not appear as `added` or
  `modified`.
- A frame that previously produced no diff entry (status `reviewed`,
  empty `verified_gt`, no spurious) still produces no diff entry after
  the change.
- A frame that previously produced a diff entry (e.g. user drew a new
  bbox after Space) still produces the same diff entry — except the
  reviewer no longer has to click "Keep all" first.

The legacy `else` branch is retained so existing review state on disk
with empty `verified_gt` continues to round-trip correctly.

## 7. Implementation

Single file: `src/data_quality_frame_level/audit_app/static/app.js`.

1. **New helper** alongside the bbox helpers (~line 350):

   ```js
   function materializeVerifiedFromOriginals(sample) {
     if (sample.verified_gt.length > 0) return;
     const sp = sample.spurious_originals || [];
     for (const o of sample.original_gt) {
       if (!containsBbox(sp, o)) sample.verified_gt.push(bboxCopy(o));
     }
   }
   ```

2. **Update `setStatus(s)`** (line 925) — call the helper before
   setting status when `s === 'reviewed'`, and add `paint()` so the
   canvas reflects the new verified rows immediately:

   ```js
   function setStatus(s) {
     if (!state.sample) return;
     if (s === 'reviewed') materializeVerifiedFromOriginals(state.sample);
     state.sample.status = s;
     markDirty();
     renderRight();
     paint();
   }
   ```

3. **Simplify `setStatusAndAdvance`** to reuse `setStatus`:

   ```js
   async function setStatusAndAdvance(s) {
     setStatus(s);
     await seqStep(+1);
   }
   ```

4. **Route the "Mark reviewed/unclear" button** through `setStatus` so
   all three entry points share one funnel. In `renderRight()`
   (line 718–724):

   ```js
   document.querySelectorAll('#status-pane button[data-status]').forEach(btn => {
     btn.onclick = () => setStatus(btn.dataset.status);
   });
   ```

5. **Update help-pane text** in `static/index.html` (line 229 and the
   `r` row at line 230) to indicate that "mark reviewed" accepts the
   originals as GT. Suggested phrasing:
   - `Space` row: "Mark *reviewed* (accept originals as GT) + advance"
   - `r` row: "Mark *reviewed* (accept originals as GT), stay"

## 8. Test plan

JS-only change, no automated frontend test harness. Verification is
manual against a local audit-app instance.

- **A. Clean approve.** Frame with original bboxes, no edits. Press
  `Space`. Navigate back. Right pane shows the originals as verified
  rows. No "will be dropped" banner.
- **B. Add missed bbox.** Frame A. Press `Space`. Navigate back. Draw a
  new bbox `B`. Right pane shows originals + `B` as verified. No
  banner. Export contains all originals + `B`.
- **C. Spurious is respected.** Mark one of two originals as spurious.
  Press `r`. Verified contains only the non-spurious original.
  Spurious original remains spurious.
- **D. User edits preserved.** Manually promote only one of two
  originals into verified. Press `Space`. `verified_gt` unchanged
  (still the single promoted bbox). The other original is still
  flagged as "will be dropped" — same as today.
- **E. Toggle re-fills.** Press `Space` (reviewed). Toggle to
  `unclear`. Clear verified (by removing the rows). Toggle back to
  `reviewed`. Originals are re-materialized.
- **F. Backward compat.** Open an existing review state with samples
  saved as `status='reviewed', verified_gt=[]`. No materialization
  happens on load. Re-pressing `Space` on such a frame populates
  `verified_gt` (and marks dirty); navigating away without further
  edits and re-exporting produces the same diff as before.
- **G. Button parity.** Click the "Mark reviewed" button in the right
  pane on a fresh frame. Verified rows appear. Same outcome as
  pressing `r`.

## 9. Out of scope

- Changes to `export.py`.
- Changes to keyboard map or status taxonomy.
- A separate "explicit accept" UI affordance — the existing
  `Space` / `r` / button entry points are reused.
- Server-side changes. The fill is purely client-side; the API contract
  in `audit_app/main.py` is unchanged.
