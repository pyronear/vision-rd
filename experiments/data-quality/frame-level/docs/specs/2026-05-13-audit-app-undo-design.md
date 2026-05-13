# Audit-app undo — design

**Status:** draft
**Date:** 2026-05-13
**Owner:** Arthur

## 1. Background

The frame-level audit app persists reviewer edits aggressively:

- `markDirty()` (`static/app.js:633`) schedules a save 1 s after the
  most recent edit via `scheduleSave()` / `persistSample()`
  (lines 643–668).
- Any navigation (Space, `←`, `→`, queue-card click, jump-to-sequence)
  calls `flushPending()` (line 671) which **synchronously** saves
  before loading the next sample.

As a result, by the time a reviewer realises they made a mistake on
the previous frame, that frame's state has almost always been written
to disk — both the materialise-on-`reviewed` behaviour from
[`2026-05-13-space-prefill-verified-design.md`](2026-05-13-space-prefill-verified-design.md)
and any manual edits.

There is no client-side undo mechanism today.

## 2. Problem

The dominant "oops" pattern in the workflow:

1. Reviewer holds `Space`, batch-approving frames quickly.
2. On frame `N+k`, they realise they Space-d frame `N` too fast — it
   needed an edit (e.g., a missed bbox, or a spurious original that
   should not have been auto-materialised).
3. Today, the only recovery is: navigate manually back to `N`, look at
   what was saved, and patch it up by hand. There is no single action
   that says "take that last save back."

A lightweight "discard unsaved edits" undo would not help here, because
auto-save means there are no unsaved edits to discard.

## 3. Goals

- One keyboard shortcut (`Ctrl+Z` / `Cmd+Z`) that undoes the **most
  recent save**, regardless of which frame it happened on.
- Undo navigates the reviewer to the frame whose save it reverted, so
  they can pick up editing where the mistake was made.
- Multi-step: pressing `Ctrl+Z` repeatedly walks back through recent
  saves (up to 50).
- Reverts all four mutable fields the save writes: `status`,
  `verified_gt`, `spurious_originals`, `note`.
- Reverted state is **persisted** to the server immediately, so the
  next export sees clean data.

## 4. Non-goals

- Redo (`Ctrl+Shift+Z`). YAGNI.
- Per-action granular undo (one bbox draw at a time). YAGNI.
- Undo across browser sessions / page reloads.
- Undo that only touches in-memory state without re-persisting. The
  whole point is that the next export reflects the revert.
- Changes to auto-save semantics on the Python side. (One additive
  server-side change is required — see §5.4: a `DELETE /api/sample`
  endpoint to revert a frame to the unreviewed state. The existing
  POST save path and persistence contract are unchanged.)

## 5. Design

### 5.1 State additions

Two new fields on the client `state` object:

- `state.loadedSnapshot: SampleSnapshot | null` — a deep copy of the
  four mutable fields of the current sample, captured at the moment
  `loadSample()` finishes. Represents "what the server had when we
  arrived here."
- `state.undoStack: Array<{stem: string, snapshot: SampleSnapshot}>` —
  capped at **50**, oldest dropped via `Array.prototype.shift` when the
  cap is exceeded. Each entry is "this frame's pre-edit state at the
  time we last saved over it."

A `SampleSnapshot` is plain JSON:

```js
{
  status: string | null,
  verified_gt: BBox[],
  spurious_originals: BBox[],
  note: string | null,
}
```

`original_gt`, `predictions`, `sequence_neighbors`, `reviewed_at`,
`stem` are not snapshotted — they are server-derived or immutable
within a sample.

### 5.2 Snapshot capture points

**At load.** End of `loadSample()` (`app.js:308–322`), after all the
state assignments and re-renders, set
`state.loadedSnapshot = snapshotOf(state.sample)`.

**After save.** End of `persistSample()` (after the awaited network
call), refresh `state.loadedSnapshot = snapshotOf(state.sample)`. The
"as-loaded baseline" for the next round of edits is now the just-saved
state.

### 5.3 Push on save

Inside `persistSample()`, **before** the network call:

- If `state.loadedSnapshot` is non-null, push
  `{stem: state.sample.stem, snapshot: state.loadedSnapshot}` onto
  `state.undoStack`.
- If `state.undoStack.length > 50`, shift the oldest off.

Skipping the push is opt-in via a parameter:
`persistSample({ recordUndo: true })` is the default; the undo flow
itself calls `persistSample({ recordUndo: false })` to avoid pushing
revert-saves onto the stack.

Each save burst (the 1 s debounced save, or the synchronous
flushPending on navigation) creates **one** stack entry, because
`scheduleSave` clears its timer between keystrokes. Rapid edits
followed by a single quiet period still produce a single save and a
single stack entry — the granularity is "per save round-trip," not
"per keystroke."

### 5.4 Undo flow

`Ctrl+Z` (or `Cmd+Z` on Mac) invokes `undoLastSave()`:

1. If `state.undoStack` is empty → no-op (silent, no toast).
2. Cancel any pending auto-save: `clearTimeout(saveTimer)`,
   `state.dirty = false`. Any in-memory edits the reviewer hasn't yet
   had auto-saved are abandoned (they're not in the undo scope).
3. Pop the top entry `{stem, snapshot}`.
4. If `state.sample?.stem !== stem`, navigate to that frame by calling
   `loadSample(stem, { preserveView: true })`. After load,
   `state.loadedSnapshot` is the just-loaded (post-edit) baseline, but
   we are about to overwrite it.
5. Apply `snapshot` onto `state.sample`'s four mutable fields.
6. Set `state.loadedSnapshot = snapshotOf(state.sample)` to reflect
   the reverted baseline.
7. **If `snapshot.status === null`** (the frame was unreviewed when we
   first loaded it), the persistence layer cannot represent that as a
   save — `review.json` only stores entries with `status ∈
   {reviewed, unclear}`. Instead, call
   `DELETE /api/sample?model&split&stem` to remove the entry from
   `review.json`, clear `state.sample.reviewed_at`, update the queue
   card's `status` field, refresh the save-bar, and re-render the
   queue/progress. Set `state.dirty = false`.
8. **Otherwise** (status is `"reviewed"` or `"unclear"`): set
   `state.dirty = true` and call
   `await persistSample({ recordUndo: false })` — the revert is
   persisted immediately, not via the 1 s debounce.
9. Re-render: `paint(); renderRight();`.

### 5.5 Stack cap

Hard cap at 50. Oldest entry is dropped silently. If a reviewer makes
more than 50 saves in a session, undo can only reach the last 50.
This is enough to recover from a fast-typed mistake without growing
memory unboundedly.

## 6. Edge cases

- **Empty stack on Ctrl+Z.** No-op. No toast, no error. The reviewer
  has nothing to undo.
- **Unsaved (dirty) edits on the current frame.** Discarded silently
  before the pop. Rationale: those edits were never saved, so they
  aren't on the stack; the user pressed Ctrl+Z meaning "undo a save,"
  not "save first."
- **Note field focused.** The existing `keydown` guard
  (`if (e.target.matches('input, textarea, select')) return;`,
  line 821) lets the browser's native `Ctrl+Z` handle textarea undo.
  Reviewers must blur the field to invoke app-level undo. This is the
  consistent pattern for every other shortcut in the app.
- **Repeated saves on the same frame.** Each save pushes its own
  entry. Multi-step undo on a single frame works correctly: every
  `Ctrl+Z` walks one save further back. (In practice, debounced saves
  collapse rapid edits into one entry — see §5.3.)
- **Stem in stack not present in queue.** Defensive: if
  `loadSample(stem)` fails (e.g., model/split changed during session),
  the popped entry is silently dropped and the function returns. The
  reviewer can try `Ctrl+Z` again to pop the next entry.
- **`reviewed_at` after undo.** Server sets this on every save. After
  undo + re-save, `reviewed_at` reflects the time of the undo. This
  is accurate ("last time this frame's state changed on disk") and
  not used in export decisions, so no special handling is needed.
- **Concurrent saves.** Existing code does not lock around
  `persistSample`. Undo follows the same pattern: it awaits its own
  save and trusts JS's single-threaded event loop. If a debounced
  save fires concurrently with Ctrl+Z, the `state.dirty = false` /
  `clearTimeout(saveTimer)` in step 2 prevents a race.

## 7. UI

- **Keyboard:** `Ctrl+Z` and `Cmd+Z` only. No on-screen button.
- **Help-pane:** new row under the Navigation section in
  `static/index.html` (~line 227):

  ```html
  <tr><td class="px-4 py-1 whitespace-nowrap"><kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Z</kbd></td><td class="px-4 py-1 text-slate-600">Undo last save (revert + jump to that frame)</td></tr>
  ```

- **Save bar:** no change. The existing "unsaved…" / "✓ saved at …"
  indicator is sufficient. We do not add a visible undo-stack counter.

## 8. Implementation

Single file: `src/data_quality_frame_level/audit_app/static/app.js`.
Plus the help-pane row in `static/index.html`.

### 8.1 Helper additions (~line 350, alongside the bbox helpers)

```js
function snapshotOf(sample) {
  return {
    status: sample.status ?? null,
    verified_gt: sample.verified_gt.map(bboxCopy),
    spurious_originals: (sample.spurious_originals || []).map(bboxCopy),
    note: sample.note ?? null,
  };
}

function applySnapshot(sample, snapshot) {
  sample.status = snapshot.status;
  sample.verified_gt = snapshot.verified_gt.map(bboxCopy);
  sample.spurious_originals = snapshot.spurious_originals.map(bboxCopy);
  sample.note = snapshot.note;
}
```

### 8.2 State init

Add two fields to the `state` object literal at the top of `app.js`
(currently lines 1–10):

```js
const state = {
  /* …existing fields… */
  dirty: false,
  loadedSnapshot: null,
  undoStack: [],
};
```

### 8.3 `loadSample` — capture snapshot at end

`app.js:308–322`, after the existing `renderTimeline();`:

```js
  state.loadedSnapshot = snapshotOf(state.sample);
```

### 8.4 `persistSample` — push undo entry + refresh snapshot

```js
async function persistSample(options = {}) {
  if (!state.dirty || !state.sample || !state.sample.status) return;
  if (options.recordUndo !== false && state.loadedSnapshot) {
    state.undoStack.push({ stem: state.sample.stem, snapshot: state.loadedSnapshot });
    if (state.undoStack.length > 50) state.undoStack.shift();
  }
  const r = await api.save({ /* unchanged */ });
  state.dirty = false;
  state.sample.reviewed_at = r.saved_at;
  state.loadedSnapshot = snapshotOf(state.sample);
  setSaveBar();
  /* …existing queue/progress updates… */
}
```

### 8.5 `undoLastSave`

New function near other navigation helpers (~line 850):

```js
async function undoLastSave() {
  if (state.undoStack.length === 0) return;
  clearTimeout(saveTimer);
  state.dirty = false;
  const { stem, snapshot } = state.undoStack.pop();
  if (state.sample?.stem !== stem) {
    try {
      await loadSample(stem, { preserveView: true });
    } catch {
      return;
    }
  }
  applySnapshot(state.sample, snapshot);
  state.loadedSnapshot = snapshotOf(state.sample);
  state.dirty = true;
  await persistSample({ recordUndo: false });
  paint();
  renderRight();
}
```

### 8.6 Keybinding

Inside the existing window `keydown` listener (~line 820), add **before**
the other key handlers so it can't be shadowed:

```js
  if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
    e.preventDefault();
    return undoLastSave();
  }
```

The handler already returns early when focus is in
`input, textarea, select`, so textarea undo works natively.

## 9. Test plan

Manual against a running audit-app instance. No JS test harness in
this project.

- **A. Single-frame undo.** On a fresh frame, draw a verified bbox.
  Wait >1 s for the debounced save (save-bar reads "✓ saved at …").
  Press `Ctrl+Z`. Bbox is gone in the UI; save-bar updates to a fresh
  "saved at" timestamp. Reload the page → bbox is still gone (server
  reflects the revert).

- **B. Undo a Space-press.** On frame `N` with originals, press
  `Space`. The app advances to `N+1`. Press `Ctrl+Z`. The app
  navigates back to `N`; `verified_gt` is empty again; status is
  whatever it was before `Space`. The server reflects this.

- **C. Multi-step.** Press `Space` three times in a row across frames
  N, N+1, N+2. You are on N+3. Press `Ctrl+Z` three times. After
  each press you are one frame further back, with that frame's
  pre-Space state restored on the server.

- **D. Empty stack.** Reload the page. Press `Ctrl+Z`. Nothing
  happens; no error in console.

- **E. Discard pending.** On a fresh frame, draw a bbox but do **not**
  wait for the 1 s debounce. Press `Ctrl+Z` quickly. The bbox is
  discarded; the app navigates back if the stack has an entry, else
  stays put.

- **F. Note field focus.** Click into the note field, type some
  text, press `Ctrl+Z`. The textarea's own undo fires; no app-level
  navigation happens.

- **G. Stack cap.** Edit + save 51 distinct frames. Press `Ctrl+Z` 51
  times. The 51st undo finds an empty stack and is a no-op (you have
  recovered the last 50 saves, but not the first).

- **H. Cross-session.** Reload the page after some edits. The undo
  stack is empty (state lives in JS memory only). `Ctrl+Z` does
  nothing. This is the documented limitation.

## 10. Out of scope

- Redo, per-action granular undo, server-side undo history,
  persistence of the undo stack across page reloads, on-screen undo
  button or stack visualisation.
- Changes to auto-save debounce, `flushPending`, or the Python
  persistence layer.
- Touching `export.py` — the export reads whatever the persistence
  layer has, and undo writes through that same layer, so no export
  changes are needed.
