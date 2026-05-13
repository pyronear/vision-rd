# Audit-app undo (Ctrl+Z) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `Ctrl+Z` / `Cmd+Z` shortcut that reverts the most recent save: pops a pre-edit snapshot off a per-session stack, navigates to that frame, restores its state, and persists the revert.

**Architecture:** Single client-side change in `static/app.js`. Two new fields on the existing `state` object (`loadedSnapshot`, `undoStack`); a pair of pure helpers (`snapshotOf`, `applySnapshot`); a hook in `loadSample` to capture the as-loaded baseline; a hook in `persistSample` to push the pre-edit snapshot **before** the network call; and a new `undoLastSave` function bound to `Ctrl+Z`. Help-pane row added to `static/index.html`.

**Tech Stack:** Vanilla JS ES modules (no JS test harness in this project). Python persistence layer is **unchanged** — undo writes through the same save endpoint and the export reads whatever is on disk.

**Spec:** [`docs/specs/2026-05-13-audit-app-undo-design.md`](../specs/2026-05-13-audit-app-undo-design.md)

---

## Pre-flight context (read before starting)

- App: `experiments/data-quality/frame-level/` (a uv sub-project). All commands below are run from this directory unless stated otherwise.
- Frontend entry point: `src/data_quality_frame_level/audit_app/static/app.js` (single file). It is loaded by `static/index.html` as an ES module.
- There is **no JS test framework** in this project. We do not introduce one. Verification is `node --check` for syntax + manual testing against the running app via `make audit-app`.
- Relevant existing code locations (confirmed at plan time):
  - `state` declaration: `app.js:1–10`.
  - `loadSample`: `app.js:308–322`.
  - `setSaveBar`: `app.js:324–329`.
  - `scheduleSave` / `saveTimer`: `app.js:643–647`.
  - `persistSample`: `app.js:649–669`.
  - `flushPending`: `app.js:671–674`.
  - Main `window.keydown` handler: `app.js:823–853`.
  - Existing bbox helpers (`bboxCopy`, etc.): `app.js:344–350`.
- Server (Python) launch command for manual testing: `make audit-app` (serves on `http://localhost:8000`).

---

## File map

- **Modify:** `src/data_quality_frame_level/audit_app/static/app.js`
  - Add two fields to the `state` object literal at the top.
  - Add `snapshotOf` and `applySnapshot` helpers (~line 350, alongside other bbox helpers).
  - Append one line to `loadSample` to capture the as-loaded snapshot.
  - Modify `persistSample` to accept `options = {}`, push the pre-edit snapshot onto `undoStack` (when `options.recordUndo !== false`), and refresh `state.loadedSnapshot` after the awaited save.
  - Add new `undoLastSave` async function (near other navigation helpers ~line 855).
  - Add a `Ctrl+Z` / `Cmd+Z` branch to the main `window.keydown` handler.
- **Modify:** `src/data_quality_frame_level/audit_app/static/index.html`
  - Add a Navigation row for the new shortcut.
- **No changes:** Python source, tests, `dvc.yaml`, `params.yaml`.

---

## Task 1: Add state fields and helper functions

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/app.js` (lines 1–10 and ~350)

- [ ] **Step 1: Confirm the current state declaration**

Open `app.js` and confirm lines 1–10 read:

```js
const state = {
  model: null, split: null,
  view: 'fp',
  conf: 0.05, iou: 0.05, reviewConf: 0.35,
  showOrig: true, showPred: true, showOnlyVerified: false,
  reviewer: localStorage.getItem('reviewer') || '',
  queue: [], queueIndex: -1,
  sample: null,
  dirty: false,
};
```

- [ ] **Step 2: Add `loadedSnapshot` and `undoStack` to the state literal**

Replace the block above with:

```js
const state = {
  model: null, split: null,
  view: 'fp',
  conf: 0.05, iou: 0.05, reviewConf: 0.35,
  showOrig: true, showPred: true, showOnlyVerified: false,
  reviewer: localStorage.getItem('reviewer') || '',
  queue: [], queueIndex: -1,
  sample: null,
  dirty: false,
  loadedSnapshot: null,
  undoStack: [],
};
```

- [ ] **Step 3: Confirm the bbox helper block**

Confirm `app.js:344–350` read:

```js
function bboxClose(a, b) {
  return Math.abs(a.cx - b.cx) < 1e-6 && Math.abs(a.cy - b.cy) < 1e-6
      && Math.abs(a.w - b.w) < 1e-6 && Math.abs(a.h - b.h) < 1e-6;
}
function bboxCopy(b) { return { class_id: 0, cx: b.cx, cy: b.cy, w: b.w, h: b.h }; }
function containsBbox(arr, b) { return arr.some(x => bboxClose(x, b)); }
function withoutBbox(arr, b) { return arr.filter(x => !bboxClose(x, b)); }
function materializeVerifiedFromOriginals(sample) {
  if (sample.verified_gt.length > 0) return;
  const sp = sample.spurious_originals || [];
  for (const o of sample.original_gt) {
    if (!containsBbox(sp, o)) sample.verified_gt.push(bboxCopy(o));
  }
}
```

- [ ] **Step 4: Add `snapshotOf` and `applySnapshot` immediately after `materializeVerifiedFromOriginals`**

Insert these two helpers right after the closing `}` of `materializeVerifiedFromOriginals`:

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

- [ ] **Step 5: Smoke-check syntax**

Run from the experiment directory:

```bash
node --check src/data_quality_frame_level/audit_app/static/app.js
```

Expected: no output, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add src/data_quality_frame_level/audit_app/static/app.js
git commit -m "feat(audit-app): add undo state fields and snapshot helpers"
```

---

## Task 2: Capture `loadedSnapshot` at end of `loadSample`

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/app.js` (lines 308–322)

- [ ] **Step 1: Confirm current `loadSample` implementation**

Confirm `app.js:308–322` read:

```js
async function loadSample(stem, opts = {}) {
  state.sample = await api.sample({
    model: state.model, split: state.split, stem,
    conf: state.conf, iou: state.iou, reviewConf: state.reviewConf,
  });
  state.queueIndex = state.queue.findIndex(q => q.stem === stem);
  state.dirty = false;
  selected = null;
  hovered = null;
  setSaveBar();
  renderQueue();
  renderCanvas(opts);
  renderRight();
  renderTimeline();
}
```

- [ ] **Step 2: Add one line capturing the as-loaded snapshot**

Replace the block with:

```js
async function loadSample(stem, opts = {}) {
  state.sample = await api.sample({
    model: state.model, split: state.split, stem,
    conf: state.conf, iou: state.iou, reviewConf: state.reviewConf,
  });
  state.queueIndex = state.queue.findIndex(q => q.stem === stem);
  state.dirty = false;
  selected = null;
  hovered = null;
  state.loadedSnapshot = snapshotOf(state.sample);
  setSaveBar();
  renderQueue();
  renderCanvas(opts);
  renderRight();
  renderTimeline();
}
```

The new line is inserted after `hovered = null;` so it runs before any of the renders see a stale snapshot.

- [ ] **Step 3: Syntax check**

```bash
node --check src/data_quality_frame_level/audit_app/static/app.js
```

Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add src/data_quality_frame_level/audit_app/static/app.js
git commit -m "feat(audit-app): capture loaded snapshot on each loadSample"
```

---

## Task 3: Push snapshot in `persistSample` + refresh after save

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/app.js` (lines 649–669)

- [ ] **Step 1: Confirm current `persistSample` implementation**

Confirm `app.js:649–669` read:

```js
async function persistSample() {
  if (!state.dirty || !state.sample || !state.sample.status) return;
  const r = await api.save({
    model: state.model, split: state.split,
    body: {
      stem: state.sample.stem,
      status: state.sample.status,
      bboxes: state.sample.verified_gt.map(bboxCopy),
      spurious_originals: (state.sample.spurious_originals || []).map(bboxCopy),
      reviewer: state.reviewer || null,
      note: state.sample.note || null,
    },
  });
  state.dirty = false;
  state.sample.reviewed_at = r.saved_at;
  setSaveBar();
  const qi = state.queue.find(q => q.stem === state.sample.stem);
  if (qi) qi.status = state.sample.status;
  renderQueue();
  renderProgress();
}
```

- [ ] **Step 2: Add the `options` parameter, the push, and the post-save snapshot refresh**

Replace the block with:

```js
async function persistSample(options = {}) {
  if (!state.dirty || !state.sample || !state.sample.status) return;
  if (options.recordUndo !== false && state.loadedSnapshot) {
    state.undoStack.push({ stem: state.sample.stem, snapshot: state.loadedSnapshot });
    if (state.undoStack.length > 50) state.undoStack.shift();
  }
  const r = await api.save({
    model: state.model, split: state.split,
    body: {
      stem: state.sample.stem,
      status: state.sample.status,
      bboxes: state.sample.verified_gt.map(bboxCopy),
      spurious_originals: (state.sample.spurious_originals || []).map(bboxCopy),
      reviewer: state.reviewer || null,
      note: state.sample.note || null,
    },
  });
  state.dirty = false;
  state.sample.reviewed_at = r.saved_at;
  state.loadedSnapshot = snapshotOf(state.sample);
  setSaveBar();
  const qi = state.queue.find(q => q.stem === state.sample.stem);
  if (qi) qi.status = state.sample.status;
  renderQueue();
  renderProgress();
}
```

Three changes:
1. `options = {}` parameter added to the signature.
2. New block immediately after the guard pushes `{stem, snapshot}` onto `undoStack` when `recordUndo` is not explicitly `false`, capping the stack at 50.
3. After the `await api.save(...)` (and after `state.sample.reviewed_at = r.saved_at;`), `state.loadedSnapshot` is refreshed to the just-saved state.

Note: `scheduleSave` (`app.js:644–647`) is **not** modified. It still calls `persistSample()` (no args), which means future debounced saves record undo by default.

- [ ] **Step 3: Syntax check**

```bash
node --check src/data_quality_frame_level/audit_app/static/app.js
```

Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add src/data_quality_frame_level/audit_app/static/app.js
git commit -m "feat(audit-app): push pre-edit snapshot onto undo stack before save"
```

---

## Task 4: Add `undoLastSave` function

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/app.js` (just before `seqStep`, ~line 855)

- [ ] **Step 1: Locate the insertion point**

Find the line `async function seqStep(d) {` (currently `app.js:855`). The new function will be inserted **directly before** it (after the closing `});` of the `window.addEventListener('keydown', …)` block on line 853).

- [ ] **Step 2: Insert `undoLastSave`**

Add this function immediately before `async function seqStep(d) {`:

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

Key correctness notes (do **not** write as code comments):
- `clearTimeout(saveTimer)` + `state.dirty = false` discards any pending in-memory edits / debounced save on the current frame so they don't slip into the revert flow.
- `loadSample` itself calls `setSaveBar`, `renderQueue`, `renderCanvas`, `renderRight`, and `renderTimeline`, so we get a fresh render after navigation. The trailing `paint(); renderRight();` after the snapshot apply refresh the canvas and right pane to reflect the reverted state on the same frame (no navigation case) and to overwrite the just-rendered post-edit state with the reverted one (navigation case).
- `persistSample({ recordUndo: false })` writes the reverted state to the server without pushing a new undo entry — otherwise repeated `Ctrl+Z` would create cycles.

- [ ] **Step 3: Syntax check**

```bash
node --check src/data_quality_frame_level/audit_app/static/app.js
```

Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add src/data_quality_frame_level/audit_app/static/app.js
git commit -m "feat(audit-app): add undoLastSave to revert and re-save"
```

---

## Task 5: Bind `Ctrl+Z` / `Cmd+Z`

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/app.js` (lines 823–830)

- [ ] **Step 1: Confirm the top of the keydown handler**

Confirm `app.js:823–830` read:

```js
window.addEventListener('keydown', async e => {
  if (e.target.matches('input, textarea, select')) return;
  if (e.key === '?') {
    e.preventDefault();
    const p = document.getElementById('help-pane');
    p.hidden = !p.hidden;
    return;
  }
```

- [ ] **Step 2: Add the `Ctrl+Z` / `Cmd+Z` branch immediately after the `input` guard**

Replace the block with:

```js
window.addEventListener('keydown', async e => {
  if (e.target.matches('input, textarea, select')) return;
  if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
    e.preventDefault();
    return undoLastSave();
  }
  if (e.key === '?') {
    e.preventDefault();
    const p = document.getElementById('help-pane');
    p.hidden = !p.hidden;
    return;
  }
```

Notes:
- The branch lives **after** the textarea/input guard, so typing `Ctrl+Z` inside the note field still triggers the browser's native undo (the handler returns early).
- Matching both `'z'` and `'Z'` handles capslock / shift edge cases.
- `e.metaKey` is the macOS Cmd modifier.

- [ ] **Step 3: Syntax check**

```bash
node --check src/data_quality_frame_level/audit_app/static/app.js
```

Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add src/data_quality_frame_level/audit_app/static/app.js
git commit -m "feat(audit-app): bind Ctrl+Z / Cmd+Z to undoLastSave"
```

---

## Task 6: Add help-pane row

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/index.html` (~line 227, under the Navigation section)

- [ ] **Step 1: Locate the Navigation section**

Find the row that ends the Navigation section (the Ctrl+arrow row):

```html
          <tr><td class="px-4 py-1 whitespace-nowrap"><kbd>Ctrl</kbd>+<kbd>←</kbd>/<kbd>→</kbd></td><td class="px-4 py-1 text-slate-600">Jump to prev / next sequence</td></tr>
```

- [ ] **Step 2: Insert the undo row immediately after it**

Add this row directly after the Ctrl+arrow row (before the `<th>Status</th>` row that opens the next section):

```html
          <tr><td class="px-4 py-1 whitespace-nowrap"><kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Z</kbd></td><td class="px-4 py-1 text-slate-600">Undo last save (revert + jump to that frame)</td></tr>
```

- [ ] **Step 3: Commit**

```bash
git add src/data_quality_frame_level/audit_app/static/index.html
git commit -m "docs(audit-app): document Ctrl+Z undo shortcut in help pane"
```

---

## Task 7: Lint + Python test sanity check

**Files:**
- Run only.

- [ ] **Step 1: Run the full Python test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass. No Python files changed; this is a safety check.

- [ ] **Step 2: Run lint**

```bash
make lint
```

Expected: clean. Only Python is linted; JS/HTML changes are not gated by ruff.

---

## Task 8: Manual verification

This task requires interaction with the running app. Each scenario corresponds to a spec test (§9 of `2026-05-13-audit-app-undo-design.md`).

**Setup:**

- [ ] **Step 1: Start the dev server**

From `experiments/data-quality/frame-level/`:

```bash
make audit-app
```

Open `http://localhost:8000`. Pick a model+split that has at least 4 unreviewed frames with original GT bboxes. Open browser devtools → Console to watch for errors.

- [ ] **Step 2: Test A — Single-frame undo**

1. Open a fresh frame, double-click + drag to draw a verified bbox.
2. Wait >1 second; the save-bar should change from `unsaved…` to `✓ saved at …`.
3. Press `Ctrl+Z`.

Expected:
- The drawn bbox disappears from the canvas and the right-pane list.
- The save-bar shows a new `✓ saved at …` timestamp.
- Browser console is error-free.
- Hard-refresh the page (Ctrl+Shift+R) and navigate back to the same frame: the bbox is still gone (server-side revert).

- [ ] **Step 3: Test B — Undo a Space-press**

1. Navigate to a fresh frame `N` with originals (verified empty).
2. Press `Space`. The app advances to `N+1`. (Behind the scenes, `N` was auto-materialized + saved with `verified_gt = [non-spurious originals]`.)
3. Press `Ctrl+Z`.

Expected:
- The app navigates back to `N`.
- `verified_gt` on `N` is empty again (the pre-Space state).
- The status row reflects whatever `N`'s status was before `Space` (likely unset).
- The save-bar shows a new `✓ saved at …` timestamp.

- [ ] **Step 4: Test C — Multi-step undo**

1. Navigate to a fresh frame `N`. Press `Space` three times (advancing through `N → N+1 → N+2 → N+3`). You are now on `N+3`.
2. Press `Ctrl+Z` once.

Expected: you are now on `N+2`, `verified_gt` empty, status pre-Space.

3. Press `Ctrl+Z` again.

Expected: you are now on `N+1`, `verified_gt` empty.

4. Press `Ctrl+Z` a third time.

Expected: you are now on `N`, `verified_gt` empty.

- [ ] **Step 5: Test D — Empty stack**

1. Hard-refresh the page.
2. Press `Ctrl+Z` immediately (without making any edits or saves).

Expected: nothing happens. No console error.

- [ ] **Step 6: Test E — Discard pending dirty edits**

1. On a fresh frame, draw a verified bbox. Do **not** wait for the 1 s debounce.
2. Within the 1 s window, press `Ctrl+Z`.

Expected:
- If the undo stack is non-empty (i.e., you have a previous save in this session): the in-progress bbox is discarded and the app navigates to the previous saved frame.
- If the stack is empty: nothing happens (no navigation, no error). The bbox **remains** on screen because we only call `clearTimeout(saveTimer)` / `state.dirty = false` inside `undoLastSave` after the empty-stack early return — confirm this matches the spec's behaviour (§6 "Empty stack on Ctrl+Z").

(If the stack is empty and you want to verify the discard-on-undo path, first edit & save a different frame, then come back and try this test.)

- [ ] **Step 7: Test F — Note field focus**

1. Click inside the note textarea, type some text (e.g., `hello`).
2. Press `Ctrl+Z`.

Expected:
- The textarea's native undo runs (some of the typed text is removed).
- The app does **not** navigate to another frame.
- The save-bar may show `unsaved…` if the note edits haven't yet been debounced; that's expected.

- [ ] **Step 8: Test G — Stack cap (optional, slow)**

If you have time and patience: edit + save 51 distinct frames in succession. Press `Ctrl+Z` 51 times.

Expected: the 51st `Ctrl+Z` is a no-op (the oldest entry was dropped when the stack exceeded 50).

If this is impractical, you can spot-check the cap by running this snippet in the browser devtools console after a save:

```js
state.undoStack.length
```

If it's ≤ 50 after many saves, the cap is working.

- [ ] **Step 9: Test H — Cross-session**

1. After making some saves and confirming Ctrl+Z works, hard-refresh the page.
2. Press `Ctrl+Z`.

Expected: nothing happens (the stack lives in JS memory only; reload wipes it). This is the documented limitation.

---

## Task 9: Final diff review and PR readiness

- [ ] **Step 1: Review the full diff**

```bash
git log --oneline main..HEAD
git diff main..HEAD -- src/data_quality_frame_level/audit_app/static/
```

Confirm only `app.js` and `index.html` are touched, with no incidental edits (formatting, unrelated comments, etc.).

- [ ] **Step 2: Stop the dev server**

If `make audit-app` is still running, Ctrl-C it.

---

## What this plan does NOT do

- Does not modify `export.py`, `persistence.py`, `state.py`, the FastAPI handler in `main.py`, or any other Python module.
- Does not add JS test infrastructure. Verification is `node --check` + manual.
- Does not change the keyboard map for any existing shortcut. Adds `Ctrl+Z` / `Cmd+Z` as a new branch in the same handler.
- Does not persist the undo stack across page reloads.
- Does not introduce redo, granular per-action undo, or a visible Undo button.
- Does not change the auto-save debounce or `flushPending` behaviour.
