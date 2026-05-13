# Pre-fill `verified_gt` on `reviewed` transition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a frame's status transitions to `'reviewed'` via any entry point (Space, `r`, or "Mark reviewed" button), pre-fill `verified_gt` with non-spurious originals when it's empty — so adding a missed bbox no longer requires clicking "Keep all".

**Architecture:** Single client-side change in `static/app.js`. Add a `materializeVerifiedFromOriginals(sample)` helper, route the three "set status" entry points through `setStatus(s)`, and have `setStatus` invoke the helper on the `'reviewed'` branch. Help-pane text in `static/index.html` updated to match.

**Tech Stack:** Vanilla JS ES modules (no JS test harness in this project). Python export logic in `audit_app/export.py` is **unchanged** — its existing IoU≥0.95 "unchanged" matcher (`compute_diff`) absorbs identical-geometry copies so no spurious diff entries appear.

**Spec:** [`docs/specs/2026-05-13-space-prefill-verified-design.md`](../specs/2026-05-13-space-prefill-verified-design.md)

---

## Pre-flight context (read before starting)

- App: `experiments/data-quality/frame-level/` (a uv sub-project). All commands below are run from this directory unless stated otherwise.
- Frontend entry point: `src/data_quality_frame_level/audit_app/static/app.js` (single file). It is loaded by `static/index.html` as an ES module.
- There is **no JS test framework** in this project. We do not introduce one. Verification is manual via the running app, plus the existing Python test suite to confirm export behavior is unaffected.
- Existing helpers in `app.js` we will reuse:
  - `bboxCopy(b)` — line 348, returns `{class_id, cx, cy, w, h}`.
  - `bboxClose(a, b)` — line 344, geometric equality within 1e-6.
  - `containsBbox(arr, b)` — line 349, uses `bboxClose`.
- Server (Python) launch command for manual testing: `make audit-app` (serves on `http://localhost:8000`).

---

## File map

- **Modify:** `src/data_quality_frame_level/audit_app/static/app.js`
  - Add `materializeVerifiedFromOriginals` helper near other bbox helpers (~line 350).
  - Update `setStatus(s)` (line 925).
  - Replace body of `setStatusAndAdvance(s)` (line 932) to reuse `setStatus`.
  - Update status-button handler inside `renderRight()` (line 718–724) to call `setStatus`.
- **Modify:** `src/data_quality_frame_level/audit_app/static/index.html`
  - Update help-pane rows for `Space` (line 229) and `r` (line 230).
- **No changes:** `audit_app/export.py`, `audit_app/persistence.py`, any Python test, `dvc.yaml`, `params.yaml`.

---

## Task 1: Add `materializeVerifiedFromOriginals` helper

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/app.js` (~line 350, after `withoutBbox`)

- [ ] **Step 1: Locate the bbox helper block**

Open `app.js` and confirm lines 344–350 look like:

```js
function bboxClose(a, b) {
  return Math.abs(a.cx - b.cx) < 1e-6 && Math.abs(a.cy - b.cy) < 1e-6
      && Math.abs(a.w - b.w) < 1e-6 && Math.abs(a.h - b.h) < 1e-6;
}
function bboxCopy(b) { return { class_id: 0, cx: b.cx, cy: b.cy, w: b.w, h: b.h }; }
function containsBbox(arr, b) { return arr.some(x => bboxClose(x, b)); }
function withoutBbox(arr, b) { return arr.filter(x => !bboxClose(x, b)); }
```

- [ ] **Step 2: Insert the new helper directly after `withoutBbox`**

After the `withoutBbox` line, add:

```js
function materializeVerifiedFromOriginals(sample) {
  if (sample.verified_gt.length > 0) return;
  const sp = sample.spurious_originals || [];
  for (const o of sample.original_gt) {
    if (!containsBbox(sp, o)) sample.verified_gt.push(bboxCopy(o));
  }
}
```

Rationale (do **not** add as a code comment):
- Guard: no-op when `verified_gt` is already non-empty — preserves user edits.
- `sp` defaults to `[]` because legacy review state may omit the field; matches the pattern used at app.js:425 and elsewhere.
- We push **copies** (via `bboxCopy`) so later edits to verified rows do not mutate `original_gt`.

- [ ] **Step 3: Smoke-check syntax**

Run from the experiment directory:

```bash
node --check src/data_quality_frame_level/audit_app/static/app.js
```

Expected: no output (exit code 0). If the file uses syntax `node --check` can't parse (top-level `export`), instead start the dev server and confirm it loads without console errors:

```bash
make audit-app
```

Then open `http://localhost:8000` and check the browser devtools console for `SyntaxError`. Ctrl-C to stop the server.

- [ ] **Step 4: Commit**

```bash
git add src/data_quality_frame_level/audit_app/static/app.js
git commit -m "feat(audit-app): add materializeVerifiedFromOriginals helper"
```

---

## Task 2: Update `setStatus` to fill verified on `'reviewed'`

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/app.js` (line 925–930)

- [ ] **Step 1: Locate current implementation**

Confirm lines 925–930 read:

```js
function setStatus(s) {
  if (!state.sample) return;
  state.sample.status = s;
  markDirty();
  renderRight();
}
```

- [ ] **Step 2: Replace with the new body**

Replace the block above with:

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

Why `paint()` is added: the helper may have inserted new verified rows, which are drawn in green on the canvas. The existing handlers that mutate `verified_gt` (e.g. app.js:786) follow the same `markDirty(); paint(); renderRight();` pattern, so this is consistent.

- [ ] **Step 3: Commit**

```bash
git add src/data_quality_frame_level/audit_app/static/app.js
git commit -m "feat(audit-app): materialize verified_gt when status becomes reviewed"
```

---

## Task 3: Simplify `setStatusAndAdvance` to reuse `setStatus`

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/app.js` (line 932–938)

- [ ] **Step 1: Locate current implementation**

Confirm lines 932–938 read:

```js
async function setStatusAndAdvance(s) {
  if (!state.sample) return;
  state.sample.status = s;
  markDirty();
  renderRight();
  await seqStep(+1);
}
```

- [ ] **Step 2: Replace with the simplified body**

Replace the block with:

```js
async function setStatusAndAdvance(s) {
  setStatus(s);
  await seqStep(+1);
}
```

This funnels the Space-key path through the same logic as `r`, so the fill applies to both. `setStatus` already guards against `!state.sample`, so the early-return there is preserved.

- [ ] **Step 3: Commit**

```bash
git add src/data_quality_frame_level/audit_app/static/app.js
git commit -m "refactor(audit-app): route setStatusAndAdvance through setStatus"
```

---

## Task 4: Route status-button handler through `setStatus`

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/app.js` (line 718–724, inside `renderRight()`)

- [ ] **Step 1: Locate current handler**

Confirm lines 718–724 read:

```js
  document.querySelectorAll('#status-pane button[data-status]').forEach(btn => {
    btn.onclick = () => {
      state.sample.status = btn.dataset.status;
      markDirty();
      updateStatusButtons();
    };
  });
```

- [ ] **Step 2: Replace with `setStatus` call**

Replace the block with:

```js
  document.querySelectorAll('#status-pane button[data-status]').forEach(btn => {
    btn.onclick = () => setStatus(btn.dataset.status);
  });
```

Notes:
- `setStatus` calls `renderRight()` itself, which re-runs `updateStatusButtons()` indirectly (the button states are part of the re-rendered right pane). If you want to confirm, look at `updateStatusButtons` and `renderRight` — they are independent functions but the right pane re-render covers the visible state we need.
- The check on line 725 (`updateStatusButtons();` immediately after the loop) is **kept as-is** — it runs once on initial `renderRight()` to set the buttons' visual state for the current sample. Do not delete it.

- [ ] **Step 3: Commit**

```bash
git add src/data_quality_frame_level/audit_app/static/app.js
git commit -m "refactor(audit-app): route status-button handler through setStatus"
```

---

## Task 5: Update help-pane text

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/index.html` (lines 229–230)

- [ ] **Step 1: Locate current rows**

Confirm lines 229–230 read:

```html
          <tr><td class="px-4 py-1 whitespace-nowrap"><kbd>Space</kbd></td><td class="px-4 py-1 text-slate-600">Mark <em>reviewed</em> + advance</td></tr>
          <tr><td class="px-4 py-1 whitespace-nowrap"><kbd>r</kbd></td><td class="px-4 py-1 text-slate-600">Mark <em>reviewed</em> (stay)</td></tr>
```

- [ ] **Step 2: Replace with updated wording**

Replace those two lines with:

```html
          <tr><td class="px-4 py-1 whitespace-nowrap"><kbd>Space</kbd></td><td class="px-4 py-1 text-slate-600">Mark <em>reviewed</em> (accept originals as GT) + advance</td></tr>
          <tr><td class="px-4 py-1 whitespace-nowrap"><kbd>r</kbd></td><td class="px-4 py-1 text-slate-600">Mark <em>reviewed</em> (accept originals as GT), stay</td></tr>
```

- [ ] **Step 3: Commit**

```bash
git add src/data_quality_frame_level/audit_app/static/index.html
git commit -m "docs(audit-app): note that reviewed status accepts originals as GT"
```

---

## Task 6: Confirm Python tests are unaffected

**Files:**
- Run: `tests/test_audit_app_export.py`

- [ ] **Step 1: Run the export tests**

From `experiments/data-quality/frame-level/`:

```bash
uv run pytest tests/test_audit_app_export.py -v
```

Expected: all tests pass. The export logic was not modified, so this is a safety check.

- [ ] **Step 2: Run the full audit-app test suite as a sanity sweep**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass. No code path on the Python side was touched.

- [ ] **Step 3: Run lint**

```bash
make lint
```

Expected: clean (only Python is linted; JS/HTML changes are not gated by ruff).

---

## Task 7: Manual verification against the running app

**Setup:**

- [ ] **Step 1: Start the dev server**

From `experiments/data-quality/frame-level/`:

```bash
make audit-app
```

Open `http://localhost:8000`. Pick a model+split that has at least one frame with original GT bboxes.

- [ ] **Step 2: Test scenario A — Clean approve**

1. Navigate to a frame with one or more `original_gt` bboxes and `verified_gt` empty.
2. Press `Space`.
3. App advances to the next frame.
4. Press `←` to come back.

Expected:
- Status shows `reviewed`.
- The right-pane "verified" section now lists rows matching the originals.
- No amber "will be dropped" banner.

- [ ] **Step 3: Test scenario B — Add missed bbox**

1. On a fresh frame (originals present, verified empty), press `Space`.
2. Navigate back to it.
3. Double-click on an empty area of the image and drag to draw a new verified bbox `B`.

Expected:
- Right pane shows all original-derived verified rows **plus** `B`.
- No "will be dropped" banner.
- Dirty indicator is set.

- [ ] **Step 4: Test scenario C — Spurious is respected**

1. Find a frame with two original bboxes, both initially in the original layer.
2. Click "🚫 Spurious" on one of them (or use the existing flow you normally use to mark spurious).
3. Press `r`.

Expected:
- `verified_gt` is populated with the **non-spurious** original only.
- The spurious original is still marked spurious (orange dashed) and is **not** in the verified rows.

- [ ] **Step 5: Test scenario D — User edits preserved**

1. Find a fresh frame with two originals.
2. Manually click "Use as GT" on **one** of the originals (so `verified_gt` has length 1).
3. Press `Space`.

Expected:
- `verified_gt` is unchanged — still just the one bbox.
- The other original is shown as "will be dropped" (the existing banner behavior). This is correct: the user made a deliberate choice and we honor it.

- [ ] **Step 6: Test scenario E — Toggle re-fills when verified is empty**

1. Press `Space` on a fresh frame so verified gets populated.
2. Press `u` to mark `unclear`. (Verified stays populated; this is expected.)
3. Delete the verified rows one by one (click ✕ on each row) until `verified_gt` is empty.
4. Press `r` to mark `reviewed` again.

Expected: originals are re-materialized into verified.

- [ ] **Step 7: Test scenario F — Backward compat with saved state**

1. Stop the dev server (Ctrl-C).
2. Find an existing `review.json` (under `data/09_review/<model>/<split>/`) that has at least one entry with `status: "reviewed"` and `bboxes: []`. If none exists, skip this step.
3. Restart `make audit-app`. Open the app, navigate to that frame.

Expected on load:
- Frame shows `reviewed` status. Verified rows are empty (we do **not** retroactively materialize on load — only on transitions).
- Pressing `Space` from this state populates `verified_gt`. The amber banner does **not** appear (because all originals get copied, with IoU=1.0 they are "in verified").
- Marking dirty + saving is fine; re-export should produce identical bytes for unchanged frames.

To verify the last point, after saving, re-run `make audit-export` and inspect `data/10_export/manifest.json` — frames that were `(reviewed, [])` before should still not appear in the `changed` list after this round-trip.

- [ ] **Step 8: Test scenario G — Button parity**

1. Open a fresh frame.
2. Click the "Mark reviewed" button in the right pane (do not use the keyboard).

Expected: verified rows appear, exactly as with `r`/`Space`.

---

## Task 8: Final commit hygiene

- [ ] **Step 1: Review the full diff**

```bash
git log --oneline main..HEAD
git diff main..HEAD -- src/data_quality_frame_level/audit_app/static/
```

Confirm only `app.js` and `index.html` are touched on the frontend, with no incidental edits (formatting, unrelated comments, etc.).

- [ ] **Step 2: Update the user-facing changelog if applicable**

Check whether the project has a `CHANGELOG.md` or release notes. If yes, add one line under the appropriate section:

```
- audit-app: pressing `Space`/`r` (or clicking "Mark reviewed") now copies originals into verified_gt, so adding a missed bbox no longer requires clicking "Keep all".
```

If no such file exists, skip — do not create one.

- [ ] **Step 3: Stop the dev server**

If still running, Ctrl-C the `make audit-app` process.

---

## What this plan does NOT do

- Does not modify `export.py`, `persistence.py`, `state.py`, or any other Python module.
- Does not add JS test infrastructure. This is a deliberate scope choice — see "Pre-flight context".
- Does not change the keyboard map (`Space`, `r`, `u`, etc. all retain their bindings).
- Does not retroactively rewrite saved review state. Existing `(reviewed, bboxes=[])` entries continue to round-trip via the legacy `else` branch in `export.py:108`.
- Does not change `unclear` semantics or any other status.
