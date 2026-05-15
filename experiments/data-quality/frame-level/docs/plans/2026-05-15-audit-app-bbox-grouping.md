# Audit-app bbox grouping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group rows in the audit-app right pane by spatial overlap (IoU ≥ 0.3) so GT bboxes and predictions for the same smoke region sit together; within each group, sort verified → orig kept → orig spurious → predictions desc by conf.

**Architecture:** Pure client-side change inside `static/app.js` and `static/index.html`. A new `clusterBboxes()` helper does union-find on the pooled rows; `renderRight()` is rewritten to consume that grouping and emit either bare rows (singletons) or framed `.bbox-group` blocks. Row markup keeps its existing `data-layer` / `data-idx` attributes so all existing handlers (`mouseover`, `mouseout`, `click` on `#bbox-list`) work unchanged.

**Tech Stack:** Vanilla JS + Tailwind (via the in-file `@layer components` block). No build step. No JS test runner exists in this project — verification is manual via `make audit-app`.

**Spec:** [`../specs/2026-05-15-audit-app-bbox-grouping-design.md`](../specs/2026-05-15-audit-app-bbox-grouping-design.md)

---

## File Structure

- **Modify** `src/data_quality_frame_level/audit_app/static/app.js`
  - Add `BBOX_GROUP_IOU` constant and `clusterBboxes()` helper next to the existing `bboxIou` (around line 432).
  - Rewrite the bbox-emitting middle section of `renderRight()` (`app.js:780-807`). The drop-warning banner block above and the status/note block below stay untouched.
- **Modify** `src/data_quality_frame_level/audit_app/static/index.html`
  - Add `.bbox-group` and `.bbox-group-header` CSS classes after `.bbox-row` (around line 64).

No backend, no API, no persistence change.

---

### Task 1: Add `clusterBboxes` helper

Pure function: given a list of items with bboxes and an IoU threshold, return connected components as arrays of item indices. No knowledge of GT/pred — that lives in `renderRight()`.

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/app.js:426-432` (insert immediately after `bboxIou`)

- [ ] **Step 1: Add the constant and helper**

Insert this block immediately after the closing `}` of `bboxIou` (currently at `app.js:432`):

```javascript
const BBOX_GROUP_IOU = 0.3;

function clusterBboxes(boxes, threshold) {
  const n = boxes.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = i => {
    while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; }
    return i;
  };
  const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; };
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (bboxIou(boxes[i], boxes[j]) >= threshold) union(i, j);
    }
  }
  const byRoot = new Map();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    if (!byRoot.has(r)) byRoot.set(r, []);
    byRoot.get(r).push(i);
  }
  return [...byRoot.values()];
}
```

- [ ] **Step 2: Sanity-check in the browser console**

Start the dev server (`make audit-app`), open `http://localhost:8000`, open DevTools console, and run:

```javascript
clusterBboxes([
  { cx: 0.5, cy: 0.5, w: 0.1, h: 0.1 },
  { cx: 0.51, cy: 0.51, w: 0.1, h: 0.1 },
  { cx: 0.9, cy: 0.9, w: 0.05, h: 0.05 },
], 0.3);
```

Expected: `[[0, 1], [2]]` (first two cluster together, third is a singleton).

Also verify:

```javascript
clusterBboxes([], 0.3);                                    // → []
clusterBboxes([{ cx: 0.5, cy: 0.5, w: 0.1, h: 0.1 }], 0.3); // → [[0]]
```

- [ ] **Step 3: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/audit_app/static/app.js
git commit -m "feat(audit-app): add clusterBboxes union-find helper"
```

---

### Task 2: Add CSS for grouped block

Two classes: a soft outline around multi-member groups and a small uppercase header. Singletons render with the existing `.bbox-row` only.

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/index.html:55-64` (insert after the `.bbox-row` rule cluster, before the `#drop-warn` rule at line 61 — keep it grouped with the other bbox-row rules)

- [ ] **Step 1: Add the CSS rules**

Insert immediately after line 60 (`.bbox-row.pred { ... }`), before line 61 (`#drop-warn .actions button`):

```css
      .bbox-group { @apply mb-2 rounded-md border border-slate-200 bg-white/40 p-1; }
      .bbox-group-header { @apply mb-1 px-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500; }
```

(Match the existing indentation — two spaces deeper than the surrounding `@layer components` block.)

- [ ] **Step 2: Sanity-check the CSS compiles**

Reload `http://localhost:8000` in the browser. Open DevTools, inspect any existing `.bbox-row` and confirm no CSS errors appear in the console. (The new rules aren't applied yet — that comes in Task 3 — but Tailwind's runtime must accept them without error.)

- [ ] **Step 3: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/audit_app/static/index.html
git commit -m "feat(audit-app): add .bbox-group CSS for grouped bbox rows"
```

---

### Task 3: Rewrite `renderRight()` bbox-emitting section

Replace the three `forEach` blocks that emit rows today (`app.js:787-807`) with a pooled + clustered + sorted emitter. The drop-warning banner above (lines 780-786) and the status/note block below (lines 808-814) stay exactly as they are.

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/static/app.js:787-807`

- [ ] **Step 1: Replace the emitter block**

Locate this current code (lines 787-807):

```javascript
  state.sample.original_gt.forEach((b, i) => {
    if (isInVerified(b)) return;
    const sp = isSpurious(b);
    const cls = sp ? 'orig spurious' : (showOrigsAsKept ? 'orig kept' : 'orig');
    const meta = sp
      ? `${b.status} · spurious`
      : (showOrigsAsKept ? `${b.status} · kept` : b.status);
    const actions = sp
      ? `<button data-act="restore-orig" data-i="${i}">↺ Restore</button>`
      : `<button data-act="promote-orig" data-i="${i}">Use as GT</button> <button data-act="spurious-orig" data-i="${i}">🚫 Spurious</button>`;
    const row = make(cls, i, `GT #${i}`, meta, actions);
    root.appendChild(row);
  });
  state.sample.predictions.forEach((p, i) => {
    const row = make('pred', i, 'pred', `${p.status} · ${p.conf.toFixed(2)}`, `<button data-act="promote-pred" data-i="${i}">Use as GT</button>`);
    root.appendChild(row);
  });
  state.sample.verified_gt.forEach((b, i) => {
    const row = make('verified', i, `verified #${i}`, '', `<button data-act="del-verified" data-i="${i}">✕</button>`);
    root.appendChild(row);
  });
```

Replace it with:

```javascript
  const pool = [];
  state.sample.verified_gt.forEach((b, i) => {
    pool.push({ kind: 'verified', idx: i, bbox: b });
  });
  state.sample.original_gt.forEach((b, i) => {
    if (isInVerified(b)) return;
    pool.push({ kind: 'orig', idx: i, bbox: b, spurious: isSpurious(b) });
  });
  state.sample.predictions.forEach((p, i) => {
    pool.push({ kind: 'pred', idx: i, bbox: p, conf: p.conf });
  });

  const clusters = clusterBboxes(pool.map(it => it.bbox), BBOX_GROUP_IOU);

  const groups = clusters.map(memberIdxs => {
    const members = memberIdxs.map(i => pool[i]);
    const verified = members.filter(m => m.kind === 'verified');
    const origKept = members.filter(m => m.kind === 'orig' && !m.spurious);
    const origSpurious = members.filter(m => m.kind === 'orig' && m.spurious);
    const preds = members.filter(m => m.kind === 'pred').sort((a, b) => b.conf - a.conf);
    const hasGt = verified.length > 0 || origKept.length > 0 || origSpurious.length > 0;
    const score = hasGt ? 1.0 : (preds.length > 0 ? preds[0].conf : 0);
    return {
      verified, origKept, origSpurious, preds,
      size: members.length,
      score,
      firstPoolIndex: Math.min(...memberIdxs),
    };
  });

  groups.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    if (b.size !== a.size) return b.size - a.size;
    return a.firstPoolIndex - b.firstPoolIndex;
  });

  const renderOrigRow = m => {
    const sp = m.spurious;
    const cls = sp ? 'orig spurious' : (showOrigsAsKept ? 'orig kept' : 'orig');
    const meta = sp
      ? `${m.bbox.status} · spurious`
      : (showOrigsAsKept ? `${m.bbox.status} · kept` : m.bbox.status);
    const actions = sp
      ? `<button data-act="restore-orig" data-i="${m.idx}">↺ Restore</button>`
      : `<button data-act="promote-orig" data-i="${m.idx}">Use as GT</button> <button data-act="spurious-orig" data-i="${m.idx}">🚫 Spurious</button>`;
    return make(cls, m.idx, `GT #${m.idx}`, meta, actions);
  };
  const renderPredRow = m =>
    make('pred', m.idx, 'pred', `${m.bbox.status} · ${m.conf.toFixed(2)}`,
         `<button data-act="promote-pred" data-i="${m.idx}">Use as GT</button>`);
  const renderVerifiedRow = m =>
    make('verified', m.idx, `verified #${m.idx}`, '',
         `<button data-act="del-verified" data-i="${m.idx}">✕</button>`);

  let multiGroupIdx = 0;
  for (const g of groups) {
    const rows = [
      ...g.verified.map(renderVerifiedRow),
      ...g.origKept.map(renderOrigRow),
      ...g.origSpurious.map(renderOrigRow),
      ...g.preds.map(renderPredRow),
    ];
    if (g.size <= 1) {
      rows.forEach(r => root.appendChild(r));
    } else {
      multiGroupIdx += 1;
      const wrap = document.createElement('div');
      wrap.className = 'bbox-group';
      const header = document.createElement('div');
      header.className = 'bbox-group-header';
      header.textContent = `Group ${multiGroupIdx} · ${g.size}`;
      wrap.appendChild(header);
      rows.forEach(r => wrap.appendChild(r));
      root.appendChild(wrap);
    }
  }
```

- [ ] **Step 2: Lint check**

Run:

```bash
cd experiments/data-quality/frame-level
make lint
```

Expected: clean (no ruff complaints — JS is not linted, but ruff still runs over Python and notebooks).

- [ ] **Step 3: Manual browser verification**

Start the dev server with `make audit-app` and open `http://localhost:8000`. Walk through these scenarios. For each, the predictions list in the right pane should match the described layout.

**Scenario A — frame with overlapping predictions:**
1. Navigate to any frame where the model emitted multiple predictions near the same smoke region (any FP-flagged or FN-flagged frame in the queue should do).
2. Right pane should show a `Group 1 · K` framed block containing those predictions, with the highest-conf prediction at the top.
3. Hovering any row inside the group highlights its bbox on the canvas (existing behaviour, must still work).
4. Clicking `Use as GT` on a pred inside the group should still promote it (existing behaviour).

**Scenario B — frame with GT + overlapping pred:**
1. Find a frame where an original GT and a prediction overlap.
2. Right pane should show a single group containing both. The GT row appears above the pred row.
3. Promoting the pred to GT should still work; on next render the pred and the (now-spurious) original should still cluster together.

**Scenario C — frame with isolated detections only:**
1. Find a frame where no two bboxes overlap.
2. Right pane should show bare rows (no group frames), ordered: verified rows, then original-kept rows, then original-spurious rows, then predictions desc by conf.

**Scenario D — empty frame:**
1. Find a frame with no GT and no predictions (or set conf slider high enough to clear preds).
2. Right pane bbox list should be empty (drop-banner stays hidden because no originals were dropped). No errors in console.

**Scenario E — drop-warning banner present:**
1. Trigger the banner by promoting an original to verified, then deleting the verified copy without restoring the original (or any path that leaves `original_gt` rows neither kept nor in verified).
2. The amber banner must still appear above the (now-grouped) row list.

- [ ] **Step 4: Commit**

```bash
git add experiments/data-quality/frame-level/src/data_quality_frame_level/audit_app/static/app.js
git commit -m "feat(audit-app): group right-pane bboxes by IoU, sort preds by conf"
```

---

## Self-Review

Cross-checked the plan against the spec:

- §5.1 Pool (verified + non-verified originals + predictions): Task 3, pool-building block.
- §5.2 Union-find at IoU 0.3: Task 1, `clusterBboxes` + `BBOX_GROUP_IOU`.
- §5.3 Within-group order (verified → orig kept → orig spurious → preds desc): Task 3, `rows = [...verified, ...kept, ...spurious, ...preds]` and `preds.sort((a, b) => b.conf - a.conf)`.
- §5.4 Group order (score desc, then size desc, then insertion): Task 3, `groups.sort(...)`.
- §5.5 Singleton bare / multi-member framed with `Group N · K`: Task 3, `if (g.size <= 1) … else …` block.
- §5.6 Event delegation unchanged: Task 3 keeps the existing `make()` helper, which preserves `data-layer` + `data-idx` on every row.
- §6 Edge cases: covered by Scenarios A–E in Task 3 Step 3.
- §8 Files touched: app.js (Tasks 1 + 3), index.html (Task 2). No other files.

No placeholders. Helper / constant / class names are consistent across tasks (`clusterBboxes`, `BBOX_GROUP_IOU`, `.bbox-group`, `.bbox-group-header`). The CSS rule in Task 2 is referenced by exact name in Task 3.
