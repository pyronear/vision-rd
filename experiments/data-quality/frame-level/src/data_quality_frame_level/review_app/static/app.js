const state = {
  model: null, split: null,
  view: 'fp',
  conf: 0.05, iou: 0.05, reviewConf: 0.35,
  showOrig: true, showPred: true,
  reviewer: localStorage.getItem('reviewer') || '',
  queue: [], queueIndex: -1,
  sample: null,
  dirty: false,
};

const api = {
  contexts: () => fetch('/api/contexts').then(r => r.json()),
  queue: ({ model, split, view, conf, iou, reviewConf }) =>
    fetch(`/api/queue?model=${encodeURIComponent(model)}&split=${encodeURIComponent(split)}&view=${view}&conf=${conf}&iou=${iou}&review_conf=${reviewConf}`).then(r => r.json()),
  sample: ({ model, split, stem, conf, iou, reviewConf }) =>
    fetch(`/api/sample?model=${encodeURIComponent(model)}&split=${encodeURIComponent(split)}&stem=${encodeURIComponent(stem)}&conf=${conf}&iou=${iou}&review_conf=${reviewConf}`).then(r => r.json()),
  save: ({ model, split, body }) =>
    fetch(`/api/sample?model=${encodeURIComponent(model)}&split=${encodeURIComponent(split)}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json()),
};

const cnv = document.getElementById('cnv');
const ctx2d = cnv.getContext('2d');
let img = new Image();
let imgLoaded = false;
let selected = null;
let drag = null;

async function init() {
  document.getElementById('reviewer').value = state.reviewer;
  document.getElementById('reviewer').addEventListener('input', e => {
    state.reviewer = e.target.value;
    localStorage.setItem('reviewer', state.reviewer);
  });

  const ctxs = await api.contexts();
  const selModel = document.getElementById('sel-model');
  const selSplit = document.getElementById('sel-split');
  ctxs.models.forEach(m => selModel.add(new Option(m, m)));
  ctxs.splits.forEach(s => selSplit.add(new Option(s, s)));
  state.model = ctxs.models[0];
  state.split = ctxs.splits.includes('val') ? 'val' : ctxs.splits[0];
  selModel.value = state.model;
  selSplit.value = state.split;
  selModel.addEventListener('change', () => { state.model = selModel.value; reloadQueue(); });
  selSplit.addEventListener('change', () => { state.split = selSplit.value; reloadQueue(); });

  ['conf', 'iou', 'review-conf'].forEach(id => {
    const el = document.getElementById(id);
    const valEl = document.getElementById(`${id}-v`);
    el.addEventListener('input', () => {
      const v = parseFloat(el.value);
      valEl.textContent = v.toFixed(2);
      if (id === 'conf') state.conf = v;
      else if (id === 'iou') state.iou = v;
      else state.reviewConf = v;
      debounceReload();
    });
  });
  document.querySelectorAll('#view-chips button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#view-chips button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.view = btn.dataset.view;
      reloadQueue();
    });
  });
  document.getElementById('show-orig').addEventListener('change', e => {
    state.showOrig = e.target.checked; paint();
  });
  document.getElementById('show-pred').addEventListener('change', e => {
    state.showPred = e.target.checked; paint();
  });
  const helpModal = document.getElementById('help-modal');
  document.getElementById('help-btn').addEventListener('click', () => { helpModal.hidden = false; });
  document.getElementById('help-close').addEventListener('click', () => { helpModal.hidden = true; });
  helpModal.addEventListener('click', e => { if (e.target === helpModal) helpModal.hidden = true; });

  await reloadQueue();
}

let reloadTimer = null;
function debounceReload() {
  clearTimeout(reloadTimer);
  reloadTimer = setTimeout(reloadQueue, 200);
}

async function reloadQueue() {
  await flushPending();
  const r = await api.queue({
    model: state.model, split: state.split, view: state.view,
    conf: state.conf, iou: state.iou, reviewConf: state.reviewConf,
  });
  state.queue = r.items;
  state.queueIndex = state.queue.length > 0 ? 0 : -1;
  renderQueue();
  renderProgress();
  if (state.queueIndex >= 0) await loadSample(state.queue[0].stem);
  else { state.sample = null; paint(); renderRight(); renderTimeline(); }
}

function renderProgress() {
  const reviewed = state.queue.filter(i => i.status === 'reviewed').length;
  document.getElementById('progress').textContent =
    `${reviewed} / ${state.queue.length} reviewed`;
}

function renderQueue() {
  const root = document.getElementById('queue');
  root.innerHTML = '';
  let lastSeq = null;
  state.queue.forEach((it, idx) => {
    if (it.sequence_id !== lastSeq) {
      const h = document.createElement('div');
      h.className = 'queue-seq';
      h.innerHTML = `<span>${escapeHtml(it.sequence_id)}</span><span></span>`;
      root.appendChild(h);
      lastSeq = it.sequence_id;
    }
    const flagged = it.kind && it.kind !== 'none';
    const row = document.createElement('div');
    row.className = 'queue-item'
      + (idx === state.queueIndex ? ' active' : '')
      + (flagged ? ` kind-${it.kind}` : ' unflagged');
    row.innerHTML = `
      <span class="stem">${escapeHtml(it.timestamp)}</span>
      <span class="kind">${flagged ? it.kind : '·'}</span>
      <span class="dot ${it.status || ''}"></span>`;
    row.addEventListener('click', () => navigateTo(idx));
    root.appendChild(row);
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

async function navigateTo(idx) {
  if (idx < 0 || idx >= state.queue.length) return;
  await flushPending();
  state.queueIndex = idx;
  renderQueue();
  await loadSample(state.queue[idx].stem);
}

async function loadSample(stem) {
  state.sample = await api.sample({
    model: state.model, split: state.split, stem,
    conf: state.conf, iou: state.iou, reviewConf: state.reviewConf,
  });
  state.queueIndex = state.queue.findIndex(q => q.stem === stem);
  state.dirty = false;
  selected = null;
  setSaveBar();
  renderQueue();
  renderCanvas();
  renderRight();
  renderTimeline();
}

function setSaveBar() {
  const b = document.getElementById('save-bar');
  if (state.dirty) { b.textContent = 'unsaved…'; b.classList.add('dirty'); }
  else if (state.sample?.reviewed_at) { b.textContent = `✓ saved at ${state.sample.reviewed_at}`; b.classList.remove('dirty'); }
  else { b.textContent = '— no edits —'; b.classList.remove('dirty'); }
}

function bboxToRect(b, W, H) {
  return { x: (b.cx - b.w / 2) * W, y: (b.cy - b.h / 2) * H, w: b.w * W, h: b.h * H };
}
function rectToBbox(r, W, H) {
  return { class_id: 0, cx: (r.x + r.w / 2) / W, cy: (r.y + r.h / 2) / H, w: r.w / W, h: r.h / H };
}
function clamp01(v) { return Math.max(0, Math.min(1, v)); }
function clampBbox(b) {
  const cx = clamp01(b.cx), cy = clamp01(b.cy);
  const w = Math.min(clamp01(b.w), 2 * cx, 2 * (1 - cx));
  const h = Math.min(clamp01(b.h), 2 * cy, 2 * (1 - cy));
  return { class_id: 0, cx, cy, w, h };
}
function bboxClose(a, b) {
  return Math.abs(a.cx - b.cx) < 1e-6 && Math.abs(a.cy - b.cy) < 1e-6
      && Math.abs(a.w - b.w) < 1e-6 && Math.abs(a.h - b.h) < 1e-6;
}

function renderCanvas() {
  if (!state.sample) { ctx2d.clearRect(0, 0, cnv.width, cnv.height); return; }
  const url = `/image?model=${encodeURIComponent(state.model)}&split=${encodeURIComponent(state.split)}&stem=${encodeURIComponent(state.sample.stem)}`;
  if (img.src !== location.origin + url) {
    imgLoaded = false;
    img.onload = () => { imgLoaded = true; sizeCanvas(); paint(); };
    img.src = url;
  } else if (imgLoaded) { sizeCanvas(); paint(); }
}

function sizeCanvas() {
  const wrap = document.getElementById('canvas-wrap');
  const maxW = wrap.clientWidth - 32, maxH = wrap.clientHeight - 32;
  const ar = img.naturalWidth / img.naturalHeight || 16 / 9;
  let w = maxW, h = maxW / ar;
  if (h > maxH) { h = maxH; w = maxH * ar; }
  cnv.width = w; cnv.height = h;
}

function paint() {
  if (!state.sample || !imgLoaded) {
    ctx2d.clearRect(0, 0, cnv.width || 1, cnv.height || 1);
    return;
  }
  ctx2d.clearRect(0, 0, cnv.width, cnv.height);
  ctx2d.drawImage(img, 0, 0, cnv.width, cnv.height);
  const corrected = state.sample.corrected_gt;
  if (state.showOrig) {
    state.sample.original_gt.forEach((b, i) => {
      const overridden = corrected.some(c => bboxClose(c, b));
      if (overridden) return;
      drawBox(b, { stroke: '#58a6ff', fill: 'rgba(88,166,255,.10)', dashed: false, label: `GT (orig) · ${b.status}`, selected: selected?.layer === 'orig' && selected.idx === i });
    });
  }
  if (state.showPred) {
    state.sample.predictions.forEach(p => {
      drawBox(p, { stroke: '#f85149', fill: 'transparent', dashed: true, label: `pred · ${p.status} · ${p.conf.toFixed(2)}` });
    });
  }
  corrected.forEach((b, i) => {
    drawBox(b, { stroke: '#3fb950', fill: 'rgba(63,185,80,.10)', dashed: false, label: 'GT (corr)', selected: selected?.layer === 'corr' && selected.idx === i, handles: true });
  });
}

function drawBox(b, { stroke, fill, dashed, label, selected: sel = false, handles = false }) {
  const r = bboxToRect(b, cnv.width, cnv.height);
  ctx2d.lineWidth = sel ? 3 : 2;
  ctx2d.strokeStyle = stroke;
  ctx2d.fillStyle = fill;
  ctx2d.setLineDash(dashed ? [6, 4] : []);
  ctx2d.fillRect(r.x, r.y, r.w, r.h);
  ctx2d.strokeRect(r.x, r.y, r.w, r.h);
  ctx2d.setLineDash([]);
  if (label) {
    ctx2d.font = '11px ui-sans-serif, system-ui';
    const tw = ctx2d.measureText(label).width + 6;
    ctx2d.fillStyle = stroke;
    ctx2d.fillRect(r.x, r.y - 14, tw, 14);
    ctx2d.fillStyle = '#fff';
    ctx2d.fillText(label, r.x + 3, r.y - 3);
  }
  if (handles) {
    ctx2d.fillStyle = stroke;
    [[r.x, r.y], [r.x + r.w, r.y], [r.x, r.y + r.h], [r.x + r.w, r.y + r.h]]
      .forEach(([x, y]) => ctx2d.fillRect(x - 4, y - 4, 8, 8));
  }
}

function hit(x, y) {
  for (let i = state.sample.corrected_gt.length - 1; i >= 0; i--) {
    const r = bboxToRect(state.sample.corrected_gt[i], cnv.width, cnv.height);
    const handle = handleAt(r, x, y);
    if (handle) return { layer: 'corr', idx: i, handle };
    if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h)
      return { layer: 'corr', idx: i, handle: 'move' };
  }
  if (state.showOrig) {
    for (let i = state.sample.original_gt.length - 1; i >= 0; i--) {
      const r = bboxToRect(state.sample.original_gt[i], cnv.width, cnv.height);
      if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h)
        return { layer: 'orig', idx: i, handle: 'click' };
    }
  }
  return null;
}

function handleAt(r, x, y) {
  const tol = 6;
  if (Math.abs(x - r.x) < tol && Math.abs(y - r.y) < tol) return 'tl';
  if (Math.abs(x - (r.x + r.w)) < tol && Math.abs(y - r.y) < tol) return 'tr';
  if (Math.abs(x - r.x) < tol && Math.abs(y - (r.y + r.h)) < tol) return 'bl';
  if (Math.abs(x - (r.x + r.w)) < tol && Math.abs(y - (r.y + r.h)) < tol) return 'br';
  return null;
}

cnv.addEventListener('mousedown', e => {
  if (!state.sample || !imgLoaded) return;
  const rect = cnv.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const h = hit(x, y);
  if (h?.layer === 'orig') {
    const o = state.sample.original_gt[h.idx];
    state.sample.corrected_gt.push({ class_id: 0, cx: o.cx, cy: o.cy, w: o.w, h: o.h });
    selected = { layer: 'corr', idx: state.sample.corrected_gt.length - 1 };
    markDirty(); paint(); renderRight(); return;
  }
  if (h?.layer === 'corr') {
    selected = { layer: 'corr', idx: h.idx };
    drag = { kind: h.handle, start: { x, y }, ref: { ...state.sample.corrected_gt[h.idx] } };
    paint(); return;
  }
  selected = null;
  drag = { kind: 'draw', start: { x, y } };
  paint();
});

cnv.addEventListener('mousemove', e => {
  if (!drag || !imgLoaded) return;
  const rect = cnv.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const W = cnv.width, H = cnv.height;
  if (drag.kind === 'draw') {
    paint();
    ctx2d.strokeStyle = '#3fb950'; ctx2d.lineWidth = 2; ctx2d.setLineDash([4, 4]);
    ctx2d.strokeRect(Math.min(drag.start.x, x), Math.min(drag.start.y, y), Math.abs(x - drag.start.x), Math.abs(y - drag.start.y));
    ctx2d.setLineDash([]);
    return;
  }
  if (drag.kind === 'move') {
    const dx = (x - drag.start.x) / W, dy = (y - drag.start.y) / H;
    state.sample.corrected_gt[selected.idx] = clampBbox({ ...drag.ref, cx: drag.ref.cx + dx, cy: drag.ref.cy + dy });
    paint(); return;
  }
  const r0 = bboxToRect(drag.ref, W, H);
  let nx = r0.x, ny = r0.y, nw = r0.w, nh = r0.h;
  if (drag.kind === 'tl') { nw = r0.x + r0.w - x; nh = r0.y + r0.h - y; nx = x; ny = y; }
  if (drag.kind === 'tr') { nw = x - r0.x; nh = r0.y + r0.h - y; ny = y; }
  if (drag.kind === 'bl') { nw = r0.x + r0.w - x; nh = y - r0.y; nx = x; }
  if (drag.kind === 'br') { nw = x - r0.x; nh = y - r0.y; }
  if (nw < 4 || nh < 4) return;
  state.sample.corrected_gt[selected.idx] = clampBbox(rectToBbox({ x: nx, y: ny, w: nw, h: nh }, W, H));
  paint();
});

cnv.addEventListener('mouseup', e => {
  if (!drag) return;
  if (drag.kind === 'draw') {
    const rect = cnv.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const W = cnv.width, H = cnv.height;
    const r = { x: Math.min(drag.start.x, x), y: Math.min(drag.start.y, y), w: Math.abs(x - drag.start.x), h: Math.abs(y - drag.start.y) };
    if (r.w >= 4 && r.h >= 4) {
      state.sample.corrected_gt.push(clampBbox(rectToBbox(r, W, H)));
      selected = { layer: 'corr', idx: state.sample.corrected_gt.length - 1 };
      markDirty();
    }
  } else {
    markDirty();
  }
  drag = null;
  paint();
  renderRight();
});

function markDirty() { state.dirty = true; setSaveBar(); scheduleSave(); }

let saveTimer = null;
function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(persistSample, 1000);
}

async function persistSample() {
  if (!state.dirty || !state.sample) return;
  const r = await api.save({
    model: state.model, split: state.split,
    body: {
      stem: state.sample.stem,
      status: state.sample.status || 'reviewed',
      bboxes: state.sample.corrected_gt.map(b => ({ class_id: 0, cx: b.cx, cy: b.cy, w: b.w, h: b.h })),
      reviewer: state.reviewer || null,
      note: state.sample.note || null,
    },
  });
  state.dirty = false;
  state.sample.status = state.sample.status || 'reviewed';
  state.sample.reviewed_at = r.saved_at;
  setSaveBar();
  const qi = state.queue.find(q => q.stem === state.sample.stem);
  if (qi) qi.status = state.sample.status;
  renderQueue();
  renderProgress();
}

async function flushPending() {
  if (state.dirty) await persistSample();
  clearTimeout(saveTimer);
}

function renderRight() {
  const root = document.getElementById('bbox-list');
  root.innerHTML = '';
  if (!state.sample) return;
  const make = (cls, src, meta, actions = '') => {
    const row = document.createElement('div');
    row.className = `bbox-row ${cls}`;
    row.innerHTML = `<span class="src">${escapeHtml(src)}</span><span class="meta-x">${escapeHtml(meta)}</span><span class="actions">${actions}</span>`;
    return row;
  };
  state.sample.original_gt.forEach((b, i) => {
    const row = make('orig', `GT #${i}`, `${b.cx.toFixed(2)} ${b.cy.toFixed(2)} · ${b.status}`, `<button data-act="promote-orig" data-i="${i}">Use as GT</button>`);
    root.appendChild(row);
  });
  state.sample.predictions.forEach((p, i) => {
    const row = make('pred', 'pred', `${p.cx.toFixed(2)} ${p.cy.toFixed(2)} · ${p.status} · ${p.conf.toFixed(2)}`, `<button data-act="promote-pred" data-i="${i}">Use as GT</button>`);
    root.appendChild(row);
  });
  state.sample.corrected_gt.forEach((b, i) => {
    const row = make('corr', `corr #${i}`, `${b.cx.toFixed(2)} ${b.cy.toFixed(2)}`, `<button data-act="del-corr" data-i="${i}">✕</button>`);
    root.appendChild(row);
  });
  document.querySelectorAll('#status-pane button[data-status]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.status === (state.sample.status || 'reviewed'));
    btn.onclick = () => {
      state.sample.status = btn.dataset.status;
      markDirty();
      document.querySelectorAll('#status-pane button[data-status]').forEach(b => b.classList.toggle('active', b === btn));
    };
  });
  const note = document.getElementById('note');
  note.value = state.sample.note || '';
  note.oninput = () => { state.sample.note = note.value || null; markDirty(); };
}

document.getElementById('bbox-list').addEventListener('click', e => {
  const btn = e.target.closest('button[data-act]');
  if (!btn) return;
  const i = +btn.dataset.i;
  if (btn.dataset.act === 'promote-orig') {
    const o = state.sample.original_gt[i];
    state.sample.corrected_gt.push({ class_id: 0, cx: o.cx, cy: o.cy, w: o.w, h: o.h });
  } else if (btn.dataset.act === 'promote-pred') {
    const p = state.sample.predictions[i];
    state.sample.corrected_gt.push({ class_id: 0, cx: p.cx, cy: p.cy, w: p.w, h: p.h });
  } else if (btn.dataset.act === 'del-corr') {
    state.sample.corrected_gt.splice(i, 1);
    if (selected?.layer === 'corr' && selected.idx === i) selected = null;
  }
  markDirty(); paint(); renderRight();
});

function renderTimeline() {
  const root = document.getElementById('timeline');
  root.innerHTML = '';
  if (!state.sample) return;
  const queueByStem = new Map(state.queue.map(it => [it.stem, it]));
  let currentEl = null;
  state.sample.sequence_neighbors.forEach(n => {
    const q = queueByStem.get(n.stem);
    const flagged = q && q.kind && q.kind !== 'none';
    const isCurrent = n.stem === state.sample.stem;
    const f = document.createElement('div');
    f.className = 'tl-frame'
      + (isCurrent ? ' current' : '')
      + (flagged ? ` kind-${q.kind}` : ' unflagged');
    const dotClass = q?.status === 'reviewed' ? 'reviewed'
      : q?.status === 'unclear' ? 'unclear'
      : flagged ? `flagged-${q.kind}` : 'none';
    f.innerHTML = `
      <img class="tl-img" src="/image?model=${encodeURIComponent(state.model)}&split=${encodeURIComponent(state.split)}&stem=${encodeURIComponent(n.stem)}" alt="">
      <div class="tl-status ${dotClass}"></div>
      <div class="tl-time">${escapeHtml(n.timestamp)}</div>`;
    f.addEventListener('click', () => loadSample(n.stem));
    root.appendChild(f);
    if (isCurrent) currentEl = f;
  });
  if (currentEl) {
    const scroll = currentEl.offsetLeft - (root.clientWidth - currentEl.offsetWidth) / 2;
    root.scrollLeft = Math.max(0, scroll);
  }
}

window.addEventListener('keydown', async e => {
  if (e.target.matches('input, textarea, select')) return;
  const helpModal = document.getElementById('help-modal');
  if (e.key === '?') { e.preventDefault(); helpModal.hidden = !helpModal.hidden; return; }
  if (e.key === 'Escape' && !helpModal.hidden) { helpModal.hidden = true; return; }
  if (!helpModal.hidden) return;
  if (e.key === 'ArrowLeft' && e.ctrlKey) { e.preventDefault(); return jumpSequence(-1); }
  if (e.key === 'ArrowRight' && e.ctrlKey) { e.preventDefault(); return jumpSequence(+1); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); return seqStep(-1); }
  if (e.key === 'ArrowRight') { e.preventDefault(); return seqStep(+1); }
  if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); return deleteSelected(); }
  if (e.key === 'Escape') { selected = null; paint(); }
  if (e.key === ' ') { e.preventDefault(); return setStatusAndAdvance('reviewed'); }
  if (e.key === 'r') return setStatus('reviewed');
  if (e.key === 'u') return setStatus('unclear');
  if (e.key === 'o') {
    state.showOrig = !state.showOrig;
    document.getElementById('show-orig').checked = state.showOrig; paint();
  }
  if (e.key === 'p') {
    state.showPred = !state.showPred;
    document.getElementById('show-pred').checked = state.showPred; paint();
  }
});

async function seqStep(d) {
  if (!state.sample) return;
  await flushPending();
  const ns = state.sample.sequence_neighbors;
  const i = ns.findIndex(n => n.stem === state.sample.stem);
  const target = ns[i + d];
  if (target) return loadSample(target.stem);
  // At the edge of the current sequence: flow into the adjacent one in queue order.
  // → goes to the first frame of the next sequence; ← goes to the last frame of the previous.
  const q = state.queue;
  if (q.length === 0) return;
  const currentSeq = state.sample.sequence_id;
  if (d > 0) {
    const start = state.queueIndex >= 0 ? state.queueIndex + 1 : 0;
    for (let j = start; j < q.length; j++) {
      if (q[j].sequence_id !== currentSeq) return loadSample(q[j].stem);
    }
  } else {
    const start = state.queueIndex >= 0 ? state.queueIndex - 1 : q.length - 1;
    for (let j = start; j >= 0; j--) {
      if (q[j].sequence_id !== currentSeq) {
        const targetSeq = q[j].sequence_id;
        let last = j;
        while (last + 1 < q.length && q[last + 1].sequence_id === targetSeq) last++;
        return loadSample(q[last].stem);
      }
    }
  }
}

async function jumpSequence(d) {
  if (!state.sample || state.queue.length === 0) return;
  await flushPending();
  const currentSeq = state.sample.sequence_id;
  const q = state.queue;
  let anchor = state.queueIndex;
  if (anchor < 0) {
    anchor = d > 0
      ? q.findIndex(i => i.sequence_id > currentSeq)
      : (() => {
          for (let i = q.length - 1; i >= 0; i--) {
            if (q[i].sequence_id < currentSeq) return i;
          }
          return -1;
        })();
    if (anchor >= 0) return loadSample(q[anchor].stem);
    return;
  }
  if (d > 0) {
    for (let i = anchor + 1; i < q.length; i++) {
      if (q[i].sequence_id !== currentSeq) return loadSample(q[i].stem);
    }
  } else {
    for (let i = anchor - 1; i >= 0; i--) {
      if (q[i].sequence_id !== currentSeq) {
        const targetSeq = q[i].sequence_id;
        let first = i;
        while (first > 0 && q[first - 1].sequence_id === targetSeq) first--;
        return loadSample(q[first].stem);
      }
    }
  }
}

function deleteSelected() {
  if (!state.sample || !selected) return;
  if (selected.layer === 'corr') {
    state.sample.corrected_gt.splice(selected.idx, 1);
    selected = null;
    markDirty(); paint(); renderRight();
  }
}

function setStatus(s) {
  if (!state.sample) return;
  state.sample.status = s;
  markDirty();
  renderRight();
}

async function setStatusAndAdvance(s) {
  if (!state.sample) return;
  state.sample.status = s;
  markDirty();
  renderRight();
  await seqStep(+1);
}

window.addEventListener('resize', () => {
  if (imgLoaded && state.sample) { sizeCanvas(); paint(); }
});

window.addEventListener('DOMContentLoaded', init);
window.addEventListener('beforeunload', e => {
  if (state.dirty) { e.preventDefault(); e.returnValue = ''; }
});
export {};
