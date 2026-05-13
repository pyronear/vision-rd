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
let hovered = null;
let zoom = 1, panX = 0, panY = 0;
let imgNaturalW = 0, imgNaturalH = 0;

function ensureReviewer() {
  if (state.reviewer) {
    updateReviewerDisplay();
    return Promise.resolve();
  }
  return promptReviewer();
}

function updateReviewerDisplay() {
  document.getElementById('reviewer-display').textContent = state.reviewer
    ? `\u{1F464} ${state.reviewer}`
    : '— set handle —';
}

function promptReviewer() {
  return new Promise(resolve => {
    const modal = document.getElementById('reviewer-modal');
    const input = document.getElementById('reviewer-input');
    const submit = document.getElementById('reviewer-submit');
    const presets = document.querySelectorAll('#reviewer-presets .preset');
    input.value = '';
    submit.disabled = true;
    modal.hidden = false;
    setTimeout(() => input.focus(), 0);
    const cleanup = () => {
      modal.hidden = true;
      submit.removeEventListener('click', onSubmit);
      input.removeEventListener('input', onInput);
      input.removeEventListener('keydown', onKey);
      presets.forEach(b => b.removeEventListener('click', onPreset));
    };
    const finish = handle => {
      state.reviewer = handle;
      localStorage.setItem('reviewer', handle);
      cleanup();
      updateReviewerDisplay();
      resolve();
    };
    const onSubmit = () => {
      const v = input.value.trim();
      if (v) finish(v);
    };
    const onInput = () => { submit.disabled = !input.value.trim(); };
    const onKey = e => { if (e.key === 'Enter') onSubmit(); };
    const onPreset = e => finish(e.currentTarget.dataset.handle);
    submit.addEventListener('click', onSubmit);
    input.addEventListener('input', onInput);
    input.addEventListener('keydown', onKey);
    presets.forEach(b => b.addEventListener('click', onPreset));
  });
}

async function init() {
  await ensureReviewer();
  document.getElementById('reviewer-display').addEventListener('click', () => {
    promptReviewer();
  });

  const ctxs = await api.contexts();
  renderDvcBanners(ctxs.dvc_warnings || []);
  renderDatasetVersion(ctxs.dataset_version);
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
  document.getElementById('filter-toggle').addEventListener('click', () => {
    document.getElementById('filters').classList.toggle('collapsed');
  });
  document.getElementById('filter-reset').addEventListener('click', () => {
    state.conf = 0.05;
    state.iou = 0.05;
    state.reviewConf = 0.35;
    document.getElementById('conf').value = '0.05';
    document.getElementById('iou').value = '0.05';
    document.getElementById('review-conf').value = '0.35';
    document.getElementById('conf-v').textContent = '0.05';
    document.getElementById('iou-v').textContent = '0.05';
    document.getElementById('review-conf-v').textContent = '0.35';
    reloadQueue();
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
  document.getElementById('show-only-verified').addEventListener('change', e => {
    state.showOnlyVerified = e.target.checked; paint();
  });
  const helpPane = document.getElementById('help-pane');
  const toggleHelp = () => { helpPane.hidden = !helpPane.hidden; };
  document.getElementById('help-btn').addEventListener('click', toggleHelp);
  document.getElementById('help-close').addEventListener('click', () => { helpPane.hidden = true; });

  document.getElementById('export-btn').addEventListener('click', doExport);

  await reloadQueue();
}

async function doExport() {
  const btn = document.getElementById('export-btn');
  if (btn.disabled) return;
  await flushPending();
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = '📦 Exporting…';
  try {
    const r = await fetch(
      `/api/export?model=${encodeURIComponent(state.model)}&split=${encodeURIComponent(state.split)}&conf=${state.conf}&iou=${state.iou}&review_conf=${state.reviewConf}`,
      { method: 'POST' }
    );
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      showToast('error', `Export failed: ${err.detail || r.status}`);
      return;
    }
    const m = await r.json();
    const t = m.totals || {};
    const contrib = m.contributors?.length ? ` · ${m.contributors.join(', ')}` : '';
    showToast(
      'success',
      `✓ Exported ${t.reviewed || 0} reviewed, ${t.changed || 0} changed (${state.split})${contrib} → data/10_export/${state.model}/${state.split}/`
    );
  } catch (e) {
    showToast('error', `Export failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

function showToast(kind, message) {
  const host = document.getElementById('toast-host');
  const div = document.createElement('div');
  div.className = `toast toast-${kind}`;
  div.textContent = message;
  host.appendChild(div);
  setTimeout(() => div.classList.add('opacity-0'), 4000);
  setTimeout(() => div.remove(), 4500);
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
  const flagged = state.queue.filter(i => i.kind && i.kind !== 'none');
  const reviewed = flagged.filter(i => i.status === 'reviewed').length;
  const total = flagged.length;
  const pct = total > 0 ? (reviewed / total) * 100 : 0;
  const root = document.getElementById('progress');
  root.querySelector('.fill').style.width = `${pct}%`;
  root.querySelector('.label').textContent = `${reviewed} / ${total}`;
}

function renderQueue() {
  const root = document.getElementById('queue');
  root.innerHTML = '';
  let lastSeq = null;
  let activeRow = null;
  state.queue.forEach((it, idx) => {
    if (it.sequence_id !== lastSeq) {
      const h = document.createElement('div');
      h.className = 'queue-seq';
      h.innerHTML = `<span class="min-w-0 flex-1 truncate" title="${escapeHtml(it.sequence_id)}">${escapeHtml(it.sequence_id)}</span><span></span>`;
      root.appendChild(h);
      lastSeq = it.sequence_id;
    }
    const flagged = it.kind && it.kind !== 'none';
    const isActive = idx === state.queueIndex;
    const row = document.createElement('div');
    row.className = 'queue-item'
      + (isActive ? ' active' : '')
      + (flagged ? ` kind-${it.kind}` : ' unflagged');
    row.innerHTML = `
      <span class="stem">${escapeHtml(it.timestamp)}</span>
      <span class="kind">${flagged ? it.kind : '·'}</span>
      <span class="dot ${it.status || ''}"></span>`;
    row.addEventListener('click', () => navigateTo(idx));
    root.appendChild(row);
    if (isActive) activeRow = row;
  });
  if (activeRow) activeRow.scrollIntoView({ block: 'nearest' });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

function renderDatasetVersion(rev) {
  const el = document.getElementById('dataset-version');
  if (!rev) { el.hidden = true; return; }
  el.textContent = `dataset ${rev}`;
  el.hidden = false;
  if (rev.startsWith('mixed:')) {
    el.classList.remove('border-slate-200', 'bg-slate-50', 'text-slate-600');
    el.classList.add('border-amber-300', 'bg-amber-50', 'text-amber-800');
  }
}

function renderDvcBanners(warnings) {
  const host = document.getElementById('dvc-banner-host');
  host.innerHTML = '';
  const dismissed = JSON.parse(sessionStorage.getItem('dvc_dismissed') || '[]');
  warnings.forEach(w => {
    const key = `${w.model}/${w.split}/${w.kind}`;
    if (dismissed.includes(key)) return;
    const div = document.createElement('div');
    div.className = 'dvc-banner';
    div.innerHTML = `
      <span class="msg">⚠️ ${escapeHtml(w.message)}</span>
      <span class="ctx">${escapeHtml(w.model)} / ${escapeHtml(w.split)}</span>
      <button type="button">Dismiss</button>`;
    div.querySelector('button').addEventListener('click', () => {
      dismissed.push(key);
      sessionStorage.setItem('dvc_dismissed', JSON.stringify(dismissed));
      div.remove();
    });
    host.appendChild(div);
  });
}

async function navigateTo(idx) {
  if (idx < 0 || idx >= state.queue.length) return;
  await flushPending();
  state.queueIndex = idx;
  renderQueue();
  await loadSample(state.queue[idx].stem);
}

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
function bboxIou(a, b) {
  const ix = Math.max(0, Math.min(a.cx + a.w/2, b.cx + b.w/2) - Math.max(a.cx - a.w/2, b.cx - b.w/2));
  const iy = Math.max(0, Math.min(a.cy + a.h/2, b.cy + b.h/2) - Math.max(a.cy - a.h/2, b.cy - b.h/2));
  const inter = ix * iy;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

function renderCanvas(opts = {}) {
  if (!state.sample) { ctx2d.clearRect(0, 0, cnv.width, cnv.height); return; }
  const preserve = !!opts.preserveView;
  const url = `/image?model=${encodeURIComponent(state.model)}&split=${encodeURIComponent(state.split)}&stem=${encodeURIComponent(state.sample.stem)}`;
  if (img.src !== location.origin + url) {
    imgLoaded = false;
    img.onload = () => {
      imgLoaded = true;
      imgNaturalW = img.naturalWidth;
      imgNaturalH = img.naturalHeight;
      sizeCanvas();
      if (preserve) clampPan(); else resetView();
      paint();
    };
    img.src = url;
  } else if (imgLoaded) {
    sizeCanvas();
    if (preserve) clampPan(); else resetView();
    paint();
  }
}

function sizeCanvas() {
  const wrap = document.getElementById('canvas-wrap');
  cnv.width = wrap.clientWidth - 32;
  cnv.height = wrap.clientHeight - 32;
}

function resetView() {
  if (!imgNaturalW || !imgNaturalH || !cnv.width || !cnv.height) return;
  zoom = Math.min(cnv.width / imgNaturalW, cnv.height / imgNaturalH);
  panX = (cnv.width - imgNaturalW * zoom) / 2;
  panY = (cnv.height - imgNaturalH * zoom) / 2;
}

function clampPan() {
  const w = imgNaturalW * zoom, h = imgNaturalH * zoom;
  panX = Math.min(cnv.width - 50, Math.max(50 - w, panX));
  panY = Math.min(cnv.height - 50, Math.max(50 - h, panY));
}

function screenToWorld(clientX, clientY) {
  const rect = cnv.getBoundingClientRect();
  return {
    x: (clientX - rect.left - panX) / zoom,
    y: (clientY - rect.top - panY) / zoom,
  };
}

function paint() {
  if (!state.sample || !imgLoaded) {
    ctx2d.clearRect(0, 0, cnv.width || 1, cnv.height || 1);
    return;
  }
  ctx2d.clearRect(0, 0, cnv.width, cnv.height);
  ctx2d.save();
  ctx2d.translate(panX, panY);
  ctx2d.scale(zoom, zoom);
  ctx2d.drawImage(img, 0, 0);
  const verified = state.sample.verified_gt;
  const isReviewed = state.sample.status === 'reviewed';
  const previewMode = state.showOnlyVerified;
  const showOrigsAsKept = isReviewed && verified.length === 0;
  const dimming = hovered !== null;
  const setAlpha = isHov => { ctx2d.globalAlpha = (dimming && !isHov) ? 0.25 : 1.0; };
  if (state.showOrig) {
    const spurious = state.sample.spurious_originals || [];
    state.sample.original_gt.forEach((b, i) => {
      const overridden = containsBbox(verified, b);
      if (overridden) return;
      const isSp = containsBbox(spurious, b);
      if (previewMode && (isSp || verified.length > 0 || !isReviewed)) return;
      const isHov = hovered?.layer === 'orig' && hovered.idx === i;
      setAlpha(isHov);
      const style = isSp
        ? { stroke: '#d97706', fill: 'rgba(217,119,6,.08)', fillHover: 'rgba(217,119,6,.20)', dashed: true, label: `GT (orig) · spurious` }
        : showOrigsAsKept
          ? { stroke: '#3fb950', fill: 'rgba(63,185,80,.10)', fillHover: 'rgba(63,185,80,.25)', dashed: false, label: 'GT (verified)' }
          : { stroke: '#58a6ff', fill: 'rgba(88,166,255,.10)', fillHover: 'rgba(88,166,255,.25)', dashed: false, label: `GT (orig) · ${b.status}` };
      drawBox(b, { ...style, selected: selected?.layer === 'orig' && selected.idx === i, highlighted: isHov });
    });
  }
  if (state.showPred && !previewMode) {
    state.sample.predictions.forEach((p, i) => {
      const isHov = hovered?.layer === 'pred' && hovered.idx === i;
      setAlpha(isHov);
      drawBox(p, { stroke: '#f85149', fill: 'transparent', fillHover: 'rgba(248,81,73,.20)', dashed: true, label: `pred · ${p.status} · ${p.conf.toFixed(2)}`, highlighted: isHov });
    });
  }
  verified.forEach((b, i) => {
    const isHov = hovered?.layer === 'verified' && hovered.idx === i;
    setAlpha(isHov);
    drawBox(b, { stroke: '#3fb950', fill: 'rgba(63,185,80,.10)', fillHover: 'rgba(63,185,80,.25)', dashed: false, label: 'GT (verified)', selected: selected?.layer === 'verified' && selected.idx === i, highlighted: isHov, handles: true });
  });
  ctx2d.globalAlpha = 1.0;
  ctx2d.restore();
}

function drawBox(b, { stroke, fill, fillHover, dashed, label, selected: sel = false, highlighted = false, handles = false }) {
  const r = bboxToRect(b, imgNaturalW, imgNaturalH);
  ctx2d.lineWidth = highlighted ? 4 : (sel ? 3 : 2);
  ctx2d.strokeStyle = stroke;
  ctx2d.fillStyle = highlighted && fillHover ? fillHover : fill;
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
  for (let i = state.sample.verified_gt.length - 1; i >= 0; i--) {
    const r = bboxToRect(state.sample.verified_gt[i], imgNaturalW, imgNaturalH);
    const handle = handleAt(r, x, y);
    if (handle) return { layer: 'verified', idx: i, handle };
    if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h)
      return { layer: 'verified', idx: i, handle: 'move' };
  }
  if (state.showOrig) {
    for (let i = state.sample.original_gt.length - 1; i >= 0; i--) {
      const r = bboxToRect(state.sample.original_gt[i], imgNaturalW, imgNaturalH);
      if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h)
        return { layer: 'orig', idx: i, handle: 'click' };
    }
  }
  return null;
}

function handleAt(r, x, y) {
  const tol = 6 / zoom;
  if (Math.abs(x - r.x) < tol && Math.abs(y - r.y) < tol) return 'tl';
  if (Math.abs(x - (r.x + r.w)) < tol && Math.abs(y - r.y) < tol) return 'tr';
  if (Math.abs(x - r.x) < tol && Math.abs(y - (r.y + r.h)) < tol) return 'bl';
  if (Math.abs(x - (r.x + r.w)) < tol && Math.abs(y - (r.y + r.h)) < tol) return 'br';
  return null;
}

function updateCursor(e) {
  if (drag?.kind === 'pan') cnv.style.cursor = 'grabbing';
  else if (drag) cnv.style.cursor = '';
  else if (e?.shiftKey) cnv.style.cursor = 'grab';
  else cnv.style.cursor = '';
}

window.addEventListener('keydown', e => { if (e.key === 'Shift') updateCursor(e); });
window.addEventListener('keyup', e => { if (e.key === 'Shift') updateCursor(e); });

cnv.addEventListener('mousedown', e => {
  if (!state.sample || !imgLoaded) return;
  if (e.shiftKey) {
    drag = {
      kind: 'pan',
      startScreen: { x: e.clientX, y: e.clientY },
      startPan: { x: panX, y: panY },
    };
    updateCursor(e);
    return;
  }
  const { x, y } = screenToWorld(e.clientX, e.clientY);
  const h = hit(x, y);
  if (h?.layer === 'orig') {
    const o = state.sample.original_gt[h.idx];
    state.sample.verified_gt.push(bboxCopy(o));
    selected = { layer: 'verified', idx: state.sample.verified_gt.length - 1 };
    markDirty(); paint(); renderRight(); return;
  }
  if (h?.layer === 'verified') {
    selected = { layer: 'verified', idx: h.idx };
    drag = { kind: h.handle, start: { x, y }, ref: { ...state.sample.verified_gt[h.idx] } };
    paint(); return;
  }
  selected = null;
  drag = { kind: 'draw', start: { x, y } };
  paint();
});

cnv.addEventListener('mousemove', e => {
  if (!drag || !imgLoaded) return;
  if (drag.kind === 'pan') {
    panX = drag.startPan.x + (e.clientX - drag.startScreen.x);
    panY = drag.startPan.y + (e.clientY - drag.startScreen.y);
    clampPan();
    paint();
    return;
  }
  const { x, y } = screenToWorld(e.clientX, e.clientY);
  const W = imgNaturalW, H = imgNaturalH;
  if (drag.kind === 'draw') {
    paint();
    ctx2d.save();
    ctx2d.translate(panX, panY);
    ctx2d.scale(zoom, zoom);
    ctx2d.strokeStyle = '#3fb950'; ctx2d.lineWidth = 2; ctx2d.setLineDash([4, 4]);
    ctx2d.strokeRect(Math.min(drag.start.x, x), Math.min(drag.start.y, y), Math.abs(x - drag.start.x), Math.abs(y - drag.start.y));
    ctx2d.setLineDash([]);
    ctx2d.restore();
    return;
  }
  if (drag.kind === 'move') {
    const dx = (x - drag.start.x) / W, dy = (y - drag.start.y) / H;
    state.sample.verified_gt[selected.idx] = clampBbox({ ...drag.ref, cx: drag.ref.cx + dx, cy: drag.ref.cy + dy });
    paint(); return;
  }
  const r0 = bboxToRect(drag.ref, W, H);
  let nx = r0.x, ny = r0.y, nw = r0.w, nh = r0.h;
  if (drag.kind === 'tl') { nw = r0.x + r0.w - x; nh = r0.y + r0.h - y; nx = x; ny = y; }
  if (drag.kind === 'tr') { nw = x - r0.x; nh = r0.y + r0.h - y; ny = y; }
  if (drag.kind === 'bl') { nw = r0.x + r0.w - x; nh = y - r0.y; nx = x; }
  if (drag.kind === 'br') { nw = x - r0.x; nh = y - r0.y; }
  if (nw < 4 || nh < 4) return;
  state.sample.verified_gt[selected.idx] = clampBbox(rectToBbox({ x: nx, y: ny, w: nw, h: nh }, W, H));
  paint();
});

cnv.addEventListener('mouseup', e => {
  if (!drag) return;
  if (drag.kind === 'pan') { drag = null; updateCursor(e); return; }
  if (drag.kind === 'draw') {
    const { x, y } = screenToWorld(e.clientX, e.clientY);
    const W = imgNaturalW, H = imgNaturalH;
    const r = { x: Math.min(drag.start.x, x), y: Math.min(drag.start.y, y), w: Math.abs(x - drag.start.x), h: Math.abs(y - drag.start.y) };
    if (r.w >= 4 && r.h >= 4) {
      state.sample.verified_gt.push(clampBbox(rectToBbox(r, W, H)));
      selected = { layer: 'verified', idx: state.sample.verified_gt.length - 1 };
      markDirty();
    }
  } else {
    markDirty();
  }
  drag = null;
  paint();
  renderRight();
});

cnv.addEventListener('wheel', e => {
  if (!imgLoaded) return;
  e.preventDefault();
  const rect = cnv.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const factor = Math.exp(-e.deltaY * 0.005);
  const newZoom = Math.max(0.25, Math.min(8, zoom * factor));
  const wx = (sx - panX) / zoom, wy = (sy - panY) / zoom;
  panX = sx - wx * newZoom;
  panY = sy - wy * newZoom;
  zoom = newZoom;
  clampPan();
  paint();
}, { passive: false });

function updateStatusButtons() {
  document.querySelectorAll('#status-pane button[data-status]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.status === state.sample?.status);
  });
}

function markDirty() {
  state.dirty = true;
  if (state.sample && !state.sample.status) {
    state.sample.status = 'reviewed';
    updateStatusButtons();
  }
  setSaveBar();
  scheduleSave();
}

let saveTimer = null;
function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(persistSample, 1000);
}

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

async function flushPending() {
  if (state.dirty) await persistSample();
  clearTimeout(saveTimer);
}

function renderRight() {
  const root = document.getElementById('bbox-list');
  root.innerHTML = '';
  if (!state.sample) return;
  const make = (cls, idx, src, meta, actions = '') => {
    const row = document.createElement('div');
    row.className = `bbox-row ${cls}`;
    row.dataset.layer = cls;
    row.dataset.idx = idx;
    row.innerHTML = `<span class="src">${escapeHtml(src)}</span><span class="meta-x">${escapeHtml(meta)}</span><span class="actions">${actions}</span>`;
    return row;
  };
  const spurious = state.sample.spurious_originals || [];
  const verified = state.sample.verified_gt;
  const isSpurious = b => containsBbox(spurious, b);
  const isInVerified = b => containsBbox(verified, b);
  const isReviewed = state.sample.status === 'reviewed';
  const showOrigsAsKept = isReviewed && verified.length === 0;
  const droppedOrigs = verified.length > 0
    ? state.sample.original_gt.filter(b => !isSpurious(b) && !isInVerified(b))
    : [];
  if (droppedOrigs.length > 0) {
    const banner = document.createElement('div');
    banner.id = 'drop-warn';
    banner.className = 'mb-2 rounded-md border border-amber-300 bg-amber-50 px-2 py-1.5 text-[11px] text-amber-900';
    banner.innerHTML = `⚠️ <strong>${droppedOrigs.length}</strong> original bbox(es) will be dropped on save. <span class="actions ml-1"><button data-act="keep-all">Keep all</button> <button data-act="spurious-all">Mark all spurious</button></span>`;
    root.appendChild(banner);
  }
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
  document.querySelectorAll('#status-pane button[data-status]').forEach(btn => {
    btn.onclick = () => setStatus(btn.dataset.status);
  });
  updateStatusButtons();
  const note = document.getElementById('note');
  note.value = state.sample.note || '';
  note.oninput = () => { state.sample.note = note.value || null; markDirty(); };
}

document.getElementById('bbox-list').addEventListener('mouseover', e => {
  const row = e.target.closest('.bbox-row');
  if (!row) return;
  const layer = row.dataset.layer;
  const idx = +row.dataset.idx;
  if (hovered?.layer === layer && hovered.idx === idx) return;
  hovered = { layer, idx };
  paint();
});
document.getElementById('bbox-list').addEventListener('mouseout', e => {
  const row = e.target.closest('.bbox-row');
  if (!row) return;
  if (row.contains(e.relatedTarget)) return;
  if (e.relatedTarget?.closest?.('.bbox-row')) return;
  if (hovered === null) return;
  hovered = null;
  paint();
});
document.getElementById('bbox-list').addEventListener('click', e => {
  const btn = e.target.closest('button[data-act]');
  if (!btn) return;
  const i = +btn.dataset.i;
  state.sample.spurious_originals = state.sample.spurious_originals || [];
  const sp = state.sample.spurious_originals;
  const verified = state.sample.verified_gt;
  if (btn.dataset.act === 'promote-orig') {
    const o = state.sample.original_gt[i];
    verified.push(bboxCopy(o));
    if (!containsBbox(sp, o)) sp.push(bboxCopy(o));
  } else if (btn.dataset.act === 'spurious-orig') {
    const o = state.sample.original_gt[i];
    if (!containsBbox(sp, o)) sp.push(bboxCopy(o));
  } else if (btn.dataset.act === 'restore-orig') {
    const o = state.sample.original_gt[i];
    state.sample.spurious_originals = withoutBbox(sp, o);
  } else if (btn.dataset.act === 'promote-pred') {
    const p = state.sample.predictions[i];
    verified.push(bboxCopy(p));
    const thr = state.iou ?? 0.05;
    state.sample.original_gt.forEach(o => {
      if (bboxIou(o, p) >= thr && !containsBbox(sp, o)) sp.push(bboxCopy(o));
    });
  } else if (btn.dataset.act === 'del-verified') {
    const removed = verified.splice(i, 1)[0];
    state.sample.spurious_originals = withoutBbox(sp, removed);
    if (selected?.layer === 'verified' && selected.idx === i) selected = null;
  } else if (btn.dataset.act === 'keep-all') {
    state.sample.original_gt.forEach(o => {
      if (!containsBbox(sp, o) && !containsBbox(verified, o)) verified.push(bboxCopy(o));
    });
  } else if (btn.dataset.act === 'spurious-all') {
    state.sample.original_gt.forEach(o => {
      if (!containsBbox(verified, o) && !containsBbox(sp, o)) sp.push(bboxCopy(o));
    });
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
    f.addEventListener('click', () => loadSample(n.stem, { preserveView: true }));
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
  if (e.key === '?') {
    e.preventDefault();
    const p = document.getElementById('help-pane');
    p.hidden = !p.hidden;
    return;
  }
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
  if (e.key === 'v') {
    state.showOnlyVerified = !state.showOnlyVerified;
    document.getElementById('show-only-verified').checked = state.showOnlyVerified; paint();
  }
  if (e.key === '0') { resetView(); paint(); }
});

async function seqStep(d) {
  if (!state.sample) return;
  await flushPending();
  const ns = state.sample.sequence_neighbors;
  const i = ns.findIndex(n => n.stem === state.sample.stem);
  const target = ns[i + d];
  if (target) return loadSample(target.stem, { preserveView: true });
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
  if (selected.layer === 'verified') {
    state.sample.verified_gt.splice(selected.idx, 1);
    selected = null;
    markDirty(); paint(); renderRight();
  }
}

function setStatus(s) {
  if (!state.sample) return;
  if (s === 'reviewed') materializeVerifiedFromOriginals(state.sample);
  state.sample.status = s;
  markDirty();
  renderRight();
  paint();
}

async function setStatusAndAdvance(s) {
  setStatus(s);
  await seqStep(+1);
}

window.addEventListener('resize', () => {
  if (imgLoaded && state.sample) { sizeCanvas(); clampPan(); paint(); }
});

window.addEventListener('DOMContentLoaded', init);
window.addEventListener('beforeunload', e => {
  if (state.dirty) { e.preventDefault(); e.returnValue = ''; }
});
export {};
