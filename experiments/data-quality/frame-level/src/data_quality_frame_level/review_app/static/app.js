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

async function init() {
  document.getElementById('reviewer').value = state.reviewer;
  document.getElementById('reviewer').addEventListener('input', e => {
    state.reviewer = e.target.value;
    localStorage.setItem('reviewer', state.reviewer);
  });

  const ctx = await api.contexts();
  const selModel = document.getElementById('sel-model');
  const selSplit = document.getElementById('sel-split');
  ctx.models.forEach(m => selModel.add(new Option(m, m)));
  ctx.splits.forEach(s => selSplit.add(new Option(s, s)));
  state.model = ctx.models[0];
  state.split = ctx.splits.includes('val') ? 'val' : ctx.splits[0];
  selModel.value = state.model;
  selSplit.value = state.split;
  selModel.addEventListener('change', () => { state.model = selModel.value; reloadQueue(); });
  selSplit.addEventListener('change', () => { state.split = selSplit.value; reloadQueue(); });

  await reloadQueue();
}

async function reloadQueue() {
  state.queue = [];
  state.queueIndex = -1;
  renderProgress();
}

function renderProgress() {
  const reviewed = state.queue.filter(i => i.status === 'reviewed').length;
  document.getElementById('progress').textContent =
    `${reviewed} / ${state.queue.length} reviewed`;
}

window.addEventListener('DOMContentLoaded', init);
export {};
