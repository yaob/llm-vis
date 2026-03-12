import { renderResidualFlow } from './residual-flow.js';

const VIEWS = [
  { id: 'architecture', label: 'Architecture', render: renderResidualFlow },
];

function formatParams(value) {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toString();
}

function formatBytes(value) {
  if (!value) return '—';
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} GB`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)} MB`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)} KB`;
  return `${value} B`;
}

function renderAttentionGeometry(model) {
  if (!model.headCount) return '';
  const gqaLabel = model.gqaRatio > 1 ? ` (${model.gqaRatio}:1 GQA)` : ' (MHA)';
  return `<div class="attn-geometry">
    <span class="attn-geo-item"><strong>${model.headCount}</strong> Q heads</span>
    <span class="attn-geo-sep">×</span>
    <span class="attn-geo-item"><strong>${model.headDim || '?'}</strong> dim</span>
    <span class="attn-geo-sep">|</span>
    <span class="attn-geo-item"><strong>${model.headCountKV || model.headCount}</strong> KV heads${gqaLabel}</span>
  </div>`;
}

function renderSummary(model) {
  const summary = document.createElement('div');
  summary.className = 'model-summary';
  summary.innerHTML = `
    <h2>${model.modelName}</h2>
    <div class="summary-grid">
      <div class="summary-item"><span class="label">Architecture</span><span class="value">${model.arch}</span></div>
      <div class="summary-item"><span class="label">Parameters</span><span class="value">${formatParams(model.totalParams)}</span></div>
      <div class="summary-item"><span class="label">Memory</span><span class="value">${formatBytes(model.totalMemory)}</span></div>
      <div class="summary-item"><span class="label">Blocks</span><span class="value">${model.blockCount}</span></div>
      <div class="summary-item"><span class="label">Embedding</span><span class="value">${model.embeddingLength}</span></div>
      <div class="summary-item"><span class="label">Heads</span><span class="value">${model.headCount}</span></div>
      <div class="summary-item"><span class="label">Context</span><span class="value">${model.contextLength}</span></div>
      <div class="summary-item"><span class="label">Vocab</span><span class="value">${model.vocabSize}</span></div>
      <div class="summary-item"><span class="label">Version</span><span class="value">v${model.version}</span></div>
    </div>
    ${renderAttentionGeometry(model)}
  `;
  return summary;
}

function renderTabs(activeView, onViewChange) {
  const tabs = document.createElement('div');
  tabs.className = 'view-tabs';
  for (const view of VIEWS) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `view-tab${view.id === activeView ? ' active' : ''}`;
    button.textContent = view.label;
    button.disabled = view.id === activeView;
    button.addEventListener('click', () => onViewChange?.(view.id));
    tabs.appendChild(button);
  }
  return tabs;
}

export function renderModelView(container, model, uiState = {}) {
  const el = typeof container === 'string' ? document.querySelector(container) : container;
  const activeView = VIEWS.some(view => view.id === uiState.activeView) ? uiState.activeView : VIEWS[0].id;
  const renderer = VIEWS.find(view => view.id === activeView)?.render || VIEWS[0].render;

  el.innerHTML = '';
  el.appendChild(renderSummary(model));
  el.appendChild(renderTabs(activeView, uiState.onViewChange));

  const panel = document.createElement('div');
  panel.className = 'view-panel';
  el.appendChild(panel);
  renderer(panel, model, uiState);
}