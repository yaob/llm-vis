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

function getMeta(model, ...keys) {
  const arch = model.arch || '';
  for (const k of keys) {
    const v = model.metadata?.[`${arch}.${k}`] ?? model.metadata?.[k];
    if (v != null) return v;
  }
  return null;
}

function escHtml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

function renderProvenance(model) {
  const md = model.metadata || {};
  const author = md['general.author'] || null;
  const url = md['general.url'] || md['general.source.url'] || null;
  const license = md['general.license'] || null;
  const desc = md['general.description'] || null;
  if (!author && !url && !license && !desc) return '';
  const items = [];
  if (author) items.push(`<span class="attn-geo-item">Author: <strong>${escHtml(author)}</strong></span>`);
  if (license) items.push(`<span class="attn-geo-item">License: <strong>${escHtml(license)}</strong></span>`);
  if (url) items.push(`<span class="attn-geo-item"><a href="${escHtml(url)}" target="_blank" rel="noopener" style="color:#61afef;text-decoration:none">${escHtml(url)}</a></span>`);
  let html = `<div class="attn-geometry">${items.join('<span class="attn-geo-sep">·</span>')}</div>`;
  if (desc) html += `<div class="model-description">${escHtml(desc)}</div>`;
  return html;
}

function renderTokenizerInfo(model) {
  const md = model.metadata || {};
  const tokModel = md['tokenizer.ggml.model'] || null;
  const bosId = md['tokenizer.ggml.bos_token_id'] ?? null;
  const eosId = md['tokenizer.ggml.eos_token_id'] ?? null;
  const padId = md['tokenizer.ggml.padding_token_id'] ?? md['tokenizer.ggml.pad_token_id'] ?? null;
  const chatTpl = md['tokenizer.chat_template'] || null;
  if (tokModel == null && bosId == null && eosId == null) return '';
  const items = [];
  if (tokModel) items.push(`<span class="attn-geo-item">Tokenizer: <strong>${tokModel.toUpperCase()}</strong></span>`);
  if (bosId != null) items.push(`<span class="attn-geo-item">BOS <strong>${bosId}</strong></span>`);
  if (eosId != null) items.push(`<span class="attn-geo-item">EOS <strong>${eosId}</strong></span>`);
  if (padId != null) items.push(`<span class="attn-geo-item">PAD <strong>${padId}</strong></span>`);
  if (chatTpl) items.push(`<span class="attn-geo-item">Chat template: <strong>✓</strong></span>`);
  if (!items.length) return '';
  return `<div class="attn-geometry">${items.join('<span class="attn-geo-sep">·</span>')}</div>`;
}

function renderRopeInfo(model) {
  const freqBase = getMeta(model, 'rope.freq_base', 'rope_freq_base');
  const scalingType = getMeta(model, 'rope.scaling.type', 'rope_scaling_type');
  const ropeDim = getMeta(model, 'rope.dimension_count', 'rope_dimension_count');
  const scalingFactor = getMeta(model, 'rope.scaling.factor', 'rope_scaling_factor');
  if (freqBase == null && scalingType == null && ropeDim == null) return '';
  const items = [];
  if (ropeDim != null) items.push(`<span class="attn-geo-item"><strong>${ropeDim}</strong> RoPE dim</span>`);
  if (freqBase != null) items.push(`<span class="attn-geo-item">base <strong>${Number(freqBase).toLocaleString()}</strong></span>`);
  if (scalingType) {
    let label = scalingType;
    if (scalingFactor != null) label += ` ×${scalingFactor}`;
    items.push(`<span class="attn-geo-item">scaling: <strong>${label}</strong></span>`);
  }
  if (!items.length) return '';
  return `<div class="attn-geometry">${items.join('<span class="attn-geo-sep">·</span>')}</div>`;
}

// Standard GGML file type IDs → human labels
const GGML_FILE_TYPES = {
  0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0',
  8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K', 11: 'Q3_K_S', 12: 'Q3_K_M',
  13: 'Q3_K_L', 14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M',
  18: 'Q6_K', 19: 'IQ2_XXS', 20: 'IQ2_XS', 21: 'IQ3_XXS',
  22: 'IQ1_S', 23: 'IQ4_NL', 24: 'IQ3_S', 25: 'IQ2_S', 26: 'IQ4_XS',
  27: 'IQ1_M', 28: 'BF16',
};

const QUANT_COLORS = {
  'F32': '#e06c75', 'F16': '#e5c07b', 'BF16': '#d19a66',
  'Q8_0': '#98c379', 'Q8_1': '#7ec87e', 'Q8_K': '#56b6c2',
  'Q6_K': '#61afef', 'Q5_K': '#5199d4', 'Q5_1': '#4a90d9', 'Q5_0': '#4585cc',
  'Q4_K': '#c678dd', 'Q4_1': '#b06ccc', 'Q4_0': '#a55fbf',
  'Q3_K': '#d19a66', 'Q2_K': '#e86767',
  'IQ4_XS': '#c678dd', 'IQ4_NL': '#b86cd5', 'IQ3_S': '#d4a157',
  'IQ3_XXS': '#cc9544', 'IQ2_XS': '#e07070', 'IQ2_XXS': '#d96060',
  'IQ2_S': '#d55858', 'IQ1_S': '#cc4c4c', 'IQ1_M': '#c04040',
};

function renderQuantProfile(model) {
  if (!model.quantProfile?.length) return '';
  const fileTypeLabel = model.fileType != null ? (GGML_FILE_TYPES[model.fileType] || `type ${model.fileType}`) : null;
  const bars = model.quantProfile.map(q => {
    const pct = (q.pct * 100).toFixed(1);
    const color = QUANT_COLORS[q.type] || '#888';
    return `<div class="quant-bar-seg" style="flex:${Math.max(q.pct, 0.01)};background:${color}" title="${q.type}: ${pct}% (${q.count} tensors, ${formatBytes(q.bytes)})"></div>`;
  }).join('');
  const legend = model.quantProfile.map(q => {
    const pct = (q.pct * 100).toFixed(1);
    const color = QUANT_COLORS[q.type] || '#888';
    return `<span class="quant-legend-item"><span class="quant-legend-dot" style="background:${color}"></span>${q.type} <span class="quant-legend-pct">${pct}%</span></span>`;
  }).join('');
  return `<div class="quant-profile">
    <div class="quant-profile-header">Quantization${fileTypeLabel ? ` · <strong>${fileTypeLabel}</strong>` : ''}</div>
    <div class="quant-bar">${bars}</div>
    <div class="quant-legend">${legend}</div>
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
    ${renderProvenance(model)}
    ${renderAttentionGeometry(model)}
    ${renderRopeInfo(model)}
    ${renderTokenizerInfo(model)}
    ${renderQuantProfile(model)}
  `;

  // Raw metadata browser (expandable)
  if (model.metadata && Object.keys(model.metadata).length) {
    const browser = document.createElement('div');
    browser.className = 'metadata-browser';
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'metadata-toggle';
    toggle.innerHTML = '<span class="metadata-toggle-icon">▸</span> Raw GGUF Metadata <span class="metadata-count">' + Object.keys(model.metadata).length + ' keys</span>';
    const table = document.createElement('div');
    table.className = 'metadata-table hidden';
    const sortedKeys = Object.keys(model.metadata).sort();
    for (const key of sortedKeys) {
      const val = model.metadata[key];
      let display;
      if (Array.isArray(val)) {
        display = val.length > 8 ? `Array[${val.length}]` : JSON.stringify(val);
      } else if (typeof val === 'string' && val.length > 200) {
        display = val.slice(0, 200) + '…';
      } else {
        display = String(val);
      }
      const row = document.createElement('div');
      row.className = 'metadata-row';
      row.innerHTML = `<span class="metadata-key">${escHtml(key)}</span><span class="metadata-val">${escHtml(display)}</span>`;
      table.appendChild(row);
    }
    toggle.addEventListener('click', () => {
      table.classList.toggle('hidden');
      toggle.querySelector('.metadata-toggle-icon').textContent = table.classList.contains('hidden') ? '▸' : '▾';
    });
    browser.appendChild(toggle);
    browser.appendChild(table);
    summary.appendChild(browser);
  }

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