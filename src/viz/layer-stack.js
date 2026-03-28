/**
 * D3.js Layer Stack Visualization
 * Renders model layers as a vertical block diagram with expandable sub-components.
 * Includes: tensor shape diagrams, quantization breakdown, attention geometry,
 * memory footprint, and dataflow mini-diagrams.
 */

const CATEGORY_COLORS = {
  embedding: '#4a90d9',
  attention: '#e6894a',
  mlp:       '#6abf69',
  norm:      '#b07cc6',
  output:    '#d94a4a',
  moe:       '#d9c74a',
  ssm:       '#4ad9c7',
  other:     '#999',
};

const QUANT_COLORS = {
  'F32': '#d94a4a', 'F16': '#e6894a', 'BF16': '#e6894a',
  'Q8_0': '#d9c74a', 'Q8_1': '#d9c74a', 'Q8_K': '#d9c74a',
  'Q6_K': '#6abf69', 'Q5_0': '#4ad9c7', 'Q5_1': '#4ad9c7',
  'Q5_K': '#4ad9c7', 'Q4_0': '#4a90d9', 'Q4_1': '#4a90d9',
  'Q4_K': '#4a90d9', 'Q3_K': '#b07cc6', 'Q2_K': '#b07cc6',
  'IQ4_NL': '#4a90d9', 'IQ4_XS': '#4a90d9', 'IQ3_S': '#b07cc6',
  'IQ3_XXS': '#b07cc6', 'IQ2_XXS': '#d94a6a', 'IQ2_XS': '#d94a6a',
  'IQ2_S': '#d94a6a', 'IQ1_S': '#d94a6a', 'IQ1_M': '#d94a6a',
};

function formatParams(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

function formatBytes(bytes) {
  if (!bytes) return '—';
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB';
  return bytes + ' B';
}

function formatDims(dims) {
  if (!dims || dims.length === 0) return '—';
  return dims.join(' × ');
}

// --- Tensor Shape Diagram ---
function renderShapeDiagram(tensors) {
  if (!tensors || tensors.length === 0) return '';
  const hasDims = tensors.some(t => t.dimensions && t.dimensions.length > 0);
  if (!hasDims) return '';

  const maxEl = Math.max(...tensors.map(t => t.numElements || 0));
  let html = '<div class="shape-diagram">';
  for (const t of tensors) {
    if (!t.dimensions || t.dimensions.length === 0) continue;
    const pct = maxEl > 0 ? Math.max(8, (t.numElements / maxEl) * 100) : 20;
    const shortName = (t.component || t.name).replace('.weight', '');
    html += `<div class="shape-row">
      <span class="shape-name">${shortName}</span>
      <div class="shape-rect" style="width:${pct}%" title="${t.name}: [${t.dimensions.join('×')}]">
        <span class="shape-dims">${t.dimensions.join('×')}</span>
      </div>
    </div>`;
  }
  html += '</div>';
  return html;
}

// --- Quantization Breakdown ---
function renderQuantBreakdown(tensors) {
  if (!tensors || tensors.length === 0) return '';
  const counts = {};
  let total = 0;
  for (const t of tensors) {
    const tn = t.typeName || 'unknown';
    counts[tn] = (counts[tn] || 0) + (t.numElements || 1);
    total += (t.numElements || 1);
  }
  if (total === 0) return '';

  let html = '<div class="quant-breakdown"><div class="quant-bar">';
  for (const [name, count] of Object.entries(counts).sort((a, b) => b[1] - a[1])) {
    const pct = (count / total * 100);
    const color = QUANT_COLORS[name] || '#777';
    html += `<div class="quant-segment" style="width:${pct}%;background:${color}" title="${name}: ${pct.toFixed(1)}%"></div>`;
  }
  html += '</div><div class="quant-legend">';
  for (const [name, count] of Object.entries(counts).sort((a, b) => b[1] - a[1])) {
    const pct = (count / total * 100).toFixed(1);
    const color = QUANT_COLORS[name] || '#777';
    html += `<span class="quant-label"><span class="quant-dot" style="background:${color}"></span>${name} ${pct}%</span>`;
  }
  html += '</div></div>';
  return html;
}

// --- Attention Geometry ---
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

// --- Dataflow Mini-Diagram ---
function renderDataflow(layer) {
  if (layer.type !== 'block') return '';
  // Detect which components are present
  const comps = new Set(layer.tensors.map(t => t.component));
  const hasGate = comps.has('ffn_gate');
  const hasMoE = comps.has('ffn_gate_inp');
  const hasMoEGate = comps.has('ffn_gate_exp');

  const nodes = [
    { id: 'in', label: 'Input' },
    { id: 'anorm', label: 'Norm' },
  ];

  if (comps.has('attn_qkv')) {
    nodes.push({ id: 'qkv', label: 'QKV' });
  } else {
    nodes.push({ id: 'q', label: 'Q' }, { id: 'kv', label: 'K,V' });
  }
  nodes.push(
    { id: 'attn', label: 'Attn' },
    { id: 'proj', label: 'Proj' },
    { id: 'res1', label: '+' },
    { id: 'fnorm', label: 'Norm' },
  );
  if (hasMoE) {
    nodes.push(
      { id: 'router', label: 'Router' },
      { id: 'up', label: 'Up' },
      ...(hasMoEGate ? [{ id: 'gate', label: 'Gate' }] : []),
      { id: 'down', label: 'Down' },
    );
  } else if (hasGate) {
    nodes.push({ id: 'gate', label: 'Gate' }, { id: 'up', label: 'Up' }, { id: 'down', label: 'Down' });
  } else {
    nodes.push({ id: 'up', label: 'Up' }, { id: 'down', label: 'Down' });
  }
  nodes.push({ id: 'res2', label: '+' }, { id: 'out', label: 'Output' });

  let html = '<div class="dataflow">';
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    const isResidual = n.label === '+';
    const cls = isResidual ? 'df-node df-residual' : 'df-node';
    html += `<span class="${cls}">${n.label}</span>`;
    if (i < nodes.length - 1) html += '<span class="df-arrow">→</span>';
  }
  html += '</div>';
  return html;
}


export function renderLayerStack(container, model) {
  const el = typeof container === 'string' ? document.querySelector(container) : container;
  el.innerHTML = '';

  // Model summary header
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
  el.appendChild(summary);

  // Layer stack
  const stack = document.createElement('div');
  stack.className = 'layer-stack';

  const maxParams = Math.max(...model.layers.map(l => l.params));

  for (const layer of model.layers) {
    const layerEl = document.createElement('div');
    layerEl.className = `layer layer-${layer.type}`;

    const widthPct = Math.max(20, (layer.params / maxParams) * 100);
    const memLabel = layer.memoryBytes ? ` · ${formatBytes(layer.memoryBytes)}` : '';
    const headerEl = document.createElement('div');
    headerEl.className = 'layer-header';
    headerEl.innerHTML = `
      <div class="layer-bar" style="width:${widthPct}%; background:${layer.type === 'block' ? '#3a3f4b' : CATEGORY_COLORS[layer.type] || '#555'}">
        <span class="layer-label">${layer.label}</span>
        <span class="layer-params">${formatParams(layer.params)}${memLabel}</span>
        ${layer.type === 'block' ? '<span class="expand-icon">▸</span>' : ''}
      </div>
    `;

    layerEl.appendChild(headerEl);

    // Subgroups for blocks
    if (layer.type === 'block' && layer.subgroups) {
      const detail = document.createElement('div');
      detail.className = 'layer-detail hidden';

      // Dataflow mini-diagram
      const dfHtml = renderDataflow(layer);
      if (dfHtml) {
        const dfEl = document.createElement('div');
        dfEl.innerHTML = dfHtml;
        detail.appendChild(dfEl.firstElementChild);
      }

      // Quant breakdown for the block
      const qbHtml = renderQuantBreakdown(layer.tensors);
      if (qbHtml) {
        const qbEl = document.createElement('div');
        qbEl.innerHTML = qbHtml;
        detail.appendChild(qbEl.firstElementChild);
      }

      // Shape diagram for the block
      const sdHtml = renderShapeDiagram(layer.tensors);
      if (sdHtml) {
        const sdEl = document.createElement('div');
        sdEl.innerHTML = sdHtml;
        detail.appendChild(sdEl.firstElementChild);
      }

      for (const sg of layer.subgroups) {
        const sgEl = document.createElement('div');
        sgEl.className = 'subgroup';
        const sgWidthPct = Math.max(15, (sg.params / layer.params) * 100);
        const sgMem = sg.memoryBytes ? ` · ${formatBytes(sg.memoryBytes)}` : '';
        sgEl.innerHTML = `
          <div class="subgroup-bar" style="width:${sgWidthPct}%; background:${CATEGORY_COLORS[sg.label.toLowerCase()] || '#777'}">
            <span class="sg-label">${sg.label}</span>
            <span class="sg-params">${formatParams(sg.params)}${sgMem}</span>
          </div>
        `;

        // Tensor details on click
        const tensorList = document.createElement('div');
        tensorList.className = 'tensor-list hidden';
        for (const t of sg.tensors) {
          const tMem = t.memoryBytes ? formatBytes(t.memoryBytes) : '';
          tensorList.innerHTML += `<div class="tensor-item">
            <span class="tensor-name">${t.name}</span>
            <span class="tensor-shape">[${formatDims(t.dimensions)}]</span>
            <span class="tensor-mem">${tMem}</span>
            <span class="tensor-type">${t.typeName}</span>
          </div>`;
        }
        sgEl.appendChild(tensorList);
        sgEl.querySelector('.subgroup-bar').addEventListener('click', () => {
          tensorList.classList.toggle('hidden');
        });
        detail.appendChild(sgEl);
      }

      layerEl.appendChild(detail);
      headerEl.addEventListener('click', () => {
        detail.classList.toggle('hidden');
        const icon = headerEl.querySelector('.expand-icon');
        if (icon) icon.textContent = detail.classList.contains('hidden') ? '▸' : '▾';
      });
    } else {
      // Non-block layers: show tensors directly on click
      const tensorList = document.createElement('div');
      tensorList.className = 'tensor-list hidden';
      for (const t of layer.tensors) {
        const tMem = t.memoryBytes ? formatBytes(t.memoryBytes) : '';
        tensorList.innerHTML += `<div class="tensor-item">
          <span class="tensor-name">${t.name}</span>
          <span class="tensor-shape">[${formatDims(t.dimensions)}]</span>
          <span class="tensor-mem">${tMem}</span>
          <span class="tensor-type">${t.typeName}</span>
        </div>`;
      }
      layerEl.appendChild(tensorList);
      headerEl.addEventListener('click', () => tensorList.classList.toggle('hidden'));
    }

    stack.appendChild(layerEl);
  }

  el.appendChild(stack);
}