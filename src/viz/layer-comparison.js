/**
 * Cross-layer comparison visualizations:
 * 1. Per-layer quantization grid
 * 2. Parameter & memory budget chart
 * 3. Weight distribution sparklines (requires ggufSource)
 */
import { decodeRows } from '../model/gguf-tensor-decoder.js';
import { computeTensorBytes } from '../parsers/gguf.js';

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
const CAT_COLORS = {
  attention: '#e6894a', mlp: '#6abf69', norm: '#b07cc6',
  embedding: '#4a90d9', output: '#d94a4a', moe: '#d9c74a',
  ssm: '#4ad9c7', other: '#999',
};
const COMP_ORDER = ['attn_norm','attn_q','attn_k','attn_v','attn_output','ffn_norm','ffn_gate','ffn_up','ffn_down'];
const COMP_SHORT = { attn_norm:'ANorm', attn_q:'Q', attn_k:'K', attn_v:'V', attn_output:'O', ffn_norm:'FNorm', ffn_gate:'Gate', ffn_up:'Up', ffn_down:'Down' };

function fmtP(n) { return n >= 1e9 ? (n/1e9).toFixed(2)+'B' : n >= 1e6 ? (n/1e6).toFixed(1)+'M' : n >= 1e3 ? (n/1e3).toFixed(1)+'K' : String(n); }
function fmtB(v) { return !v ? '—' : v >= 1e9 ? (v/1e9).toFixed(2)+' GB' : v >= 1e6 ? (v/1e6).toFixed(1)+' MB' : v >= 1e3 ? (v/1e3).toFixed(1)+' KB' : v+' B'; }
function getBlocks(m) { return (m.layers||[]).filter(l => l.type === 'block'); }
function mk(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }

// ── 1. Quantization Grid ────────────────────────────────────────────
function renderQuantGrid(wrap, model) {
  const blocks = getBlocks(model);
  if (!blocks.length) return;
  const sec = mk('div','layer-cmp-section');
  sec.innerHTML = '<h3 class="layer-cmp-title">Per-Layer Quantization</h3>';
  const cols = COMP_ORDER.filter(c => blocks.some(b => b.tensors?.some(t => t.component === c)));
  if (!cols.length) { sec.innerHTML += '<p class="layer-cmp-empty">No per-component quantization data available.</p>'; wrap.appendChild(sec); return; }
  let h = '<div class="quant-grid-wrap"><table class="quant-grid"><thead><tr><th>Blk</th>';
  for (const c of cols) h += `<th>${COMP_SHORT[c]||c}</th>`;
  h += '</tr></thead><tbody>';
  for (const b of blocks) {
    h += `<tr><td class="qg-label">${b.index}</td>`;
    for (const c of cols) {
      const t = b.tensors?.find(t => t.component === c);
      if (t) { const tn = t.typeName||'?'; h += `<td class="qg-cell" style="background:${QUANT_COLORS[tn]||'#555'}" title="Block ${b.index} ${c}: ${tn}">${tn}</td>`; }
      else h += '<td class="qg-cell qg-empty">—</td>';
    }
    h += '</tr>';
  }
  h += '</tbody></table></div>';
  sec.innerHTML += h; wrap.appendChild(sec);
}

// ── 2. Budget Chart ─────────────────────────────────────────────────
function renderBudgetChart(wrap, model) {
  const blocks = getBlocks(model);
  if (!blocks.length) return;
  const sec = mk('div','layer-cmp-section');
  sec.innerHTML = '<h3 class="layer-cmp-title">Parameter & Memory Budget</h3>';
  const cats = ['attention','mlp','norm','moe','ssm','other'];
  const maxP = Math.max(...blocks.map(b => b.params||0));
  const maxM = Math.max(...blocks.map(b => b.memoryBytes||0));
  let h = '<div class="budget-chart"><div class="budget-legend">';
  for (const c of cats) { if (blocks.some(b => b.subgroups?.some(s => s.label.toLowerCase()===c && s.params>0))) h += `<span class="budget-legend-item"><span class="budget-legend-dot" style="background:${CAT_COLORS[c]}"></span>${c}</span>`; }
  h += '</div><div class="budget-group"><div class="budget-group-label">Parameters</div>';
  for (const b of blocks) h += budgetBar(b, cats, maxP, 'params');
  h += '</div><div class="budget-group"><div class="budget-group-label">Memory</div>';
  for (const b of blocks) h += budgetBar(b, cats, maxM, 'memoryBytes');
  h += '</div></div>';
  sec.innerHTML += h; wrap.appendChild(sec);
}
function budgetBar(block, cats, maxVal, field) {
  const total = block[field]||0, pct = maxVal > 0 ? (total/maxVal)*100 : 0;
  let segs = '';
  for (const c of cats) { const sg = block.subgroups?.find(s => s.label.toLowerCase()===c); if (sg && sg[field]>0) { const sp = total>0?(sg[field]/total)*100:0; segs += `<div class="budget-seg" style="width:${sp}%;background:${CAT_COLORS[c]}" title="${c}: ${field==='params'?fmtP(sg[field]):fmtB(sg[field])}"></div>`; } }
  return `<div class="budget-row"><span class="budget-row-label">${block.index}</span><div class="budget-bar" style="width:${Math.max(pct,2)}%">${segs}</div><span class="budget-row-val">${field==='params'?fmtP(total):fmtB(total)}</span></div>`;
}

// ── 3. Weight Distribution Sparklines ───────────────────────────────
const BINS = 40, SAMPLE_ROWS = 8;

function renderSparklines(wrap, model) {
  const blocks = getBlocks(model);
  if (!blocks.length) return;
  const sec = mk('div','layer-cmp-section');
  sec.innerHTML = '<h3 class="layer-cmp-title">Weight Distribution by Layer</h3>';
  if (!model.ggufSource?.slice) { sec.innerHTML += '<p class="layer-cmp-empty">Weight distributions require an uploaded GGUF file.</p>'; wrap.appendChild(sec); return; }
  const box = mk('div','sparkline-container');
  sec.appendChild(box); wrap.appendChild(sec);
  decodeSparklines(box, blocks, model).catch(e => { box.innerHTML = `<p class="layer-cmp-empty">Decode error: ${e.message}</p>`; });
}

async function decodeSparklines(box, blocks, model) {
  const file = model.ggufSource, tds = model.ggufInfo?.tensorDataStart;
  if (!Number.isFinite(tds)) { box.innerHTML = '<p class="layer-cmp-empty">Tensor data offset unavailable.</p>'; return; }
  const targets = blocks.map(b => {
    for (const c of ['attn_q','ffn_up','attn_output','ffn_down']) { const t = b.tensors?.find(t => t.component===c && t.type>=0 && t.dimensions?.length>=2); if (t) return {b,t}; }
    const t = b.tensors?.find(t => t.type>=0 && t.dimensions?.length>=2); return t ? {b,t} : null;
  }).filter(Boolean);
  if (!targets.length) { box.innerHTML = '<p class="layer-cmp-empty">No decodable tensors.</p>'; return; }
  box.innerHTML = `<p class="layer-cmp-loading">Decoding samples from ${targets.length} layers…</p>`;
  const results = [];
  for (const {b,t} of targets) {
    try {
      const cols = Number(t.dimensions[0]), stride = computeTensorBytes(t.type, cols);
      if (!stride || !cols) continue;
      const rows = Math.min(SAMPLE_ROWS, Number(t.dimensions[1]||1));
      const off = Number.isFinite(t.absoluteOffset) ? t.absoluteOffset : (t.offset + tds);
      const buf = await file.slice(off, off + rows * stride).arrayBuffer();
      results.push({b, t, values: decodeRows(t.type, buf, rows, cols)});
    } catch { /* skip */ }
  }
  if (!results.length) { box.innerHTML = '<p class="layer-cmp-empty">Could not decode weights.</p>'; return; }
  box.innerHTML = '';
  const gMax = Math.max(...results.map(r => { let m=0; for (let i=0;i<r.values.length;i++) m=Math.max(m,Math.abs(r.values[i])); return m; }));
  for (const {b, t, values} of results) {
    const row = mk('div','sparkline-row');
    row.innerHTML = `<span class="sparkline-label">${b.index}</span>`;
    const cv = document.createElement('canvas'); cv.className = 'sparkline-canvas'; cv.width = BINS; cv.height = 24;
    drawHist(cv, values, gMax); row.appendChild(cv);
    let sum=0; for (let i=0;i<values.length;i++) sum+=values[i];
    row.innerHTML += `<span class="sparkline-stats" title="${t.component} · ${t.typeName}">μ=${(sum/values.length).toFixed(4)} ±${gMax.toFixed(2)}</span>`;
    box.appendChild(row);
  }
}
function drawHist(canvas, values, absMax) {
  const bins = new Float64Array(BINS);
  for (let i=0;i<values.length;i++) { const n = absMax>0?(values[i]/absMax+1)*0.5:0.5; bins[Math.max(0,Math.min(BINS-1,Math.floor(n*BINS)))]++; }
  const mx = Math.max(...bins), ctx = canvas.getContext('2d'); if (!ctx) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for (let i=0;i<BINS;i++) { const bh = mx>0?(bins[i]/mx)*24:0, t=i/(BINS-1); ctx.fillStyle = `rgb(${Math.round(59+121*t)},${Math.round(76-72*t*t)},${Math.round(192-154*t)})`; ctx.fillRect(i,24-bh,1,bh); }
}

// ── Main entry ──────────────────────────────────────────────────────
export function renderLayerComparison(container, model) {
  container.innerHTML = '';
  const wrap = mk('div','layer-comparison');
  container.appendChild(wrap);
  renderBudgetChart(wrap, model);
  renderQuantGrid(wrap, model);
  renderSparklines(wrap, model);
}
