import { buildResidualFlowGraph } from '../model/residual-flow-graph.js';
import { decodeHeadSelection } from '../model/gguf-tensor-decoder.js';

const NS = 'http://www.w3.org/2000/svg';
const COLORS = {
  input: '#58606f',
  embedding: '#4a90d9',
  block: '#3a3f4b',
  norm: '#b07cc6',
  output: '#d94a4a',
  attention: '#e6894a',
  mlp: '#6abf69',
  moe: '#d9c74a',
  ssm: '#4ad9c7',
  residual: '#93a0b4',
  selected: '#71b7ff',
};

const headDecodeCache = new WeakMap();

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

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

function formatFloat(value) {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs >= 1000 || (abs > 0 && abs < 0.001)) return value.toExponential(2);
  if (abs >= 10) return value.toFixed(2);
  if (abs >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

function svgElement(name, attrs = {}, text = '') {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text) node.textContent = text;
  return node;
}

function createButton(label, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'toolbar-btn';
  button.textContent = label;
  button.addEventListener('click', onClick);
  return button;
}

function renderLegend(container) {
  const legend = document.createElement('div');
  legend.className = 'architecture-legend';
  const items = [
    ['Residual stream', COLORS.residual],
    ['Attention', COLORS.attention],
    ['MLP', COLORS.mlp],
    ['Norm', COLORS.norm],
    ['Output', COLORS.output],
  ];
  for (const [label, color] of items) {
    const item = document.createElement('span');
    item.className = 'legend-item';
    item.innerHTML = `<span class="legend-swatch" style="background:${color}"></span>${label}`;
    legend.appendChild(item);
  }
  container.appendChild(legend);
}

function formatMaybeParams(value) {
  return Number.isFinite(value) && value > 0 ? formatParams(Math.round(value)) : '—';
}

function formatMaybeBytes(value) {
  return Number.isFinite(value) && value > 0 ? formatBytes(value) : '—';
}

function formatHeadRange(start, end) {
  if (!Number.isFinite(start) || !Number.isFinite(end)) return '—';
  return start === end ? `${start}` : `${start}–${end}`;
}

function getAttentionTypeLabel(headInfo, verbose = false) {
  if (!headInfo?.headCount) return 'Attention';
  const isMHA = headInfo.headCountKV === headInfo.headCount;
  const isGQA = headInfo.headCountKV > 1 && headInfo.headCountKV < headInfo.headCount;
  const isMQA = headInfo.headCountKV === 1 && headInfo.headCount > 1;
  if (verbose) {
    return isMHA ? 'Multi-Head (MHA)' : isGQA ? 'Grouped-Query (GQA)' : isMQA ? 'Multi-Query (MQA)' : 'Attention';
  }
  return isMHA ? 'MHA' : isGQA ? 'GQA' : isMQA ? 'MQA' : 'Attention';
}

function getHeadGroups(headInfo) {
  if (!headInfo?.headCount) return [];
  const headCount = headInfo.headCount;
  const groupCount = Math.max(1, headInfo.headCountKV || 1);
  return Array.from({ length: groupCount }, (_, kvIndex) => {
    const qStart = Math.floor(kvIndex * headCount / groupCount);
    const qEnd = Math.max(qStart, Math.min(headCount - 1, Math.floor((kvIndex + 1) * headCount / groupCount) - 1));
    const qHeads = Array.from({ length: qEnd - qStart + 1 }, (_, idx) => qStart + idx);
    return { kvIndex, qStart, qEnd, qHeads };
  });
}

function getKVGroupIndex(headInfo, qHeadIndex) {
  if (!headInfo?.headCount) return 0;
  const groupCount = Math.max(1, headInfo.headCountKV || 1);
  return clamp(Math.floor(qHeadIndex * groupCount / headInfo.headCount), 0, groupCount - 1);
}

function getHeadSelection(headInfo, selectedHead) {
  if (!headInfo?.headCount || !selectedHead) return { selectedQIndex: null, selectedKVIndex: null };
  if (selectedHead.kind === 'q') {
    const selectedQIndex = clamp(selectedHead.index, 0, headInfo.headCount - 1);
    return { selectedQIndex, selectedKVIndex: getKVGroupIndex(headInfo, selectedQIndex) };
  }
  if (selectedHead.kind === 'kv') {
    return {
      selectedQIndex: null,
      selectedKVIndex: clamp(selectedHead.index, 0, Math.max(0, (headInfo.headCountKV || 1) - 1)),
    };
  }
  return { selectedQIndex: null, selectedKVIndex: null };
}

function formatDimsLabel(dims) {
  const values = (dims || []).filter((value) => Number.isFinite(value) && value > 0);
  return values.length ? `[${values.join(' × ')}]` : '—';
}

function getTensorRowCount(tensor) {
  const dimensions = tensor?.dimensions || tensor?.shape || [];
  const directRows = Number(dimensions[1]) || 0;
  if (directRows > 0) return directRows;
  const cols = Number(dimensions[0]) || 0;
  const numElements = Number(tensor?.numElements) || 0;
  if (!cols || !numElements) return 0;
  const totalRows = Math.round(numElements / cols);
  return Number.isFinite(totalRows) && totalRows > 0 ? totalRows : 0;
}

function createHeadCellButton(kind, index, label, selected, related, title, onSelect) {
  const cell = document.createElement('button');
  cell.type = 'button';
  const kindClass = kind === 'q' ? 'q-head' : kind === 'kv' ? 'kv-head' : 'wo-head';
  cell.className = `head-cell ${kindClass}`;
  if (selected) cell.classList.add('selected');
  else if (related) cell.classList.add('related');
  if (label !== undefined && label !== null) cell.textContent = `${label}`;
  if (title) {
    cell.title = title;
    cell.setAttribute('aria-label', title);
  }
  cell.setAttribute('aria-pressed', selected ? 'true' : 'false');
  cell.addEventListener('click', () => onSelect?.({ kind, index }));
  return cell;
}

function getSelectedHeadDetail(stage, selectedHead) {
  if (!stage?.headInfo?.headCount || !selectedHead) return null;
  const hi = stage.headInfo;
  const ad = stage.attentionDetail;
  const groups = getHeadGroups(hi);

  if (selectedHead.kind === 'q') {
    const headIndex = clamp(selectedHead.index, 0, hi.headCount - 1);
    const kvGroupIndex = getKVGroupIndex(hi, headIndex);
    const group = groups[kvGroupIndex];
    const rowStart = headIndex * hi.headDim;
    const rowEnd = rowStart + hi.headDim - 1;
    let params = null;
    let memoryBytes = null;
    let quantText = '—';
    let sliceText = `Rows ${rowStart}–${rowEnd}`;
    let decodePlan = null;

    if (ad?.q) {
      params = ad.q.params / hi.headCount;
      memoryBytes = ad.q.memoryBytes / hi.headCount;
      quantText = ad.q.typeName || '—';
      sliceText = `Rows ${rowStart}–${rowEnd} in attn_q`;
      decodePlan = {
        cacheKey: `${stage.index}:q:${headIndex}`,
        slices: [{ label: 'Q projection', tensorLabel: 'attn_q', tensor: ad.q, rowStart, rowCount: hi.headDim }],
      };
    } else if (ad?.qkv) {
      const logicalSlices = hi.headCount + (hi.headCountKV || 1) * 2;
      params = logicalSlices ? ad.qkv.params / logicalSlices : null;
      memoryBytes = logicalSlices ? ad.qkv.memoryBytes / logicalSlices : null;
      quantText = ad.qkv.typeName || '—';
      sliceText = `Rows ${rowStart}–${rowEnd} in attn_qkv (Q segment)`;
      decodePlan = {
        cacheKey: `${stage.index}:q:${headIndex}`,
        slices: [{ label: 'Q projection', tensorLabel: 'attn_qkv · Q segment', tensor: ad.qkv, rowStart, rowCount: hi.headDim }],
      };
    }

    const outputParams = ad?.output ? ad.output.params / hi.headCount : null;
    const outputMemory = ad?.output ? ad.output.memoryBytes / hi.headCount : null;
    return {
      title: `Q head ${headIndex}`,
      badge: 'Query slice',
      note: group ? `Shares KV head ${kvGroupIndex} with Q heads ${formatHeadRange(group.qStart, group.qEnd)}.` : 'Selected query head.',
      fields: [
        { label: 'Tensor slice', value: sliceText },
        { label: 'Estimated params', value: formatMaybeParams(params) },
        { label: 'Estimated memory', value: formatMaybeBytes(memoryBytes) },
        { label: 'Quantization', value: quantText },
        { label: 'KV mapping', value: `Q ${headIndex} → KV ${kvGroupIndex}` },
        { label: 'Output share', value: `${formatMaybeParams(outputParams)} · ${formatMaybeBytes(outputMemory)}` },
      ],
      decodePlan,
    };
  }

  if (selectedHead.kind === 'kv') {
    const kvIndex = clamp(selectedHead.index, 0, Math.max(0, (hi.headCountKV || 1) - 1));
    const group = groups[kvIndex];
    const rowStart = kvIndex * hi.headDim;
    const rowEnd = rowStart + hi.headDim - 1;
    let params = null;
    let memoryBytes = null;
    let quantText = '—';
    let sliceText = `Rows ${rowStart}–${rowEnd}`;
    let decodePlan = null;

    if (ad?.k || ad?.v) {
      const kParams = ad?.k ? ad.k.params / (hi.headCountKV || 1) : 0;
      const vParams = ad?.v ? ad.v.params / (hi.headCountKV || 1) : 0;
      const kMemory = ad?.k ? ad.k.memoryBytes / (hi.headCountKV || 1) : 0;
      const vMemory = ad?.v ? ad.v.memoryBytes / (hi.headCountKV || 1) : 0;
      params = kParams + vParams;
      memoryBytes = kMemory + vMemory;
      quantText = ad?.k?.typeName === ad?.v?.typeName ? (ad?.k?.typeName || ad?.v?.typeName || '—') : `K ${ad?.k?.typeName || '—'} • V ${ad?.v?.typeName || '—'}`;
      sliceText = `K rows ${rowStart}–${rowEnd} • V rows ${rowStart}–${rowEnd}`;
      decodePlan = {
        cacheKey: `${stage.index}:kv:${kvIndex}`,
        slices: [
          ad?.k ? { label: 'K projection', tensorLabel: 'attn_k', tensor: ad.k, rowStart, rowCount: hi.headDim } : null,
          ad?.v ? { label: 'V projection', tensorLabel: 'attn_v', tensor: ad.v, rowStart, rowCount: hi.headDim } : null,
        ].filter(Boolean),
      };
    } else if (ad?.qkv) {
      const groupCount = hi.headCountKV || 1;
      const logicalSlices = hi.headCount + groupCount * 2;
      const qRows = hi.headCount * hi.headDim;
      const kStart = qRows + kvIndex * hi.headDim;
      const kEnd = kStart + hi.headDim - 1;
      const vStart = qRows + groupCount * hi.headDim + kvIndex * hi.headDim;
      const vEnd = vStart + hi.headDim - 1;
      params = logicalSlices ? (ad.qkv.params / logicalSlices) * 2 : null;
      memoryBytes = logicalSlices ? (ad.qkv.memoryBytes / logicalSlices) * 2 : null;
      quantText = ad.qkv.typeName || '—';
      sliceText = `K rows ${kStart}–${kEnd} • V rows ${vStart}–${vEnd} in attn_qkv`;
      decodePlan = {
        cacheKey: `${stage.index}:kv:${kvIndex}`,
        slices: [
          { label: 'K projection', tensorLabel: 'attn_qkv · K segment', tensor: ad.qkv, rowStart: kStart, rowCount: hi.headDim },
          { label: 'V projection', tensorLabel: 'attn_qkv · V segment', tensor: ad.qkv, rowStart: vStart, rowCount: hi.headDim },
        ],
      };
    }

    return {
      title: `KV head ${kvIndex}`,
      badge: 'Shared key/value slice',
      note: group ? `Serves Q heads ${formatHeadRange(group.qStart, group.qEnd)} (${group.qHeads.length} query heads).` : 'Selected KV head.',
      fields: [
        { label: 'Tensor slice', value: sliceText },
        { label: 'Estimated params', value: formatMaybeParams(params) },
        { label: 'Estimated memory', value: formatMaybeBytes(memoryBytes) },
        { label: 'Quantization', value: quantText },
        { label: 'Q fan-in', value: group ? `Q heads ${formatHeadRange(group.qStart, group.qEnd)}` : '—' },
        { label: 'Group size', value: group ? `${group.qHeads.length} queries per KV head` : '—' },
      ],
      decodePlan,
    };
  }

  if (selectedHead.kind === 'wo') {
    const outputRows = getTensorRowCount(ad?.output);
    const previewRows = Math.min(outputRows || 0, 96);
    const previewEnd = Math.max(0, previewRows - 1);
    return {
      title: 'Wo',
      badge: 'Output projection',
      note: previewRows && outputRows > previewRows
        ? `Showing rows 0–${previewEnd} of Wo as a bounded preview so large output projections stay responsive.`
        : 'Attention output projection applied after concatenating all head outputs.',
      fields: [
        { label: 'Tensor slice', value: previewRows ? `Rows 0–${previewEnd} in attn_output` : 'attn_output' },
        { label: 'Tensor shape', value: ad?.output?.shape?.length ? `[${ad.output.shape.join(' × ')}]` : '—' },
        { label: 'Parameters', value: formatMaybeParams(ad?.output?.params) },
        { label: 'Memory', value: formatMaybeBytes(ad?.output?.memoryBytes) },
        { label: 'Quantization', value: ad?.output?.typeName || '—' },
        { label: 'Heatmap preview', value: previewRows ? `${previewRows} of ${outputRows} rows` : 'Unavailable' },
      ],
      decodePlan: previewRows
        ? {
          cacheKey: `${stage.index}:wo`,
          slices: [{ label: 'Wo projection', tensorLabel: 'attn_output', tensor: ad.output, rowStart: 0, rowCount: previewRows }],
        }
        : null,
    };
  }

  return null;
}

function getHeadDecodeState(model, decodePlan, onInvalidate) {
  if (!decodePlan?.cacheKey) return { status: 'unavailable', message: 'No decodable tensor slice is attached to this selection.', warnings: [], slices: [] };
  let modelCache = headDecodeCache.get(model);
  if (!modelCache) {
    modelCache = new Map();
    headDecodeCache.set(model, modelCache);
  }
  const existing = modelCache.get(decodePlan.cacheKey);
  if (existing) return existing;
  const loading = { status: 'loading', warnings: [], slices: [] };
  modelCache.set(decodePlan.cacheKey, loading);
  decodeHeadSelection(model, decodePlan)
    .then((result) => {
      modelCache.set(decodePlan.cacheKey, result);
      onInvalidate?.();
    })
    .catch((error) => {
      modelCache.set(decodePlan.cacheKey, { status: 'error', message: error.message || 'Failed to decode GGUF tensor slice.', warnings: [], slices: [] });
      onInvalidate?.();
    });
  return loading;
}

function createInfoBlock(className, title, message) {
  const block = document.createElement('div');
  block.className = className;
  const strong = document.createElement('strong');
  strong.textContent = title;
  const p = document.createElement('p');
  p.textContent = message;
  block.append(strong, p);
  return block;
}

function createHeatmapCanvas(values, width, height, absMax) {
  const canvas = document.createElement('canvas');
  canvas.className = 'head-heatmap-canvas';
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return canvas;
  const image = ctx.createImageData(width, height);
  for (let i = 0; i < values.length; i++) {
    const scaled = absMax ? values[i] / absMax : 0;
    const magnitude = Math.sqrt(Math.min(1, Math.abs(scaled)));
    const base = Math.round(248 - 150 * magnitude);
    const idx = i * 4;
    if (scaled >= 0) {
      image.data[idx] = 255;
      image.data[idx + 1] = base;
      image.data[idx + 2] = base;
    } else {
      image.data[idx] = base;
      image.data[idx + 1] = Math.min(255, base + 12);
      image.data[idx + 2] = 255;
    }
    image.data[idx + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function renderSelectedHeadDecode(host, model, selectedHeadDetail, onInvalidate) {
  host.innerHTML = '';
  if (!selectedHeadDetail?.decodePlan?.slices?.length) {
    host.appendChild(createInfoBlock('head-decode-status unavailable', 'Heatmap unavailable', 'This head selection does not expose a decodable GGUF tensor slice.'));
    return;
  }
  const state = getHeadDecodeState(model, selectedHeadDetail.decodePlan, onInvalidate);
  if (state.status === 'loading') {
    host.appendChild(createInfoBlock('head-decode-status loading', 'Decoding selected weights…', 'Reading GGUF tensor bytes and dequantizing approximate weight values.'));
    return;
  }
  if (state.status === 'error') {
    host.appendChild(createInfoBlock('head-decode-status error', 'Heatmap decode failed', state.message || 'Unable to decode the selected tensor slice.'));
    return;
  }
  if (state.status === 'unavailable') {
    host.appendChild(createInfoBlock('head-decode-status unavailable', 'Heatmap unavailable', state.message || 'This model cannot currently expose learned tensor values.'));
    return;
  }

  const intro = document.createElement('div');
  intro.className = 'head-decode-meta';
  intro.textContent = `Best-effort dequantized values from ${state.sourceName || 'uploaded GGUF'}. Positive weights are red, negative weights are blue.`;
  host.appendChild(intro);

  if (state.warnings?.length) {
    const warningList = document.createElement('div');
    warningList.className = 'head-warning-list';
    state.warnings.forEach((warning) => {
      const item = document.createElement('div');
      item.className = 'head-warning-item';
      item.textContent = warning;
      warningList.appendChild(item);
    });
    host.appendChild(warningList);
  }

  const list = document.createElement('div');
  list.className = 'head-heatmap-list';
  for (const slice of state.slices || []) {
    const panel = document.createElement('div');
    panel.className = 'head-heatmap-panel';
    const header = document.createElement('div');
    header.className = 'head-heatmap-header';
    header.innerHTML = `<div><strong>${slice.label}</strong><span>${slice.tensorLabel} · ${slice.tensor.typeName || 'unknown'} · rows ${slice.rowStart}–${slice.rowStart + slice.rowCount - 1}</span></div><div class="head-heatmap-shape">${slice.rows} × ${slice.cols}</div>`;
    panel.appendChild(header);

    const stats = document.createElement('div');
    stats.className = 'head-stat-grid';
    [
      ['Min', formatFloat(slice.stats.min)],
      ['Max', formatFloat(slice.stats.max)],
      ['Mean', formatFloat(slice.stats.mean)],
      ['Std', formatFloat(slice.stats.std)],
      ['|Max|', formatFloat(slice.stats.absMax)],
    ].forEach(([label, value]) => {
      const card = document.createElement('div');
      card.className = 'head-stat-card';
      card.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
      stats.appendChild(card);
    });
    panel.appendChild(stats);

    const heatmapWrap = document.createElement('div');
    heatmapWrap.className = 'head-heatmap-wrap';
    heatmapWrap.appendChild(createHeatmapCanvas(slice.values, slice.cols, slice.rows, slice.stats.absMax));
    panel.appendChild(heatmapWrap);

    const caption = document.createElement('p');
    caption.className = 'head-heatmap-caption';
    caption.textContent = `Symmetric color scale: ±${formatFloat(slice.stats.absMax)} across ${slice.values.length} decoded weights.`;
    panel.appendChild(caption);

    list.appendChild(panel);
  }
  host.appendChild(list);
}

function getHeadStructureLayout(headInfo, panelWidth) {
  if (!headInfo?.headCount) return null;
  const groups = getHeadGroups(headInfo);
  const maxGroupSize = Math.max(...groups.map(group => group.qHeads.length), 1);
  const bridgeCellSize = 16;
  const bridgeCellGap = 4;
  const groupPaddingX = 10;
  const groupGap = 12;
  const groupHeaderHeight = 18;
  const groupFooterHeight = 36;
  const groupCols = Math.min(maxGroupSize, 6);
  const bridgeRowsPerGroup = Math.ceil(maxGroupSize / groupCols);
  const groupInnerWidth = groupCols * (bridgeCellSize + bridgeCellGap) - bridgeCellGap;
  const bandWidth = Math.max(92, groupInnerWidth + groupPaddingX * 2);
  const bandHeight = groupHeaderHeight + bridgeRowsPerGroup * (bridgeCellSize + bridgeCellGap) - bridgeCellGap + groupFooterHeight;
  const groupsPerRow = Math.max(1, Math.floor((panelWidth - 40 + groupGap) / (bandWidth + groupGap)));
  const bridgeRowCount = Math.ceil(groups.length / groupsPerRow);
  const bridgeHeight = bridgeRowCount * bandHeight + Math.max(0, bridgeRowCount - 1) * groupGap;
  const matrixCellSize = 15;
  const matrixCellGap = 4;
  const matrixCols = Math.max(1, Math.min(12, Math.floor((panelWidth - 56 + matrixCellGap) / (matrixCellSize + matrixCellGap))));
  const qMatrixRows = Math.ceil(headInfo.headCount / matrixCols);
  const kvMatrixRows = Math.ceil((headInfo.headCountKV || 1) / matrixCols);
  const qMatrixHeight = 14 + qMatrixRows * (matrixCellSize + matrixCellGap) - matrixCellGap;
  const kvMatrixHeight = 14 + kvMatrixRows * (matrixCellSize + matrixCellGap) - matrixCellGap;
  const matrixHeight = qMatrixHeight + 14 + kvMatrixHeight;
  return {
    groups,
    bridgeCellSize,
    bridgeCellGap,
    groupPaddingX,
    groupGap,
    bandWidth,
    bandHeight,
    groupsPerRow,
    bridgeHeight,
    matrixCellSize,
    matrixCellGap,
    matrixCols,
    qMatrixHeight,
    kvMatrixHeight,
    matrixHeight,
    totalHeight: 28 + bridgeHeight + 26 + matrixHeight + 18,
  };
}

function renderHTMLHeadGrid(container, headInfo, selectedHead, onSelectHead) {
  if (!headInfo?.headCount) return;
  const { selectedQIndex, selectedKVIndex } = getHeadSelection(headInfo, selectedHead);
  const groups = getHeadGroups(headInfo);

  const section = document.createElement('div');
  section.className = 'head-grid-section';

  const title = document.createElement('h5');
  title.textContent = `${getAttentionTypeLabel(headInfo)}: ${headInfo.headCount} Q × ${headInfo.headDim}d → ${headInfo.headCountKV} KV`;
  section.appendChild(title);

  // Q heads
  const qLabel = document.createElement('div');
  qLabel.className = 'head-grid-label';
  qLabel.textContent = `Q heads (${headInfo.headCount})`;
  section.appendChild(qLabel);

  const qGrid = document.createElement('div');
  qGrid.className = 'head-grid';
  for (let i = 0; i < headInfo.headCount; i++) {
    qGrid.appendChild(createHeadCellButton(
      'q',
      i,
      headInfo.headCount <= 128 ? i : null,
      selectedQIndex === i,
      selectedKVIndex === getKVGroupIndex(headInfo, i) && selectedQIndex !== i,
      `Q head ${i} → KV head ${getKVGroupIndex(headInfo, i)}`,
      onSelectHead,
    ));
  }
  section.appendChild(qGrid);

  // KV heads
  const kvLabel = document.createElement('div');
  kvLabel.className = 'head-grid-label';
  kvLabel.textContent = `KV heads (${headInfo.headCountKV})`;
  section.appendChild(kvLabel);

  const kvGrid = document.createElement('div');
  kvGrid.className = 'head-grid';
  for (let i = 0; i < (headInfo.headCountKV || 1); i++) {
    const group = groups[i];
    kvGrid.appendChild(createHeadCellButton(
      'kv',
      i,
      (headInfo.headCountKV || 1) <= 128 ? i : null,
      selectedHead?.kind === 'kv' && selectedKVIndex === i,
      selectedHead?.kind === 'q' && selectedKVIndex === i,
      `KV head ${i} — serves Q heads ${formatHeadRange(group?.qStart, group?.qEnd)}`,
      onSelectHead,
    ));
  }
  section.appendChild(kvGrid);

  const woLabel = document.createElement('div');
  woLabel.className = 'head-grid-label';
  woLabel.textContent = 'Output projection';
  section.appendChild(woLabel);

  const woGrid = document.createElement('div');
  woGrid.className = 'head-grid';
  woGrid.appendChild(createHeadCellButton(
    'wo',
    undefined,
    'Wo',
    selectedHead?.kind === 'wo',
    false,
    'Output projection Wo',
    onSelectHead,
  ));
  section.appendChild(woGrid);

  const hint = document.createElement('div');
  hint.className = 'head-grid-hint';
  hint.textContent = 'Use these selectors—or click nodes in the SVG wiring—to inspect tensor slices, params, and weight heatmaps.';
  section.appendChild(hint);

  container.appendChild(section);
}

function renderInspector(container, stage, model, selectedHead = null, uiState = {}) {
  if (!stage) {
    container.innerHTML = `
      <h3>Architecture inspector</h3>
      <p class="inspector-note">Click any transformer block to inspect its residual branches, parameter footprint, and component breakdown. This model has <strong>${model.blockCount}</strong> transformer blocks.</p>
    `;
    return;
  }

  const breakdown = (stage.breakdown || []).map(group => `
    <span class="subgroup-pill"><strong>${group.label}</strong> ${formatParams(group.params)} · ${formatBytes(group.memoryBytes)}</span>
  `).join('');

  // Build attention detail table if available
  let attnHTML = '';
  const ad = stage.attentionDetail;
  if (ad) {
    const rows = ad.fused
      ? [['QKV Projection', ad.qkv], ['Output Projection', ad.output]]
      : [['Q Projection', ad.q], ['K Projection', ad.k], ['V Projection', ad.v], ['Output Projection', ad.output]];
    const tableRows = rows.filter(([, d]) => d).map(([name, d]) => `
      <tr>
        <td style="font-weight:600;color:#e0e6f0">${name}</td>
        <td style="color:#9aa4b8;font-family:monospace;font-size:12px">${d.shape?.length ? '[' + d.shape.join(' × ') + ']' : '—'}</td>
        <td>${formatParams(d.params)}</td>
        <td>${formatBytes(d.memoryBytes)}</td>
        <td style="color:#7a8599">${d.typeName || '—'}</td>
      </tr>
    `).join('');
    attnHTML = `
      <div class="attn-detail-section">
        <h4 style="margin:10px 0 6px;color:#c6cfdd;font-size:13px">Attention Components</h4>
        <table class="attn-detail-table">
          <thead><tr><th>Component</th><th>Shape</th><th>Params</th><th>Memory</th><th>Quant</th></tr></thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
    `;
  }

  // Build head info section
  let headHTML = '';
  let headSelectorHTML = '';
  let selectedHeadDetail = null;
  let selectedHeadHTML = '';
  const hi = stage.headInfo;
  if (hi && hi.headCount) {
    const typeLabel = getAttentionTypeLabel(hi, true);
    selectedHeadDetail = getSelectedHeadDetail(stage, selectedHead);
    headSelectorHTML = '<div data-head-grid></div>';
    selectedHeadHTML = selectedHeadDetail
      ? `
        <div class="selected-head-card">
          <div class="selected-head-header">
            <span class="head-chip">${selectedHeadDetail.title}</span>
            <span class="head-chip secondary">${selectedHeadDetail.badge}</span>
          </div>
          <div class="selected-head-decode" data-head-decode></div>
          <p class="selected-head-note">${selectedHeadDetail.note}</p>
          <div class="head-detail-grid">
            ${selectedHeadDetail.fields.map(field => `
              <div class="head-detail-item">
                <span>${field.label}</span>
                <strong>${field.value}</strong>
              </div>
            `).join('')}
          </div>
        </div>
      `
      : `
        <div class="selected-head-card placeholder">
          <div class="selected-head-header">
            <span class="head-chip">Selected head inspector</span>
          </div>
          <p class="selected-head-note">Click a Q, KV, or Wo node in the SVG wiring diagram—or any head cell in the grid—to inspect its slice, mapping, params, memory, quantization, and weight heatmap.</p>
        </div>
      `;
    headHTML = `
      <div class="attn-detail-section">
        <h4 style="margin:10px 0 6px;color:#c6cfdd;font-size:13px">Attention Heads — ${typeLabel}</h4>
        <div class="inspector-grid">
          <div class="inspector-card"><span class="label">Q Heads</span><span class="value">${hi.headCount}</span></div>
          <div class="inspector-card"><span class="label">KV Heads</span><span class="value">${hi.headCountKV}</span></div>
          <div class="inspector-card"><span class="label">Head Dim</span><span class="value">${hi.headDim}</span></div>
          <div class="inspector-card"><span class="label">GQA Ratio</span><span class="value">${hi.gqaRatio}:1</span></div>
        </div>
      </div>
    `;
  }

  container.innerHTML = `
    <h3>${stage.label}</h3>
    ${headSelectorHTML}
    ${selectedHeadHTML}
    <div class="inspector-grid">
      <div class="inspector-card"><span class="label">Pattern</span><span class="value">${stage.pattern}</span></div>
      <div class="inspector-card"><span class="label">Parameters</span><span class="value">${formatParams(stage.params)}</span></div>
      <div class="inspector-card"><span class="label">Memory</span><span class="value">${formatBytes(stage.memoryBytes)}</span></div>
      <div class="inspector-card"><span class="label">Components</span><span class="value">${stage.badges.join(' • ') || 'Residual'}</span></div>
    </div>
    <div class="subgroup-pills">${breakdown || '<span class="inspector-note">No subgroup breakdown available.</span>'}</div>
    ${attnHTML}
    ${headHTML}
  `;

  const headGridHost = container.querySelector('[data-head-grid]');
  if (headGridHost && hi?.headCount) {
    renderHTMLHeadGrid(headGridHost, hi, selectedHead, uiState.onSelectHead || null);
  }

  const decodeHost = container.querySelector('[data-head-decode]');
  if (decodeHost && selectedHeadDetail) {
    renderSelectedHeadDecode(decodeHost, model, selectedHeadDetail, uiState.onInvalidate || null);
  }
}

function drawStage(camera, stage, isSelected, onSelect) {
  const fill = stage.type === 'block' ? COLORS.block : (COLORS[stage.type] || '#4a4f5b');
  const stroke = isSelected ? COLORS.selected : (stage.type === 'block' ? '#515a6b' : fill);
  const group = svgElement('g', { transform: `translate(${stage.x}, ${stage.y})` });
  const body = svgElement('rect', {
    x: 0, y: 0, width: stage.width, height: stage.height, rx: 16,
    fill: isSelected ? '#2e3a53' : fill, stroke, 'stroke-width': isSelected ? 2.5 : 1.5,
  });
  const accent = svgElement('rect', {
    x: 0, y: 0, width: stage.width, height: 6, rx: 6,
    fill: stage.type === 'block' ? (stage.hasMoE ? COLORS.moe : stage.hasSSM ? COLORS.ssm : stage.hasAttention ? COLORS.attention : COLORS.residual) : fill,
  });
  const label = svgElement('text', {
    x: stage.width / 2, y: stage.height / 2 - 4, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 600, fill: '#fff',
  }, stage.label);
  const summary = svgElement('text', {
    x: stage.width / 2, y: stage.height / 2 + 14, 'text-anchor': 'middle', 'font-size': 10.5, fill: '#c6cfdd',
  }, stage.type === 'block' ? formatParams(stage.params) : stage.summary);
  const title = svgElement('title', {}, [stage.label, stage.summary, formatParams(stage.params), formatBytes(stage.memoryBytes)].filter(Boolean).join('\n'));

  group.append(body, accent, label, summary, title);
  if (stage.type === 'block') {
    group.style.cursor = 'pointer';
    group.addEventListener('click', () => onSelect(stage.index));
  }
  camera.appendChild(group);
}

function formatShape(dims) {
  if (!dims || !dims.length) return '';
  return dims.join(' × ');
}

function nodeTooltip(node) {
  if (!node.detail) return node.label;
  const parts = [node.label];
  if (node.detail.shape?.length) parts.push(`Shape: [${formatShape(node.detail.shape)}]`);
  parts.push(`Params: ${formatParams(node.detail.params)}`);
  parts.push(`Memory: ${formatBytes(node.detail.memoryBytes)}`);
  if (node.detail.typeName) parts.push(`Quant: ${node.detail.typeName}`);
  return parts.join('\n');
}

function drawNodeBox(group, cx, cy, w, h, node) {
  const g = svgElement('g');
  g.appendChild(svgElement('rect', {
    x: cx - w / 2, y: cy - h / 2, width: w, height: h, rx: 10,
    fill: '#202530', stroke: COLORS[node.type] || '#6f7a8e', 'stroke-width': 1.5,
  }));
  g.appendChild(svgElement('text', {
    x: cx, y: cy + 4, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 600, fill: '#fff',
  }, node.label));
  g.appendChild(svgElement('title', {}, nodeTooltip(node)));
  if (node.detail?.shape?.length) {
    g.appendChild(svgElement('text', {
      x: cx, y: cy + h / 2 + 11, 'text-anchor': 'middle', 'font-size': 8.5, fill: '#7a8599',
    }, `[${formatShape(node.detail.shape)}]`));
  }
  group.appendChild(g);
}

function drawFlowNode(group, cx, cy, width, height, options) {
  const {
    title,
    subtitle = '',
    fill = '#202a38',
    stroke = '#4d607e',
    titleFill = '#ffffff',
    subtitleFill = '#9fb0c7',
  } = options;
  const node = svgElement('g');
  node.appendChild(svgElement('rect', {
    x: cx - width / 2,
    y: cy - height / 2,
    width,
    height,
    rx: 10,
    fill,
    stroke,
    'stroke-width': 1.4,
  }));
  node.appendChild(svgElement('text', {
    x: cx,
    y: cy - (subtitle ? 3 : -4),
    'text-anchor': 'middle',
    'font-size': 10,
    'font-weight': 700,
    fill: titleFill,
  }, title));
  if (subtitle) {
    node.appendChild(svgElement('text', {
      x: cx,
      y: cy + 11,
      'text-anchor': 'middle',
      'font-size': 8.5,
      fill: subtitleFill,
    }, subtitle));
  }
  group.appendChild(node);
  return node;
}

function drawDimPill(group, cx, y, text, options = {}) {
  if (!text) return;
  const fill = options.fill || '#223247';
  const stroke = options.stroke || '#4b6487';
  const textFill = options.textFill || '#9fc7ff';
  const width = Math.max(34, text.length * 5.8 + 12);
  const height = 16;
  group.appendChild(svgElement('rect', {
    x: cx - width / 2,
    y,
    width,
    height,
    rx: 8,
    fill,
    stroke,
    'stroke-width': 1,
  }));
  group.appendChild(svgElement('text', {
    x: cx,
    y: y + 11,
    'text-anchor': 'middle',
    'font-size': 8,
    'font-weight': 600,
    fill: textFill,
  }, text));
}

function drawFlowConnector(group, points, options = {}) {
  const pathData = points.map(([x, y], index) => `${index ? 'L' : 'M'} ${x} ${y}`).join(' ');
  const attrs = {
    d: pathData,
    fill: 'none',
    stroke: options.stroke || '#7f9ac7',
    'stroke-width': options.strokeWidth || 1.6,
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
    opacity: options.opacity || 0.95,
  };
  if (options.dash) attrs['stroke-dasharray'] = options.dash;
  if (options.arrow !== false) attrs['marker-end'] = 'url(#architecture-arrowhead)';
  group.appendChild(svgElement('path', attrs));
}

function drawCurvedConnector(group, start, end, options = {}) {
  const [startX, startY] = start;
  const [endX, endY] = end;
  const controlOffset = options.controlOffset || Math.max(18, Math.abs(endX - startX) * 0.45);
  const pathData = `M ${startX} ${startY} C ${startX + controlOffset} ${startY}, ${endX - controlOffset} ${endY}, ${endX} ${endY}`;
  const attrs = {
    d: pathData,
    fill: 'none',
    stroke: options.stroke || '#7f9ac7',
    'stroke-width': options.strokeWidth || 1.6,
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
    opacity: options.opacity || 0.95,
  };
  if (options.dash) attrs['stroke-dasharray'] = options.dash;
  if (options.arrow !== false) attrs['marker-end'] = 'url(#architecture-arrowhead)';
  group.appendChild(svgElement('path', attrs));
}

function getSVGHeadWiringLayout(headInfo, panelWidth) {
  if (!headInfo?.headCount) return null;
  const rowEntries = Array.from({ length: headInfo.headCount }, (_, qIndex) => ({
    qIndex,
    kvIndex: getKVGroupIndex(headInfo, qIndex),
  }));
  const leftSectionWidth = 302;
  const combineSectionWidth = 168;
  const columnGap = 18;
  const minBandWidth = 244;
  const availableWidth = Math.max(minBandWidth, panelWidth - leftSectionWidth - combineSectionWidth - 32);
  const columns = rowEntries.length > 12 && availableWidth >= minBandWidth * 2 + columnGap ? 2 : 1;
  const rowsPerColumn = Math.ceil(rowEntries.length / columns);
  const bandWidth = Math.floor((availableWidth - columnGap * (columns - 1)) / columns);
  const bandHeight = 86;
  const rowGap = 10;
  const rowAreaHeight = rowsPerColumn * bandHeight + Math.max(0, rowsPerColumn - 1) * rowGap;
  const mergeHeight = 0;
  const totalHeight = rowAreaHeight + mergeHeight;

  return {
    rowEntries,
    leftSectionWidth,
    combineSectionWidth,
    columnGap,
    columns,
    rowsPerColumn,
    bandWidth,
    bandHeight,
    rowGap,
    rowAreaHeight,
    mergeHeight,
    totalHeight,
  };
}

function getDetailPanelMetrics(stage) {
  const hasAttnDetail = stage.detailRows.some((row) => row.layout === 'attention-qkv');
  const diagramYOffset = 48;
  const panelWidth = hasAttnDetail ? 1020 : 390;
  const basePanelHeight = hasAttnDetail ? 260 : 240;
  const headDiagramTopGap = 168;
  const wiringLayout = hasAttnDetail && stage.headInfo?.headCount
    ? getSVGHeadWiringLayout(stage.headInfo, panelWidth)
    : null;
  const headDiagramHeight = wiringLayout ? headDiagramTopGap + wiringLayout.totalHeight : 0;

  return {
    hasAttnDetail,
    panelWidth,
    panelHeight: basePanelHeight + diagramYOffset + headDiagramHeight,
  };
}

function drawSVGHeadWiring(group, stage, panelX, chartTopY, panelWidth, selectedHead, onSelectHead) {
  const headInfo = stage.headInfo;
  if (!headInfo?.headCount) return 0;

  const { selectedQIndex, selectedKVIndex } = getHeadSelection(headInfo, selectedHead);
  const layout = getSVGHeadWiringLayout(headInfo, panelWidth);
  if (!layout) return 0;
  const {
    rowEntries,
    leftSectionWidth,
    combineSectionWidth,
    columnGap,
    columns,
    rowsPerColumn,
    bandWidth,
    bandHeight,
    rowGap,
    rowAreaHeight,
    mergeHeight,
    totalHeight,
  } = layout;
  const inputDims = formatDimsLabel([headInfo.embeddingLength]);
  const qDims = formatDimsLabel([headInfo.headCount, headInfo.headDim]);
  const kvDims = formatDimsLabel([headInfo.headCountKV || 1, headInfo.headDim]);
  const scoreDims = headInfo.contextLength
    ? formatDimsLabel([headInfo.contextLength, headInfo.contextLength])
    : '[ctx × ctx]';
  const woSelected = selectedHead?.kind === 'wo';
  const hasWoDetail = !!stage.attentionDetail?.output;

  group.appendChild(svgElement('text', {
    x: panelX + 20,
    y: chartTopY,
    'font-size': 10.5,
    fill: '#c6cfdd',
    'font-weight': 700,
  }, 'Attention wiring'));
  group.appendChild(svgElement('text', {
    x: panelX + panelWidth - 20,
    y: chartTopY,
    'text-anchor': 'end',
    'font-size': 9,
    fill: '#7f8da3',
  }, 'Per-head calculation flow'));

  const layoutTopY = chartTopY + 18;
  const inputCx = panelX + 76;
  const inputCy = layoutTopY + rowAreaHeight / 2;
  drawFlowNode(group, inputCx, inputCy, 114, 34, {
    title: 'Input x',
    subtitle: inputDims,
    fill: '#24303d',
    stroke: '#4f637c',
  });

  const projCx = panelX + 242;
  const projWidth = 82;
  const projHeight = 28;
  const projFractions = rowAreaHeight < 150 ? [0.25, 0.5, 0.75] : [0.18, 0.5, 0.82];
  const projections = [
    { key: 'q', cx: projCx, cy: layoutTopY + rowAreaHeight * projFractions[0], title: 'Wq', stroke: COLORS.attention, outDims: qDims, subtitle: 'query slice' },
    { key: 'k', cx: projCx, cy: layoutTopY + rowAreaHeight * projFractions[1], title: 'Wk', stroke: '#5fa8f2', outDims: kvDims, subtitle: 'key slice' },
    { key: 'v', cx: projCx, cy: layoutTopY + rowAreaHeight * projFractions[2], title: 'Wv', stroke: '#78b8ff', outDims: kvDims, subtitle: 'value slice' },
  ];
  const projectionMap = Object.fromEntries(projections.map((projection) => [projection.key, projection]));

  projections.forEach((projection) => {
    drawCurvedConnector(group, [inputCx + 57, inputCy], [projection.cx - projWidth / 2, projection.cy], {
      controlOffset: Math.max(24, (projection.cx - projWidth / 2 - (inputCx + 57)) * 0.42),
    });
    drawFlowNode(group, projection.cx, projection.cy, projWidth, projHeight, {
      title: projection.title,
      subtitle: projection.subtitle,
      fill: '#1f2835',
      stroke: projection.stroke,
    });
    drawDimPill(group, projection.cx, projection.cy - 33, inputDims);
  });

  const groupsStartX = panelX + leftSectionWidth;
  const bridgeLabelY = chartTopY + 16;
  group.appendChild(svgElement('text', {
    x: groupsStartX,
    y: bridgeLabelY,
    'font-size': 10,
    fill: '#c6cfdd',
    'font-weight': 600,
  }, 'One row per Q head: Q_i + K_g(i) → score_i → softmax_i → O_i → concat → Wo'));
  group.appendChild(svgElement('text', {
    x: panelX + panelWidth - 20,
    y: bridgeLabelY,
    'text-anchor': 'end',
    'font-size': 8.8,
    fill: '#7f8da3',
  }, `${headInfo.headCount} Q heads • ${headInfo.headCountKV || 1} KV projections`));

  const columnsLeftX = groupsStartX;
  const columnsRightX = groupsStartX + columns * bandWidth + (columns - 1) * columnGap;
  const concatY = layoutTopY + rowAreaHeight / 2;
  const combineCenterX = columnsRightX + combineSectionWidth / 2;
  const concatCx = combineCenterX - 44;
  const woCx = combineCenterX + 44;
  const combineLabelY = Math.max(bridgeLabelY + 12, concatY - 26);
  const concatDims = formatDimsLabel([headInfo.headCount, headInfo.headDim]);

  for (const rowEntry of rowEntries) {
    const col = Math.floor(rowEntry.qIndex / rowsPerColumn);
    const row = rowEntry.qIndex % rowsPerColumn;
    const bandX = groupsStartX + col * (bandWidth + columnGap);
    const bandY = layoutTopY + row * (bandHeight + rowGap);
    const rowCenterY = bandY + bandHeight / 2;
    const qSelected = selectedHead?.kind === 'q' && selectedQIndex === rowEntry.qIndex;
    const kvSelected = selectedHead?.kind === 'kv' && selectedKVIndex === rowEntry.kvIndex;
    const rowActive = qSelected || kvSelected;
    const innerLeft = bandX + 12;
    const innerRight = bandX + bandWidth - 22;
    const qWidth = headInfo.headCount <= 64 ? 28 : 24;
    const qHeight = 18;
    const kvWidth = 30;
    const kvHeight = 16;
    const scoreWidth = 42;
    const scoreHeight = 42;
    const softmaxWidth = 54;
    const outWidth = headInfo.headCount <= 64 ? 34 : 30;
    const availableGap = innerRight - innerLeft - qWidth - scoreWidth - softmaxWidth - outWidth;
    const segmentGap = clamp(Math.floor(availableGap / 3), 10, 34);
    const contentWidth = qWidth + scoreWidth + softmaxWidth + outWidth + segmentGap * 3;
    const startX = innerLeft + Math.max(0, Math.floor((innerRight - innerLeft - contentWidth) / 2));
    const qX = startX;
    const qY = rowCenterY - qHeight / 2;
    const qRight = qX + qWidth;
    const scoreX = qRight + segmentGap;
    const scoreCx = scoreX + scoreWidth / 2;
    const softmaxX = scoreX + scoreWidth + segmentGap;
    const softmaxCx = softmaxX + softmaxWidth / 2;
    const outX = softmaxX + softmaxWidth + segmentGap;
    const outCx = outX + outWidth / 2;
    const kX = scoreCx - kvWidth / 2;
    const kY = bandY + 4;
    const vX = outCx - kvWidth / 2;
    const vY = bandY + bandHeight - kvHeight - 4;
    const kCenterY = kY + kvHeight / 2;
    const vCenterY = vY + kvHeight / 2;

    group.appendChild(svgElement('rect', {
      x: bandX,
      y: bandY,
      width: bandWidth,
      height: bandHeight,
      rx: 10,
      fill: qSelected ? '#202d3d' : kvSelected ? '#1d2734' : '#1b2230',
      stroke: qSelected ? COLORS.selected : kvSelected ? '#8db6e7' : '#313d50',
      'stroke-width': qSelected ? 1.8 : kvSelected ? 1.35 : 1,
    }));

    drawCurvedConnector(group, [projectionMap.q.cx + projWidth / 2, projectionMap.q.cy], [qX, rowCenterY], {
      arrow: false,
      opacity: rowActive ? (qSelected ? 0.58 : 0.42) : 0.12,
      strokeWidth: rowActive ? (qSelected ? 1.45 : 1.1) : 1,
      stroke: rowActive ? '#f0a565' : '#67758b',
      controlOffset: Math.max(22, (qX - (projectionMap.q.cx + projWidth / 2)) * 0.38),
    });
    drawCurvedConnector(group, [projectionMap.k.cx + projWidth / 2, projectionMap.k.cy], [kX, kCenterY], {
      arrow: false,
      dash: '4,3',
      opacity: rowActive ? 0.46 : 0.12,
      strokeWidth: rowActive ? 1.35 : 1,
      stroke: rowActive ? '#78b8ff' : '#67758b',
      controlOffset: Math.max(18, (kX - (projectionMap.k.cx + projWidth / 2)) * 0.45),
    });
    drawCurvedConnector(group, [projectionMap.v.cx + projWidth / 2, projectionMap.v.cy], [vX, vCenterY], {
      arrow: false,
      dash: '4,3',
      opacity: rowActive ? 0.46 : 0.12,
      strokeWidth: rowActive ? 1.35 : 1,
      stroke: rowActive ? '#78b8ff' : '#67758b',
      controlOffset: Math.max(18, (vX - (projectionMap.v.cx + projWidth / 2)) * 0.45),
    });

    drawFlowConnector(group, [
      [qRight, rowCenterY],
      [scoreX, rowCenterY],
    ], {
      arrow: false,
      opacity: rowActive ? 0.84 : 0.28,
      strokeWidth: rowActive ? 1.5 : 1.1,
      stroke: rowActive ? '#8db6e7' : '#57667c',
    });
    drawFlowConnector(group, [
      [scoreX + scoreWidth, rowCenterY],
      [softmaxX, rowCenterY],
    ], {
      arrow: false,
      opacity: rowActive ? 0.84 : 0.28,
      strokeWidth: rowActive ? 1.5 : 1.1,
      stroke: rowActive ? '#8db6e7' : '#57667c',
    });
    drawFlowConnector(group, [
      [softmaxX + softmaxWidth, rowCenterY],
      [outX, rowCenterY],
    ], {
      arrow: false,
      opacity: rowActive ? 0.84 : 0.28,
      strokeWidth: rowActive ? 1.5 : 1.1,
      stroke: rowActive ? '#8db6e7' : '#57667c',
    });
    drawFlowConnector(group, [
      [kX + kvWidth / 2, kY + kvHeight],
      [scoreCx, rowCenterY],
    ], {
      arrow: false,
      opacity: rowActive ? 0.84 : 0.28,
      strokeWidth: rowActive ? 1.5 : 1.1,
      stroke: rowActive ? '#8db6e7' : '#57667c',
    });
    drawFlowConnector(group, [
      [vX + kvWidth / 2, vY],
      [outCx, rowCenterY],
    ], {
      arrow: false,
      opacity: rowActive ? 0.84 : 0.28,
      strokeWidth: rowActive ? 1.5 : 1.1,
      stroke: rowActive ? '#8db6e7' : '#57667c',
    });
    drawCurvedConnector(group, [outX + outWidth, rowCenterY], [concatCx - 31, concatY], {
      arrow: false,
      dash: '3,3',
      opacity: rowActive ? 0.44 : 0.12,
      strokeWidth: rowActive ? 1.25 : 1,
      stroke: rowActive ? '#8db6e7' : '#57667c',
      controlOffset: Math.max(22, (concatCx - 31 - (outX + outWidth)) * 0.36),
    });

    drawHeadCell(group, {
      x: kX,
      y: kY,
      width: kvWidth,
      height: kvHeight,
      label: `K${rowEntry.kvIndex}`,
      fill: '#4a90d9',
      textSize: headInfo.headCount <= 64 ? 7 : 6.5,
      title: `K input for Q head ${rowEntry.qIndex} (KV head ${rowEntry.kvIndex})`,
      selected: kvSelected,
      related: qSelected,
      onClick: () => onSelectHead?.({ kind: 'kv', index: rowEntry.kvIndex }),
    });
    drawHeadCell(group, {
      x: vX,
      y: vY,
      width: kvWidth,
      height: kvHeight,
      label: `V${rowEntry.kvIndex}`,
      fill: '#5d9fe3',
      textSize: headInfo.headCount <= 64 ? 7 : 6.5,
      title: `V input for Q head ${rowEntry.qIndex} (KV head ${rowEntry.kvIndex})`,
      selected: kvSelected,
      related: qSelected,
      onClick: () => onSelectHead?.({ kind: 'kv', index: rowEntry.kvIndex }),
    });
    drawHeadCell(group, {
      x: qX,
      y: qY,
      width: qWidth,
      height: qHeight,
      label: `Q${rowEntry.qIndex}`,
      fill: COLORS.attention,
      textSize: headInfo.headCount <= 64 ? 7 : 6.5,
      title: `Q head ${rowEntry.qIndex} uses KV head ${rowEntry.kvIndex}`,
      selected: qSelected,
      related: kvSelected,
      onClick: () => onSelectHead?.({ kind: 'q', index: rowEntry.qIndex }),
    });

    drawFlowNode(group, scoreCx, rowCenterY, scoreWidth, scoreHeight, {
      title: 'score',
      subtitle: scoreDims,
      fill: rowActive ? '#243244' : '#1f2734',
      stroke: rowActive ? '#6b95c9' : '#495a71',
      titleFill: rowActive ? '#f2f7ff' : '#dce5f3',
    });
    drawFlowNode(group, softmaxCx, rowCenterY, softmaxWidth, 18, {
      title: 'softmax',
      fill: rowActive ? '#253649' : '#1f2734',
      stroke: rowActive ? '#6b95c9' : '#495a71',
      titleFill: rowActive ? '#f2f7ff' : '#dce5f3',
    });
    drawFlowNode(group, outCx, rowCenterY, outWidth, 18, {
      title: `O${rowEntry.qIndex}`,
      fill: rowActive ? '#243244' : '#1f2734',
      stroke: rowActive ? '#6b95c9' : '#495a71',
      titleFill: rowActive ? '#f2f7ff' : '#dce5f3',
    });
  }

  group.appendChild(svgElement('text', {
    x: columnsRightX + combineSectionWidth / 2,
    y: combineLabelY,
    'text-anchor': 'middle',
    'font-size': 8.8,
    fill: '#7f8da3',
  }, 'combine head outputs'));
  drawDimPill(group, concatCx, concatY + 20, concatDims);
  drawDimPill(group, woCx, concatY + 20, inputDims, woSelected
    ? { fill: '#264059', stroke: COLORS.selected, textFill: '#d8ecff' }
    : undefined);
  drawFlowNode(group, concatCx, concatY, 62, 22, {
    title: 'concat',
    fill: '#223247',
    stroke: '#6b95c9',
  });
  const woNode = drawFlowNode(group, woCx, concatY, 46, 22, {
    title: 'Wo',
    fill: woSelected ? '#24384f' : '#1f2835',
    stroke: woSelected ? COLORS.selected : '#8b9bb4',
    titleFill: woSelected ? '#f2f8ff' : '#ffffff',
  });
  if (hasWoDetail) {
    woNode.style.cursor = 'pointer';
    woNode.addEventListener('click', () => onSelectHead?.({ kind: 'wo' }));
    woNode.appendChild(svgElement('title', {}, 'Inspect Wo projection heatmap'));
  }
  drawFlowConnector(group, [
    [concatCx + 31, concatY],
    [woCx - 23, concatY],
  ], {
    arrow: false,
    opacity: woSelected ? 0.9 : 0.74,
    strokeWidth: woSelected ? 1.7 : 1.35,
    stroke: woSelected ? COLORS.selected : '#8db6e7',
  });

  return 30 + totalHeight;
}

function drawAttentionPath(group, detail, panelX, mainY, panelWidth) {
  const rowCenterY = mainY - 50;
  const nw = 56; // node width
  const nh = 24; // node height
  const qkvSpacing = 48;
  const color = COLORS.attention;

  // Horizontal positions
  const normX = panelX + 116;
  const qkvX = normX + nw / 2 + 56;
  const softmaxX = qkvX + nw / 2 + 56;
  const outProjX = softmaxX + nw / 2 + 56;
  const mergeX = Math.min(outProjX + nw / 2 + 28, panelX + panelWidth - 50);

  group.appendChild(svgElement('text', { x: panelX + 20, y: rowCenterY - 38, 'font-size': 10.5, fill: '#9aa4b8' }, 'Attention path'));

  // Branch from residual up to norm
  group.appendChild(svgElement('path', {
    d: `M ${panelX + 44} ${mainY} C ${panelX + 60} ${mainY} ${normX - nw / 2 - 12} ${rowCenterY} ${normX - nw / 2} ${rowCenterY}`,
    fill: 'none', stroke: color, 'stroke-width': 2.2,
  }));

  // Attn Norm box
  drawNodeBox(group, normX, rowCenterY, nw, nh, { label: 'Attn Norm', type: 'norm' });

  // Fork lines from Norm → Q, K, V
  const forkStartX = normX + nw / 2;
  const projections = detail.fused
    ? [{ label: 'QKV', type: 'attention', detail: detail.qkv, yOff: 0 }]
    : [
      { label: 'Q', type: 'attention', detail: detail.q, yOff: -qkvSpacing },
      { label: 'K', type: 'attention', detail: detail.k, yOff: 0 },
      { label: 'V', type: 'attention', detail: detail.v, yOff: qkvSpacing },
    ];

  for (const proj of projections) {
    const projY = rowCenterY + proj.yOff;
    // Fork line
    group.appendChild(svgElement('path', {
      d: `M ${forkStartX} ${rowCenterY} C ${forkStartX + 16} ${rowCenterY} ${qkvX - nw / 2 - 16} ${projY} ${qkvX - nw / 2} ${projY}`,
      fill: 'none', stroke: color, 'stroke-width': 1.8,
    }));
    // Projection box
    drawNodeBox(group, qkvX, projY, nw, nh, proj);
    // Merge line from projection → softmax
    group.appendChild(svgElement('path', {
      d: `M ${qkvX + nw / 2} ${projY} C ${qkvX + nw / 2 + 16} ${projY} ${softmaxX - nw / 2 - 16} ${rowCenterY} ${softmaxX - nw / 2} ${rowCenterY}`,
      fill: 'none', stroke: color, 'stroke-width': 1.8,
    }));
  }

  // Softmax / Attention box
  drawNodeBox(group, softmaxX, rowCenterY, nw, nh, { label: 'Softmax', type: 'attention' });

  // Arrow: Softmax → Out Proj
  group.appendChild(svgElement('line', {
    x1: softmaxX + nw / 2, y1: rowCenterY, x2: outProjX - nw / 2, y2: rowCenterY,
    stroke: color, 'stroke-width': 1.8,
  }));

  // Out Proj box
  drawNodeBox(group, outProjX, rowCenterY, nw, nh, { label: 'Out Proj', type: 'attention', detail: detail.output });

  // Merge back to residual
  group.appendChild(svgElement('path', {
    d: `M ${outProjX + nw / 2} ${rowCenterY} C ${outProjX + nw / 2 + 14} ${rowCenterY} ${mergeX - 10} ${mainY} ${mergeX} ${mainY}`,
    fill: 'none', stroke: color, 'stroke-width': 2.2,
  }));
  group.appendChild(svgElement('circle', { cx: mergeX, cy: mainY, r: 11, fill: '#2d3441', stroke: '#5a6478', 'stroke-width': 1.5 }));
  group.appendChild(svgElement('text', { x: mergeX, y: mainY + 4, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, fill: '#fff' }, '+'));
}

function drawLinearRow(group, row, panelX, mainY, panelWidth, rowIndex) {
  const nodeWidth = 76;
  const nodeHeight = 32;
  const nodeHalfWidth = nodeWidth / 2;
  const mergeGap = 24;
  const isCompactPanel = panelWidth <= 420;
  const mergeX = isCompactPanel
    ? panelX + panelWidth - 74
    : (rowIndex === 0 ? panelX + panelWidth * 0.62 : panelX + panelWidth * 0.84);
  const rowY = rowIndex === 0 ? mainY - 58 : mainY + 58;
  const startX = isCompactPanel ? panelX + 108 : panelX + 124;
  const endX = mergeX - (nodeHalfWidth + mergeGap);
  const span = row.nodes.length > 1 ? (endX - startX) / (row.nodes.length - 1) : 0;
  const firstNodeCenterX = startX;
  const firstNodeLeftX = firstNodeCenterX - nodeHalfWidth;
  const lastNodeCenterX = startX + span * Math.max(0, row.nodes.length - 1);

  group.appendChild(svgElement('text', { x: panelX + 20, y: rowY - 22, 'font-size': 10.5, fill: '#9aa4b8' }, row.label));
  group.appendChild(svgElement('path', {
    d: `M ${panelX + 44} ${mainY} C ${panelX + 62} ${mainY} ${firstNodeLeftX - 20} ${rowY} ${firstNodeLeftX} ${rowY}`,
    fill: 'none', stroke: COLORS[row.nodes[row.nodes.length - 1].type] || COLORS.residual, 'stroke-width': 2.2,
  }));

  row.nodes.forEach((node, index) => {
    const nodeCenterX = startX + span * index;
    const nodeGroup = svgElement('g');
    const rect = svgElement('rect', {
      x: nodeCenterX - nodeHalfWidth, y: rowY - nodeHeight / 2, width: nodeWidth, height: nodeHeight, rx: 12,
      fill: '#202530', stroke: COLORS[node.type] || '#6f7a8e', 'stroke-width': 1.5,
    });
    const text = svgElement('text', {
      x: nodeCenterX, y: rowY + 4, 'text-anchor': 'middle', 'font-size': 10.5, 'font-weight': 600, fill: '#fff',
    }, node.label);
    nodeGroup.append(rect, text);
    nodeGroup.appendChild(svgElement('title', {}, nodeTooltip(node)));
    if (index > 0) {
      nodeGroup.appendChild(svgElement('line', {
        x1: nodeCenterX - span + nodeHalfWidth, y1: rowY, x2: nodeCenterX - nodeHalfWidth, y2: rowY,
        stroke: COLORS[node.type] || COLORS.residual, 'stroke-width': 2,
      }));
    }
    group.appendChild(nodeGroup);
  });

  group.appendChild(svgElement('path', {
    d: `M ${lastNodeCenterX + nodeHalfWidth} ${rowY} C ${lastNodeCenterX + nodeHalfWidth + 18} ${rowY} ${mergeX - 12} ${mainY} ${mergeX} ${mainY}`,
    fill: 'none', stroke: COLORS[row.nodes[row.nodes.length - 1].type] || COLORS.residual, 'stroke-width': 2.2,
  }));
  group.appendChild(svgElement('circle', { cx: mergeX, cy: mainY, r: 11, fill: '#2d3441', stroke: '#5a6478', 'stroke-width': 1.5 }));
  group.appendChild(svgElement('text', { x: mergeX, y: mainY + 4, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, fill: '#fff' }, '+'));
}

function drawHeadCell(group, options) {
  const {
    x,
    y,
    width,
    height,
    label,
    fill,
    title,
    textSize = 8,
    selected = false,
    related = false,
    onClick,
  } = options;
  const cellGroup = svgElement('g');
  const rect = svgElement('rect', {
    x,
    y,
    width,
    height,
    rx: Math.min(4, height / 3),
    fill: selected ? '#314f72' : fill,
    stroke: selected ? COLORS.selected : related ? '#8db6e7' : '#4e5a6d',
    'stroke-width': selected ? 2 : related ? 1.35 : 1,
    opacity: selected ? 1 : related ? 0.96 : 0.88,
  });
  cellGroup.appendChild(rect);
  if (label !== undefined && label !== null) {
    cellGroup.appendChild(svgElement('text', {
      x: x + width / 2,
      y: y + height / 2 + 3,
      'text-anchor': 'middle',
      'font-size': textSize,
      fill: '#fff',
      'pointer-events': 'none',
    }, `${label}`));
  }
  if (title) cellGroup.appendChild(svgElement('title', {}, title));
  if (onClick) {
    cellGroup.style.cursor = 'pointer';
    cellGroup.addEventListener('click', (event) => {
      event.stopPropagation();
      onClick();
    });
  }
  group.appendChild(cellGroup);
  return cellGroup;
}

function drawHeadGrid(group, headInfo, panelX, gridTopY, panelWidth, selectedHead, onSelectHead, layout = null) {
  if (!headInfo?.headCount) return 0;
  const metrics = layout || getHeadStructureLayout(headInfo, panelWidth);
  if (!metrics) return 0;

  const {
    groups,
    bridgeCellSize,
    bridgeCellGap,
    groupPaddingX,
    groupGap,
    bandWidth,
    bandHeight,
    groupsPerRow,
    bridgeHeight,
    matrixCellSize,
    matrixCellGap,
    matrixCols,
    qMatrixHeight,
    totalHeight,
  } = metrics;
  const { selectedQIndex, selectedKVIndex } = getHeadSelection(headInfo, selectedHead);
  const summaryText = `${getAttentionTypeLabel(headInfo)}: ${headInfo.headCount} Q × ${headInfo.headDim}d → ${headInfo.headCountKV} KV`;

  group.appendChild(svgElement('text', {
    x: panelX + 20, y: gridTopY, 'font-size': 10.5, fill: '#9aa4b8',
  }, summaryText));
  group.appendChild(svgElement('text', {
    x: panelX + panelWidth - 20, y: gridTopY, 'text-anchor': 'end', 'font-size': 9.5, fill: '#6f7b90',
  }, 'click heads to pin the inspector'));

  const bridgeLabelY = gridTopY + 18;
  group.appendChild(svgElement('text', {
    x: panelX + 20, y: bridgeLabelY, 'font-size': 10, fill: '#c6cfdd', 'font-weight': 600,
  }, 'Q → KV grouped bridge'));

  const bridgeTopY = bridgeLabelY + 10;
  for (const band of groups) {
    const row = Math.floor(band.kvIndex / groupsPerRow);
    const col = band.kvIndex % groupsPerRow;
    const bandX = panelX + 20 + col * (bandWidth + groupGap);
    const bandY = bridgeTopY + row * (bandHeight + groupGap);
    const bandSelected = selectedKVIndex === band.kvIndex;
    const qCols = Math.min(Math.max(1, band.qHeads.length), 6);
    const qRows = Math.ceil(band.qHeads.length / qCols);
    const qGridWidth = qCols * (bridgeCellSize + bridgeCellGap) - bridgeCellGap;
    const qLeft = bandX + (bandWidth - qGridWidth) / 2;
    const qTop = bandY + 20;
    const qBottom = qTop + qRows * (bridgeCellSize + bridgeCellGap) - bridgeCellGap;
    const busY = qBottom + 8;
    const kvY = bandY + bandHeight - 22;
    const kvWidth = Math.min(42, bandWidth - groupPaddingX * 2);
    const kvX = bandX + (bandWidth - kvWidth) / 2;
    const kvCenterX = kvX + kvWidth / 2;

    group.appendChild(svgElement('rect', {
      x: bandX,
      y: bandY,
      width: bandWidth,
      height: bandHeight,
      rx: 12,
      fill: bandSelected ? '#202d3d' : '#212733',
      stroke: bandSelected ? COLORS.selected : '#374255',
      'stroke-width': bandSelected ? 1.8 : 1,
    }));
    group.appendChild(svgElement('text', {
      x: bandX + 10, y: bandY + 14, 'font-size': 9.5, fill: '#9aa4b8', 'font-weight': 600,
    }, `KV ${band.kvIndex}`));
    group.appendChild(svgElement('text', {
      x: bandX + bandWidth - 10, y: bandY + 14, 'text-anchor': 'end', 'font-size': 9, fill: '#6f7b90',
    }, `Q ${formatHeadRange(band.qStart, band.qEnd)}`));

    band.qHeads.forEach((headIndex, headOffset) => {
      const cellCol = headOffset % qCols;
      const cellRow = Math.floor(headOffset / qCols);
      const x = qLeft + cellCol * (bridgeCellSize + bridgeCellGap);
      const y = qTop + cellRow * (bridgeCellSize + bridgeCellGap);
      const exactSelected = selectedQIndex === headIndex;
      const related = selectedKVIndex === band.kvIndex && selectedQIndex !== headIndex;
      group.appendChild(svgElement('line', {
        x1: x + bridgeCellSize / 2,
        y1: y + bridgeCellSize,
        x2: x + bridgeCellSize / 2,
        y2: busY,
        stroke: exactSelected || related ? '#8db6e7' : '#556174',
        'stroke-width': exactSelected ? 1.6 : 1,
        opacity: exactSelected || related ? 0.95 : 0.72,
      }));
      drawHeadCell(group, {
        x,
        y,
        width: bridgeCellSize,
        height: bridgeCellSize,
        label: headInfo.headCount <= 64 ? headIndex : null,
        fill: COLORS.attention,
        title: `Q head ${headIndex} → KV head ${band.kvIndex}`,
        selected: exactSelected,
        related,
        onClick: () => onSelectHead?.({ kind: 'q', index: headIndex }),
      });
    });

    group.appendChild(svgElement('line', {
      x1: qLeft,
      y1: busY,
      x2: qLeft + qGridWidth,
      y2: busY,
      stroke: bandSelected ? COLORS.selected : '#5a677c',
      'stroke-width': bandSelected ? 1.8 : 1.1,
    }));
    group.appendChild(svgElement('line', {
      x1: kvCenterX,
      y1: busY,
      x2: kvCenterX,
      y2: kvY,
      stroke: bandSelected ? COLORS.selected : '#5a677c',
      'stroke-width': bandSelected ? 1.8 : 1.1,
      'stroke-dasharray': '4,3',
    }));
    drawHeadCell(group, {
      x: kvX,
      y: kvY,
      width: kvWidth,
      height: bridgeCellSize,
      label: band.kvIndex,
      fill: '#4a90d9',
      title: `KV head ${band.kvIndex} serves Q heads ${formatHeadRange(band.qStart, band.qEnd)}`,
      selected: selectedHead?.kind === 'kv' && selectedKVIndex === band.kvIndex,
      related: selectedHead?.kind === 'q' && selectedKVIndex === band.kvIndex,
      onClick: () => onSelectHead?.({ kind: 'kv', index: band.kvIndex }),
    });
  }

  const matrixLabelY = bridgeTopY + bridgeHeight + 20;
  group.appendChild(svgElement('text', {
    x: panelX + 20, y: matrixLabelY, 'font-size': 10, fill: '#c6cfdd', 'font-weight': 600,
  }, 'Clickable head matrix'));

  const qMatrixLabelY = matrixLabelY + 14;
  group.appendChild(svgElement('text', {
    x: panelX + 20, y: qMatrixLabelY, 'font-size': 9.5, fill: '#9aa4b8',
  }, `Q heads (${headInfo.headCount})`));
  const qMatrixTop = qMatrixLabelY + 6;
  for (let headIndex = 0; headIndex < headInfo.headCount; headIndex++) {
    const cellCol = headIndex % matrixCols;
    const cellRow = Math.floor(headIndex / matrixCols);
    const x = panelX + 20 + cellCol * (matrixCellSize + matrixCellGap);
    const y = qMatrixTop + cellRow * (matrixCellSize + matrixCellGap);
    drawHeadCell(group, {
      x,
      y,
      width: matrixCellSize,
      height: matrixCellSize,
      label: headInfo.headCount <= 96 ? headIndex : null,
      fill: COLORS.attention,
      title: `Q head ${headIndex}`,
      selected: selectedQIndex === headIndex,
      related: selectedKVIndex === getKVGroupIndex(headInfo, headIndex) && selectedQIndex !== headIndex,
      onClick: () => onSelectHead?.({ kind: 'q', index: headIndex }),
    });
  }

  const kvMatrixLabelY = qMatrixTop + qMatrixHeight + 8;
  group.appendChild(svgElement('text', {
    x: panelX + 20, y: kvMatrixLabelY, 'font-size': 9.5, fill: '#9aa4b8',
  }, `KV heads (${headInfo.headCountKV})`));
  const kvMatrixTop = kvMatrixLabelY + 6;
  for (let kvIndex = 0; kvIndex < (headInfo.headCountKV || 1); kvIndex++) {
    const cellCol = kvIndex % matrixCols;
    const cellRow = Math.floor(kvIndex / matrixCols);
    const x = panelX + 20 + cellCol * (matrixCellSize + matrixCellGap);
    const y = kvMatrixTop + cellRow * (matrixCellSize + matrixCellGap);
    drawHeadCell(group, {
      x,
      y,
      width: matrixCellSize,
      height: matrixCellSize,
      label: (headInfo.headCountKV || 1) <= 96 ? kvIndex : null,
      fill: '#4a90d9',
      title: `KV head ${kvIndex}`,
      selected: selectedHead?.kind === 'kv' && selectedKVIndex === kvIndex,
      related: selectedHead?.kind === 'q' && selectedKVIndex === kvIndex,
      onClick: () => onSelectHead?.({ kind: 'kv', index: kvIndex }),
    });
  }

  return totalHeight;
}

function drawDetailPanel(camera, stage, contentWidth, selectedHead = null, onSelectHead = null) {
  const { hasAttnDetail, panelWidth, panelHeight } = getDetailPanelMetrics(stage);
  const diagramYOffset = 48;
  const panelMinX = 36;
  const panelMaxX = Math.max(panelMinX, contentWidth - panelWidth - 36);
  const panelX = clamp(stage.x + stage.width / 2 - panelWidth / 2, panelMinX, panelMaxX);
  const panelY = 208;
  const mainY = panelY + 120 + diagramYOffset;
  const group = svgElement('g');
  const connector = svgElement('path', {
    d: `M ${stage.x + stage.width / 2} ${stage.y + stage.height} L ${stage.x + stage.width / 2} ${panelY - 18} L ${panelX + panelWidth / 2} ${panelY - 18}`,
    fill: 'none', stroke: '#536079', 'stroke-width': 2,
  });
  const panel = svgElement('rect', {
    x: panelX, y: panelY, width: panelWidth, height: panelHeight, rx: 18,
    fill: '#262b35', stroke: '#3d4656', 'stroke-width': 1.5,
  });
  const title = svgElement('text', { x: panelX + 20, y: panelY + 28, 'font-size': 14, 'font-weight': 700, fill: '#fff' }, `${stage.label} detail`);
  const subtitle = svgElement('text', { x: panelX + 20, y: panelY + 48, 'font-size': 11.5, fill: '#9aa4b8' }, `${stage.pattern} • ${formatParams(stage.params)} • ${formatBytes(stage.memoryBytes)}`);
  const residual = svgElement('line', {
    x1: panelX + 34, y1: mainY, x2: panelX + panelWidth - 34, y2: mainY,
    stroke: COLORS.residual, 'stroke-width': 8, 'stroke-linecap': 'round', opacity: 0.7,
  });

  group.append(connector, panel, title, subtitle, residual);
  group.appendChild(svgElement('text', { x: panelX + 18, y: mainY - 10, 'font-size': 10.5, fill: '#bac3d2' }, 'input'));
  group.appendChild(svgElement('text', { x: panelX + panelWidth - 46, y: mainY - 10, 'font-size': 10.5, fill: '#bac3d2' }, 'output'));

  stage.detailRows.forEach((row, rowIndex) => {
    if (row.layout === 'attention-qkv' && row.attentionDetail) {
      drawAttentionPath(group, row.attentionDetail, panelX, mainY, panelWidth);
    } else {
      drawLinearRow(group, row, panelX, mainY, panelWidth, rowIndex);
    }
  });

  if (hasAttnDetail && stage.headInfo?.headCount) {
    const separatorY = mainY + 52;
    drawSVGHeadWiring(group, stage, panelX, separatorY + 78, panelWidth, selectedHead, onSelectHead);
  }

  camera.appendChild(group);
}

export function renderResidualFlow(container, model, uiState = {}) {
  const graph = buildResidualFlowGraph(model);
  const selectedStage = graph.stages.find(stage => stage.type === 'block' && stage.index === uiState.selectedLayerIndex) || null;
  const selectedHead = selectedStage?.headInfo?.headCount ? uiState.selectedHead || null : null;
  const wrapper = document.createElement('section');
  wrapper.className = 'architecture-view';

  const toolbar = document.createElement('div');
  toolbar.className = 'architecture-toolbar';
  const leftTools = document.createElement('div');
  leftTools.className = 'toolbar-group';
  leftTools.innerHTML = '<span class="inspector-note">Zoom and pan the full residual stream, then click a block to expand its internal branches.</span>';
  const rightTools = document.createElement('div');
  rightTools.className = 'toolbar-group';
  toolbar.append(leftTools, rightTools);
  wrapper.appendChild(toolbar);
  renderLegend(wrapper);

  const canvas = document.createElement('div');
  canvas.className = 'architecture-canvas';
  wrapper.appendChild(canvas);

  const inspector = document.createElement('section');
  inspector.className = 'architecture-inspector';
  wrapper.appendChild(inspector);

  container.appendChild(wrapper);
  renderInspector(inspector, selectedStage, model, selectedHead, uiState);

  const detailPanelHeight = selectedStage
    ? getDetailPanelMetrics(selectedStage).panelHeight
    : 560;
  const svgHeight = selectedStage ? Math.max(560, detailPanelHeight) : 420;
  const svg = svgElement('svg', { class: 'architecture-svg', width: '100%', height: svgHeight });
  canvas.appendChild(svg);
  const defs = svgElement('defs');
  const arrowhead = svgElement('marker', {
    id: 'architecture-arrowhead',
    markerWidth: 8,
    markerHeight: 8,
    refX: 7,
    refY: 4,
    orient: 'auto',
    markerUnits: 'strokeWidth',
  });
  arrowhead.appendChild(svgElement('path', { d: 'M 0 0 L 8 4 L 0 8 z', fill: '#7f9ac7' }));
  defs.appendChild(arrowhead);
  svg.appendChild(defs);
  const viewportWidth = canvas.clientWidth || container.clientWidth || 960;
  const viewportHeight = svgHeight;
  svg.setAttribute('viewBox', `0 0 ${viewportWidth} ${viewportHeight}`);

  const camera = svgElement('g');
  svg.appendChild(camera);

  const baseY = 116;
  let cursor = 72;
  for (const stage of graph.stages) {
    stage.x = cursor;
    stage.y = baseY - stage.height / 2;
    cursor += stage.width + 48;
  }
  const contentWidth = cursor + 48;
  const contentHeight = selectedStage ? Math.max(470, detailPanelHeight - 90) : 210;
  const background = svgElement('rect', {
    x: 0, y: 0, width: contentWidth, height: Math.max(contentHeight + 80, viewportHeight), fill: 'transparent',
  });
  camera.appendChild(background);
  camera.appendChild(svgElement('line', {
    x1: graph.stages[0].x + graph.stages[0].width / 2,
    y1: baseY,
    x2: graph.stages[graph.stages.length - 1].x + graph.stages[graph.stages.length - 1].width / 2,
    y2: baseY,
    stroke: COLORS.residual,
    'stroke-width': 10,
    'stroke-linecap': 'round',
    opacity: 0.65,
  }));

  for (const stage of graph.stages) drawStage(camera, stage, selectedStage?.index === stage.index, uiState.onSelectLayer || (() => {}));
  if (selectedStage) drawDetailPanel(camera, selectedStage, contentWidth, selectedHead, uiState.onSelectHead || null);

  const transform = uiState.transform ? { ...uiState.transform } : { scale: 1, x: 0, y: 0 };
  function applyTransform() {
    camera.setAttribute('transform', `translate(${transform.x} ${transform.y}) scale(${transform.scale})`);
    uiState.onTransformChange?.({ ...transform });
  }

  function fitToScreen() {
    transform.scale = clamp(Math.min((viewportWidth - 40) / contentWidth, (viewportHeight - 40) / (contentHeight + 40), 1.35), 0.35, 2.6);
    transform.x = Math.max(20, (viewportWidth - contentWidth * transform.scale) / 2);
    transform.y = Math.max(24, (viewportHeight - contentHeight * transform.scale) / 2);
    applyTransform();
  }

  function resetTransform() {
    transform.scale = 1;
    transform.x = 24;
    transform.y = 28;
    applyTransform();
  }

  function zoomBy(factor, clientX = viewportWidth / 2, clientY = viewportHeight / 2) {
    const nextScale = clamp(transform.scale * factor, 0.35, 2.8);
    const pointX = (clientX - transform.x) / transform.scale;
    const pointY = (clientY - transform.y) / transform.scale;
    transform.scale = nextScale;
    transform.x = clientX - pointX * transform.scale;
    transform.y = clientY - pointY * transform.scale;
    applyTransform();
  }

  function panBy(dx, dy = 0) {
    transform.x += dx;
    transform.y += dy;
    applyTransform();
  }

  rightTools.append(
    createButton('Fit', fitToScreen),
    createButton('Reset', resetTransform),
    createButton('Zoom +', () => zoomBy(1.15)),
    createButton('Zoom −', () => zoomBy(1 / 1.15)),
    createButton('Pan ↑', () => panBy(0, 180)),
    createButton('Pan ↓', () => panBy(0, -180)),
    createButton('← Pan', () => panBy(180)),
    createButton('Pan →', () => panBy(-180)),
  );

  let dragState = null;
  background.style.cursor = 'grab';
  background.addEventListener('pointerdown', (event) => {
    dragState = { x: event.clientX, y: event.clientY, tx: transform.x, ty: transform.y };
    background.style.cursor = 'grabbing';
    background.setPointerCapture(event.pointerId);
  });
  background.addEventListener('pointermove', (event) => {
    if (!dragState) return;
    transform.x = dragState.tx + (event.clientX - dragState.x);
    transform.y = dragState.ty + (event.clientY - dragState.y);
    applyTransform();
  });
  background.addEventListener('pointerup', () => {
    dragState = null;
    background.style.cursor = 'grab';
  });
  background.addEventListener('pointerleave', () => {
    if (!dragState) background.style.cursor = 'grab';
  });
  svg.addEventListener('wheel', (event) => {
    event.preventDefault();
    const rect = svg.getBoundingClientRect();
    zoomBy(event.deltaY < 0 ? 1.08 : 1 / 1.08, event.clientX - rect.left, event.clientY - rect.top);
  }, { passive: false });

  if (uiState.transform) applyTransform();
  else fitToScreen();
}