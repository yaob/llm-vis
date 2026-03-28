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

const MOE_SUBDIAGRAM_ROW_OFFSET = -48;

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

const MLP_SELECTIONS = {
  'mlp-up': {
    key: 'up',
    title: 'FFN Up',
    shortLabel: 'Up',
    tensorLabel: 'ffn_up',
    badge: 'Expansion projection',
    note: 'Projects the model-state features into the intermediate FFN width before gating and contraction.',
  },
  'mlp-gate': {
    key: 'gate',
    title: 'FFN Gate',
    shortLabel: 'Gate',
    tensorLabel: 'ffn_gate',
    badge: 'Gate projection',
    note: 'Produces gate values that modulate the Up branch before the FFN contracts back down.',
  },
  'mlp-down': {
    key: 'down',
    title: 'FFN Down',
    shortLabel: 'Down',
    tensorLabel: 'ffn_down',
    badge: 'Output projection',
    note: 'Projects the intermediate FFN activations back to model width for the residual merge.',
  },
};

function getMLPSelectionConfig(kind) {
  return kind ? MLP_SELECTIONS[kind] || null : null;
}

const MOE_SELECTIONS = {
  'moe-router': {
    key: 'router',
    title: 'MoE Router',
    shortLabel: 'Router',
    tensorLabel: 'ffn_gate_inp',
    badge: 'Routing projection',
    note: 'Projects token features into routing logits used to score and choose the active experts.',
  },
  'moe-up': {
    key: 'expertUp',
    title: 'MoE Expert Up',
    shortLabel: 'Up',
    tensorLabel: 'ffn_up_exp',
    badge: 'Expert expansion',
    note: 'Packed per-expert expansion weights that lift routed token activations into each expert hidden space.',
  },
  'moe-gate': {
    key: 'expertGate',
    title: 'MoE Expert Gate',
    shortLabel: 'Gate',
    tensorLabel: 'ffn_gate_exp',
    badge: 'Expert gating',
    note: 'Packed per-expert gating weights used by gated MoE feed-forward variants before expert contraction.',
  },
  'moe-down': {
    key: 'expertDown',
    title: 'MoE Expert Down',
    shortLabel: 'Down',
    tensorLabel: 'ffn_down_exp',
    badge: 'Expert projection',
    note: 'Packed per-expert projection weights that map expert activations back to the model width.',
  },
  'moe-experts': {
    key: 'experts',
    title: 'MoE Experts',
    shortLabel: 'Experts',
    badge: 'Packed expert bank',
    note: 'Packed MoE expert tensors holding the per-expert Up / Gate / Down projections that run after routing.',
  },
};

function getMoESelectionConfig(kind) {
  return kind ? MOE_SELECTIONS[kind] || null : null;
}

const SSM_SELECTIONS = {
  'ssm-norm': {
    key: 'norm',
    title: 'SSM Norm',
    tensorLabel: 'norm',
    badge: 'Normalization',
    note: 'Normalization weights applied before the state-space mixing path.',
  },
  'ssm-in': {
    key: 'input',
    title: 'SSM In',
    tensorLabel: 'ssm_in',
    badge: 'Input projection',
    note: 'Projects model-state features into the SSM path before local mixing and selective scan.',
  },
  'ssm-conv': {
    key: 'conv1d',
    title: 'SSM Conv1D',
    tensorLabel: 'ssm_conv1d',
    badge: 'Local mixing',
    note: 'Applies the short convolution used by the SSM block before selective state updates.',
  },
  'ssm-selective': {
    key: 'selective',
    title: 'SSM Selective',
    tensorLabel: 'ssm_x',
    badge: 'Selective scan',
    note: 'Primary selective state-space tensor used to drive the recurrent scan/update path.',
  },
  'ssm-a': {
    key: 'a',
    title: 'SSM A',
    tensorLabel: 'ssm_a',
    badge: 'State matrix',
    note: 'Learned state transition parameter controlling how the SSM state evolves over sequence positions.',
  },
  'ssm-dt': {
    key: 'dt',
    title: 'SSM Δt',
    tensorLabel: 'ssm_dt',
    badge: 'Step size',
    note: 'Time-step / delta parameter that modulates the selective scan dynamics.',
  },
  'ssm-d': {
    key: 'd',
    title: 'SSM D',
    tensorLabel: 'ssm_d',
    badge: 'Skip parameter',
    note: 'Direct or skip parameter that blends immediate signal flow into the SSM output path.',
  },
  'ssm-out': {
    key: 'output',
    title: 'SSM Out',
    tensorLabel: 'ssm_out',
    badge: 'Output projection',
    note: 'Projects the SSM path back into model width for the residual merge.',
  },
};

function getSSMSelectionConfig(kind) {
  return kind ? SSM_SELECTIONS[kind] || null : null;
}

function getSSMSelectionTitle(stage, selection, tensor) {
  if (selection?.key === 'norm') return stage?.ssmDetail?.normLabel || tensor?.label || selection.title;
  return selection?.title || tensor?.label || 'SSM Component';
}

function getSSMSelectionTensorLabel(stage, selection, tensor) {
  if (selection?.key === 'norm') return tensor?.component || stage?.ssmDetail?.normLabel || selection.tensorLabel;
  return selection?.tensorLabel || tensor?.component || tensor?.label || selection?.key || 'tensor';
}

function getMoEExpertEntries(moeDetail) {
  return [
    ['Expert Up', 'ffn_up_exp', moeDetail?.expertUp],
    ['Expert Gate', 'ffn_gate_exp', moeDetail?.expertGate],
    ['Expert Down', 'ffn_down_exp', moeDetail?.expertDown],
  ].filter(([, , tensor]) => tensor);
}

function getMoEExpertCount(moeDetail) {
  const counts = getMoEExpertEntries(moeDetail)
    .map(([, , tensor]) => Number(tensor?.shape?.[2] ?? tensor?.dimensions?.[2]))
    .filter((value) => Number.isFinite(value) && value > 0);
  return counts.length ? counts[0] : null;
}

function getMoEPathMetrics(moeDetail) {
  const expertCount = Math.max(1, getMoEExpertCount(moeDetail) || 1);
  const denseLayout = expertCount > 12;
  return {
    expertCount,
    laneGap: denseLayout ? 24 : 26,
    leadNodeWidth: 72,
    leadNodeHeight: 28,
    laneNodeWidth: denseLayout ? 44 : 48,
    laneNodeHeight: 20,
    laneTextSize: denseLayout ? 8.6 : 9.2,
    bankPadTop: 26,
    bankPadBottom: 18,
    firstLaneOffset: 42,
  };
}

function getMoEVerticalFootprint(moeDetail, rowYOffset = -48, centerLanesOnRow = false) {
  const metrics = getMoEPathMetrics(moeDetail);
  const laneSpread = (metrics.expertCount - 1) * metrics.laneGap;
  const laneClusterCenterOffset = centerLanesOnRow
    ? rowYOffset
    : rowYOffset + metrics.firstLaneOffset + laneSpread / 2;
  const lastLaneOffset = centerLanesOnRow
    ? laneClusterCenterOffset + laneSpread / 2
    : metrics.firstLaneOffset + laneSpread;
  return {
    ...metrics,
    lastLaneOffset,
    bankBottomOffset: laneClusterCenterOffset + laneSpread / 2 + metrics.laneNodeHeight / 2 + metrics.bankPadBottom,
  };
}

function isMoERow(row) {
  return row?.layout === 'linear' && row.nodes?.some((node) => node.type === 'moe');
}

function getSelectionKindClass(kind) {
  if (kind === 'q') return 'q-head';
  if (kind === 'kv') return 'kv-head';
  if (kind === 'wo') return 'wo-head';
  if (kind?.startsWith('mlp-')) return 'mlp-head';
  if (kind?.startsWith('moe-')) return 'moe-head';
  return 'wo-head';
}

function createHeadCellButton(kind, index, label, selected, related, title, onSelect) {
  const cell = document.createElement('button');
  cell.type = 'button';
  const kindClass = getSelectionKindClass(kind);
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
  if (!selectedHead) return null;
  const mlpSelection = getMLPSelectionConfig(selectedHead.kind);
  if (mlpSelection) {
    const tensor = stage?.mlpDetail?.[mlpSelection.key];
    if (!tensor) return null;
    const totalRows = getTensorRowCount(tensor);
    const previewRows = Math.min(totalRows || 0, 96);
    const previewEnd = Math.max(0, previewRows - 1);
    return {
      title: mlpSelection.title,
      badge: mlpSelection.badge,
      note: previewRows && totalRows > previewRows
        ? `Showing rows 0–${previewEnd} of ${mlpSelection.tensorLabel} as a bounded preview so large FFN matrices stay responsive.`
        : mlpSelection.note,
      fields: [
        { label: 'Tensor slice', value: previewRows ? `Rows 0–${previewEnd} in ${mlpSelection.tensorLabel}` : mlpSelection.tensorLabel },
        { label: 'Tensor shape', value: formatDimsLabel(tensor.shape) },
        { label: 'Parameters', value: formatMaybeParams(tensor.params) },
        { label: 'Memory', value: formatMaybeBytes(tensor.memoryBytes) },
        { label: 'Quantization', value: tensor.typeName || '—' },
        { label: 'FFN style', value: stage?.mlpDetail?.gated ? 'Gated FFN' : 'Plain FFN' },
      ],
      decodePlan: previewRows
        ? {
          cacheKey: `${stage.index}:${selectedHead.kind}`,
          slices: [{ label: mlpSelection.title, tensorLabel: mlpSelection.tensorLabel, tensor, rowStart: 0, rowCount: previewRows }],
        }
        : null,
    };
  }

  const moeSelection = getMoESelectionConfig(selectedHead.kind);
  if (moeSelection) {
    if (moeSelection.key === 'experts') {
      const expertEntries = getMoEExpertEntries(stage?.moeDetail);
      if (!expertEntries.length) return null;
      const previewLimit = 48;
      const slices = expertEntries
        .map(([label, tensorLabel, tensor]) => {
          const rowCount = Math.min(getTensorRowCount(tensor) || 0, previewLimit);
          return rowCount ? { label, tensorLabel, tensor, rowStart: 0, rowCount } : null;
        })
        .filter(Boolean);
      const expertPack = expertEntries.map(([, tensorLabel]) => tensorLabel).join(' • ');
      const expertShapes = expertEntries.map(([label, , tensor]) => `${label} ${formatDimsLabel(tensor?.shape)}`).join(' • ');
      const expertCount = getMoEExpertCount(stage?.moeDetail);
      return {
        title: moeSelection.title,
        badge: moeSelection.badge,
        note: slices.length
          ? `Showing up to the first ${previewLimit} rows from each packed expert tensor so large MoE banks stay responsive.`
          : moeSelection.note,
        fields: [
          { label: 'Tensor pack', value: expertPack || '—' },
          { label: 'Tensor shapes', value: expertShapes || '—' },
          { label: 'Parameters', value: formatMaybeParams(stage?.moeDetail?.experts?.params) },
          { label: 'Memory', value: formatMaybeBytes(stage?.moeDetail?.experts?.memoryBytes) },
          { label: 'Quantization', value: stage?.moeDetail?.experts?.typeName || '—' },
          { label: 'Expert layout', value: `${stage?.moeDetail?.gated ? 'Up / Gate / Down' : 'Up / Down'}${expertCount ? ` • ${expertCount} packed experts` : ''}` },
        ],
        decodePlan: slices.length
          ? {
            cacheKey: `${stage.index}:${selectedHead.kind}`,
            slices,
          }
          : null,
      };
    }

    const tensor = stage?.moeDetail?.[moeSelection.key];
    if (!tensor) return null;
    const totalRows = getTensorRowCount(tensor);
    const previewRows = Math.min(totalRows || 0, moeSelection.key === 'router' ? 96 : 48);
    const previewEnd = Math.max(0, previewRows - 1);
    const expertCount = getMoEExpertCount(stage?.moeDetail);
    const roleLabel = moeSelection.key === 'router' ? 'Routing role' : 'Expert role';
    const roleValue = moeSelection.key === 'router'
      ? 'Scores experts for top-k dispatch'
      : `${moeSelection.badge}${expertCount ? ` • ${expertCount} packed experts` : ''}`;
    return {
      title: moeSelection.title,
      badge: moeSelection.badge,
      note: previewRows && totalRows > previewRows
        ? `Showing rows 0–${previewEnd} of ${moeSelection.tensorLabel} as a bounded preview so large ${moeSelection.key === 'router' ? 'routing' : 'expert'} matrices stay responsive.`
        : moeSelection.note,
      fields: [
        { label: 'Tensor slice', value: previewRows ? `Rows 0–${previewEnd} in ${moeSelection.tensorLabel}` : moeSelection.tensorLabel },
        { label: 'Tensor shape', value: formatDimsLabel(tensor.shape) },
        { label: 'Parameters', value: formatMaybeParams(tensor.params) },
        { label: 'Memory', value: formatMaybeBytes(tensor.memoryBytes) },
        { label: 'Quantization', value: tensor.typeName || '—' },
        { label: roleLabel, value: roleValue },
      ],
      decodePlan: previewRows
        ? {
          cacheKey: `${stage.index}:${selectedHead.kind}`,
          slices: [{ label: moeSelection.title, tensorLabel: moeSelection.tensorLabel, tensor, rowStart: 0, rowCount: previewRows }],
        }
        : null,
    };
  }

  const ssmSelection = getSSMSelectionConfig(selectedHead.kind);
  if (ssmSelection) {
    const tensor = stage?.ssmDetail?.[ssmSelection.key];
    if (!tensor) return null;
    const totalRows = getTensorRowCount(tensor);
    const previewRows = Math.min(totalRows || 0, 96);
    const previewEnd = Math.max(0, previewRows - 1);
    const title = getSSMSelectionTitle(stage, ssmSelection, tensor);
    const tensorLabel = getSSMSelectionTensorLabel(stage, ssmSelection, tensor);
    return {
      title,
      badge: ssmSelection.badge,
      note: previewRows && totalRows > previewRows
        ? `Showing rows 0–${previewEnd} of ${tensorLabel} as a bounded preview so large SSM tensors stay responsive.`
        : ssmSelection.note,
      fields: [
        { label: 'Tensor slice', value: previewRows ? `Rows 0–${previewEnd} in ${tensorLabel}` : tensorLabel },
        { label: 'Tensor shape', value: formatDimsLabel(tensor.shape) },
        { label: 'Parameters', value: formatMaybeParams(tensor.params) },
        { label: 'Memory', value: formatMaybeBytes(tensor.memoryBytes) },
        { label: 'Quantization', value: tensor.typeName || '—' },
        { label: 'SSM role', value: ssmSelection.badge },
      ],
      decodePlan: previewRows
        ? {
          cacheKey: `${stage.index}:${selectedHead.kind}`,
          slices: [{ label: title, tensorLabel, tensor, rowStart: 0, rowCount: previewRows }],
        }
        : null,
    };
  }

  if (!stage?.headInfo?.headCount) return null;
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

function renderHTMLMLPGrid(container, mlpDetail, selectedHead, onSelectHead) {
  const buttons = [
    ['mlp-up', mlpDetail?.up],
    ['mlp-gate', mlpDetail?.gate],
    ['mlp-down', mlpDetail?.down],
  ].filter(([, tensor]) => tensor);
  if (!buttons.length) return;

  const section = document.createElement('div');
  section.className = 'head-grid-section';

  const title = document.createElement('h5');
  title.textContent = `MLP Components${mlpDetail?.gated ? ' · gated FFN' : ''}`;
  section.appendChild(title);

  const label = document.createElement('div');
  label.className = 'head-grid-label';
  label.textContent = 'Inspectable weight matrices';
  section.appendChild(label);

  const grid = document.createElement('div');
  grid.className = 'head-grid';
  buttons.forEach(([kind, tensor]) => {
    const config = getMLPSelectionConfig(kind);
    grid.appendChild(createHeadCellButton(
      kind,
      undefined,
      config?.shortLabel || tensor?.label || kind,
      selectedHead?.kind === kind,
      false,
      `${config?.title || tensor?.label || kind} ${formatDimsLabel(tensor?.shape)}`,
      onSelectHead,
    ));
  });
  section.appendChild(grid);

  const hint = document.createElement('div');
  hint.className = 'head-grid-hint';
  hint.textContent = 'Use these selectors—or click the Up / Gate / Down boxes in the SVG wiring—to inspect tensor metadata and weight heatmaps.';
  section.appendChild(hint);

  container.appendChild(section);
}

function renderHTMLMoEGrid(container, moeDetail, selectedHead, onSelectHead) {
  const buttons = [
    ['moe-router', moeDetail?.router],
    ['moe-up', moeDetail?.expertUp],
    ['moe-gate', moeDetail?.expertGate],
    ['moe-down', moeDetail?.expertDown],
  ].filter(([, tensor]) => tensor);
  if (!buttons.length) return;

  const section = document.createElement('div');
  section.className = 'head-grid-section';

  const title = document.createElement('h5');
  const expertCount = getMoEExpertCount(moeDetail);
  title.textContent = `MoE Components${expertCount ? ` · ${expertCount} experts packed` : ''}`;
  section.appendChild(title);

  const label = document.createElement('div');
  label.className = 'head-grid-label';
  label.textContent = 'Inspectable routing and expert tensors';
  section.appendChild(label);

  const grid = document.createElement('div');
  grid.className = 'head-grid';
  buttons.forEach(([kind, tensor]) => {
    const config = getMoESelectionConfig(kind);
    const tooltip = kind === 'moe-experts'
      ? `${config?.title || tensor?.label || kind}${expertCount ? ` • ${expertCount} packed experts` : ''}`
      : `${config?.title || tensor?.label || kind} ${formatDimsLabel(tensor?.shape)}${kind === 'moe-router' || !expertCount ? '' : ` • ${expertCount} packed experts`}`;
    grid.appendChild(createHeadCellButton(
      kind,
      undefined,
      config?.shortLabel || tensor?.label || kind,
      selectedHead?.kind === kind,
      false,
      tooltip,
      onSelectHead,
    ));
  });
  section.appendChild(grid);

  const hint = document.createElement('div');
  hint.className = 'head-grid-hint';
  hint.textContent = 'Use these selectors—or click the Router / Up / Gate / Down regions in the SVG wiring—to inspect tensor metadata and MoE weight heatmaps.';
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
  const hi = stage.headInfo;
  const hasMLPSelection = !!(stage.mlpDetail?.up || stage.mlpDetail?.gate || stage.mlpDetail?.down);
  const hasMoESelection = !!(stage.moeDetail?.router || stage.moeDetail?.experts);
  const hasSSMSelection = !!(stage.ssmDetail?.norm || stage.ssmDetail?.input || stage.ssmDetail?.conv1d || stage.ssmDetail?.selective || stage.ssmDetail?.a || stage.ssmDetail?.dt || stage.ssmDetail?.d || stage.ssmDetail?.output);
  const hasSelectorGrid = !!(hi?.headCount || hasMLPSelection || hasMoESelection);
  const hasSelectableDetail = !!(hi?.headCount || hasMLPSelection || hasMoESelection || hasSSMSelection);
  const selectedHeadDetail = hasSelectableDetail ? getSelectedHeadDetail(stage, selectedHead) : null;
  let selectedHeadHTML = '';
  if (hasSelectableDetail) {
    headSelectorHTML = hasSelectorGrid ? '<div data-head-grid></div>' : '';
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
            <span class="head-chip">Selected component inspector</span>
          </div>
          <p class="selected-head-note">Click a Q, KV, Wo, MLP, MoE, or SSM node in the SVG wiring diagram${hasSelectorGrid ? ', or any selector pill above,' : ''} to inspect tensor slices, params, memory, quantization, and weight heatmaps.</p>
        </div>
      `;
  }
  if (hi && hi.headCount) {
    const typeLabel = getAttentionTypeLabel(hi, true);
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
  if (headGridHost) {
    if (hi?.headCount) renderHTMLHeadGrid(headGridHost, hi, selectedHead, uiState.onSelectHead || null);
    if (hasMLPSelection) renderHTMLMLPGrid(headGridHost, stage.mlpDetail, selectedHead, uiState.onSelectHead || null);
    if (hasMoESelection) renderHTMLMoEGrid(headGridHost, stage.moeDetail, selectedHead, uiState.onSelectHead || null);
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
  if (node.detail.tensors?.length) {
    parts.push(`Tensors: ${node.detail.tensors.map((tensor) => tensor.label || tensor.component || tensor.name).join(' • ')}`);
    if (node.detail.expertCount) parts.push(`Packed experts: ${node.detail.expertCount}`);
  }
  parts.push(`Params: ${formatParams(node.detail.params)}`);
  parts.push(`Memory: ${formatBytes(node.detail.memoryBytes)}`);
  if (node.detail.typeName) parts.push(`Quant: ${node.detail.typeName}`);
  return parts.join('\n');
}

function drawNodeBox(group, cx, cy, w, h, node) {
  const g = svgElement('g');
  const selected = !!node.selected;
  const related = !!node.related;
  const stroke = selected ? COLORS.selected : related ? '#8db6e7' : (COLORS[node.type] || '#6f7a8e');
  g.appendChild(svgElement('rect', {
    x: cx - w / 2, y: cy - h / 2, width: w, height: h, rx: 10,
    fill: selected ? '#253446' : '#202530', stroke, 'stroke-width': selected ? 2.2 : related ? 1.8 : 1.5,
  }));
  g.appendChild(svgElement('text', {
    x: cx, y: cy + 4, 'text-anchor': 'middle', 'font-size': node.textSize || 10, 'font-weight': 600, fill: '#fff',
  }, node.label));
  g.appendChild(svgElement('title', {}, nodeTooltip(node)));
  if (node.showShape !== false && node.detail?.shape?.length) {
    g.appendChild(svgElement('text', {
      x: cx, y: cy + h / 2 + 11, 'text-anchor': 'middle', 'font-size': node.shapeTextSize || 8.5, fill: '#7a8599',
    }, `[${formatShape(node.detail.shape)}]`));
  }
  if (node.onClick) {
    g.style.cursor = 'pointer';
    g.addEventListener('click', node.onClick);
  }
  group.appendChild(g);
  return g;
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
  const hasSSMDetail = stage.detailRows.some((row) => row.layout === 'ssm');
  const hasMLPDetail = stage.detailRows.some((row) => row.layout === 'mlp');
  const hasMoEDetail = stage.detailRows.some((row) => isMoERow(row));
  const hasTripleDetail = hasAttnDetail && hasSSMDetail && hasMLPDetail;
  const hasSequentialAttnMoE = hasAttnDetail && hasMoEDetail && !hasTripleDetail && !hasSSMDetail && !hasMLPDetail;
  const hasSequentialSSMMLP = hasSSMDetail && hasMLPDetail;
  let diagramYOffset = 68;
  if (hasMoEDetail) {
    const moeRowYOffset = hasSequentialAttnMoE ? MOE_SUBDIAGRAM_ROW_OFFSET : -48;
    const moeFootprint = getMoEVerticalFootprint(stage.moeDetail, moeRowYOffset, hasSequentialAttnMoE);
    const laneSpread = (moeFootprint.expertCount - 1) * moeFootprint.laneGap;
    const bottomOverlap = laneSpread / 2 + moeFootprint.laneNodeHeight / 2 + Math.abs(moeRowYOffset) + 16;
    if (bottomOverlap > diagramYOffset) diagramYOffset = bottomOverlap;
  }
  const panelWidth = hasTripleDetail ? 1280 : (hasAttnDetail && hasMoEDetail) ? 1280 : hasAttnDetail ? 1120 : hasSequentialSSMMLP ? 1080 : hasSSMDetail ? 820 : 450;
  let basePanelHeight = hasTripleDetail ? 340 : hasAttnDetail ? 260 : hasSSMDetail ? 280 : 240;
  let headDiagramTopGap = hasTripleDetail ? 216 : 140;
  if (hasMoEDetail) {
    const moeRowYOffset = hasSequentialAttnMoE ? MOE_SUBDIAGRAM_ROW_OFFSET : -48;
    const moeFootprint = getMoEVerticalFootprint(stage.moeDetail, moeRowYOffset, hasSequentialAttnMoE);
    basePanelHeight = Math.max(basePanelHeight, 176 + moeFootprint.bankBottomOffset);
    if (hasAttnDetail) headDiagramTopGap = Math.max(headDiagramTopGap, 84 + moeFootprint.bankBottomOffset);
  }
  const wiringLayout = hasAttnDetail && stage.headInfo?.headCount
    ? getSVGHeadWiringLayout(stage.headInfo, panelWidth)
    : null;
  const headDiagramHeight = wiringLayout ? headDiagramTopGap + wiringLayout.totalHeight : 0;

  return {
    hasAttnDetail,
    hasTripleDetail,
    headDiagramTopGap,
    diagramYOffset,
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
    ? formatDimsLabel([headInfo.contextLength])
    : '[ctx]';
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

function drawResidualMerge(group, mergeX, mainY) {
  group.appendChild(svgElement('circle', { cx: mergeX, cy: mainY, r: 11, fill: '#2d3441', stroke: '#5a6478', 'stroke-width': 1.5 }));
  group.appendChild(svgElement('text', { x: mergeX, y: mainY + 4, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, fill: '#fff' }, '+'));
}

function drawAttentionPath(group, detail, panelX, mainY, panelWidth, layout = {}) {
  const rowCenterY = layout.rowCenterY ?? (mainY - 50);
  const nw = 56; // node width
  const nh = 24; // node height
  const qkvSpacing = 48;
  const color = COLORS.attention;
  const leadInOffset = 32;

  // Horizontal positions
  const entryX = layout.entryX ?? (panelX + 44);
  const normX = layout.normX ?? (panelX + 116 + leadInOffset);
  const qkvX = layout.qkvX ?? (normX + nw / 2 + 56);
  const softmaxX = layout.softmaxX ?? (qkvX + nw / 2 + 56);
  const outProjX = layout.outProjX ?? (softmaxX + nw / 2 + 56);
  const mergeX = layout.mergeX ?? Math.min(outProjX + nw / 2 + 28, panelX + panelWidth - 50);
  const label = layout.label || 'Attention path';

  group.appendChild(svgElement('text', { x: panelX + 20, y: rowCenterY - 38, 'font-size': 10.5, fill: '#9aa4b8' }, label));

  // Branch from residual up to norm
  group.appendChild(svgElement('path', {
    d: `M ${entryX} ${mainY} C ${entryX + 16} ${mainY} ${normX - nw / 2 - 12} ${rowCenterY} ${normX - nw / 2} ${rowCenterY}`,
    fill: 'none', stroke: color, 'stroke-width': 2.2,
  }));

  // Attn Norm box
  drawNodeBox(group, normX, rowCenterY, nw, nh, { label: 'Attn Norm', type: 'norm', detail: detail.norm });

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
  drawResidualMerge(group, mergeX, mainY);
}

function drawMLPRow(group, row, panelX, mainY, panelWidth, rowIndex, selectedHead, onSelectHead, layout = {}) {
  const detail = row.mlpDetail || {};
  const color = COLORS.mlp;
  const nodeWidth = panelWidth <= 420 ? 72 : 76;
  const nodeHeight = 32;
  const nodeHalfWidth = nodeWidth / 2;
  const isCompactPanel = panelWidth <= 420;
  const leadInOffset = isCompactPanel ? 0 : 32;
  const entryX = layout.entryX ?? (panelX + 44);
  const mergeX = layout.mergeX ?? (isCompactPanel
    ? panelX + panelWidth - 74
    : (rowIndex === 0 ? panelX + panelWidth * 0.62 : panelX + panelWidth * 0.84));
  const rowY = layout.rowY ?? (rowIndex === 0 ? mainY - 58 : mainY + 58);
  const labelY = layout.labelY ?? (rowY - (detail.gate ? 40 : 24));
  const normX = layout.normX ?? (isCompactPanel ? panelX + 108 : panelX + 124 + leadInOffset);
  const branchX = layout.branchX ?? (isCompactPanel ? panelX + 196 : panelX + 244 + leadInOffset);
  const downX = layout.downX ?? (isCompactPanel ? panelX + 282 : panelX + 360 + leadInOffset);
  const branchSpread = detail.gate ? 22 : 0;
  const upY = rowY - branchSpread;
  const gateY = rowY + branchSpread;
  const mixX = downX - nodeHalfWidth - 22;
  const label = layout.label || row.label;
  const labelX = layout.labelX ?? (panelX + 20);
  const labelAnchor = layout.labelAnchor || 'start';

  group.appendChild(svgElement('text', { x: labelX, y: labelY, 'text-anchor': labelAnchor, 'font-size': 10.5, fill: '#9aa4b8' }, label));
  drawCurvedConnector(group, [entryX, mainY], [normX - nodeHalfWidth, rowY], {
    stroke: color,
    strokeWidth: 2.2,
    arrow: false,
  });

  drawNodeBox(group, normX, rowY, nodeWidth, nodeHeight, { label: 'FFN Norm', type: 'norm', detail: detail.norm });

  drawCurvedConnector(group, [normX + nodeHalfWidth, rowY], [branchX - nodeHalfWidth, upY], {
    stroke: color,
    strokeWidth: 1.8,
    arrow: false,
  });
  drawNodeBox(group, branchX, upY, nodeWidth, nodeHeight, {
    label: 'Up',
    type: 'mlp',
    detail: detail.up,
    selected: selectedHead?.kind === 'mlp-up',
    onClick: detail.up ? () => onSelectHead?.({ kind: 'mlp-up' }) : null,
  });

  if (detail.gate) {
    drawCurvedConnector(group, [normX + nodeHalfWidth, rowY], [branchX - nodeHalfWidth, gateY], {
      stroke: color,
      strokeWidth: 1.8,
      arrow: false,
    });
    drawNodeBox(group, branchX, gateY, nodeWidth, nodeHeight, {
      label: 'Gate',
      type: 'mlp',
      detail: detail.gate,
      selected: selectedHead?.kind === 'mlp-gate',
      onClick: detail.gate ? () => onSelectHead?.({ kind: 'mlp-gate' }) : null,
    });
    drawCurvedConnector(group, [branchX + nodeHalfWidth, upY], [mixX, rowY], {
      stroke: color,
      strokeWidth: 1.8,
      arrow: false,
    });
    drawCurvedConnector(group, [branchX + nodeHalfWidth, gateY], [mixX, rowY], {
      stroke: color,
      strokeWidth: 1.8,
      arrow: false,
    });
    group.appendChild(svgElement('circle', {
      cx: mixX,
      cy: rowY,
      r: 8.5,
      fill: '#243128',
      stroke: '#5fba6d',
      'stroke-width': 1.4,
    }));
    group.appendChild(svgElement('text', {
      x: mixX,
      y: rowY + 3.5,
      'text-anchor': 'middle',
      'font-size': 10,
      'font-weight': 700,
      fill: '#ecfff0',
    }, '×'));
    group.appendChild(svgElement('line', {
      x1: mixX + 8.5,
      y1: rowY,
      x2: downX - nodeHalfWidth,
      y2: rowY,
      stroke: color,
      'stroke-width': 1.8,
    }));
  } else {
    drawCurvedConnector(group, [branchX + nodeHalfWidth, upY], [downX - nodeHalfWidth, rowY], {
      stroke: color,
      strokeWidth: 1.8,
      arrow: false,
    });
  }

  drawNodeBox(group, downX, rowY, nodeWidth, nodeHeight, {
    label: 'Down',
    type: 'mlp',
    detail: detail.down,
    selected: selectedHead?.kind === 'mlp-down',
    onClick: detail.down ? () => onSelectHead?.({ kind: 'mlp-down' }) : null,
  });

  drawCurvedConnector(group, [downX + nodeHalfWidth, rowY], [mergeX, mainY], {
    stroke: color,
    strokeWidth: 2.2,
    arrow: false,
  });
  drawResidualMerge(group, mergeX, mainY);
}

function drawSSMRow(group, row, panelX, mainY, panelWidth, rowIndex, selectedHead, onSelectHead, layout = {}) {
  const detail = row.ssmDetail || {};
  const color = COLORS.ssm;
  const nodeWidth = 74;
  const nodeHeight = 30;
  const sideWidth = 54;
  const sideHeight = 22;
  const nodeHalfWidth = nodeWidth / 2;
  const sideHalfWidth = sideWidth / 2;
  const leadInOffset = 32;
  const entryX = layout.entryX ?? (panelX + 44);
  const mergeX = layout.mergeX ?? (panelX + panelWidth - 44);
  const rowY = layout.rowY ?? (rowIndex === 0 ? mainY - 58 : mainY + 58);
  const label = layout.label || row.label;
  const labelY = layout.labelY ?? (rowY - 78);
  const normX = layout.normX ?? (panelX + 110 + leadInOffset);
  const inX = layout.inX ?? (panelX + 204 + leadInOffset);
  const convX = layout.convX ?? (panelX + 298 + leadInOffset);
  const selectiveX = layout.selectiveX ?? (panelX + 414 + leadInOffset);
  const paramX = layout.paramX ?? (panelX + 520 + leadInOffset);
  const outX = layout.outX ?? (panelX + 620 + leadInOffset);
  const sideNodes = [
    { label: 'A', detail: detail.a, y: rowY - 58 },
    { label: 'Δt', detail: detail.dt, y: rowY },
    { label: 'D', detail: detail.d, y: rowY + 58 },
  ];

  group.appendChild(svgElement('text', { x: panelX + 20, y: labelY, 'font-size': 10.5, fill: '#9aa4b8' }, label));

  drawCurvedConnector(group, [entryX, mainY], [normX - nodeHalfWidth, rowY], {
    stroke: color,
    strokeWidth: 2.2,
    arrow: false,
  });
  drawNodeBox(group, normX, rowY, nodeWidth, nodeHeight, {
    label: detail.normLabel || 'Norm',
    type: 'norm',
    detail: detail.norm,
    selected: selectedHead?.kind === 'ssm-norm',
    onClick: detail.norm ? () => onSelectHead?.({ kind: 'ssm-norm' }) : null,
  });

  drawCurvedConnector(group, [normX + nodeHalfWidth, rowY], [inX - nodeHalfWidth, rowY], {
    stroke: color,
    strokeWidth: 1.8,
    arrow: false,
  });
  drawNodeBox(group, inX, rowY, nodeWidth, nodeHeight, {
    label: 'In',
    type: 'ssm',
    detail: detail.input,
    selected: selectedHead?.kind === 'ssm-in',
    onClick: detail.input ? () => onSelectHead?.({ kind: 'ssm-in' }) : null,
  });

  drawCurvedConnector(group, [inX + nodeHalfWidth, rowY], [convX - nodeHalfWidth, rowY], {
    stroke: color,
    strokeWidth: 1.8,
    arrow: false,
  });
  drawNodeBox(group, convX, rowY, nodeWidth, nodeHeight, {
    label: 'Conv1D',
    type: 'ssm',
    detail: detail.conv1d,
    selected: selectedHead?.kind === 'ssm-conv',
    onClick: detail.conv1d ? () => onSelectHead?.({ kind: 'ssm-conv' }) : null,
  });

  drawCurvedConnector(group, [convX + nodeHalfWidth, rowY], [selectiveX - nodeHalfWidth, rowY], {
    stroke: color,
    strokeWidth: 1.8,
    arrow: false,
  });
  drawNodeBox(group, selectiveX, rowY, nodeWidth, nodeHeight, {
    label: 'Selective',
    type: 'ssm',
    detail: detail.selective,
    selected: selectedHead?.kind === 'ssm-selective',
    onClick: detail.selective ? () => onSelectHead?.({ kind: 'ssm-selective' }) : null,
  });

  sideNodes.forEach((node) => {
    const kind = node.label === 'A'
      ? 'ssm-a'
      : node.label === 'Δt'
        ? 'ssm-dt'
        : 'ssm-d';
    drawNodeBox(group, paramX, node.y, sideWidth, sideHeight, {
      label: node.label,
      type: 'ssm',
      detail: node.detail,
      selected: selectedHead?.kind === kind,
      onClick: node.detail ? () => onSelectHead?.({ kind }) : null,
    });
    group.appendChild(svgElement('line', {
      x1: selectiveX + nodeHalfWidth,
      y1: rowY,
      x2: paramX - sideHalfWidth,
      y2: node.y,
      stroke: color,
      'stroke-width': 1.5,
      opacity: 0.9,
    }));
  });

  drawCurvedConnector(group, [selectiveX + nodeHalfWidth, rowY], [outX - nodeHalfWidth, rowY], {
    stroke: color,
    strokeWidth: 1.8,
    arrow: false,
  });
  drawNodeBox(group, outX, rowY, nodeWidth, nodeHeight, {
    label: 'Out',
    type: 'ssm',
    detail: detail.output,
    selected: selectedHead?.kind === 'ssm-out',
    onClick: detail.output ? () => onSelectHead?.({ kind: 'ssm-out' }) : null,
  });

  drawCurvedConnector(group, [outX + nodeHalfWidth, rowY], [mergeX, mainY], {
    stroke: color,
    strokeWidth: 2.2,
    arrow: false,
  });
  drawResidualMerge(group, mergeX, mainY);
}

function drawLinearRow(group, row, panelX, mainY, panelWidth, rowIndex, selectedHead = null, onSelectHead = null, layout = {}) {
  const nodeWidth = layout.nodeWidth ?? 76;
  const nodeHeight = layout.nodeHeight ?? 32;
  const nodeHalfWidth = nodeWidth / 2;
  const mergeGap = layout.mergeGap ?? 24;
  const isCompactPanel = panelWidth <= 420;
  const leadInOffset = isCompactPanel ? 0 : 32;
  const mergeX = layout.mergeX ?? (isCompactPanel
    ? panelX + panelWidth - 74
    : (rowIndex === 0 ? panelX + panelWidth * 0.62 : panelX + panelWidth * 0.84));
  const rowY = layout.rowY ?? (rowIndex === 0 ? mainY - 58 : mainY + 58);
  const startX = layout.startX ?? (isCompactPanel ? panelX + 108 : panelX + 124 + leadInOffset);
  const endX = mergeX - (nodeHalfWidth + mergeGap);
  const span = row.nodes.length > 1 ? (endX - startX) / (row.nodes.length - 1) : 0;
  const firstNodeCenterX = startX;
  const firstNodeLeftX = firstNodeCenterX - nodeHalfWidth;
  const lastNodeCenterX = startX + span * Math.max(0, row.nodes.length - 1);
  const nodeCenters = row.nodes.map((_, index) => startX + span * index);
  const label = layout.label || row.label;
  const labelX = layout.labelX ?? (panelX + 20);
  const labelAnchor = layout.labelAnchor || 'start';
  const entryX = layout.entryX ?? (panelX + 44);

  group.appendChild(svgElement('text', { x: labelX, y: rowY - 22, 'text-anchor': labelAnchor, 'font-size': 10.5, fill: '#9aa4b8' }, label));
  group.appendChild(svgElement('path', {
    d: `M ${entryX} ${mainY} C ${entryX + 18} ${mainY} ${firstNodeLeftX - 20} ${rowY} ${firstNodeLeftX} ${rowY}`,
    fill: 'none', stroke: COLORS[row.nodes[row.nodes.length - 1].type] || COLORS.residual, 'stroke-width': 2.2,
  }));

  (layout.groups || []).forEach((groupSpec) => {
    const startIndex = groupSpec.startIndex ?? row.nodes.findIndex((node) => node.kind === groupSpec.startKind);
    const endIndex = groupSpec.endIndex ?? row.nodes.findIndex((node) => node.kind === groupSpec.endKind);
    if (startIndex < 0 || endIndex < 0 || endIndex < startIndex) return;
    const left = nodeCenters[startIndex] - nodeHalfWidth - (groupSpec.padX ?? 14);
    const right = nodeCenters[endIndex] + nodeHalfWidth + (groupSpec.padX ?? 14);
    const top = rowY - nodeHeight / 2 - (groupSpec.padTop ?? 22);
    const height = nodeHeight + (groupSpec.padTop ?? 22) + (groupSpec.padBottom ?? 12);
    const selected = !!(groupSpec.kind && selectedHead?.kind === groupSpec.kind);
    const stroke = selected ? COLORS.selected : (COLORS[groupSpec.type] || COLORS.residual);
    const groupNode = svgElement('g');
    const headerWidth = groupSpec.headerWidth ?? 84;
    const headerHeight = groupSpec.headerHeight ?? 24;
    const headerX = left + ((right - left) - headerWidth) / 2;
    const headerY = top + 10;
    groupNode.appendChild(svgElement('rect', {
      x: left,
      y: top,
      width: right - left,
      height,
      rx: 14,
      fill: selected ? '#2a3645' : (groupSpec.fill || '#2a312f'),
      opacity: groupSpec.opacity ?? 0.96,
      stroke,
      'stroke-width': selected ? 2.1 : 1.4,
      'stroke-dasharray': groupSpec.strokeDasharray || '',
    }));
    if (groupSpec.label) {
      groupNode.appendChild(svgElement('rect', {
        x: headerX,
        y: headerY,
        width: headerWidth,
        height: headerHeight,
        rx: 10,
        fill: selected ? '#31455d' : '#313c30',
        stroke,
        'stroke-width': selected ? 1.9 : 1.3,
      }));
      groupNode.appendChild(svgElement('text', {
        x: headerX + headerWidth / 2,
        y: headerY + 15,
        'text-anchor': 'middle',
        'font-size': 10.5,
        'font-weight': 700,
        fill: selected ? '#d9edff' : '#eef5e8',
      }, groupSpec.label));
    }
    if (groupSpec.detail) groupNode.appendChild(svgElement('title', {}, nodeTooltip({ label: groupSpec.label, detail: groupSpec.detail })));
    if (groupSpec.kind && groupSpec.detail && onSelectHead) {
      groupNode.style.cursor = 'pointer';
      groupNode.addEventListener('click', () => onSelectHead({ kind: groupSpec.kind }));
    }
    group.appendChild(groupNode);
  });

  row.nodes.forEach((node, index) => {
    const nodeCenterX = nodeCenters[index];
    const nodeGroup = svgElement('g');
    const isSelected = !!(node.kind && selectedHead?.kind === node.kind);
    const stroke = isSelected ? COLORS.selected : (COLORS[node.type] || '#6f7a8e');
    const clickHandler = node.onClick || (node.kind && node.detail ? () => onSelectHead?.({ kind: node.kind }) : null);
    const rect = svgElement('rect', {
      x: nodeCenterX - nodeHalfWidth, y: rowY - nodeHeight / 2, width: nodeWidth, height: nodeHeight, rx: 12,
      fill: isSelected ? '#253446' : '#202530', stroke, 'stroke-width': isSelected ? 2.2 : 1.5,
    });
    const text = svgElement('text', {
      x: nodeCenterX, y: rowY + 4, 'text-anchor': 'middle', 'font-size': 10.5, 'font-weight': 600, fill: '#fff',
    }, node.label);
    nodeGroup.append(rect, text);
    nodeGroup.appendChild(svgElement('title', {}, nodeTooltip(node)));
    if (clickHandler) {
      nodeGroup.style.cursor = 'pointer';
      nodeGroup.addEventListener('click', clickHandler);
    }
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

function drawMoERow(group, row, panelX, mainY, panelWidth, rowIndex, selectedHead = null, onSelectHead = null, layout = {}) {
  const moeDetail = layout.moeDetail || {};
  const metrics = getMoEPathMetrics(moeDetail);
  const {
    expertCount,
    laneGap,
    leadNodeWidth,
    leadNodeHeight,
    laneNodeWidth,
    laneNodeHeight,
    laneTextSize,
    firstLaneOffset,
  } = metrics;
  const rowY = layout.rowY ?? (rowIndex === 0 ? mainY - 58 : mainY + 58);
  const mergeX = layout.mergeX ?? (panelX + panelWidth - 76);
  const entryX = layout.entryX ?? (panelX + 44);
  const label = layout.label || row.label;
  const labelX = layout.labelX ?? (panelX + 20);
  const labelAnchor = layout.labelAnchor || 'start';
  const labelY = layout.labelY ?? (rowY - 38);
  const color = COLORS.moe;
  const laneStep = laneNodeWidth + (layout.componentGap ?? 20);

  const normNode = row.nodes.find((node) => node.type === 'norm') || { label: 'FFN Norm', type: 'norm', detail: moeDetail.norm || null };
  const routerNode = row.nodes.find((node) => node.kind === 'moe-router') || { label: 'Router', type: 'moe', kind: 'moe-router', detail: moeDetail.router || null };
  const upNode = row.nodes.find((node) => node.kind === 'moe-up') || { label: 'Up', type: 'moe', kind: 'moe-up', detail: moeDetail.expertUp || null };
  const gateNode = row.nodes.find((node) => node.kind === 'moe-gate') || (moeDetail.expertGate ? { label: 'Gate', type: 'moe', kind: 'moe-gate', detail: moeDetail.expertGate } : null);
  const downNode = row.nodes.find((node) => node.kind === 'moe-down') || { label: 'Down', type: 'moe', kind: 'moe-down', detail: moeDetail.expertDown || null };

  const normX = layout.normX ?? (entryX + 92);
  const routerX = layout.routerX ?? (normX + 112);
  const fanoutX = layout.fanoutX ?? (routerX + 110);
  const bankLeft = layout.bankLeft ?? (fanoutX + 14);
  const laneLabelX = layout.laneLabelX ?? (bankLeft + 48);
  const upX = layout.upX ?? (laneLabelX + 52);
  const gateX = gateNode ? (layout.gateX ?? (upX + laneStep)) : null;
  const downX = layout.downX ?? ((gateNode ? gateX : upX) + laneStep);
  const lastNodeX = downX;
  const combineX = layout.combineX ?? (mergeX - 52);
  const collectorRadius = 8.5;
  const collectorX = layout.collectorX ?? (combineX + 18);

  const laneClusterCenterY = layout.laneClusterCenterY
    ?? (rowY + firstLaneOffset + ((expertCount - 1) * laneGap) / 2);
  const firstLaneY = laneClusterCenterY - ((expertCount - 1) * laneGap) / 2;
  const lastLaneY = laneClusterCenterY + ((expertCount - 1) * laneGap) / 2;
  const collectorY = laneClusterCenterY;
  const expertsSelected = selectedHead?.kind === 'moe-experts';

  group.appendChild(svgElement('text', {
    x: labelX,
    y: labelY,
    'text-anchor': labelAnchor,
    'font-size': 10.5,
    fill: '#9aa4b8',
  }, label));

  group.appendChild(svgElement('path', {
    d: `M ${entryX} ${mainY} C ${entryX + 18} ${mainY} ${normX - leadNodeWidth / 2 - 20} ${rowY} ${normX - leadNodeWidth / 2} ${rowY}`,
    fill: 'none',
    stroke: color,
    'stroke-width': 2.2,
  }));

  drawNodeBox(group, normX, rowY, leadNodeWidth, leadNodeHeight, {
    label: normNode.label,
    type: normNode.type,
    detail: normNode.detail,
  });

  group.appendChild(svgElement('line', {
    x1: normX + leadNodeWidth / 2,
    y1: rowY,
    x2: routerX - leadNodeWidth / 2,
    y2: rowY,
    stroke: color,
    'stroke-width': 2,
  }));

  drawNodeBox(group, routerX, rowY, leadNodeWidth, leadNodeHeight, {
    label: routerNode.label,
    type: routerNode.type,
    detail: routerNode.detail,
    selected: selectedHead?.kind === routerNode.kind,
    onClick: routerNode.detail ? () => onSelectHead?.({ kind: routerNode.kind }) : null,
  });

  for (let expertIndex = 0; expertIndex < expertCount; expertIndex += 1) {
    const laneY = firstLaneY + expertIndex * laneGap;
    const laneEntryX = upX - laneNodeWidth / 2;
    drawCurvedConnector(group, [routerX + leadNodeWidth / 2, rowY], [laneEntryX, laneY], {
      stroke: expertsSelected ? COLORS.selected : color,
      strokeWidth: expertsSelected ? 1.75 : 1.45,
      arrow: false,
      opacity: 0.88,
    });
  }

  if (expertCount > 1) {
    group.appendChild(svgElement('line', {
      x1: combineX,
      y1: firstLaneY,
      x2: combineX,
      y2: lastLaneY,
      stroke: color,
      'stroke-width': 1.6,
      opacity: 0.86,
    }));
  }

  for (let expertIndex = 0; expertIndex < expertCount; expertIndex += 1) {
    const laneY = firstLaneY + expertIndex * laneGap;
    const laneEntryX = upX - laneNodeWidth / 2;
    const branchStroke = expertsSelected ? COLORS.selected : color;

    group.appendChild(svgElement('text', {
      x: laneLabelX,
      y: laneY + 3,
      'text-anchor': 'middle',
      'font-size': 8.5,
      'font-weight': 600,
      fill: '#c5cfde',
    }, `E${expertIndex}`));

    drawNodeBox(group, upX, laneY, laneNodeWidth, laneNodeHeight, {
      label: upNode.label,
      type: upNode.type,
      detail: upNode.detail,
      selected: selectedHead?.kind === upNode.kind,
      onClick: upNode.detail ? () => onSelectHead?.({ kind: upNode.kind }) : null,
      showShape: false,
      textSize: laneTextSize,
    });

    const laneBoxes = [{ node: upNode, x: upX }];
    if (gateNode) laneBoxes.push({ node: gateNode, x: gateX });
    laneBoxes.push({ node: downNode, x: downX });

    for (let index = 1; index < laneBoxes.length; index += 1) {
      const prev = laneBoxes[index - 1];
      const next = laneBoxes[index];
      group.appendChild(svgElement('line', {
        x1: prev.x + laneNodeWidth / 2,
        y1: laneY,
        x2: next.x - laneNodeWidth / 2,
        y2: laneY,
        stroke: color,
        'stroke-width': 1.6,
      }));
      drawNodeBox(group, next.x, laneY, laneNodeWidth, laneNodeHeight, {
        label: next.node.label,
        type: next.node.type,
        detail: next.node.detail,
        selected: selectedHead?.kind === next.node.kind,
        onClick: next.node.detail ? () => onSelectHead?.({ kind: next.node.kind }) : null,
        showShape: false,
        textSize: laneTextSize,
      });
    }

    group.appendChild(svgElement('line', {
      x1: lastNodeX + laneNodeWidth / 2,
      y1: laneY,
      x2: combineX,
      y2: laneY,
      stroke: color,
      'stroke-width': 1.6,
      opacity: 0.9,
    }));
  }

  group.appendChild(svgElement('line', {
    x1: combineX,
    y1: collectorY,
    x2: collectorX - collectorRadius,
    y2: collectorY,
    stroke: color,
    'stroke-width': 1.8,
    opacity: 0.92,
  }));
  group.appendChild(svgElement('circle', {
    cx: collectorX,
    cy: collectorY,
    r: collectorRadius,
    fill: expertsSelected ? '#314055' : '#3a3320',
    stroke: expertsSelected ? COLORS.selected : '#d7c55a',
    'stroke-width': expertsSelected ? 1.8 : 1.35,
  }));
  group.appendChild(svgElement('text', {
    x: collectorX,
    y: collectorY + 3.5,
    'text-anchor': 'middle',
    'font-size': 10,
    'font-weight': 700,
    fill: expertsSelected ? '#e8f3ff' : '#fff6bf',
  }, 'Σ'));

  drawCurvedConnector(group, [collectorX + collectorRadius, collectorY], [mergeX, mainY], {
    stroke: color,
    strokeWidth: 2.2,
    opacity: 0.95,
    arrow: false,
  });
  drawResidualMerge(group, mergeX, mainY);
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
  const { hasAttnDetail, hasTripleDetail, headDiagramTopGap, diagramYOffset, panelWidth, panelHeight } = getDetailPanelMetrics(stage);
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

  const attentionRow = stage.detailRows.find((row) => row.layout === 'attention-qkv' && row.attentionDetail);
  const ssmRow = stage.detailRows.find((row) => row.layout === 'ssm' && row.ssmDetail);
  const mlpRow = stage.detailRows.find((row) => row.layout === 'mlp' && row.mlpDetail);
  const moeRow = stage.detailRows.find((row) => isMoERow(row));
  const renderParallelAttnSSMMLP = !!(attentionRow && ssmRow && mlpRow);
  const renderSequentialAttnMoE = !!(!renderParallelAttnSSMMLP && attentionRow && moeRow);
  const renderSequentialAttnMLP = !!(!renderParallelAttnSSMMLP && !renderSequentialAttnMoE && attentionRow && mlpRow);
  const renderSequentialSSMMLP = !!(!renderParallelAttnSSMMLP && !renderSequentialAttnMoE && !renderSequentialAttnMLP && ssmRow && mlpRow);

  if (renderParallelAttnSSMMLP) {
    const firstMergeX = panelX + 700;
    const secondMergeX = panelX + panelWidth - 82;
    const mixerRowY = mainY - 72;
    const ssmRowY = mainY + 78;
    drawAttentionPath(group, attentionRow.attentionDetail, panelX, mainY, panelWidth, {
      label: 'Attention sublayer',
      rowCenterY: mixerRowY,
      mergeX: firstMergeX,
    });
    drawSSMRow(group, ssmRow, panelX, mainY, panelWidth, 1, selectedHead, onSelectHead, {
      label: 'SSM sublayer',
      rowY: ssmRowY,
      labelY: mainY + 18,
      mergeX: firstMergeX,
      normX: panelX + 150,
      inX: panelX + 252,
      convX: panelX + 354,
      selectiveX: panelX + 474,
      paramX: panelX + 584,
      outX: panelX + 650,
    });
    drawMLPRow(group, mlpRow, panelX, mainY, panelWidth, 1, selectedHead, onSelectHead, {
      label: 'MLP sublayer',
      labelX: panelX + panelWidth - 22,
      labelAnchor: 'end',
      entryX: firstMergeX + 28,
      mergeX: secondMergeX,
      rowY: mixerRowY,
      normX: firstMergeX + 160,
      branchX: firstMergeX + 296,
      downX: firstMergeX + 432,
    });
  } else if (renderSequentialAttnMoE) {
    const firstMergeX = panelX + 430;
    const secondMergeX = panelX + panelWidth - 76;
    drawAttentionPath(group, attentionRow.attentionDetail, panelX, mainY, panelWidth, {
      label: 'Attention sublayer',
      rowCenterY: mainY - 48,
      mergeX: firstMergeX,
    });
    drawMoERow(group, moeRow, panelX, mainY, panelWidth, 1, selectedHead, onSelectHead, {
      label: 'MoE sublayer',
      entryX: firstMergeX + 28,
      mergeX: secondMergeX,
      rowY: mainY + MOE_SUBDIAGRAM_ROW_OFFSET,
      laneClusterCenterY: mainY + MOE_SUBDIAGRAM_ROW_OFFSET,
      moeDetail: stage.moeDetail,
    });
  } else if (renderSequentialAttnMLP) {
    const firstMergeX = panelX + 520;
    const secondMergeX = panelX + panelWidth - 82;
    drawAttentionPath(group, attentionRow.attentionDetail, panelX, mainY, panelWidth, {
      label: 'Attention sublayer',
      rowCenterY: mainY - 48,
      mergeX: firstMergeX,
    });
    drawMLPRow(group, mlpRow, panelX, mainY, panelWidth, 1, selectedHead, onSelectHead, {
      label: 'MLP sublayer',
      labelX: panelX + panelWidth - 22,
      labelAnchor: 'end',
      entryX: firstMergeX + 28,
      mergeX: secondMergeX,
      rowY: mainY - 48,
      normX: firstMergeX + 158,
      branchX: firstMergeX + 284,
      downX: firstMergeX + 418,
    });
  } else if (renderSequentialSSMMLP) {
    const firstMergeX = panelX + 640;
    const secondMergeX = panelX + panelWidth - 82;
    const mlpRowY = mainY - 72;
    const ssmRowY = mainY + 72;
    drawSSMRow(group, ssmRow, panelX, mainY, panelWidth, 0, selectedHead, onSelectHead, {
      label: 'SSM sublayer',
      rowY: ssmRowY,
      mergeX: firstMergeX,
      normX: panelX + 142,
      inX: panelX + 260,
      convX: panelX + 378,
      selectiveX: panelX + 518,
      paramX: panelX + 652,
      outX: panelX + 768,
    });
    drawMLPRow(group, mlpRow, panelX, mainY, panelWidth, 1, selectedHead, onSelectHead, {
      label: 'MLP sublayer',
      entryX: firstMergeX + 28,
      mergeX: secondMergeX,
      rowY: mlpRowY,
      normX: firstMergeX + 140,
      branchX: firstMergeX + 268,
      downX: firstMergeX + 404,
    });
  } else {
    stage.detailRows.forEach((row, rowIndex) => {
      if (row.layout === 'attention-qkv' && row.attentionDetail) {
        drawAttentionPath(group, row.attentionDetail, panelX, mainY, panelWidth);
      } else if (row.layout === 'mlp' && row.mlpDetail) {
        drawMLPRow(group, row, panelX, mainY, panelWidth, rowIndex, selectedHead, onSelectHead);
      } else if (row.layout === 'ssm' && row.ssmDetail) {
        drawSSMRow(group, row, panelX, mainY, panelWidth, rowIndex, selectedHead, onSelectHead);
      } else if (isMoERow(row)) {
        drawMoERow(group, row, panelX, mainY, panelWidth, rowIndex, selectedHead, onSelectHead, {
          moeDetail: stage.moeDetail,
        });
      } else {
        drawLinearRow(group, row, panelX, mainY, panelWidth, rowIndex, selectedHead, onSelectHead);
      }
    });
  }

  if (hasAttnDetail && stage.headInfo?.headCount) {
    const wiringTopY = mainY + (headDiagramTopGap - 38);
    drawSVGHeadWiring(group, stage, panelX, wiringTopY, panelWidth, selectedHead, onSelectHead);
  }

  camera.appendChild(group);
}

export function renderResidualFlow(container, model, uiState = {}) {
  const graph = buildResidualFlowGraph(model);
  const selectedStage = graph.stages.find(stage => stage.type === 'block' && stage.index === uiState.selectedLayerIndex) || null;
  const selectedHead = selectedStage ? uiState.selectedHead || null : null;
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
  const stageLeftPadding = 72;
  const stageGap = 48;
  const contentRightPadding = 48;
  let cursor = stageLeftPadding;
  for (const stage of graph.stages) {
    stage.x = cursor;
    stage.y = baseY - stage.height / 2;
    cursor += stage.width + stageGap;
  }
  const detailPanelWidth = selectedStage ? getDetailPanelMetrics(selectedStage).panelWidth : 0;
  const contentWidth = Math.max(cursor + contentRightPadding, detailPanelWidth + 72);
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