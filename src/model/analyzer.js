/**
 * Model Structure Analyzer
 * Groups GGUF tensors into a logical layer tree using standardized tensor names.
 * Ref: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#standardized-tensor-names
 */

import { computeTensorBytes } from '../parsers/gguf.js';

const COMPONENT_LABELS = {
  'token_embd':   'Token Embedding',
  'pos_embd':     'Position Embedding',
  'output_norm':  'Output Norm',
  'output':       'Output Head',
  'attn_norm':    'Attention Norm',
  'attn_norm_2':  'Attention Norm 2',
  'attn_qkv':     'Attention QKV',
  'attn_q':       'Attention Q',
  'attn_k':       'Attention K',
  'attn_v':       'Attention V',
  'attn_output':  'Attention Output',
  'ffn_norm':     'FFN Norm',
  'ffn_up':       'FFN Up',
  'ffn_gate':     'FFN Gate',
  'ffn_down':     'FFN Down',
  'ffn_gate_inp': 'MoE Router',
  'ffn_gate_exp': 'MoE FFN Gate',
  'ffn_down_exp': 'MoE FFN Down',
  'ffn_up_exp':   'MoE FFN Up',
  'ssm_in':       'SSM Input',
  'ssm_conv1d':   'SSM Conv1D',
  'ssm_x':        'SSM Selective',
  'ssm_a':        'SSM State Compress',
  'ssm_d':        'SSM Skip',
  'ssm_dt':       'SSM Time Step',
  'ssm_out':      'SSM Output',
};

const CATEGORY_MAP = {
  'token_embd': 'embedding', 'pos_embd': 'embedding',
  'output_norm': 'norm', 'output': 'output',
  'attn_norm': 'norm', 'attn_norm_2': 'norm',
  'attn_qkv': 'attention', 'attn_q': 'attention', 'attn_k': 'attention',
  'attn_v': 'attention', 'attn_output': 'attention',
  'ffn_norm': 'norm', 'ffn_up': 'mlp', 'ffn_gate': 'mlp', 'ffn_down': 'mlp',
  'ffn_gate_inp': 'moe', 'ffn_gate_exp': 'moe', 'ffn_down_exp': 'moe', 'ffn_up_exp': 'moe',
  'ssm_in': 'ssm', 'ssm_conv1d': 'ssm', 'ssm_x': 'ssm', 'ssm_a': 'ssm',
  'ssm_d': 'ssm', 'ssm_dt': 'ssm', 'ssm_out': 'ssm',
};

const COMPONENT_ALIASES = {
  'ffn_up_exps': 'ffn_up_exp',
  'ffn_gate_exps': 'ffn_gate_exp',
  'ffn_down_exps': 'ffn_down_exp',
};

function normalizeComponent(component) {
  return COMPONENT_ALIASES[component] || component;
}

function parseTensorName(name) {
  // blk.N.component.weight/bias  OR  base_component.weight/bias
  const blockMatch = name.match(/^blk\.(\d+)\.(.+?)(?:\.(weight|bias))?$/);
  if (blockMatch) {
    return { block: parseInt(blockMatch[1]), component: blockMatch[2], suffix: blockMatch[3] || 'weight' };
  }
  const baseMatch = name.match(/^(.+?)(?:\.(weight|bias))?$/);
  if (baseMatch) {
    return { block: null, component: baseMatch[1], suffix: baseMatch[2] || 'weight' };
  }
  return { block: null, component: name, suffix: 'weight' };
}

/**
 * Coerce a GGUF metadata value to a scalar number.
 * Handles: plain numbers, single-element arrays, and uint8-byte arrays
 * (some GGUF files store uint32 scalars as type-9 uint8 arrays of 4 bytes).
 */
function toNum(v) {
  if (v == null) return 0;
  if (typeof v === 'number') return v;
  if (Array.isArray(v) || ArrayBuffer.isView(v)) {
    if (v.length === 1) return Number(v[0]) || 0;
    if (v.length >= 4) {
      // Try interpreting first 4 bytes as a little-endian uint32
      const le = (v[0] | (v[1] << 8) | (v[2] << 16) | ((v[3] << 24) >>> 0)) >>> 0;
      if (le > 0 && le < 1e6) return le;
      // Try big-endian
      const be = ((v[0] << 24) | (v[1] << 16) | (v[2] << 8) | v[3]) >>> 0;
      if (be > 0 && be < 1e6) return be;
    }
    // Fallback: first non-zero element, or first element
    for (let i = 0; i < v.length; i++) { if (v[i]) return Number(v[i]); }
    return Number(v[0]) || 0;
  }
  return Number(v) || 0;
}

export function analyzeModel(gguf) {
  const { metadata, tensors } = gguf;

  const arch = metadata['general.architecture'] || 'unknown';
  const modelName = metadata['general.name'] || 'Unknown Model';
  const blockCount = toNum(metadata[`${arch}.block_count`]);
  const embeddingLength = toNum(metadata[`${arch}.embedding_length`]);
  const headCount = toNum(metadata[`${arch}.attention.head_count`]);
  const headCountKV = toNum(metadata[`${arch}.attention.head_count_kv`]) || headCount;
  const contextLength = toNum(metadata[`${arch}.context_length`]);
  const vocabSize = metadata['tokenizer.ggml.tokens']?.length || 0;
  const headDim = headCount > 0 ? Math.round(embeddingLength / headCount) : 0;
  const gqaRatio = headCountKV > 0 ? Math.round(headCount / headCountKV) : 1;

  // Detect norm type
  const ARCH_NORM = {
    llama: 'RMSNorm', mistral: 'RMSNorm', qwen2: 'RMSNorm', qwen2moe: 'RMSNorm',
    gemma: 'RMSNorm', gemma2: 'RMSNorm', phi3: 'RMSNorm',
    internlm2: 'RMSNorm', yi: 'RMSNorm', deepseek: 'RMSNorm', deepseek2: 'RMSNorm',
    starcoder2: 'LayerNorm', gpt2: 'LayerNorm', falcon: 'LayerNorm',
    phi: 'LayerNorm', bloom: 'LayerNorm', mpt: 'LayerNorm',
    mamba: 'RMSNorm', jamba: 'RMSNorm', command_r: 'RMSNorm', cohere: 'RMSNorm',
    olmo: 'LayerNorm', stablelm: 'LayerNorm',
  };
  let normType = ARCH_NORM[arch] || null;
  if (!normType) {
    // Heuristic: RMSNorm weight tensors are 1D; LayerNorm also has a bias tensor
    const normTensors = tensors.filter(t => {
      const c = parseTensorName(t.name).component;
      return c === 'attn_norm' || c === 'ffn_norm' || c === 'output_norm';
    });
    const hasBias = normTensors.some(t => t.name.endsWith('.bias') || t.name.includes('_bias'));
    normType = normTensors.length ? (hasBias ? 'LayerNorm' : 'RMSNorm') : null;
  }

  // Detect activation function from architecture
  const ARCH_ACTIVATION = {
    llama: 'SiLU', mistral: 'SiLU', qwen2: 'SiLU', qwen2moe: 'SiLU',
    gemma: 'GELU', gemma2: 'GELU', phi3: 'SiLU', phi: 'GELU',
    internlm2: 'SiLU', yi: 'SiLU', deepseek: 'SiLU', deepseek2: 'SiLU',
    gpt2: 'GELU', gpt_neox: 'GELU', opt: 'ReLU', falcon: 'GELU',
    bloom: 'GELU', mpt: 'GELU', stablelm: 'SiLU', starcoder: 'GELU',
    starcoder2: 'GELU', command_r: 'SiLU', cohere: 'SiLU', olmo: 'SiLU',
    arctic: 'SiLU', dbrx: 'SiLU', jais: 'SwiGLU', mixtral: 'SiLU',
  };
  const baseActivation = ARCH_ACTIVATION[arch] || 'SiLU';
  const hasGate = tensors.some(t =>
    t.component === 'ffn_gate' || t.component === 'ffn_gate_exp' || t.component === 'ffn_gate_exps'
  );
  // If gated, derive the GLU variant name; otherwise use the base activation
  const GATED_NAME = { SiLU: 'SwiGLU', GELU: 'GeGLU', ReLU: 'ReGLU' };
  const activationFunction = hasGate ? (GATED_NAME[baseActivation] || `${baseActivation} (gated)`) : baseActivation;

  // Group tensors
  const baseLayers = [];  // non-block tensors (embed, output, etc.)
  const blocks = new Map(); // block index -> tensors

  for (const t of tensors) {
    const parsed = parseTensorName(t.name);
    const component = normalizeComponent(parsed.component);
    const memoryBytes = computeTensorBytes(t.type, t.numElements);
    const entry = {
      ...t, ...parsed,
      rawComponent: parsed.component,
      component,
      memoryBytes,
      label: COMPONENT_LABELS[component] || component,
      category: CATEGORY_MAP[component] || 'other',
    };
    if (parsed.block !== null) {
      if (!blocks.has(parsed.block)) blocks.set(parsed.block, []);
      blocks.get(parsed.block).push(entry);
    } else {
      baseLayers.push(entry);
    }
  }

  // Build layer tree
  const layers = [];

  // Helper: sum memory for a tensor list
  const sumMem = (list) => list.reduce((s, t) => s + (t.memoryBytes || 0), 0);

  // Embedding layers first
  const embedLayers = baseLayers.filter(t => t.category === 'embedding');
  if (embedLayers.length) {
    layers.push({ type: 'embedding', label: 'Embedding', tensors: embedLayers,
      params: embedLayers.reduce((s, t) => s + t.numElements, 0),
      memoryBytes: sumMem(embedLayers) });
  }

  // Transformer blocks
  const sortedBlocks = [...blocks.entries()].sort((a, b) => a[0] - b[0]);
  for (const [idx, blockTensors] of sortedBlocks) {
    const subgroups = {};
    for (const t of blockTensors) {
      const cat = t.category;
      if (!subgroups[cat]) subgroups[cat] = { label: cat.charAt(0).toUpperCase() + cat.slice(1), tensors: [], params: 0, memoryBytes: 0 };
      subgroups[cat].tensors.push(t);
      subgroups[cat].params += t.numElements;
      subgroups[cat].memoryBytes += (t.memoryBytes || 0);
    }
    layers.push({
      type: 'block', label: `Block ${idx}`, index: idx,
      subgroups: Object.values(subgroups),
      tensors: blockTensors,
      params: blockTensors.reduce((s, t) => s + t.numElements, 0),
      memoryBytes: sumMem(blockTensors),
    });
  }

  // Output layers
  const outputLayers = baseLayers.filter(t => t.category === 'norm' || t.category === 'output');
  if (outputLayers.length) {
    layers.push({ type: 'output', label: 'Output', tensors: outputLayers,
      params: outputLayers.reduce((s, t) => s + t.numElements, 0),
      memoryBytes: sumMem(outputLayers) });
  }

  const totalParams = tensors.reduce((s, t) => s + t.numElements, 0);
  const totalMemory = tensors.reduce((s, t) => s + (t.memoryBytes || 0), 0);

  // Quantization profile: count tensors and bytes per quant type
  const quantCounts = {};
  for (const t of tensors) {
    const typeName = t.typeName || `unknown(${t.type})`;
    if (!quantCounts[typeName]) quantCounts[typeName] = { count: 0, bytes: 0, params: 0 };
    quantCounts[typeName].count += 1;
    quantCounts[typeName].bytes += (computeTensorBytes(t.type, t.numElements) || 0);
    quantCounts[typeName].params += t.numElements;
  }
  // Sort by byte share descending
  const quantProfile = Object.entries(quantCounts)
    .map(([type, d]) => ({ type, ...d, pct: totalMemory > 0 ? d.bytes / totalMemory : 0 }))
    .sort((a, b) => b.bytes - a.bytes);

  const fileType = metadata['general.file_type'] ?? null;

  // FFN hidden dimension: find from ffn_up tensor shape or metadata
  let ffnHiddenDim = metadata[`${arch}.feed_forward_length`] ?? metadata[`${arch}.intermediate_size`] ?? null;
  if (ffnHiddenDim == null) {
    const upTensor = tensors.find(t => t.component === 'ffn_up' || t.component === 'ffn_up_exp');
    if (upTensor && upTensor.dimensions && upTensor.dimensions.length >= 2) {
      ffnHiddenDim = Number(upTensor.dimensions[0]); // first dim = hidden size
    }
  }
  const ffnExpansionRatio = (ffnHiddenDim && embeddingLength) ? (ffnHiddenDim / embeddingLength) : null;

  return {
    arch, modelName, blockCount, embeddingLength, headCount, headCountKV, headDim, gqaRatio,
    contextLength, vocabSize, activationFunction, normType, ffnHiddenDim, ffnExpansionRatio,
    totalParams, totalMemory, layers, version: gguf.version,
    quantProfile, fileType,
    metadata,
    ggufSource: gguf.source || null,
    ggufInfo: {
      alignment: gguf.alignment || 32,
      tensorDataStart: Number.isFinite(gguf.tensorDataStart) ? gguf.tensorDataStart : null,
    },
  };
}

