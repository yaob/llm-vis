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

export function analyzeModel(gguf) {
  const { metadata, tensors } = gguf;

  const arch = metadata['general.architecture'] || 'unknown';
  const modelName = metadata['general.name'] || 'Unknown Model';
  const blockCount = metadata[`${arch}.block_count`] || 0;
  const embeddingLength = metadata[`${arch}.embedding_length`] || 0;
  const headCount = metadata[`${arch}.attention.head_count`] || 0;
  const headCountKV = metadata[`${arch}.attention.head_count_kv`] || headCount;
  const contextLength = metadata[`${arch}.context_length`] || 0;
  const vocabSize = metadata['tokenizer.ggml.tokens']?.length || 0;
  const headDim = headCount > 0 ? Math.round(embeddingLength / headCount) : 0;
  const gqaRatio = headCountKV > 0 ? Math.round(headCount / headCountKV) : 1;

  // Group tensors
  const baseLayers = [];  // non-block tensors (embed, output, etc.)
  const blocks = new Map(); // block index -> tensors

  for (const t of tensors) {
    const parsed = parseTensorName(t.name);
    const memoryBytes = computeTensorBytes(t.type, t.numElements);
    const entry = {
      ...t, ...parsed,
      memoryBytes,
      label: COMPONENT_LABELS[parsed.component] || parsed.component,
      category: CATEGORY_MAP[parsed.component] || 'other',
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

  return {
    arch, modelName, blockCount, embeddingLength, headCount, headCountKV, headDim, gqaRatio,
    contextLength, vocabSize,
    totalParams, totalMemory, layers, version: gguf.version,
    ggufSource: gguf.source || null,
    ggufInfo: {
      alignment: gguf.alignment || 32,
      tensorDataStart: Number.isFinite(gguf.tensorDataStart) ? gguf.tensorDataStart : null,
    },
  };
}

