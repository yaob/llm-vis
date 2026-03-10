/**
 * Ollama Model Adapter
 * Converts Ollama /api/show response into the same model tree format
 * that analyzeModel() produces, so the visualization works identically.
 *
 * Since Ollama's API doesn't expose individual tensor info, we synthesize
 * the layer tree from architecture metadata (block_count, embedding_length, etc.)
 */

/**
 * Architecture templates: define which components exist per block
 * and how to estimate their parameter counts from model dimensions.
 */
const ARCH_TEMPLATES = {
  llama: {
    blockComponents: [
      { name: 'attn_norm',   category: 'norm',      label: 'Attention Norm',   calc: (e) => e },
      { name: 'attn_q',      category: 'attention',  label: 'Attention Q',      calc: (e) => e * e },
      { name: 'attn_k',      category: 'attention',  label: 'Attention K',      calc: (e, _, hkv, h) => e * (e * (hkv || h) / h) },
      { name: 'attn_v',      category: 'attention',  label: 'Attention V',      calc: (e, _, hkv, h) => e * (e * (hkv || h) / h) },
      { name: 'attn_output', category: 'attention',  label: 'Attention Output', calc: (e) => e * e },
      { name: 'ffn_norm',    category: 'norm',       label: 'FFN Norm',         calc: (e) => e },
      { name: 'ffn_gate',    category: 'mlp',        label: 'FFN Gate',         calc: (e, ff) => e * ff },
      { name: 'ffn_up',      category: 'mlp',        label: 'FFN Up',           calc: (e, ff) => e * ff },
      { name: 'ffn_down',    category: 'mlp',        label: 'FFN Down',         calc: (e, ff) => ff * e },
    ],
    embedComponents: [
      { name: 'token_embd', category: 'embedding', label: 'Token Embedding', calc: (e, _, __, ___, v) => v * e },
    ],
    outputComponents: [
      { name: 'output_norm', category: 'norm',   label: 'Output Norm', calc: (e) => e },
      { name: 'output',      category: 'output', label: 'Output Head', calc: (e, _, __, ___, v) => v * e },
    ],
  },
};

// Aliases: many architectures share the llama template
['gemma', 'gemma2', 'phi3', 'qwen2', 'mistral', 'starcoder2', 'command-r', 'internlm2', 'deepseek2'].forEach(a => {
  ARCH_TEMPLATES[a] = ARCH_TEMPLATES.llama;
});

function getTemplate(arch) {
  return ARCH_TEMPLATES[arch] || ARCH_TEMPLATES.llama;
}

/**
 * Convert Ollama /api/show response to our model tree.
 * @param {string} modelName - display name
 * @param {object} showResponse - from OllamaClient.showModel()
 */
export function adaptOllamaModel(modelName, showResponse) {
  const info = showResponse.modelInfo;
  const details = showResponse.details;

  const arch = info['general.architecture'] || details.family || 'unknown';
  const blockCount = info[`${arch}.block_count`] || 0;
  const embeddingLength = info[`${arch}.embedding_length`] || 0;
  const ffLength = info[`${arch}.feed_forward_length`] || Math.round(embeddingLength * 2.68);
  const headCount = info[`${arch}.attention.head_count`] || 0;
  const headCountKV = info[`${arch}.attention.head_count_kv`] || headCount;
  const contextLength = info[`${arch}.context_length`] || 0;
  const vocabSize = info[`${arch}.vocab_size`] || 0;
  const headDim = headCount > 0 ? Math.round(embeddingLength / headCount) : 0;
  const gqaRatio = headCountKV > 0 ? Math.round(headCount / headCountKV) : 1;

  const template = getTemplate(arch);
  const layers = [];
  let totalParams = 0;

  // Estimate bits-per-weight from quantization level string for memory calc
  const QUANT_BPW = {
    'Q2_K': 3.35, 'Q3_K_S': 3.5, 'Q3_K_M': 3.9, 'Q3_K_L': 4.3,
    'Q4_0': 4.5, 'Q4_1': 5.0, 'Q4_K_S': 4.5, 'Q4_K_M': 4.8,
    'Q5_0': 5.5, 'Q5_1': 6.0, 'Q5_K_S': 5.5, 'Q5_K_M': 5.7,
    'Q6_K': 6.6, 'Q8_0': 8.5, 'F16': 16, 'BF16': 16, 'F32': 32,
  };
  const quantLevel = details.quantization_level || '';
  const bpw = QUANT_BPW[quantLevel] || 4.5; // default to ~Q4

  // Build synthetic tensors for each component
  function makeTensors(components) {
    return components.map(c => {
      const params = c.calc(embeddingLength, ffLength, headCountKV, headCount, vocabSize);
      const memoryBytes = Math.ceil(params * bpw / 8);
      return {
        name: c.name + '.weight',
        dimensions: [],  // unknown exact dims from API
        type: -1,
        typeName: quantLevel || 'unknown',
        numElements: params,
        memoryBytes,
        component: c.name,
        label: c.label,
        category: c.category,
      };
    });
  }

  const sumMem = (list) => list.reduce((s, t) => s + (t.memoryBytes || 0), 0);

  // Embedding
  if (template.embedComponents) {
    const tensors = makeTensors(template.embedComponents);
    const params = tensors.reduce((s, t) => s + t.numElements, 0);
    layers.push({ type: 'embedding', label: 'Embedding', tensors, params, memoryBytes: sumMem(tensors) });
    totalParams += params;
  }

  // Transformer blocks
  for (let i = 0; i < blockCount; i++) {
    const blockTensors = makeTensors(template.blockComponents).map(t => ({
      ...t, name: `blk.${i}.${t.component}.weight`,
    }));

    const subgroups = {};
    for (const t of blockTensors) {
      const cat = t.category;
      if (!subgroups[cat]) subgroups[cat] = { label: cat.charAt(0).toUpperCase() + cat.slice(1), tensors: [], params: 0, memoryBytes: 0 };
      subgroups[cat].tensors.push(t);
      subgroups[cat].params += t.numElements;
      subgroups[cat].memoryBytes += (t.memoryBytes || 0);
    }

    const params = blockTensors.reduce((s, t) => s + t.numElements, 0);
    layers.push({
      type: 'block', label: `Block ${i}`, index: i,
      subgroups: Object.values(subgroups),
      tensors: blockTensors,
      params,
      memoryBytes: sumMem(blockTensors),
    });
    totalParams += params;
  }

  // Output
  if (template.outputComponents) {
    const tensors = makeTensors(template.outputComponents);
    const params = tensors.reduce((s, t) => s + t.numElements, 0);
    layers.push({ type: 'output', label: 'Output', tensors, params, memoryBytes: sumMem(tensors) });
    totalParams += params;
  }

  // Use official param count if available
  const officialParams = info['general.parameter_count'] || totalParams;
  const totalMemory = layers.reduce((s, l) => s + (l.memoryBytes || 0), 0);

  return {
    arch, modelName, blockCount, embeddingLength, headCount, headCountKV, headDim, gqaRatio,
    contextLength, vocabSize,
    totalParams: officialParams, totalMemory, layers, version: 'ollama',
    source: 'ollama',
    quantizationLevel: quantLevel,
    capabilities: showResponse.capabilities || [],
  };
}

