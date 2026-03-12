function sumElements(tensors) {
  return tensors.reduce((sum, tensor) => sum + (tensor.numElements || 0), 0);
}

function sumMemory(tensors) {
  return tensors.reduce((sum, tensor) => sum + (tensor.memoryBytes || 0), 0);
}

function getBlockProfile(layer) {
  const categories = new Set(layer.tensors.map(tensor => tensor.category));
  const hasAttention = categories.has('attention');
  const hasMLP = categories.has('mlp');
  const hasMoE = categories.has('moe');
  const hasSSM = categories.has('ssm');
  const hasNorm = categories.has('norm');

  // Extract attention component details
  let attentionDetail = null;
  if (hasAttention) {
    const attnTensors = layer.tensors.filter(t => t.category === 'attention');
    const getComp = (comp) => attnTensors.find(t => t.component === comp);
    const makeDetail = (t) => t ? {
      name: t.name || '',
      component: t.component || '',
      type: t.type,
      offset: t.offset,
      absoluteOffset: t.absoluteOffset,
      byteLength: t.byteLength || 0,
      numElements: t.numElements || 0,
      params: t.numElements, memoryBytes: t.memoryBytes || 0,
      shape: t.dimensions || [], dimensions: t.dimensions || [], typeName: t.typeName || '', label: t.label || '',
    } : null;
    attentionDetail = {
      fused: !!getComp('attn_qkv'),
      q: makeDetail(getComp('attn_q')),
      k: makeDetail(getComp('attn_k')),
      v: makeDetail(getComp('attn_v')),
      qkv: makeDetail(getComp('attn_qkv')),
      output: makeDetail(getComp('attn_output')),
    };
  }

  const detailRows = [];
  if (hasAttention || hasSSM) {
    if (hasSSM) {
      detailRows.push({
        label: 'State path',
        layout: 'linear',
        nodes: [
          { label: hasNorm ? 'Attn Norm' : 'Norm', type: 'norm' },
          { label: 'SSM', type: 'ssm' },
        ],
      });
    } else {
      detailRows.push({
        label: 'Attention path',
        layout: 'attention-qkv',
        attentionDetail,
      });
    }
  }
  if (hasMoE) {
    detailRows.push({
      label: 'Expert path',
      layout: 'linear',
      nodes: [
        { label: 'FFN Norm', type: 'norm' },
        { label: 'Router', type: 'moe' },
        { label: 'Experts', type: 'moe' },
      ],
    });
  } else if (hasMLP) {
    detailRows.push({
      label: 'MLP path',
      layout: 'linear',
      nodes: [
        { label: 'FFN Norm', type: 'norm' },
        { label: 'MLP', type: 'mlp' },
      ],
    });
  }

  let pattern = 'Residual block';
  if (hasAttention && hasMoE) pattern = 'Attention • MoE';
  else if (hasAttention && hasMLP) pattern = 'Attention • MLP';
  else if (hasSSM) pattern = 'SSM block';
  else if (hasMoE) pattern = 'MoE block';

  const badges = [
    hasAttention && 'Attention',
    hasMLP && 'MLP',
    hasMoE && 'MoE',
    hasSSM && 'SSM',
  ].filter(Boolean);

  return { hasAttention, hasMLP, hasMoE, hasSSM, pattern, badges, detailRows, attentionDetail };
}

function makeStage(stage) {
  const widths = { input: 90, embedding: 120, block: 128, norm: 112, output: 118 };
  const heights = { input: 52, embedding: 56, block: 64, norm: 52, output: 56 };
  return {
    width: widths[stage.type] || 112,
    height: heights[stage.type] || 56,
    ...stage,
  };
}

export function buildResidualFlowGraph(model) {
  const stages = [makeStage({ id: 'input', label: 'Input', type: 'input', summary: model.contextLength ? `ctx ${model.contextLength}` : 'Tokens', params: 0, memoryBytes: 0 })];
  const embedding = model.layers.find(layer => layer.type === 'embedding');
  const blocks = model.layers.filter(layer => layer.type === 'block');
  const output = model.layers.find(layer => layer.type === 'output');

  if (embedding) {
    stages.push(makeStage({
      id: 'embedding',
      label: embedding.label,
      type: 'embedding',
      summary: model.embeddingLength ? `d ${model.embeddingLength}` : 'Lookup',
      params: embedding.params,
      memoryBytes: embedding.memoryBytes,
      layer: embedding,
    }));
  }

  const headInfo = {
    headCount: model.headCount || 0,
    headCountKV: model.headCountKV || model.headCount || 0,
    headDim: model.headDim || 0,
    gqaRatio: model.gqaRatio || 1,
    embeddingLength: model.embeddingLength || 0,
  };

  for (const block of blocks) {
    const profile = getBlockProfile(block);
    stages.push(makeStage({
      id: `block-${block.index}`,
      label: block.label,
      type: 'block',
      index: block.index,
      summary: profile.pattern,
      params: block.params,
      memoryBytes: block.memoryBytes,
      breakdown: block.subgroups || [],
      ...profile,
      headInfo: profile.hasAttention ? headInfo : null,
      layer: block,
    }));
  }

  if (output) {
    const normTensors = output.tensors.filter(tensor => tensor.category === 'norm');
    const outputTensors = output.tensors.filter(tensor => tensor.category === 'output');
    if (normTensors.length) {
      stages.push(makeStage({
        id: 'output-norm',
        label: 'Output Norm',
        type: 'norm',
        summary: 'Normalize',
        params: sumElements(normTensors),
        memoryBytes: sumMemory(normTensors),
      }));
    }
    if (outputTensors.length) {
      stages.push(makeStage({
        id: 'output-head',
        label: 'LM Head',
        type: 'output',
        summary: model.vocabSize ? `${model.vocabSize} vocab` : 'Logits',
        params: sumElements(outputTensors),
        memoryBytes: sumMemory(outputTensors),
      }));
    }
    if (!normTensors.length && !outputTensors.length) {
      stages.push(makeStage({
        id: 'output',
        label: output.label,
        type: 'output',
        summary: model.vocabSize ? `${model.vocabSize} vocab` : 'Output',
        params: output.params,
        memoryBytes: output.memoryBytes,
      }));
    }
  }

  return { stages };
}