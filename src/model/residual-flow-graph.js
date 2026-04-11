function sumElements(tensors) {
  return tensors.reduce((sum, tensor) => sum + (tensor.numElements || 0), 0);
}

function sumMemory(tensors) {
  return tensors.reduce((sum, tensor) => sum + (tensor.memoryBytes || 0), 0);
}

function makeTensorDetail(tensor) {
  return tensor ? {
    name: tensor.name || '',
    component: tensor.component || '',
    type: tensor.type,
    offset: tensor.offset,
    absoluteOffset: tensor.absoluteOffset,
    byteLength: tensor.byteLength || 0,
    numElements: tensor.numElements || 0,
    params: tensor.numElements,
    memoryBytes: tensor.memoryBytes || 0,
    shape: tensor.dimensions || [],
    dimensions: tensor.dimensions || [],
    typeName: tensor.typeName || '',
    label: tensor.label || '',
  } : null;
}

function makeTensorGroupDetail(label, component, tensors) {
  const details = tensors.filter(Boolean);
  if (!details.length) return null;
  const typeNames = [...new Set(details.map((tensor) => tensor.typeName).filter(Boolean))];
  const expertCounts = details
    .map((tensor) => Number(tensor.shape?.[2] ?? tensor.dimensions?.[2]))
    .filter((value) => Number.isFinite(value) && value > 0);
  return {
    name: details.map((tensor) => tensor.name).filter(Boolean).join(' • '),
    component,
    type: details.length === 1 ? details[0].type : null,
    offset: null,
    absoluteOffset: null,
    byteLength: details.reduce((sum, tensor) => sum + (tensor.byteLength || 0), 0),
    numElements: details.reduce((sum, tensor) => sum + (tensor.numElements || 0), 0),
    params: details.reduce((sum, tensor) => sum + (tensor.params || tensor.numElements || 0), 0),
    memoryBytes: details.reduce((sum, tensor) => sum + (tensor.memoryBytes || 0), 0),
    shape: [],
    dimensions: [],
    typeName: typeNames.length <= 1 ? (typeNames[0] || '') : typeNames.join(' • '),
    label,
    tensors: details,
    expertCount: expertCounts.length ? expertCounts[0] : null,
  };
}

function baseActivationName(gluName) {
  if (!gluName) return 'Gate';
  const base = gluName.replace(/GLU$/, '').replace(/^Swi$/, 'SiLU').replace(/^Ge$/, 'GELU').replace(/^Re$/, 'ReLU');
  return base || gluName;
}

function getBlockProfile(layer, activationFunction) {
  const categories = new Set(layer.tensors.map(tensor => tensor.category));
  const hasAttention = categories.has('attention');
  const hasMLP = categories.has('mlp');
  const hasMoE = categories.has('moe');
  const hasSSM = categories.has('ssm');
  const hasNorm = categories.has('norm');
  const findLayerComp = (...components) => layer.tensors.find(t => components.includes(t.component));

  // Extract attention component details
  let attentionDetail = null;
  if (hasAttention) {
    const attnTensors = layer.tensors.filter(t => t.category === 'attention');
    const getComp = (comp) => attnTensors.find(t => t.component === comp);
    attentionDetail = {
      norm: makeTensorDetail(findLayerComp('attn_norm', 'attn_norm_2')),
      fused: !!getComp('attn_qkv'),
      q: makeTensorDetail(getComp('attn_q')),
      k: makeTensorDetail(getComp('attn_k')),
      v: makeTensorDetail(getComp('attn_v')),
      qkv: makeTensorDetail(getComp('attn_qkv')),
      output: makeTensorDetail(getComp('attn_output')),
    };
  }

  let mlpDetail = null;
  if (hasMLP) {
    const mlpTensors = layer.tensors.filter(t => t.category === 'mlp');
    const getComp = (comp) => mlpTensors.find(t => t.component === comp);
    const up = makeTensorDetail(getComp('ffn_up'));
    const gate = makeTensorDetail(getComp('ffn_gate'));
    const down = makeTensorDetail(getComp('ffn_down'));
    mlpDetail = {
      norm: makeTensorDetail(findLayerComp('ffn_norm')),
      up,
      gate,
      down,
      gated: !!gate,
      activationFunction: activationFunction || null,
    };
  }

  let moeDetail = null;
  if (hasMoE) {
    const moeTensors = layer.tensors.filter(t => t.category === 'moe');
    const getComp = (...components) => moeTensors.find(t => components.includes(t.component) || components.includes(t.rawComponent));
    const router = makeTensorDetail(getComp('ffn_gate_inp'));
    const expertUp = makeTensorDetail(getComp('ffn_up_exp', 'ffn_up_exps'));
    const expertGate = makeTensorDetail(getComp('ffn_gate_exp', 'ffn_gate_exps'));
    const expertDown = makeTensorDetail(getComp('ffn_down_exp', 'ffn_down_exps'));
    moeDetail = {
      norm: makeTensorDetail(findLayerComp('ffn_norm')),
      router,
      expertUp,
      expertGate,
      expertDown,
      experts: makeTensorGroupDetail('MoE Experts', 'moe_experts', [expertUp, expertGate, expertDown]),
      gated: !!expertGate,
      activationFunction: activationFunction || null,
    };
  }

  let ssmDetail = null;
  if (hasSSM) {
    const ssmTensors = layer.tensors.filter(t => t.category === 'ssm');
    const getComp = (comp) => ssmTensors.find(t => t.component === comp);
    const normTensor = findLayerComp('attn_norm', 'attn_norm_2', 'ffn_norm');
    const normLabel = normTensor?.component === 'attn_norm_2'
      ? 'Attn Norm 2'
      : normTensor?.component === 'ffn_norm'
        ? 'FFN Norm'
        : normTensor
          ? 'Attn Norm'
          : (hasNorm ? 'Norm' : 'Norm');
    ssmDetail = {
      normLabel,
      norm: makeTensorDetail(normTensor),
      input: makeTensorDetail(getComp('ssm_in')),
      conv1d: makeTensorDetail(getComp('ssm_conv1d')),
      selective: makeTensorDetail(getComp('ssm_x')),
      a: makeTensorDetail(getComp('ssm_a')),
      d: makeTensorDetail(getComp('ssm_d')),
      dt: makeTensorDetail(getComp('ssm_dt')),
      output: makeTensorDetail(getComp('ssm_out')),
    };
  }

  const detailRows = [];
  if (hasAttention) {
    detailRows.push({
      label: 'Attention path',
      layout: 'attention-qkv',
      attentionDetail,
    });
  }
  if (hasSSM) {
    detailRows.push({
      label: 'State path',
      layout: 'ssm',
      ssmDetail,
    });
  }
  if (hasMoE) {
    detailRows.push({
      label: 'Expert path',
      layout: 'linear',
      nodes: [
        { label: 'FFN Norm', type: 'norm', detail: moeDetail?.norm || null },
        { label: 'Router', type: 'moe', kind: 'moe-router', detail: moeDetail?.router || null },
        { label: 'Up', type: 'moe', kind: 'moe-up', detail: moeDetail?.expertUp || null },
        { label: baseActivationName(activationFunction), type: 'moe', kind: 'moe-gate', detail: moeDetail?.expertGate || null },
        { label: 'Down', type: 'moe', kind: 'moe-down', detail: moeDetail?.expertDown || null },
      ].filter((node) => node.type === 'norm' || !!node.detail),
    });
  } else if (hasMLP) {
    detailRows.push({
      label: 'MLP path',
      layout: 'mlp',
      mlpDetail,
    });
  }

  const hasAttentionSSMMLP = hasAttention && hasSSM && hasMLP;
  let pattern = 'Residual block';
  if (hasAttentionSSMMLP) pattern = 'Attention • SSM • MLP';
  else if (hasAttention && hasSSM && hasMoE) pattern = 'Attention • SSM • MoE';
  else if (hasAttention && hasMoE) pattern = 'Attention • MoE';
  else if (hasAttention && hasMLP) pattern = 'Attention • MLP';
  else if (hasAttention && hasSSM) pattern = 'Attention • SSM';
  else if (hasSSM && hasMoE) pattern = 'SSM • MoE';
  else if (hasSSM && hasMLP) pattern = 'SSM • MLP';
  else if (hasSSM) pattern = 'SSM block';
  else if (hasMoE) pattern = 'MoE block';

  const badges = [
    hasAttention && 'Attention',
    hasMLP && 'MLP',
    hasMoE && 'MoE',
    hasSSM && 'SSM',
  ].filter(Boolean);

  return { hasAttention, hasMLP, hasMoE, hasSSM, pattern, badges, detailRows, attentionDetail, mlpDetail, moeDetail, ssmDetail };
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

  // Ensure all head counts are proper scalar numbers (defense against array metadata)
  const safeNum = (v) => typeof v === 'number' ? v : (Array.isArray(v) ? Number(v[0]) || 0 : Number(v) || 0);
  const headInfo = {
    headCount: safeNum(model.headCount) || 0,
    headCountKV: safeNum(model.headCountKV) || safeNum(model.headCount) || 0,
    headDim: safeNum(model.headDim) || 0,
    gqaRatio: safeNum(model.gqaRatio) || 1,
    embeddingLength: safeNum(model.embeddingLength) || 0,
    contextLength: safeNum(model.contextLength) || 0,
  };

  for (const block of blocks) {
    const profile = getBlockProfile(block, model.activationFunction);
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