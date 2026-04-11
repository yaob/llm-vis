import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildResidualFlowGraph } from '../src/model/residual-flow-graph.js';

function makeTensor(category, component, dimensions = [64, 64], extra = {}) {
  const numElements = dimensions.reduce((product, value) => product * value, 1);
  return {
    name: `blk.0.${component}.weight`,
    category,
    component,
    dimensions,
    numElements,
    params: numElements,
    memoryBytes: numElements * 2,
    typeName: 'F16',
    ...extra,
  };
}

describe('buildResidualFlowGraph', () => {
  it('keeps the attention row when attention, SSM, and MoE coexist', () => {
    const model = {
      activationFunction: 'SiLU',
      headCount: 8,
      headCountKV: 8,
      headDim: 16,
      gqaRatio: 1,
      embeddingLength: 128,
      contextLength: 4096,
      layers: [
        {
          type: 'block',
          index: 0,
          label: 'Block 0',
          params: 0,
          memoryBytes: 0,
          tensors: [
            makeTensor('norm', 'attn_norm', [128]),
            makeTensor('attention', 'attn_q'),
            makeTensor('attention', 'attn_k'),
            makeTensor('attention', 'attn_v'),
            makeTensor('attention', 'attn_output'),
            makeTensor('norm', 'ffn_norm', [128]),
            makeTensor('ssm', 'ssm_in'),
            makeTensor('ssm', 'ssm_conv1d'),
            makeTensor('ssm', 'ssm_x'),
            makeTensor('ssm', 'ssm_a', [16]),
            makeTensor('ssm', 'ssm_d', [16]),
            makeTensor('ssm', 'ssm_dt', [16]),
            makeTensor('ssm', 'ssm_out'),
            makeTensor('moe', 'ffn_gate_inp', [8, 128]),
            makeTensor('moe', 'ffn_up_exp', [64, 128, 8]),
            makeTensor('moe', 'ffn_gate_exp', [64, 128, 8]),
            makeTensor('moe', 'ffn_down_exp', [128, 64, 8]),
          ],
        },
      ],
    };

    const { stages } = buildResidualFlowGraph(model);
    const blockStage = stages.find((stage) => stage.id === 'block-0');

    assert.ok(blockStage);
    assert.equal(blockStage.pattern, 'Attention • SSM • MoE');
    assert.deepEqual(
      blockStage.detailRows.map((row) => row.label),
      ['Attention path', 'State path', 'Expert path'],
    );
    assert.ok(blockStage.detailRows.some((row) => row.layout === 'attention-qkv'));
    assert.ok(blockStage.detailRows.some((row) => row.layout === 'ssm'));
    assert.ok(blockStage.detailRows.some((row) => row.layout === 'linear'));
  });
});