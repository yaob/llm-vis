import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { parseTensorName, normalizeComponent, toNum } from '../src/model/analyzer.js';

// ─── parseTensorName ────────────────────────────────────────────────

describe('parseTensorName', () => {
  it('parses block tensor with weight suffix', () => {
    const r = parseTensorName('blk.0.attn_q.weight');
    assert.deepStrictEqual(r, { block: 0, component: 'attn_q', suffix: 'weight' });
  });

  it('parses block tensor with bias suffix', () => {
    const r = parseTensorName('blk.12.ffn_down.bias');
    assert.deepStrictEqual(r, { block: 12, component: 'ffn_down', suffix: 'bias' });
  });

  it('parses block tensor without explicit suffix (defaults to weight)', () => {
    const r = parseTensorName('blk.3.attn_norm');
    assert.deepStrictEqual(r, { block: 3, component: 'attn_norm', suffix: 'weight' });
  });

  it('parses base tensor with weight suffix', () => {
    const r = parseTensorName('token_embd.weight');
    assert.deepStrictEqual(r, { block: null, component: 'token_embd', suffix: 'weight' });
  });

  it('parses base tensor with bias suffix', () => {
    const r = parseTensorName('output_norm.bias');
    assert.deepStrictEqual(r, { block: null, component: 'output_norm', suffix: 'bias' });
  });

  it('parses base tensor without suffix', () => {
    const r = parseTensorName('output');
    assert.deepStrictEqual(r, { block: null, component: 'output', suffix: 'weight' });
  });

  it('parses high block index', () => {
    const r = parseTensorName('blk.127.ffn_up.weight');
    assert.deepStrictEqual(r, { block: 127, component: 'ffn_up', suffix: 'weight' });
  });

  it('handles MoE expert tensor names', () => {
    const r = parseTensorName('blk.5.ffn_gate_exps.weight');
    assert.deepStrictEqual(r, { block: 5, component: 'ffn_gate_exps', suffix: 'weight' });
  });

  it('handles unknown bare name', () => {
    const r = parseTensorName('some_random_thing');
    assert.deepStrictEqual(r, { block: null, component: 'some_random_thing', suffix: 'weight' });
  });
});

// ─── normalizeComponent ─────────────────────────────────────────────

describe('normalizeComponent', () => {
  it('normalizes ffn_up_exps to ffn_up_exp', () => {
    assert.equal(normalizeComponent('ffn_up_exps'), 'ffn_up_exp');
  });

  it('normalizes ffn_gate_exps to ffn_gate_exp', () => {
    assert.equal(normalizeComponent('ffn_gate_exps'), 'ffn_gate_exp');
  });

  it('normalizes ffn_down_exps to ffn_down_exp', () => {
    assert.equal(normalizeComponent('ffn_down_exps'), 'ffn_down_exp');
  });

  it('passes through non-aliased component', () => {
    assert.equal(normalizeComponent('attn_q'), 'attn_q');
  });

  it('passes through unknown component', () => {
    assert.equal(normalizeComponent('custom_layer'), 'custom_layer');
  });
});

// ─── toNum ──────────────────────────────────────────────────────────

describe('toNum', () => {
  it('returns 0 for null', () => {
    assert.equal(toNum(null), 0);
  });

  it('returns 0 for undefined', () => {
    assert.equal(toNum(undefined), 0);
  });

  it('returns plain numbers as-is', () => {
    assert.equal(toNum(42), 42);
    assert.equal(toNum(3.14), 3.14);
    assert.equal(toNum(0), 0);
  });

  it('returns 0 for NaN-coercible strings', () => {
    assert.equal(toNum('not a number'), 0);
  });

  it('coerces numeric strings', () => {
    assert.equal(toNum('128'), 128);
  });

  it('extracts value from single-element array', () => {
    assert.equal(toNum([64]), 64);
  });

  it('extracts value from single-element typed array', () => {
    assert.equal(toNum(new Uint8Array([7])), 7);
  });

  it('interprets 4-byte LE uint32 array', () => {
    // 4096 in LE = [0x00, 0x10, 0x00, 0x00]
    assert.equal(toNum(new Uint8Array([0x00, 0x10, 0x00, 0x00])), 4096);
  });

  it('interprets 4-byte BE uint32 when LE is zero/huge', () => {
    // 256 in BE = [0x00, 0x00, 0x01, 0x00]  LE would be 0x00010000 = 65536
    // Both LE (65536) and BE (256) are < 1e6 so LE wins
    const val = toNum(new Uint8Array([0x00, 0x00, 0x01, 0x00]));
    assert.equal(val, 65536); // LE takes priority
  });

  it('4-byte array interpreted as BE uint32 when LE is out of range', () => {
    // [0, 0, 5, 3] -> LE = 50659328 (>1e6), BE = 1283 (<1e6) -> returns BE
    assert.equal(toNum([0, 0, 5, 3]), 1283);
  });

  it('returns first non-zero element for short arrays (< 4 bytes)', () => {
    assert.equal(toNum([0, 0, 5]), 5);
  });

  it('returns 0 for empty array', () => {
    assert.equal(toNum([]), 0);
  });
});

