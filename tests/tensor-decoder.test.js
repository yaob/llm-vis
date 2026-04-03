import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { fp16ToFloat, bf16ToFloat, computeStats, decodeRows } from '../src/model/gguf-tensor-decoder.js';

// ─── fp16ToFloat ────────────────────────────────────────────────────

describe('fp16ToFloat', () => {
  it('positive zero (0x0000)', () => {
    assert.equal(fp16ToFloat(0x0000), 0);
    assert.equal(Object.is(fp16ToFloat(0x0000), 0), true); // +0
  });

  it('negative zero (0x8000)', () => {
    assert.equal(fp16ToFloat(0x8000), -0);
    assert.equal(Object.is(fp16ToFloat(0x8000), -0), true);
  });

  it('1.0 (0x3C00)', () => {
    assert.equal(fp16ToFloat(0x3C00), 1.0);
  });

  it('-1.0 (0xBC00)', () => {
    assert.equal(fp16ToFloat(0xBC00), -1.0);
  });

  it('0.5 (0x3800)', () => {
    assert.equal(fp16ToFloat(0x3800), 0.5);
  });

  it('2.0 (0x4000)', () => {
    assert.equal(fp16ToFloat(0x4000), 2.0);
  });

  it('+Infinity (0x7C00)', () => {
    assert.equal(fp16ToFloat(0x7C00), Infinity);
  });

  it('-Infinity (0xFC00)', () => {
    assert.equal(fp16ToFloat(0xFC00), -Infinity);
  });

  it('NaN (0x7C01 — exponent=31, fraction!=0)', () => {
    assert.equal(Number.isNaN(fp16ToFloat(0x7C01)), true);
  });

  it('subnormal: smallest positive subnormal (0x0001)', () => {
    // 2^-14 * (1/1024) = 2^-14 / 1024 ≈ 5.96e-8
    const val = fp16ToFloat(0x0001);
    assert.ok(val > 0, 'should be positive');
    assert.ok(Math.abs(val - 5.960464477539063e-8) < 1e-15);
  });

  it('65504 — max finite fp16 (0x7BFF)', () => {
    const val = fp16ToFloat(0x7BFF);
    assert.ok(Math.abs(val - 65504) < 0.1);
  });

  it('round-trips common small values', () => {
    // 0.1 is not exactly representable, but 0x2E66 ≈ 0.0999755859375
    const val = fp16ToFloat(0x2E66);
    assert.ok(Math.abs(val - 0.1) < 0.001);
  });
});

// ─── bf16ToFloat ────────────────────────────────────────────────────

describe('bf16ToFloat', () => {
  it('1.0 (0x3F80)', () => {
    assert.equal(bf16ToFloat(0x3F80), 1.0);
  });

  it('-1.0 (0xBF80)', () => {
    assert.equal(bf16ToFloat(0xBF80), -1.0);
  });

  it('0.0 (0x0000)', () => {
    assert.equal(bf16ToFloat(0x0000), 0.0);
  });

  it('2.0 (0x4000)', () => {
    assert.equal(bf16ToFloat(0x4000), 2.0);
  });

  it('+Infinity (0x7F80)', () => {
    assert.equal(bf16ToFloat(0x7F80), Infinity);
  });

  it('-Infinity (0xFF80)', () => {
    assert.equal(bf16ToFloat(0xFF80), -Infinity);
  });

  it('NaN (0x7FC0)', () => {
    assert.equal(Number.isNaN(bf16ToFloat(0x7FC0)), true);
  });

  it('0.5 (0x3F00)', () => {
    assert.equal(bf16ToFloat(0x3F00), 0.5);
  });

  it('approximates 3.14 (0x4048 ≈ 3.125)', () => {
    const val = bf16ToFloat(0x4049);
    assert.ok(Math.abs(val - 3.14) < 0.1);
  });
});

// ─── computeStats ───────────────────────────────────────────────────

describe('computeStats', () => {
  it('returns zeros for empty array', () => {
    const s = computeStats(new Float32Array(0));
    assert.deepStrictEqual(s, { min: 0, max: 0, mean: 0, std: 0, absMax: 0 });
  });

  it('single element', () => {
    const s = computeStats(new Float32Array([5.0]));
    assert.equal(s.min, 5.0);
    assert.equal(s.max, 5.0);
    assert.equal(s.mean, 5.0);
    assert.equal(s.std, 0);
    assert.equal(s.absMax, 5.0);
  });

  it('known values [1, 2, 3, 4, 5]', () => {
    const s = computeStats(new Float32Array([1, 2, 3, 4, 5]));
    assert.equal(s.min, 1);
    assert.equal(s.max, 5);
    assert.equal(s.mean, 3);
    assert.ok(Math.abs(s.std - Math.sqrt(2)) < 1e-6);
    assert.equal(s.absMax, 5);
  });

  it('negative values [-3, -1, 0, 2]', () => {
    const s = computeStats(new Float32Array([-3, -1, 0, 2]));
    assert.equal(s.min, -3);
    assert.equal(s.max, 2);
    assert.equal(s.mean, -0.5);
    assert.equal(s.absMax, 3);
  });

  it('all same values', () => {
    const s = computeStats(new Float32Array([7, 7, 7, 7]));
    assert.equal(s.min, 7);
    assert.equal(s.max, 7);
    assert.equal(s.mean, 7);
    assert.equal(s.std, 0);
    assert.equal(s.absMax, 7);
  });
});


// ─── decodeRows helpers ─────────────────────────────────────────────

/** Build an ArrayBuffer from a builder callback that writes via DataView. */
function buildBuffer(byteLength, fn) {
  const buf = new ArrayBuffer(byteLength);
  fn(new DataView(buf));
  return buf;
}

// fp16 encoding helper (inverse of fp16ToFloat for normal values)
function floatToFp16(val) {
  const buf = new ArrayBuffer(4);
  const f32 = new DataView(buf);
  f32.setFloat32(0, val, true);
  const bits = f32.getUint32(0, true);
  const sign = (bits >> 31) & 1;
  let exp = ((bits >> 23) & 0xff) - 127 + 15;
  let frac = (bits >> 13) & 0x3ff;
  if (exp <= 0) { exp = 0; frac = 0; }
  if (exp >= 31) { exp = 31; frac = 0; }
  return (sign << 15) | (exp << 10) | frac;
}

describe('decodeRows — F32 (type 0)', () => {
  it('decodes a single row of 4 floats', () => {
    const cols = 4;
    const buf = buildBuffer(cols * 4, (v) => {
      v.setFloat32(0, 1.0, true);
      v.setFloat32(4, -2.5, true);
      v.setFloat32(8, 0.0, true);
      v.setFloat32(12, 3.14, true);
    });
    const out = decodeRows(0, buf, 1, cols);
    assert.equal(out.length, 4);
    assert.ok(Math.abs(out[0] - 1.0) < 1e-6);
    assert.ok(Math.abs(out[1] - (-2.5)) < 1e-6);
    assert.equal(out[2], 0.0);
    assert.ok(Math.abs(out[3] - 3.14) < 1e-5);
  });

  it('decodes two rows', () => {
    const cols = 2;
    const buf = buildBuffer(cols * 4 * 2, (v) => {
      v.setFloat32(0, 10.0, true);
      v.setFloat32(4, 20.0, true);
      v.setFloat32(8, 30.0, true);
      v.setFloat32(12, 40.0, true);
    });
    const out = decodeRows(0, buf, 2, cols);
    assert.equal(out.length, 4);
    assert.ok(Math.abs(out[0] - 10.0) < 1e-6);
    assert.ok(Math.abs(out[1] - 20.0) < 1e-6);
    assert.ok(Math.abs(out[2] - 30.0) < 1e-6);
    assert.ok(Math.abs(out[3] - 40.0) < 1e-6);
  });
});

describe('decodeRows — F16 (type 1)', () => {
  it('decodes a single row of 4 fp16 values', () => {
    const cols = 4;
    const buf = buildBuffer(cols * 2, (v) => {
      v.setUint16(0, 0x3C00, true); // 1.0
      v.setUint16(2, 0xBC00, true); // -1.0
      v.setUint16(4, 0x0000, true); // 0.0
      v.setUint16(6, 0x4000, true); // 2.0
    });
    const out = decodeRows(1, buf, 1, cols);
    assert.equal(out.length, 4);
    assert.equal(out[0], 1.0);
    assert.equal(out[1], -1.0);
    assert.equal(out[2], 0.0);
    assert.equal(out[3], 2.0);
  });
});

describe('decodeRows — BF16 (type 30)', () => {
  it('decodes a single row of 4 bf16 values', () => {
    const cols = 4;
    const buf = buildBuffer(cols * 2, (v) => {
      v.setUint16(0, 0x3F80, true); // 1.0
      v.setUint16(2, 0xBF80, true); // -1.0
      v.setUint16(4, 0x0000, true); // 0.0
      v.setUint16(6, 0x4000, true); // 2.0
    });
    const out = decodeRows(30, buf, 1, cols);
    assert.equal(out.length, 4);
    assert.equal(out[0], 1.0);
    assert.equal(out[1], -1.0);
    assert.equal(out[2], 0.0);
    assert.equal(out[3], 2.0);
  });
});

describe('decodeRows — Q4_0 (type 2)', () => {
  it('decodes one block of 32 elements', () => {
    // Q4_0 block: 2 bytes fp16 scale + 16 bytes quants = 18 bytes
    const d_fp16 = floatToFp16(1.0); // scale = 1.0
    const buf = buildBuffer(18, (v) => {
      v.setUint16(0, d_fp16, true);
      // All quant bytes = 0x88 → low nibble = 8, high nibble = 8
      // dequant: (nibble - 8) * d = 0 for all
      for (let j = 0; j < 16; j++) v.setUint8(2 + j, 0x88);
    });
    const out = decodeRows(2, buf, 1, 32);
    assert.equal(out.length, 32);
    for (let i = 0; i < 32; i++) assert.equal(out[i], 0.0);
  });

  it('correctly dequantizes non-zero quants', () => {
    const d_fp16 = floatToFp16(0.5); // scale = 0.5
    const buf = buildBuffer(18, (v) => {
      v.setUint16(0, d_fp16, true);
      // First byte: low nibble=0xF(15), high nibble=0x0(0)
      // dequant low: (15-8)*0.5 = 3.5, dequant high: (0-8)*0.5 = -4.0
      v.setUint8(2, 0x0F);
      for (let j = 1; j < 16; j++) v.setUint8(2 + j, 0x88);
    });
    const out = decodeRows(2, buf, 1, 32);
    assert.ok(Math.abs(out[0] - 3.5) < 0.01);   // low nibble of byte 0
    assert.ok(Math.abs(out[16] - (-4.0)) < 0.01); // high nibble of byte 0
  });
});

describe('decodeRows — Q8_0 (type 8)', () => {
  it('decodes one block of 32 elements', () => {
    // Q8_0 block: 2 bytes fp16 scale + 32 bytes int8 quants = 34 bytes
    const d_fp16 = floatToFp16(2.0);
    const buf = buildBuffer(34, (v) => {
      v.setUint16(0, d_fp16, true);
      // quant[0] = 1 (signed int8) → 1 * 2.0 = 2.0
      v.setUint8(2, 1);
      // quant[1] = 255 → signed = -1 → -1 * 2.0 = -2.0
      v.setUint8(3, 255);
      // rest = 0
      for (let j = 2; j < 32; j++) v.setUint8(2 + j, 0);
    });
    const out = decodeRows(8, buf, 1, 32);
    assert.equal(out.length, 32);
    assert.ok(Math.abs(out[0] - 2.0) < 0.01);
    assert.ok(Math.abs(out[1] - (-2.0)) < 0.01);
    for (let i = 2; i < 32; i++) assert.equal(out[i], 0.0);
  });
});

describe('decodeRows — unsupported type', () => {
  it('throws for unknown type', () => {
    const buf = new ArrayBuffer(4);
    assert.throws(() => decodeRows(999, buf, 1, 1), /Unsupported GGML tensor type/);
  });
});
