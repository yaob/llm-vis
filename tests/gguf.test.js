import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { computeTensorBytes } from '../src/parsers/gguf.js';

// ─── computeTensorBytes ─────────────────────────────────────────────

describe('computeTensorBytes', () => {
  it('F32: 1 element = 4 bytes', () => {
    assert.equal(computeTensorBytes(0, 1), 4);
  });

  it('F32: 1024 elements = 4096 bytes', () => {
    assert.equal(computeTensorBytes(0, 1024), 4096);
  });

  it('F16: 1 element = 2 bytes', () => {
    assert.equal(computeTensorBytes(1, 1), 2);
  });

  it('F16: 512 elements = 1024 bytes', () => {
    assert.equal(computeTensorBytes(1, 512), 1024);
  });

  it('BF16 (type 30): 1 element = 2 bytes', () => {
    assert.equal(computeTensorBytes(30, 1), 2);
  });

  it('Q4_0 (type 2): 32 elements = 18 bytes (1 block)', () => {
    assert.equal(computeTensorBytes(2, 32), 18);
  });

  it('Q4_0: 64 elements = 36 bytes (2 blocks)', () => {
    assert.equal(computeTensorBytes(2, 64), 36);
  });

  it('Q4_0: non-block-aligned rounds up', () => {
    // 33 elements -> ceil(33/32) = 2 blocks -> 2 * 18 = 36
    assert.equal(computeTensorBytes(2, 33), 36);
  });

  it('Q8_0 (type 8): 32 elements = 34 bytes', () => {
    assert.equal(computeTensorBytes(8, 32), 34);
  });

  it('Q4_K (type 12): 256 elements = 144 bytes', () => {
    assert.equal(computeTensorBytes(12, 256), 144);
  });

  it('Q4_K: 512 elements = 288 bytes (2 blocks)', () => {
    assert.equal(computeTensorBytes(12, 512), 288);
  });

  it('Q6_K (type 14): 256 elements = 210 bytes', () => {
    assert.equal(computeTensorBytes(14, 256), 210);
  });

  it('Q2_K (type 10): 256 elements = 84 bytes', () => {
    assert.equal(computeTensorBytes(10, 256), 84);
  });

  it('I8 (type 24): 100 elements = 100 bytes', () => {
    assert.equal(computeTensorBytes(24, 100), 100);
  });

  it('I32 (type 26): 10 elements = 40 bytes', () => {
    assert.equal(computeTensorBytes(26, 10), 40);
  });

  it('unknown type returns 0', () => {
    assert.equal(computeTensorBytes(999, 100), 0);
  });

  it('0 elements returns 0 bytes', () => {
    assert.equal(computeTensorBytes(0, 0), 0);
  });

  it('large tensor: F32 with 1M elements', () => {
    assert.equal(computeTensorBytes(0, 1_000_000), 4_000_000);
  });
});

