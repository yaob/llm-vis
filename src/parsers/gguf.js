/**
 * GGUF Binary Parser - reads header, metadata, and tensor info from .gguf files.
 * Only parses metadata (no weight data loaded into memory).
 * Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */

const GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian

export const GGML_TYPE_NAMES = {
  0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
  8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K',
  14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS',
  19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS',
  24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M',
  30: 'BF16', 34: 'TQ1_0', 35: 'TQ2_0',
};

// Block sizes for quantized types (elements per block)
export const GGML_BLOCK_SIZES = {
  0: 1, 1: 1, 2: 32, 3: 32, 6: 32, 7: 32, 8: 32, 9: 32,
  10: 256, 11: 256, 12: 256, 13: 256, 14: 256, 15: 256,
  24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 30: 1,
};

// Bytes per block for quantized types
export const GGML_TYPE_SIZES = {
  0: 4, 1: 2, 2: 18, 3: 20, 6: 22, 7: 24, 8: 34, 9: 36,
  10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 15: 292,
  24: 1, 25: 2, 26: 4, 27: 8, 28: 8, 30: 2,
};

/** Compute bytes for a tensor given its type and element count. */
export function computeTensorBytes(type, numElements) {
  const blockSize = GGML_BLOCK_SIZES[type];
  const typeSize = GGML_TYPE_SIZES[type];
  if (blockSize == null || typeSize == null) return 0;
  return Math.ceil(numElements / blockSize) * typeSize;
}

function alignOffset(offset, alignment) {
  if (!alignment || alignment <= 1) return offset;
  return Math.ceil(offset / alignment) * alignment;
}

class GGUFReader {
  constructor(buffer) {
    this.view = new DataView(buffer);
    this.offset = 0;
  }

  readUint8()  { const v = this.view.getUint8(this.offset); this.offset += 1; return v; }
  readInt8()   { const v = this.view.getInt8(this.offset); this.offset += 1; return v; }
  readUint16() { const v = this.view.getUint16(this.offset, true); this.offset += 2; return v; }
  readInt16()  { const v = this.view.getInt16(this.offset, true); this.offset += 2; return v; }
  readUint32() { const v = this.view.getUint32(this.offset, true); this.offset += 4; return v; }
  readInt32()  { const v = this.view.getInt32(this.offset, true); this.offset += 4; return v; }
  readFloat32(){ const v = this.view.getFloat32(this.offset, true); this.offset += 4; return v; }
  readFloat64(){ const v = this.view.getFloat64(this.offset, true); this.offset += 8; return v; }

  readUint64() {
    const lo = this.view.getUint32(this.offset, true);
    const hi = this.view.getUint32(this.offset + 4, true);
    this.offset += 8;
    return lo + hi * 0x100000000;
  }

  readInt64() {
    const lo = this.view.getUint32(this.offset, true);
    const hi = this.view.getInt32(this.offset + 4, true);
    this.offset += 8;
    return lo + hi * 0x100000000;
  }

  readString() {
    const len = this.readUint64();
    const bytes = new Uint8Array(this.view.buffer, this.offset, len);
    this.offset += len;
    return new TextDecoder().decode(bytes);
  }

  readMetadataValue(type) {
    switch (type) {
      case 0: return this.readUint8();
      case 1: return this.readInt8();
      case 2: return this.readUint16();
      case 3: return this.readInt16();
      case 4: return this.readUint32();
      case 5: return this.readInt32();
      case 6: return this.readFloat32();
      case 7: return Boolean(this.readUint8());
      case 8: return this.readString();
      case 9: {
        const arrType = this.readUint32();
        const arrLen = this.readUint64();
        const arr = [];
        for (let i = 0; i < arrLen; i++) arr.push(this.readMetadataValue(arrType));
        return arr;
      }
      case 10: return this.readUint64();
      case 11: return this.readInt64();
      case 12: return this.readFloat64();
      default: throw new Error(`Unknown metadata value type: ${type}`);
    }
  }
}

export function parseGGUF(arrayBuffer) {
  const reader = new GGUFReader(arrayBuffer);

  // Header
  const magic = reader.readUint32();
  if (magic !== GGUF_MAGIC) throw new Error('Not a valid GGUF file');

  const version = reader.readUint32();
  const tensorCount = reader.readUint64();
  const metadataKVCount = reader.readUint64();

  // Metadata
  const metadata = {};
  for (let i = 0; i < metadataKVCount; i++) {
    const key = reader.readString();
    const valueType = reader.readUint32();
    const value = reader.readMetadataValue(valueType);
    metadata[key] = value;
  }

  const alignment = Math.max(1, Number(metadata['general.alignment']) || 32);

  // Tensor infos
  const tensors = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = reader.readString();
    const nDims = reader.readUint32();
    const dimensions = [];
    for (let d = 0; d < nDims; d++) dimensions.push(reader.readUint64());
    const type = reader.readUint32();
    const offset = reader.readUint64();
    const numElements = dimensions.reduce((a, b) => a * b, 1);
    const byteLength = computeTensorBytes(type, numElements);
    tensors.push({
      name, dimensions, type,
      typeName: GGML_TYPE_NAMES[type] || `unknown(${type})`,
      offset, numElements, byteLength,
    });
  }

  const tensorDataStart = alignOffset(reader.offset, alignment);
  for (const tensor of tensors) {
    tensor.absoluteOffset = tensorDataStart + tensor.offset;
  }

  return { version, tensorCount, metadata, tensors, alignment, tensorDataStart };
}

