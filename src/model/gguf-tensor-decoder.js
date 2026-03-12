import { computeTensorBytes } from '../parsers/gguf.js';

const QK_K = 256;

function fp16ToFloat(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return fraction ? sign * 2 ** -14 * (fraction / 1024) : sign * 0;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function bf16ToFloat(bits) {
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setUint32(0, bits << 16, true);
  return view.getFloat32(0, true);
}

function int8At(view, offset) {
  const value = view.getUint8(offset);
  return value > 127 ? value - 256 : value;
}

function getScaleMinK4(bytes, index) {
  if (index < 4) return { scale: bytes[index] & 63, min: bytes[index + 4] & 63 };
  return {
    scale: (bytes[index + 4] & 0x0f) | ((bytes[index - 4] >> 6) << 4),
    min: (bytes[index + 4] >> 4) | ((bytes[index] >> 6) << 4),
  };
}

function decodeRowF32(view, offset, cols, out, outOffset) {
  for (let i = 0; i < cols; i++) out[outOffset + i] = view.getFloat32(offset + i * 4, true);
}

function decodeRowF16(view, offset, cols, out, outOffset) {
  for (let i = 0; i < cols; i++) out[outOffset + i] = fp16ToFloat(view.getUint16(offset + i * 2, true));
}

function decodeRowBF16(view, offset, cols, out, outOffset) {
  for (let i = 0; i < cols; i++) out[outOffset + i] = bf16ToFloat(view.getUint16(offset + i * 2, true));
}

function decodeRowQ4_0(view, offset, cols, out, outOffset) {
  for (let block = 0; block < cols / 32; block++) {
    const base = offset + block * 18;
    const d = fp16ToFloat(view.getUint16(base, true));
    const qBase = base + 2;
    const outBase = outOffset + block * 32;
    for (let j = 0; j < 16; j++) {
      const q = view.getUint8(qBase + j);
      out[outBase + j] = ((q & 0x0f) - 8) * d;
      out[outBase + j + 16] = ((q >> 4) - 8) * d;
    }
  }
}

function decodeRowQ4_1(view, offset, cols, out, outOffset) {
  for (let block = 0; block < cols / 32; block++) {
    const base = offset + block * 20;
    const d = fp16ToFloat(view.getUint16(base, true));
    const m = fp16ToFloat(view.getUint16(base + 2, true));
    const qBase = base + 4;
    const outBase = outOffset + block * 32;
    for (let j = 0; j < 16; j++) {
      const q = view.getUint8(qBase + j);
      out[outBase + j] = (q & 0x0f) * d + m;
      out[outBase + j + 16] = (q >> 4) * d + m;
    }
  }
}

function decodeRowQ5_0(view, offset, cols, out, outOffset) {
  for (let block = 0; block < cols / 32; block++) {
    const base = offset + block * 22;
    const d = fp16ToFloat(view.getUint16(base, true));
    const qh = view.getUint32(base + 2, true);
    const qBase = base + 6;
    const outBase = outOffset + block * 32;
    for (let j = 0; j < 16; j++) {
      const q = view.getUint8(qBase + j);
      const xh0 = ((qh >>> (j + 0)) << 4) & 0x10;
      const xh1 = ((qh >>> (j + 12))) & 0x10;
      out[outBase + j] = (((q & 0x0f) | xh0) - 16) * d;
      out[outBase + j + 16] = (((q >> 4) | xh1) - 16) * d;
    }
  }
}

function decodeRowQ5_1(view, offset, cols, out, outOffset) {
  for (let block = 0; block < cols / 32; block++) {
    const base = offset + block * 24;
    const d = fp16ToFloat(view.getUint16(base, true));
    const m = fp16ToFloat(view.getUint16(base + 2, true));
    const qh = view.getUint32(base + 4, true);
    const qBase = base + 8;
    const outBase = outOffset + block * 32;
    for (let j = 0; j < 16; j++) {
      const q = view.getUint8(qBase + j);
      const xh0 = ((qh >>> (j + 0)) << 4) & 0x10;
      const xh1 = ((qh >>> (j + 12))) & 0x10;
      out[outBase + j] = (((q & 0x0f) | xh0) * d) + m;
      out[outBase + j + 16] = (((q >> 4) | xh1) * d) + m;
    }
  }
}

function decodeRowQ8_0(view, offset, cols, out, outOffset) {
  for (let block = 0; block < cols / 32; block++) {
    const base = offset + block * 34;
    const d = fp16ToFloat(view.getUint16(base, true));
    const qBase = base + 2;
    const outBase = outOffset + block * 32;
    for (let j = 0; j < 32; j++) out[outBase + j] = int8At(view, qBase + j) * d;
  }
}

function decodeRowQ2_K(view, offset, out, outOffset) {
  const d = fp16ToFloat(view.getUint16(offset + 80, true));
  const min = fp16ToFloat(view.getUint16(offset + 82, true));
  let qBase = offset + 16;
  let scaleIndex = 0;
  let write = outOffset;
  for (let n = 0; n < QK_K; n += 128) {
    let shift = 0;
    for (let j = 0; j < 4; j++) {
      let sc = view.getUint8(offset + scaleIndex++);
      let dl = d * (sc & 0x0f);
      let ml = min * (sc >> 4);
      for (let l = 0; l < 16; l++) out[write++] = dl * ((view.getUint8(qBase + l) >> shift) & 3) - ml;
      sc = view.getUint8(offset + scaleIndex++);
      dl = d * (sc & 0x0f);
      ml = min * (sc >> 4);
      for (let l = 0; l < 16; l++) out[write++] = dl * ((view.getUint8(qBase + 16 + l) >> shift) & 3) - ml;
      shift += 2;
    }
    qBase += 32;
  }
}

function decodeRowQ3_K(view, offset, out, outOffset) {
  const auxBuffer = new ArrayBuffer(16);
  const auxView = new DataView(auxBuffer);
  for (let i = 0; i < 12; i++) auxView.setUint8(i, view.getUint8(offset + 96 + i));
  const kmask1 = 0x03030303;
  const kmask2 = 0x0f0f0f0f;
  const a0 = auxView.getUint32(0, true);
  const a1 = auxView.getUint32(4, true);
  const a2 = auxView.getUint32(8, true);
  auxView.setUint32(0, (a0 & kmask2) | (((a2 >>> 0) & kmask1) << 4), true);
  auxView.setUint32(4, (a1 & kmask2) | (((a2 >>> 2) & kmask1) << 4), true);
  auxView.setUint32(8, ((a0 >>> 4) & kmask2) | (((a2 >>> 4) & kmask1) << 4), true);
  auxView.setUint32(12, ((a1 >>> 4) & kmask2) | (((a2 >>> 6) & kmask1) << 4), true);
  const scales = new Int8Array(auxBuffer);
  const dAll = fp16ToFloat(view.getUint16(offset + 108, true));
  let qBase = offset + 32;
  let mask = 1;
  let scaleIndex = 0;
  let write = outOffset;
  for (let n = 0; n < QK_K; n += 128) {
    let shift = 0;
    for (let j = 0; j < 4; j++) {
      let dl = dAll * (scales[scaleIndex++] - 32);
      for (let l = 0; l < 16; l++) {
        const q = (view.getUint8(qBase + l) >> shift) & 3;
        const h = view.getUint8(offset + l) & mask ? 0 : 4;
        out[write++] = dl * (q - h);
      }
      dl = dAll * (scales[scaleIndex++] - 32);
      for (let l = 0; l < 16; l++) {
        const q = (view.getUint8(qBase + 16 + l) >> shift) & 3;
        const h = view.getUint8(offset + 16 + l) & mask ? 0 : 4;
        out[write++] = dl * (q - h);
      }
      shift += 2;
      mask <<= 1;
    }
    qBase += 32;
  }
}

function decodeRowQ4_K(view, offset, out, outOffset) {
  const d = fp16ToFloat(view.getUint16(offset, true));
  const min = fp16ToFloat(view.getUint16(offset + 2, true));
  const scales = new Uint8Array(view.buffer, view.byteOffset + offset + 4, 12);
  let qBase = offset + 16;
  let write = outOffset;
  let scaleIndex = 0;
  for (let j = 0; j < QK_K; j += 64) {
    let sm = getScaleMinK4(scales, scaleIndex++);
    const d1 = d * sm.scale;
    const m1 = min * sm.min;
    sm = getScaleMinK4(scales, scaleIndex++);
    const d2 = d * sm.scale;
    const m2 = min * sm.min;
    for (let l = 0; l < 32; l++) out[write++] = d1 * (view.getUint8(qBase + l) & 0x0f) - m1;
    for (let l = 0; l < 32; l++) out[write++] = d2 * (view.getUint8(qBase + l) >> 4) - m2;
    qBase += 32;
  }
}

function decodeRowQ5_K(view, offset, out, outOffset) {
  const d = fp16ToFloat(view.getUint16(offset, true));
  const min = fp16ToFloat(view.getUint16(offset + 2, true));
  const scales = new Uint8Array(view.buffer, view.byteOffset + offset + 4, 12);
  const qhBase = offset + 16;
  let qlBase = offset + 48;
  let write = outOffset;
  let scaleIndex = 0;
  let u1 = 1;
  let u2 = 2;
  for (let j = 0; j < QK_K; j += 64) {
    let sm = getScaleMinK4(scales, scaleIndex++);
    const d1 = d * sm.scale;
    const m1 = min * sm.min;
    sm = getScaleMinK4(scales, scaleIndex++);
    const d2 = d * sm.scale;
    const m2 = min * sm.min;
    for (let l = 0; l < 32; l++) {
      out[write++] = d1 * ((view.getUint8(qlBase + l) & 0x0f) + (view.getUint8(qhBase + l) & u1 ? 16 : 0)) - m1;
    }
    for (let l = 0; l < 32; l++) {
      out[write++] = d2 * ((view.getUint8(qlBase + l) >> 4) + (view.getUint8(qhBase + l) & u2 ? 16 : 0)) - m2;
    }
    qlBase += 32;
    u1 <<= 2;
    u2 <<= 2;
  }
}

function decodeRowQ6_K(view, offset, out, outOffset) {
  const d = fp16ToFloat(view.getUint16(offset + 208, true));
  let qlBase = offset;
  let qhBase = offset + 128;
  let scaleBase = offset + 192;
  let write = outOffset;
  for (let n = 0; n < QK_K; n += 128) {
    for (let l = 0; l < 32; l++) {
      const is = Math.floor(l / 16);
      const qh = view.getUint8(qhBase + l);
      const q1 = ((view.getUint8(qlBase + l) & 0x0f) | (((qh >> 0) & 3) << 4)) - 32;
      const q2 = ((view.getUint8(qlBase + 32 + l) & 0x0f) | (((qh >> 2) & 3) << 4)) - 32;
      const q3 = ((view.getUint8(qlBase + l) >> 4) | (((qh >> 4) & 3) << 4)) - 32;
      const q4 = ((view.getUint8(qlBase + 32 + l) >> 4) | (((qh >> 6) & 3) << 4)) - 32;
      out[write + l] = d * int8At(view, scaleBase + is + 0) * q1;
      out[write + l + 32] = d * int8At(view, scaleBase + is + 2) * q2;
      out[write + l + 64] = d * int8At(view, scaleBase + is + 4) * q3;
      out[write + l + 96] = d * int8At(view, scaleBase + is + 6) * q4;
    }
    write += 128;
    qlBase += 64;
    qhBase += 32;
    scaleBase += 8;
  }
}

function decodeRows(type, buffer, rowCount, cols) {
  const view = new DataView(buffer);
  const out = new Float32Array(rowCount * cols);
  const rowStride = computeTensorBytes(type, cols);
  for (let row = 0; row < rowCount; row++) {
    const rowOffset = row * rowStride;
    const outOffset = row * cols;
    if (type === 0) decodeRowF32(view, rowOffset, cols, out, outOffset);
    else if (type === 1) decodeRowF16(view, rowOffset, cols, out, outOffset);
    else if (type === 2) decodeRowQ4_0(view, rowOffset, cols, out, outOffset);
    else if (type === 3) decodeRowQ4_1(view, rowOffset, cols, out, outOffset);
    else if (type === 6) decodeRowQ5_0(view, rowOffset, cols, out, outOffset);
    else if (type === 7) decodeRowQ5_1(view, rowOffset, cols, out, outOffset);
    else if (type === 8 || type === 9) decodeRowQ8_0(view, rowOffset, cols, out, outOffset);
    else if (type === 10) decodeRowQ2_K(view, rowOffset, out, outOffset);
    else if (type === 11) decodeRowQ3_K(view, rowOffset, out, outOffset);
    else if (type === 12) decodeRowQ4_K(view, rowOffset, out, outOffset);
    else if (type === 13) decodeRowQ5_K(view, rowOffset, out, outOffset);
    else if (type === 14) decodeRowQ6_K(view, rowOffset, out, outOffset);
    else if (type === 30) decodeRowBF16(view, rowOffset, cols, out, outOffset);
    else throw new Error(`Unsupported GGML tensor type ${type}`);
  }
  return out;
}

function computeStats(values) {
  if (!values.length) return { min: 0, max: 0, mean: 0, std: 0, absMax: 0 };
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSq = 0;
  let absMax = 0;
  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
    const abs = Math.abs(value);
    if (abs > absMax) absMax = abs;
    sum += value;
    sumSq += value * value;
  }
  const mean = sum / values.length;
  const variance = Math.max(0, sumSq / values.length - mean * mean);
  return { min, max, mean, std: Math.sqrt(variance), absMax };
}

async function decodeTensorSlice(file, tensor, tensorDataStart, rowStart, rowCount) {
  const dimensions = tensor.dimensions || tensor.shape || [];
  const cols = Number(dimensions[0]) || 0;
  if (!cols) throw new Error('Tensor does not expose a decodable row width.');
  const totalRows = Math.round((tensor.numElements || 0) / cols);
  if (!Number.isFinite(totalRows) || totalRows <= 0) throw new Error('Tensor row count is invalid.');
  if (rowStart < 0 || rowStart + rowCount > totalRows) throw new Error(`Requested rows ${rowStart}–${rowStart + rowCount - 1} exceed tensor bounds.`);
  const rowStride = computeTensorBytes(tensor.type, cols);
  if (!rowStride) throw new Error(`Unsupported row stride for tensor type ${tensor.typeName || tensor.type}.`);
  const absoluteOffset = Number.isFinite(tensor.absoluteOffset)
    ? tensor.absoluteOffset
    : (Number.isFinite(tensor.offset) && Number.isFinite(tensorDataStart) ? tensor.offset + tensorDataStart : NaN);
  if (!Number.isFinite(absoluteOffset)) throw new Error('Tensor does not expose an absolute GGUF byte offset.');
  const start = absoluteOffset + rowStart * rowStride;
  const end = start + rowCount * rowStride;
  const buffer = await file.slice(start, end).arrayBuffer();
  const values = decodeRows(tensor.type, buffer, rowCount, cols);
  return { cols, rows: rowCount, values, stats: computeStats(values) };
}

export async function decodeHeadSelection(model, decodePlan) {
  if (!decodePlan?.slices?.length) {
    return { status: 'unavailable', message: 'No tensor slice is available for this head selection.', warnings: [], slices: [] };
  }
  if (!model?.ggufSource?.slice) {
    return {
      status: 'unavailable',
      message: 'Learned-weight heatmaps currently require an uploaded GGUF file. Ollama models only expose structural metadata in this app.',
      warnings: [],
      slices: [],
    };
  }
  const decodedSlices = [];
  const warnings = [];
  for (const slice of decodePlan.slices) {
    try {
      const decoded = await decodeTensorSlice(model.ggufSource, slice.tensor, model.ggufInfo?.tensorDataStart ?? null, slice.rowStart, slice.rowCount);
      decodedSlices.push({ ...slice, ...decoded });
    } catch (error) {
      warnings.push(`${slice.label}: ${error.message}`);
    }
  }
  if (!decodedSlices.length) {
    return { status: 'unavailable', message: warnings[0] || 'Unable to decode the selected tensor slice.', warnings, slices: [] };
  }
  return {
    status: 'loaded',
    warnings,
    slices: decodedSlices,
    sourceName: model.ggufSource?.name || 'uploaded GGUF',
  };
}