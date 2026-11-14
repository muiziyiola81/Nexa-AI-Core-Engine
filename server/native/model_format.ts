/**
 * NEXA Native Runtime - Model Format Specification (.nexa)
 * Binary serialization structures and layout constants.
 */

export const NEXA_MAGIC = 0x4E455841; // 'N', 'E', 'X', 'A' (1313167425 in decimal)
export const NEXA_VERSION = 1;

export enum NexaArchType {
  TRANSFORMER_DECODER = 1,
  MIXTURE_OF_EXPERTS = 2
}

export interface NexaHeaderConfig {
  vocabSize: number;
  dim: number;
  intermediateDim: number;
  nLayers: number;
  nHeads: number;
  nKvHeads: number;
  maxSeqLen: number;
  normEps: number;
  ropeTheta: number;
  quantType: number; // 0: FP32, 1: FP16, 2: INT8, 3: INT4
  nTensors: number;
}

export interface NexaTensorMeta {
  name: string;
  dtype: number;
  dims: number[];
  offset: number;
  sizeBytes: number;
}

export interface NexaParsedModel {
  magic: number;
  version: number;
  archType: number;
  config: NexaHeaderConfig;
  tensors: NexaTensorMeta[];
  totalWeightBytes: number;
  isValid: boolean;
  error?: string;
}

/**
 * Encodes a .nexa binary header buffer.
 */
export function encodeNexaHeader(config: NexaHeaderConfig, tensors: NexaTensorMeta[]): Buffer {
  // Fixed header size = 14 fields * 4 bytes = 56 bytes
  const headerBuf = Buffer.alloc(56);
  let pos = 0;

  headerBuf.writeUInt32LE(NEXA_MAGIC, pos); pos += 4;
  headerBuf.writeUInt32LE(NEXA_VERSION, pos); pos += 4;
  headerBuf.writeUInt32LE(NexaArchType.TRANSFORMER_DECODER, pos); pos += 4;
  headerBuf.writeUInt32LE(config.vocabSize, pos); pos += 4;
  headerBuf.writeUInt32LE(config.dim, pos); pos += 4;
  headerBuf.writeUInt32LE(config.intermediateDim, pos); pos += 4;
  headerBuf.writeUInt32LE(config.nLayers, pos); pos += 4;
  headerBuf.writeUInt32LE(config.nHeads, pos); pos += 4;
  headerBuf.writeUInt32LE(config.nKvHeads, pos); pos += 4;
  headerBuf.writeUInt32LE(config.maxSeqLen, pos); pos += 4;
  headerBuf.writeFloatLE(config.normEps, pos); pos += 4;
  headerBuf.writeFloatLE(config.ropeTheta, pos); pos += 4;
  headerBuf.writeUInt32LE(config.quantType, pos); pos += 4;
  headerBuf.writeUInt32LE(tensors.length, pos); pos += 4;

  // Build tensor table
  const tensorBuffers: Buffer[] = [headerBuf];

  for (const t of tensors) {
    const nameBytes = Buffer.from(t.name, 'utf-8');
    const descHeader = Buffer.alloc(4 + nameBytes.length + 4 + 4 + (t.dims.length * 8) + 8 + 8);
    let dPos = 0;

    descHeader.writeUInt32LE(nameBytes.length, dPos); dPos += 4;
    nameBytes.copy(descHeader, dPos); dPos += nameBytes.length;
    descHeader.writeUInt32LE(t.dtype, dPos); dPos += 4;
    descHeader.writeUInt32LE(t.dims.length, dPos); dPos += 4;

    for (const d of t.dims) {
      descHeader.writeBigUInt64LE(BigInt(d), dPos); dPos += 8;
    }

    descHeader.writeBigUInt64LE(BigInt(t.offset), dPos); dPos += 8;
    descHeader.writeBigUInt64LE(BigInt(t.sizeBytes), dPos); dPos += 8;

    tensorBuffers.push(descHeader);
  }

  return Buffer.concat(tensorBuffers);
}

/**
 * Decodes a .nexa binary model header from buffer.
 */
export function decodeNexaHeader(buffer: Buffer): NexaParsedModel {
  if (buffer.length < 56) {
    return {
      magic: 0,
      version: 0,
      archType: 0,
      config: {} as any,
      tensors: [],
      totalWeightBytes: 0,
      isValid: false,
      error: 'Buffer is smaller than minimum 56-byte .nexa header'
    };
  }

  let pos = 0;
  const magic = buffer.readUInt32LE(pos); pos += 4;
  const version = buffer.readUInt32LE(pos); pos += 4;
  const archType = buffer.readUInt32LE(pos); pos += 4;

  if (magic !== NEXA_MAGIC) {
    return {
      magic,
      version,
      archType,
      config: {} as any,
      tensors: [],
      totalWeightBytes: 0,
      isValid: false,
      error: `Invalid magic bytes: 0x${magic.toString(16).toUpperCase()} (expected 0x4E455841 'NEXA')`
    };
  }

  const config: NexaHeaderConfig = {
    vocabSize: buffer.readUInt32LE(pos),
    dim: buffer.readUInt32LE(pos + 4),
    intermediateDim: buffer.readUInt32LE(pos + 8),
    nLayers: buffer.readUInt32LE(pos + 12),
    nHeads: buffer.readUInt32LE(pos + 16),
    nKvHeads: buffer.readUInt32LE(pos + 20),
    maxSeqLen: buffer.readUInt32LE(pos + 24),
    normEps: buffer.readFloatLE(pos + 28),
    ropeTheta: buffer.readFloatLE(pos + 32),
    quantType: buffer.readUInt32LE(pos + 36),
    nTensors: buffer.readUInt32LE(pos + 40)
  };
  pos += 44;

  const tensors: NexaTensorMeta[] = [];
  let totalWeightBytes = 0;

  for (let i = 0; i < config.nTensors; i++) {
    if (pos >= buffer.length) break;
    const nameLen = buffer.readUInt32LE(pos); pos += 4;
    const name = buffer.toString('utf-8', pos, pos + nameLen); pos += nameLen;
    const dtype = buffer.readUInt32LE(pos); pos += 4;
    const nDims = buffer.readUInt32LE(pos); pos += 4;

    const dims: number[] = [];
    for (let d = 0; d < nDims; d++) {
      dims.push(Number(buffer.readBigUInt64LE(pos))); pos += 8;
    }

    const offset = Number(buffer.readBigUInt64LE(pos)); pos += 8;
    const sizeBytes = Number(buffer.readBigUInt64LE(pos)); pos += 8;

    tensors.push({ name, dtype, dims, offset, sizeBytes });
    totalWeightBytes += sizeBytes;
  }

  return {
    magic,
    version,
    archType,
    config,
    tensors,
    totalWeightBytes,
    isValid: true
  };
}
