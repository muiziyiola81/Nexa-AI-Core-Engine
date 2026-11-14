/**
 * NEXA Native Runtime - Model Loader (TypeScript)
 * Validates .nexa model binary files, computes memory footprint, and checks host capacity.
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import { decodeNexaHeader, NexaParsedModel, NexaHeaderConfig } from './model_format.js';

export interface NexaMemoryBudget {
  weightBytes: number;
  kvCacheBytes: number;
  activationBytes: number;
  totalRequiredBytes: number;
  totalRequiredMb: number;
  availableHostRamMb: number;
  fitsInRam: boolean;
}

export class NexaModelLoader {
  static inspect(filePath: string): NexaParsedModel {
    const resolved = path.isAbsolute(filePath) ? filePath : path.resolve(process.cwd(), filePath);
    if (!fs.existsSync(resolved)) {
      return {
        magic: 0,
        version: 0,
        archType: 0,
        config: {} as any,
        tensors: [],
        totalWeightBytes: 0,
        isValid: false,
        error: `File does not exist at "${resolved}"`
      };
    }

    const stat = fs.statSync(resolved);
    if (stat.size < 56) {
      return {
        magic: 0,
        version: 0,
        archType: 0,
        config: {} as any,
        tensors: [],
        totalWeightBytes: 0,
        isValid: false,
        error: `File size (${stat.size} bytes) is too small to be a valid .nexa model`
      };
    }

    // Read initial 64KB for header and tensor table
    const fd = fs.openSync(resolved, 'r');
    const readSize = Math.min(stat.size, 65536);
    const buffer = Buffer.alloc(readSize);
    fs.readSync(fd, buffer, 0, readSize, 0);
    fs.closeSync(fd);

    return decodeNexaHeader(buffer);
  }

  static calculateMemory(config: NexaHeaderConfig, availableRamBytes?: number): NexaMemoryBudget {
    const freeBytes = availableRamBytes || os.freemem();
    const headDim = config.nHeads > 0 ? Math.floor(config.dim / config.nHeads) : 64;

    let bytesPerParam = 4;
    if (config.quantType === 1) bytesPerParam = 2; // FP16
    if (config.quantType === 2) bytesPerParam = 1; // INT8
    if (config.quantType === 3) bytesPerParam = 0.5; // INT4

    // Weights: vocab * dim + layers * (4 * dim^2 + 3 * dim * interDim)
    const embedBytes = config.vocabSize * config.dim * bytesPerParam;
    const layerBytes = config.nLayers * (
      (config.dim * config.dim * 4) +
      (config.dim * config.intermediateDim * 3)
    ) * bytesPerParam;
    const weightBytes = Math.round(embedBytes + layerBytes);

    // KV Cache: 2 (K, V) * nLayers * nKvHeads * headDim * maxSeqLen * 4 (FP32)
    const kvCacheBytes = 2 * config.nLayers * config.nKvHeads * headDim * config.maxSeqLen * 4;

    // Working activation scratchpad: 4 * maxSeqLen * dim * 4
    const activationBytes = 4 * config.maxSeqLen * config.dim * 4;

    const totalRequiredBytes = weightBytes + kvCacheBytes + activationBytes;
    const totalRequiredMb = Math.round(totalRequiredBytes / (1024 * 1024));
    const availableHostRamMb = Math.round(freeBytes / (1024 * 1024));

    return {
      weightBytes,
      kvCacheBytes,
      activationBytes,
      totalRequiredBytes,
      totalRequiredMb,
      availableHostRamMb,
      fitsInRam: totalRequiredBytes <= freeBytes
    };
  }

  static validateInvariants(config: NexaHeaderConfig): { valid: boolean; error?: string } {
    if (config.dim <= 0 || config.nLayers <= 0 || config.nHeads <= 0) {
      return { valid: false, error: 'Dimension, layers, or heads count cannot be zero.' };
    }
    if (config.dim % config.nHeads !== 0) {
      return { valid: false, error: `Hidden dimension (${config.dim}) must be divisible by attention heads (${config.nHeads}).` };
    }
    if (config.nKvHeads <= 0 || config.nHeads % config.nKvHeads !== 0) {
      return { valid: false, error: `Query heads (${config.nHeads}) must be an exact multiple of KV heads (${config.nKvHeads}).` };
    }
    return { valid: true };
  }
}
