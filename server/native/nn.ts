/**
 * NEXA Native Runtime - Neural Network Layers (TypeScript)
 * RMSNorm, RoPE, Scaled Dot-Product Attention, SwiGLU, and Transformer Block.
 */

import { NexaTensor, NexaShape } from './tensor.js';

export interface NexaTransformerHyperparams {
  vocabSize: number;
  dim: number;             // Hidden size
  intermediateDim: number; // MLP dim
  nLayers: number;
  nHeads: number;          // Query heads
  nKvHeads: number;        // Key/Value heads (GQA)
  maxSeqLen: number;
  normEps: number;
  ropeTheta: number;
}

export class NexaRMSNorm {
  static forward(input: NexaTensor, weight: NexaTensor, output: NexaTensor, eps: number = 1e-5): void {
    const dim = input.shape.dims[input.shape.rank - 1];
    const totalTokens = Math.floor(input.shape.numel / dim);

    const inData = input.data;
    const wData = weight.data;
    const outData = output.data;
    const inOff = input.offset;
    const wOff = weight.offset;
    const outOff = output.offset;

    for (let t = 0; t < totalTokens; t++) {
      const tIn = inOff + t * dim;
      const tOut = outOff + t * dim;

      // Calculate Root Mean Square
      let sumSq = 0;
      for (let i = 0; i < dim; i++) {
        const val = inData[tIn + i];
        sumSq += val * val;
      }
      const invRms = 1 / Math.sqrt((sumSq / dim) + eps);

      for (let i = 0; i < dim; i++) {
        outData[tOut + i] = inData[tIn + i] * invRms * wData[wOff + i];
      }
    }
  }
}

export class NexaRoPE {
  static forward(qOrK: NexaTensor, pos: number, headDim: number, theta: number = 10000.0): void {
    const data = qOrK.data;
    const offset = qOrK.offset;
    const totalElements = qOrK.shape.numel;
    const nHeads = Math.floor(totalElements / headDim);

    for (let h = 0; h < nHeads; h++) {
      const headOff = offset + h * headDim;
      for (let i = 0; i < Math.floor(headDim / 2); i++) {
        const freq = 1.0 / Math.pow(theta, (2 * i) / headDim);
        const val = pos * freq;
        const cosVal = Math.cos(val);
        const sinVal = Math.sin(val);

        const v0 = data[headOff + 2 * i];
        const v1 = data[headOff + 2 * i + 1];

        data[headOff + 2 * i] = v0 * cosVal - v1 * sinVal;
        data[headOff + 2 * i + 1] = v0 * sinVal + v1 * cosVal;
      }
    }
  }
}

export class NexaEmbedding {
  static forward(tokens: number[], embedTable: NexaTensor, output: NexaTensor): void {
    const dim = embedTable.shape.dims[1];
    const tableData = embedTable.data;
    const outData = output.data;
    const tableOff = embedTable.offset;
    const outOff = output.offset;
    const vocabSize = embedTable.shape.dims[0];

    for (let t = 0; t < tokens.length; t++) {
      let tok = tokens[t];
      if (tok < 0 || tok >= vocabSize) tok = 0;
      const rowStart = tableOff + tok * dim;
      const outStart = outOff + t * dim;

      for (let i = 0; i < dim; i++) {
        outData[outStart + i] = tableData[rowStart + i];
      }
    }
  }
}

export class NexaSwiGLU {
  static forward(
    x: NexaTensor,
    wGate: NexaTensor,
    wUp: NexaTensor,
    wDown: NexaTensor,
    output: NexaTensor
  ): void {
    const seqLen = x.shape.dims[0];
    const interDim = wGate.shape.dims[1];

    const gate = new NexaTensor(new NexaShape([seqLen, interDim]));
    const up = new NexaTensor(new NexaShape([seqLen, interDim]));

    NexaTensor.matmul(x, wGate, gate);
    NexaTensor.matmul(x, wUp, up);

    const gData = gate.data;
    const uData = up.data;
    const gOff = gate.offset;
    const uOff = up.offset;
    const total = seqLen * interDim;

    // SiLU(gate) * up
    for (let i = 0; i < total; i++) {
      const val = gData[gOff + i];
      const silu = val / (1.0 + Math.exp(-val));
      gData[gOff + i] = silu * uData[uOff + i];
    }

    NexaTensor.matmul(gate, wDown, output);
  }
}

export class NexaAttention {
  static forward(
    q: NexaTensor,
    k: NexaTensor,
    v: NexaTensor,
    wOut: NexaTensor,
    output: NexaTensor,
    isCausal: boolean = true
  ): void {
    const seqLen = q.shape.dims[0];
    const dim = q.shape.dims[1];
    const headDim = 64;
    const nHeads = Math.floor(dim / headDim);

    const context = new NexaTensor(new NexaShape([seqLen, dim]));
    context.fillZero();

    const scale = 1.0 / Math.sqrt(headDim);
    const qData = q.data;
    const kData = k.data;
    const vData = v.data;
    const ctxData = context.data;

    for (let h = 0; h < nHeads; h++) {
      for (let i = 0; i < seqLen; i++) {
        const scores = new Float32Array(seqLen);
        let maxScore = -1e9;

        const jLimit = isCausal ? i + 1 : seqLen;
        for (let j = 0; j < jLimit; j++) {
          let dot = 0;
          const qPtr = q.offset + i * dim + h * headDim;
          const kPtr = k.offset + j * dim + h * headDim;
          for (let d = 0; d < headDim; d++) {
            dot += qData[qPtr + d] * kData[kPtr + d];
          }
          const score = dot * scale;
          scores[j] = score;
          if (score > maxScore) maxScore = score;
        }

        // Softmax
        let sumExp = 0;
        for (let j = 0; j < jLimit; j++) {
          scores[j] = Math.exp(scores[j] - maxScore);
          sumExp += scores[j];
        }
        const invSum = sumExp > 0 ? 1 / sumExp : 0;
        for (let j = 0; j < jLimit; j++) {
          scores[j] *= invSum;
        }

        // Weighted accumulation of V
        const ctxPtr = context.offset + i * dim + h * headDim;
        for (let j = 0; j < jLimit; j++) {
          const vPtr = v.offset + j * dim + h * headDim;
          const weight = scores[j];
          for (let d = 0; d < headDim; d++) {
            ctxData[ctxPtr + d] += weight * vData[vPtr + d];
          }
        }
      }
    }

    NexaTensor.matmul(context, wOut, output);
  }
}

export class NexaTransformerBlock {
  static forward(
    input: NexaTensor,
    attnNormWeight: NexaTensor,
    wQ: NexaTensor,
    wK: NexaTensor,
    wV: NexaTensor,
    wOut: NexaTensor,
    ffnNormWeight: NexaTensor,
    wGate: NexaTensor,
    wUp: NexaTensor,
    wDown: NexaTensor,
    output: NexaTensor,
    hp: NexaTransformerHyperparams,
    startPos: number = 0
  ): void {
    const seqLen = input.shape.dims[0];
    const dim = input.shape.dims[1];

    // 1. Attention Pre-Normalization
    const xNorm = new NexaTensor(new NexaShape([seqLen, dim]));
    NexaRMSNorm.forward(input, attnNormWeight, xNorm, hp.normEps);

    // 2. Q, K, V Projections
    const q = new NexaTensor(new NexaShape([seqLen, dim]));
    const k = new NexaTensor(new NexaShape([seqLen, dim]));
    const v = new NexaTensor(new NexaShape([seqLen, dim]));
    NexaTensor.matmul(xNorm, wQ, q);
    NexaTensor.matmul(xNorm, wK, k);
    NexaTensor.matmul(xNorm, wV, v);

    // 3. RoPE on Q & K
    const headDim = Math.floor(dim / hp.nHeads);
    for (let i = 0; i < seqLen; i++) {
      const qSlice = NexaTensor.createView(q, new NexaShape([1, dim]), i * dim);
      const kSlice = NexaTensor.createView(k, new NexaShape([1, dim]), i * dim);
      NexaRoPE.forward(qSlice, startPos + i, headDim, hp.ropeTheta);
      NexaRoPE.forward(kSlice, startPos + i, headDim, hp.ropeTheta);
    }

    // 4. Attention
    const attnOut = new NexaTensor(new NexaShape([seqLen, dim]));
    NexaAttention.forward(q, k, v, wOut, attnOut, true);

    // 5. First Residual: x1 = input + attnOut
    const x1 = new NexaTensor(new NexaShape([seqLen, dim]));
    NexaTensor.vecAdd(input, attnOut, x1);

    // 6. FFN Pre-Normalization
    const x1Norm = new NexaTensor(new NexaShape([seqLen, dim]));
    NexaRMSNorm.forward(x1, ffnNormWeight, x1Norm, hp.normEps);

    // 7. SwiGLU Feed-Forward
    const ffnOut = new NexaTensor(new NexaShape([seqLen, dim]));
    NexaSwiGLU.forward(x1Norm, wGate, wUp, wDown, ffnOut);

    // 8. Second Residual: output = x1 + ffnOut
    NexaTensor.vecAdd(x1, ffnOut, output);
  }
}
