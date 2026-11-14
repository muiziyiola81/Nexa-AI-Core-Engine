/**
 * NEXA Native Runtime - TypeScript Tensor Math Kernel
 * Mirrors the C++ tensor computation layer for direct Node.js execution.
 */

export enum DataType {
  FLOAT32 = 0,
  FLOAT16 = 1,
  INT8 = 2,
  INT4 = 3
}

export class NexaShape {
  readonly dims: number[];

  constructor(dims: number[]) {
    this.dims = [...dims];
  }

  get rank(): number {
    return this.dims.length;
  }

  get numel(): number {
    if (this.dims.length === 0) return 0;
    return this.dims.reduce((a, b) => a * b, 1);
  }

  computeStrides(): number[] {
    const strides = new Array(this.dims.length).fill(1);
    for (let i = this.dims.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * this.dims[i + 1];
    }
    return strides;
  }

  toString(): string {
    return `[${this.dims.join(', ')}]`;
  }
}

export class NexaTensor {
  readonly shape: NexaShape;
  readonly dtype: DataType;
  readonly data: Float32Array;
  readonly offset: number;

  constructor(shape: NexaShape, dataOrLength?: Float32Array | number, offset: number = 0, dtype: DataType = DataType.FLOAT32) {
    this.shape = shape;
    this.dtype = dtype;
    this.offset = offset;

    if (dataOrLength instanceof Float32Array) {
      this.data = dataOrLength;
    } else {
      const len = typeof dataOrLength === 'number' ? dataOrLength : shape.numel;
      this.data = new Float32Array(len);
    }
  }

  static createView(base: NexaTensor, newShape: NexaShape, elementOffset: number = 0): NexaTensor {
    return new NexaTensor(newShape, base.data, base.offset + elementOffset, base.dtype);
  }

  fillConstant(val: number): void {
    const start = this.offset;
    const end = start + this.shape.numel;
    for (let i = start; i < end; i++) {
      this.data[i] = val;
    }
  }

  fillZero(): void {
    this.fillConstant(0);
  }

  // --------------------------------------------------------------------------
  // Core Tensor Mathematics
  // --------------------------------------------------------------------------

  /**
   * 2D Matrix Multiplication: C = A x B
   * A: [M, K], B: [K, N], C: [M, N]
   */
  static matmul(A: NexaTensor, B: NexaTensor, C: NexaTensor): void {
    if (A.shape.rank !== 2 || B.shape.rank !== 2 || C.shape.rank !== 2) {
      throw new Error(`matmul expects 2D tensors, got ${A.shape.toString()} and ${B.shape.toString()}`);
    }
    const M = A.shape.dims[0];
    const K = A.shape.dims[1];
    const BK = B.shape.dims[0];
    const N = B.shape.dims[1];

    if (K !== BK) {
      throw new Error(`Inner dimension mismatch in matmul: ${K} != ${BK}`);
    }
    if (C.shape.dims[0] !== M || C.shape.dims[1] !== N) {
      throw new Error(`Output tensor dimensions mismatch: expected [${M}, ${N}], got ${C.shape.toString()}`);
    }

    C.fillZero();

    const aData = A.data;
    const bData = B.data;
    const cData = C.data;
    const aOff = A.offset;
    const bOff = B.offset;
    const cOff = C.offset;

    // Cache-tiled matrix multiplication loop
    const BLOCK = 32;
    for (let ii = 0; ii < M; ii += BLOCK) {
      const iMax = Math.min(M, ii + BLOCK);
      for (let kk = 0; kk < K; kk += BLOCK) {
        const kMax = Math.min(K, kk + BLOCK);
        for (let jj = 0; jj < N; jj += BLOCK) {
          const jMax = Math.min(N, jj + BLOCK);

          for (let i = ii; i < iMax; i++) {
            const aRow = aOff + i * K;
            const cRow = cOff + i * N;
            for (let k = kk; k < kMax; k++) {
              const aVal = aData[aRow + k];
              const bRow = bOff + k * N;
              for (let j = jj; j < jMax; j++) {
                cData[cRow + j] += aVal * bData[bRow + j];
              }
            }
          }
        }
      }
    }
  }

  static vecAdd(A: NexaTensor, B: NexaTensor, out: NexaTensor): void {
    const n = A.shape.numel;
    if (B.shape.numel !== n || out.shape.numel !== n) {
      throw new Error('Dimension mismatch in vecAdd');
    }
    const a = A.data;
    const b = B.data;
    const o = out.data;
    const aOff = A.offset;
    const bOff = B.offset;
    const oOff = out.offset;

    for (let i = 0; i < n; i++) {
      o[oOff + i] = a[aOff + i] + b[bOff + i];
    }
  }

  static vecMul(A: NexaTensor, B: NexaTensor, out: NexaTensor): void {
    const n = A.shape.numel;
    const a = A.data;
    const b = B.data;
    const o = out.data;
    const aOff = A.offset;
    const bOff = B.offset;
    const oOff = out.offset;

    for (let i = 0; i < n; i++) {
      o[oOff + i] = a[aOff + i] * b[bOff + i];
    }
  }

  static vecScale(A: NexaTensor, scalar: number, out: NexaTensor): void {
    const n = A.shape.numel;
    const a = A.data;
    const o = out.data;
    const aOff = A.offset;
    const oOff = out.offset;

    for (let i = 0; i < n; i++) {
      o[oOff + i] = a[aOff + i] * scalar;
    }
  }

  static dotProduct(a: Float32Array, aOff: number, b: Float32Array, bOff: number, n: number): number {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += a[aOff + i] * b[bOff + i];
    }
    return sum;
  }

  static softmax(input: NexaTensor, output: NexaTensor): void {
    const inData = input.data;
    const outData = output.data;
    const inOff = input.offset;
    const outOff = output.offset;

    const total = input.shape.numel;
    const lastDim = input.shape.dims[input.shape.rank - 1];
    const batches = Math.floor(total / lastDim);

    for (let b = 0; b < batches; b++) {
      const bIn = inOff + b * lastDim;
      const bOut = outOff + b * lastDim;

      // 1. Numerically stable max
      let maxVal = inData[bIn];
      for (let i = 1; i < lastDim; i++) {
        if (inData[bIn + i] > maxVal) maxVal = inData[bIn + i];
      }

      // 2. Exponential & sum
      let sum = 0;
      for (let i = 0; i < lastDim; i++) {
        const expVal = Math.exp(inData[bIn + i] - maxVal);
        outData[bOut + i] = expVal;
        sum += expVal;
      }

      // 3. Normalization
      const invSum = sum > 0 ? 1 / sum : 0;
      for (let i = 0; i < lastDim; i++) {
        outData[bOut + i] *= invSum;
      }
    }
  }
}
