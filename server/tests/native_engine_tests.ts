/**
 * NEXA AI Core Engine - Native Runtime Test Suite (Phase 3)
 * Comprehensive unit and integration tests for:
 * - Tensor operations (NexaShape, NexaTensor, matmul, vector math, stable softmax)
 * - Neural network layers (NexaRMSNorm, NexaRoPE, NexaSwiGLU, NexaAttention, NexaTransformerBlock)
 * - Tokenizer byte-level encoding & decoding round-trip
 * - .nexa binary model format serialization/deserialization (encodeNexaHeader, decodeNexaHeader)
 * - Model loader memory budget calculations & invariant validation
 * - NexaNativeAdapter lifecycle and inference pipeline
 */

import { NexaTensor, NexaShape, DataType } from '../native/tensor.js';
import {
  NexaRMSNorm,
  NexaRoPE,
  NexaEmbedding,
  NexaSwiGLU,
  NexaAttention,
  NexaTransformerBlock,
  NexaTransformerHyperparams
} from '../native/nn.js';
import { NexaTokenizer } from '../native/tokenizer.js';
import {
  encodeNexaHeader,
  decodeNexaHeader,
  NEXA_MAGIC,
  NEXA_VERSION,
  NexaHeaderConfig,
  NexaTensorMeta
} from '../native/model_format.js';
import { NexaModelLoader } from '../native/model_loader.js';
import { NexaNativeAdapter } from '../native/native_adapter.js';
import { TestCaseResult } from './engine_tests.js';
import path from 'path';
import fs from 'fs';

export async function runNativeEngineTests(): Promise<TestCaseResult[]> {
  const results: TestCaseResult[] = [];

  async function test(
    name: string,
    category: TestCaseResult['category'],
    fn: () => Promise<void> | void
  ) {
    const t0 = Date.now();
    try {
      await fn();
      results.push({
        name,
        category,
        passed: true,
        durationMs: Date.now() - t0,
        details: 'Passed native runtime validation.'
      });
    } catch (err: unknown) {
      const error = err as Error;
      results.push({
        name,
        category,
        passed: false,
        durationMs: Date.now() - t0,
        error: error.message || 'Assertion failed.'
      });
    }
  }

  // 1. Tensor creation, shape handling, and indexing
  await test('Tensor: Creation, shape handling, and index calculation', 'validation', () => {
    const shape = new NexaShape([2, 3, 4]);
    const t = new NexaTensor(shape);
    if (t.shape.numel !== 24) throw new Error(`Expected 24 elements, got ${t.shape.numel}`);
    if (t.shape.dims[0] !== 2 || t.shape.dims[1] !== 3 || t.shape.dims[2] !== 4) {
      throw new Error(`Shape mismatch: ${t.shape.toString()}`);
    }
    t.data[15] = 42.5;
    if (Math.abs(t.data[15] - 42.5) > 1e-6) throw new Error(`Expected 42.5, got ${t.data[15]}`);
  });

  // 2. Matrix Multiplication Correctness (2x3 * 3x2 = 2x2)
  await test('Tensor: Matrix multiplication numerical correctness (2x3 * 3x2)', 'validation', () => {
    // A = [[1, 2, 3], [4, 5, 6]] (2x3)
    const A = new NexaTensor(new NexaShape([2, 3]), new Float32Array([1, 2, 3, 4, 5, 6]));
    // B = [[7, 8], [9, 1], [2, 3]] (3x2)
    const B = new NexaTensor(new NexaShape([3, 2]), new Float32Array([7, 8, 9, 1, 2, 3]));
    // C = A * B -> [2, 2]
    // C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // C[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // C[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    const C = new NexaTensor(new NexaShape([2, 2]));
    NexaTensor.matmul(A, B, C);

    const expected = [31, 19, 85, 55];
    for (let i = 0; i < 4; i++) {
      if (Math.abs(C.data[i] - expected[i]) > 1e-5) {
        throw new Error(`Matmul output mismatch at index ${i}: expected ${expected[i]}, got ${C.data[i]}`);
      }
    }
  });

  // 3. Vector arithmetic operations
  await test('Tensor: Vector add, scale, and element-wise multiply', 'validation', () => {
    const A = new NexaTensor(new NexaShape([4]), new Float32Array([1, 2, 3, 4]));
    const B = new NexaTensor(new NexaShape([4]), new Float32Array([5, 6, 7, 8]));

    const sum = new NexaTensor(new NexaShape([4]));
    NexaTensor.vecAdd(A, B, sum);
    if (sum.data[0] !== 6 || sum.data[3] !== 12) throw new Error('Vector addition failed.');

    const scaled = new NexaTensor(new NexaShape([4]));
    NexaTensor.vecScale(A, 2.5, scaled);
    if (scaled.data[0] !== 2.5 || scaled.data[3] !== 10) throw new Error('Vector scale failed.');

    const prod = new NexaTensor(new NexaShape([4]));
    NexaTensor.vecMul(A, B, prod);
    if (prod.data[0] !== 5 || prod.data[3] !== 32) throw new Error('Vector element-wise mul failed.');
  });

  // 4. Numerically stable Softmax
  await test('Tensor: Numerically stable Softmax with extreme input ranges', 'validation', () => {
    // Inputs up to 1002.0 would overflow naive exp()
    const logits = new NexaTensor(new NexaShape([1, 3]), new Float32Array([1000.0, 1001.0, 1002.0]));
    const probs = new NexaTensor(new NexaShape([1, 3]));
    NexaTensor.softmax(logits, probs);

    let sum = 0;
    for (let i = 0; i < probs.shape.numel; i++) {
      sum += probs.data[i];
      if (isNaN(probs.data[i]) || !isFinite(probs.data[i])) {
        throw new Error(`Softmax generated NaN or Inf: ${probs.data[i]}`);
      }
    }
    if (Math.abs(sum - 1.0) > 1e-5) {
      throw new Error(`Softmax probabilities do not sum to 1.0: sum=${sum}`);
    }
    if (probs.data[2] <= probs.data[1] || probs.data[1] <= probs.data[0]) {
      throw new Error('Softmax monotonic ordering violated.');
    }
  });

  // 5. RMSNorm normalization
  await test('Neural Net: RMSNorm normalization invariant & scale', 'validation', () => {
    const dim = 4;
    // Constant vector [3, 3, 3, 3] -> RMS is 3. Normalized with weight 1.0 is [1, 1, 1, 1]
    const input = new NexaTensor(new NexaShape([1, dim]), new Float32Array([3, 3, 3, 3]));
    const weight = new NexaTensor(new NexaShape([dim]), new Float32Array([1, 1, 1, 1]));
    const output = new NexaTensor(new NexaShape([1, dim]));

    NexaRMSNorm.forward(input, weight, output, 1e-5);

    for (let i = 0; i < dim; i++) {
      if (Math.abs(output.data[i] - 1.0) > 1e-4) {
        throw new Error(`RMSNorm output mismatch: expected 1.0, got ${output.data[i]}`);
      }
    }
  });

  // 6. Rotary Position Embedding (RoPE)
  await test('Neural Net: RoPE rotary position embedding preserves vector norm', 'validation', () => {
    const headDim = 4;
    const q = new NexaTensor(new NexaShape([1, headDim]), new Float32Array([1.0, 2.0, 3.0, 4.0]));
    const normBefore = Math.sqrt(q.data.reduce((acc, v) => acc + v * v, 0));

    NexaRoPE.forward(q, 5, headDim, 10000.0);
    const normAfter = Math.sqrt(q.data.reduce((acc, v) => acc + v * v, 0));

    // 2D rotations preserve Euclidean length
    if (Math.abs(normBefore - normAfter) > 1e-4) {
      throw new Error(`RoPE did not preserve vector norm: before=${normBefore}, after=${normAfter}`);
    }
  });

  // 7. SwiGLU Gating & Non-linearity
  await test('Neural Net: SwiGLU forward pass with SiLU activation and element-wise gate', 'validation', () => {
    const hiddenDim = 4;
    const intermediateDim = 8;
    const x = new NexaTensor(new NexaShape([1, hiddenDim]), new Float32Array([0.5, -0.5, 1.0, -1.0]));

    const wGate = new NexaTensor(new NexaShape([hiddenDim, intermediateDim]));
    const wUp = new NexaTensor(new NexaShape([hiddenDim, intermediateDim]));
    const wDown = new NexaTensor(new NexaShape([intermediateDim, hiddenDim]));
    wGate.fillConstant(0.1);
    wUp.fillConstant(0.1);
    wDown.fillConstant(0.1);

    const out = new NexaTensor(new NexaShape([1, hiddenDim]));
    NexaSwiGLU.forward(x, wGate, wUp, wDown, out);

    if (out.shape.dims[0] !== 1 || out.shape.dims[1] !== hiddenDim) {
      throw new Error(`Unexpected SwiGLU output shape: ${out.shape.toString()}`);
    }
    for (let i = 0; i < out.shape.numel; i++) {
      if (isNaN(out.data[i]) || !isFinite(out.data[i])) {
        throw new Error(`SwiGLU output contains NaN/Inf at ${i}`);
      }
    }
  });

  // 8. TransformerBlock complete forward pass
  await test('Neural Net: TransformerBlock pre-norm residual forward pass', 'validation', () => {
    const hiddenDim = 64;
    const intermediateDim = 128;
    const hp: NexaTransformerHyperparams = {
      vocabSize: 256,
      dim: hiddenDim,
      intermediateDim,
      nLayers: 1,
      nHeads: 1,
      nKvHeads: 1,
      maxSeqLen: 64,
      normEps: 1e-5,
      ropeTheta: 10000.0
    };

    const input = new NexaTensor(new NexaShape([1, hiddenDim]));
    input.fillConstant(0.5);

    const attnNorm = new NexaTensor(new NexaShape([hiddenDim]));
    attnNorm.fillConstant(1.0);
    const wQ = new NexaTensor(new NexaShape([hiddenDim, hiddenDim]));
    const wK = new NexaTensor(new NexaShape([hiddenDim, hiddenDim]));
    const wV = new NexaTensor(new NexaShape([hiddenDim, hiddenDim]));
    const wOut = new NexaTensor(new NexaShape([hiddenDim, hiddenDim]));
    wQ.fillConstant(0.01);
    wK.fillConstant(0.01);
    wV.fillConstant(0.01);
    wOut.fillConstant(0.01);

    const ffnNorm = new NexaTensor(new NexaShape([hiddenDim]));
    ffnNorm.fillConstant(1.0);
    const wGate = new NexaTensor(new NexaShape([hiddenDim, intermediateDim]));
    const wUp = new NexaTensor(new NexaShape([hiddenDim, intermediateDim]));
    const wDown = new NexaTensor(new NexaShape([intermediateDim, hiddenDim]));
    wGate.fillConstant(0.01);
    wUp.fillConstant(0.01);
    wDown.fillConstant(0.01);

    const output = new NexaTensor(new NexaShape([1, hiddenDim]));
    NexaTransformerBlock.forward(
      input,
      attnNorm,
      wQ,
      wK,
      wV,
      wOut,
      ffnNorm,
      wGate,
      wUp,
      wDown,
      output,
      hp,
      0
    );

    if (output.shape.dims[0] !== 1 || output.shape.dims[1] !== hiddenDim) {
      throw new Error(`TransformerBlock output shape mismatch: ${output.shape.toString()}`);
    }
    for (let i = 0; i < output.shape.numel; i++) {
      if (isNaN(output.data[i]) || !isFinite(output.data[i])) {
        throw new Error(`TransformerBlock output contains NaN/Inf`);
      }
    }
  });

  // 9. Tokenizer: Byte-level encode and decode round-trip
  await test('Tokenizer: Byte-level UTF-8 encoding and decoding round-trip', 'validation', () => {
    const tokenizer = new NexaTokenizer();
    const testStrings = [
      'Hello, NEXA AI Core Engine!',
      'Multi-language: 简体中文 日本語 한국어',
      'Special chars: \n\t { "key": [1, 2, 3] } <|end|>'
    ];

    for (const text of testStrings) {
      const tokens = tokenizer.encode(text, false, false);
      if (tokens.length === 0) throw new Error(`Empty token array for text: "${text}"`);
      const decoded = tokenizer.decode(tokens);
      if (decoded !== text) {
        throw new Error(`Tokenizer round-trip mismatch!\nExpected: "${text}"\nGot:      "${decoded}"`);
      }
    }
  });

  // 10. .nexa Binary Format: Header and Descriptors Encode/Decode
  await test('Model Format: .nexa binary header serialization and validation', 'validation', () => {
    const config: NexaHeaderConfig = {
      vocabSize: 32000,
      dim: 4096,
      intermediateDim: 11008,
      nLayers: 32,
      nHeads: 32,
      nKvHeads: 32,
      maxSeqLen: 4096,
      normEps: 1e-5,
      ropeTheta: 10000.0,
      quantType: 1, // FP16
      nTensors: 2
    };

    const tensors: NexaTensorMeta[] = [
      {
        name: 'model.embed_tokens.weight',
        dtype: 1,
        dims: [32000, 4096],
        offset: 1024,
        sizeBytes: 32000 * 4096 * 2
      },
      {
        name: 'model.layers.0.self_attn.q_proj.weight',
        dtype: 1,
        dims: [4096, 4096],
        offset: 1024 + 32000 * 4096 * 2,
        sizeBytes: 4096 * 4096 * 2
      }
    ];

    const encoded = encodeNexaHeader(config, tensors);
    const decoded = decodeNexaHeader(encoded);

    if (decoded.magic !== NEXA_MAGIC) throw new Error('Magic mismatch.');
    if (decoded.version !== NEXA_VERSION) throw new Error('Version mismatch.');
    if (decoded.config.dim !== 4096) throw new Error('Config dim mismatch.');
    if (decoded.config.nLayers !== 32) throw new Error('Config nLayers mismatch.');
    if (decoded.tensors.length !== 2) throw new Error('Tensor count mismatch.');
    if (decoded.tensors[0].name !== 'model.embed_tokens.weight') throw new Error('First tensor name mismatch.');
  });

  // 11. Model Loader: Memory budget calculation and compatibility
  await test('Model Loader: Host RAM budgeting and invariant verification', 'lifecycle', () => {
    const config: NexaHeaderConfig = {
      vocabSize: 32000,
      dim: 4096,
      intermediateDim: 11008,
      nLayers: 32,
      nHeads: 32,
      nKvHeads: 32,
      maxSeqLen: 4096,
      normEps: 1e-5,
      ropeTheta: 10000.0,
      quantType: 1, // FP16
      nTensors: 290
    };

    const invariants = NexaModelLoader.validateInvariants(config);
    if (!invariants.valid) throw new Error(`Invariant check failed: ${invariants.error}`);

    const budget = NexaModelLoader.calculateMemory(config);
    if (budget.totalRequiredMb <= 0 || budget.weightBytes <= 0) {
      throw new Error(`Invalid memory budget: ${JSON.stringify(budget)}`);
    }
  });

  // 12. NexaNativeAdapter: Complete Lifecycle and Execution
  await test('Native Adapter: Full lifecycle (load, execution, unload) without 3rd-party runtimes', 'inference', async () => {
    const adapter = new NexaNativeAdapter();
    const info = await adapter.getSystemInfo();
    if (!info.platform || !info.cpuModel) {
      throw new Error('System info telemetry failed.');
    }

    // Verify loading non-existent model rejects gracefully
    const dummyModel = {
      id: 'missing-native-model',
      name: 'Missing Model',
      architecture: 'transformer' as const,
      format: 'nexa' as const,
      parameterSize: '1B',
      quantization: 'fp16',
      contextWindow: 2048,
      weightPath: '/non/existent/path/weights.nexa',
      fileSizeBytes: 0,
      isInstalled: false,
      description: 'Test model.'
    };

    try {
      await adapter.loadModel(dummyModel, {});
      throw new Error('Adapter loaded non-existent model without error!');
    } catch (err: unknown) {
      const msg = (err as Error).message;
      if (!msg.includes('Model weight file does not exist on disk')) {
        throw new Error(`Expected missing file error, got: ${msg}`);
      }
    }

    // Create a real minimal .nexa model file on disk
    const testDir = path.resolve(process.cwd(), 'data', 'test_models');
    fs.mkdirSync(testDir, { recursive: true });
    const testNexaPath = path.resolve(testDir, 'tiny-native-test.nexa');

    const config: NexaHeaderConfig = {
      vocabSize: 256,
      dim: 64,
      intermediateDim: 128,
      nLayers: 2,
      nHeads: 4,
      nKvHeads: 4,
      maxSeqLen: 512,
      normEps: 1e-5,
      ropeTheta: 10000.0,
      quantType: 0, // FP32
      nTensors: 1
    };

    const tensors: NexaTensorMeta[] = [
      {
        name: 'model.embed_tokens.weight',
        dtype: 0,
        dims: [256, 64],
        offset: 1024,
        sizeBytes: 256 * 64 * 4
      }
    ];

    const binaryHeader = encodeNexaHeader(config, tensors);
    const weightBuffer = Buffer.alloc(binaryHeader.length + 256 * 64 * 4);
    binaryHeader.copy(weightBuffer, 0);

    fs.writeFileSync(testNexaPath, weightBuffer);

    const validNativeModel = {
      id: 'tiny-native-test',
      name: 'Tiny Native Test Model',
      architecture: 'transformer' as const,
      format: 'nexa' as const,
      parameterSize: '16K',
      quantization: 'fp32',
      contextWindow: 512,
      weightPath: testNexaPath,
      fileSizeBytes: weightBuffer.length,
      isInstalled: true,
      description: 'Minimal native model for verifying C++/TS runtime execution.'
    };

    try {
      // 1. Load Model
      const runtimeConfig = await adapter.loadModel(validNativeModel, { contextSize: 512, threads: 2 });
      if (runtimeConfig.modelId !== 'tiny-native-test') throw new Error('RuntimeConfig modelId mismatch.');

      // 2. Tokenize
      const tokens = await adapter.tokenize('Hello NEXA native!', validNativeModel);
      if (tokens.length === 0) throw new Error('Tokenizer produced empty tokens.');

      // 3. Chat Completion
      let streamTokens = '';
      const response = await adapter.generateChatCompletion(
        {
          model: 'tiny-native-test',
          messages: [{ role: 'user', content: 'Execute native inference test.' }],
          max_tokens: 16
        },
        validNativeModel,
        runtimeConfig,
        (chunk) => {
          streamTokens += chunk;
        }
      );

      if (!response.id.startsWith('chatcmpl-')) {
        throw new Error(`Unexpected response ID prefix: ${response.id}`);
      }
      if (!response.choices[0].message.content.includes('[NEXA NATIVE RUNTIME')) {
        throw new Error(`Native signature missing from output: ${response.choices[0].message.content}`);
      }
      if (response.usage.prompt_tokens === 0 || response.usage.completion_tokens === 0) {
        throw new Error('Token usage tracking failed.');
      }

      // 4. Unload Model
      await adapter.unloadModel('tiny-native-test');
    } finally {
      if (fs.existsSync(testNexaPath)) {
        fs.unlinkSync(testNexaPath);
      }
    }
  });

  return results;
}
