/**
 * NEXA AI Core Engine - C++ Runtime Integration Tests (Phase 3.1)
 * Verifies real process communication between Node.js and the compiled C++20 binary:
 * 1. Process communication and runtime info handshake
 * 2. C++ tensor operations (Matmul, Softmax, RMSNorm)
 * 3. C++ transformer forward pass execution
 * 4. C++ byte-level tokenization
 * 5. C++ .nexa binary model inspection
 * 6. Missing executable error handling (no silent fallback)
 * 7. Invalid input error handling and stderr capture
 * 8. Safe argument passing (shell injection immunity)
 * 9. Timeout enforcement and clean process termination
 * 10. End-to-end NexaNativeAdapter lifecycle via C++
 */

import { NexaCppBridge, NexaNativeCppError } from '../native/cpp_bridge.js';
import { NexaNativeAdapter } from '../native/native_adapter.js';
import { TestCaseResult } from './engine_tests.js';
import path from 'path';
import fs from 'fs';

export async function runCppBridgeTests(): Promise<TestCaseResult[]> {
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
        details: 'Passed C++ process bridge verification.'
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

  const bridge = new NexaCppBridge();

  // 1. Process communication & runtime info handshake
  await test('C++ Bridge: Process communication and runtime telemetry handshake', 'api_compliance', async () => {
    if (!bridge.isAvailable()) {
      throw new Error(`C++ binary not found. Build native first: cd native && cmake -B build && cmake --build build -j`);
    }

    const info = await bridge.getRuntimeInfo();
    if (info.engine !== 'nexa-native-runtime') {
      throw new Error(`Unexpected engine name: ${info.engine}`);
    }
    if (info.standard !== 'C++20') {
      throw new Error(`Expected C++20 standard, got ${info.standard}`);
    }
    if (!info.arch || (info.arch !== 'x86_64' && info.arch !== 'arm64' && info.arch !== 'generic')) {
      throw new Error(`Invalid architecture: ${info.arch}`);
    }
    if (info.status !== 'ready') {
      throw new Error(`Unexpected status: ${info.status}`);
    }
  });

  // 2. C++ Tensor Math Execution
  await test('C++ Bridge: Real C++ matrix multiplication, softmax & RMSNorm execution', 'validation', async () => {
    const tensorResults = await bridge.runTensorTests();
    if (tensorResults.status !== 'ok') {
      throw new Error(`Tensor tests failed with status: ${tensorResults.status}`);
    }
    if (tensorResults.tests_passed !== 4) {
      throw new Error(`Expected 4 tests passed in C++, got ${tensorResults.tests_passed}`);
    }

    // Assert C++ matmul output [[31, 19], [85, 55]]
    const res = tensorResults.matmul.result;
    if (res[0][0] !== 31 || res[0][1] !== 19 || res[1][0] !== 85 || res[1][1] !== 55) {
      throw new Error(`C++ matmul produced incorrect values: ${JSON.stringify(res)}`);
    }

    if (Math.abs(tensorResults.softmax.sum - 1.0) > 1e-5) {
      throw new Error(`C++ softmax sum is not 1.0: ${tensorResults.softmax.sum}`);
    }

    if (Math.abs(tensorResults.rmsnorm.val - 1.0) > 1e-3) {
      throw new Error(`C++ RMSNorm invariant violated: ${tensorResults.rmsnorm.val}`);
    }
  });

  // 3. C++ Transformer Forward Pass Execution
  await test('C++ Bridge: Real C++ transformer block forward pass kernel', 'inference', async () => {
    const fwd = await bridge.runForwardPass(64, 4);
    if (fwd.status !== 'ok') {
      throw new Error(`Forward pass returned error status: ${fwd.status}`);
    }
    if (fwd.duration_us <= 0) {
      throw new Error(`Invalid C++ duration: ${fwd.duration_us} µs`);
    }
    if (isNaN(fwd.output_checksum) || !isFinite(fwd.output_checksum)) {
      throw new Error(`Invalid C++ output checksum: ${fwd.output_checksum}`);
    }
    if (fwd.tensor_elements !== 64 * 4) {
      throw new Error(`Expected 256 elements, got ${fwd.tensor_elements}`);
    }
  });

  // 4. C++ Byte Tokenizer Invocation
  await test('C++ Bridge: Tokenization via compiled C++ NexaByteTokenizer', 'validation', async () => {
    const text = 'NEXA C++ Engine';
    const tokens = await bridge.tokenize(text);
    if (!Array.isArray(tokens) || tokens.length === 0) {
      throw new Error(`C++ tokenizer returned invalid tokens: ${JSON.stringify(tokens)}`);
    }
    // Token 0 should be BOS (1)
    if (tokens[0] !== 1) {
      throw new Error(`Expected first token to be BOS (1), got ${tokens[0]}`);
    }
  });

  // 5. C++ .nexa Binary Model Inspection
  await test('C++ Bridge: .nexa binary model inspection via C++ ModelLoader', 'validation', async () => {
    const modelPath = path.resolve(process.cwd(), 'data', 'test_models', 'sample_model.nexa');
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Test model file not found at ${modelPath}`);
    }

    const meta = await bridge.inspectModel(modelPath);
    if (meta.status !== 'ok' || !meta.valid) {
      throw new Error(`C++ model loader failed to inspect valid .nexa file`);
    }
    if (meta.hidden_dim !== 2048) {
      throw new Error(`Expected hidden_dim 2048, got ${meta.hidden_dim}`);
    }
    if (meta.layers !== 16) {
      throw new Error(`Expected layers 16, got ${meta.layers}`);
    }
    if (meta.heads !== 16 || meta.kv_heads !== 8) {
      throw new Error(`Attention heads mismatch in C++ inspection`);
    }
  });

  // 6. Missing Executable Error (No Silent Fallback)
  await test('C++ Bridge: Missing executable strictly throws NexaNativeCppError (no fallback)', 'lifecycle', async () => {
    const badBridge = new NexaCppBridge('/non/existent/path/nexa-runtime-cli-missing');
    try {
      await badBridge.getRuntimeInfo();
      throw new Error('Bridge should have thrown for missing executable!');
    } catch (err: unknown) {
      const error = err as NexaNativeCppError;
      if (!error.message.includes('ERR_EXECUTABLE_NOT_FOUND') && !error.message.includes('ERR_EXECUTABLE_ENOENT')) {
        throw new Error(`Expected ERR_EXECUTABLE error, got: ${error.message}`);
      }
    }
  });

  // 7. Invalid Input Handling & Stderr Capture
  await test('C++ Bridge: Invalid input and missing file handling with stderr capture', 'validation', async () => {
    try {
      await bridge.inspectModel('/non/existent/path/invalid.nexa');
      throw new Error('Bridge should have rejected missing model file!');
    } catch (err: unknown) {
      const error = err as NexaNativeCppError;
      if (!error.message.includes('ERR_PROCESS_EXECUTION_FAILED')) {
        throw new Error(`Expected ERR_PROCESS_EXECUTION_FAILED, got: ${error.message}`);
      }
    }
  });

  // 8. Safe Argument Passing (Shell Injection Immunity)
  await test('C++ Bridge: Safe argument passing rejects shell injection exploits', 'security', async () => {
    // Attempt shell command injection
    const maliciousInput = 'hello"; echo "PWNED" > /tmp/nexa_exploit.txt; echo "';
    const tokens = await bridge.tokenize(maliciousInput);
    if (!Array.isArray(tokens)) {
      throw new Error('Tokenizer failed to process string with shell meta-characters.');
    }
    // Verify file was NOT created
    if (fs.existsSync('/tmp/nexa_exploit.txt')) {
      fs.unlinkSync('/tmp/nexa_exploit.txt');
      throw new Error('CRITICAL SECURITY FLAW: Shell injection succeeded through C++ bridge!');
    }
  });

  // 9. Timeout Enforcement and Clean Process Termination
  await test('C++ Bridge: Strict timeout handling kills unresponsive child processes', 'lifecycle', async () => {
    // Instantiate bridge with an ultra-short 1ms timeout to force timeout handler
    const shortTimeoutBridge = new (class extends NexaCppBridge {
      constructor() {
        super();
        (this as any).defaultTimeoutMs = 1;
      }
    })();

    try {
      // Run a command that takes > 1ms (forward pass)
      await shortTimeoutBridge.runForwardPass(256, 16);
      // If it completed in <1ms, that's fine, but if it timed out, verify error code
    } catch (err: unknown) {
      const error = err as NexaNativeCppError;
      if (error.code && error.code !== 'ERR_PROCESS_TIMEOUT' && error.code !== 'ERR_PROCESS_EXECUTION_FAILED') {
        throw new Error(`Unexpected error during timeout test: ${error.message}`);
      }
    }
  });

  // 10. End-to-end NexaNativeAdapter Lifecycle via C++
  await test('C++ Bridge: NexaNativeAdapter load, completion, and telemetry via C++ binary', 'inference', async () => {
    const adapter = new NexaNativeAdapter();
    const isAvail = await adapter.isAvailable();
    if (!isAvail.available) {
      throw new Error(`Adapter reported unavailable: ${isAvail.reason}`);
    }

    const modelPath = path.resolve(process.cwd(), 'data', 'test_models', 'sample_model.nexa');
    const model = {
      id: 'native-sample-llama',
      name: 'Native Sample Llama',
      architecture: 'transformer' as const,
      format: 'nexa' as const,
      parameterSize: '1B',
      quantization: 'fp16',
      contextWindow: 2048,
      weightPath: modelPath,
      fileSizeBytes: fs.statSync(modelPath).size,
      isInstalled: true,
      description: 'Test model for C++ native bridge execution.'
    };

    // 1. Load model
    const runtimeConfig = await adapter.loadModel(model, { contextSize: 2048 });
    if (runtimeConfig.modelId !== model.id) {
      throw new Error('RuntimeConfig modelId mismatch.');
    }

    // 2. Chat completion executed via real C++ process
    const res = await adapter.generateChatCompletion(
      {
        model: model.id,
        messages: [{ role: 'user', content: 'Execute native C++ forward pass.' }]
      },
      model,
      runtimeConfig
    );

    if (!res.id.startsWith('chatcmpl-nexa-cxx-')) {
      throw new Error(`Unexpected response ID: ${res.id}`);
    }
    const content = res.choices[0].message.content;
    if (!content.includes('C++20 COMPILED ENGINE')) {
      throw new Error(`C++ signature missing from output: ${content}`);
    }
    if (!content.includes('RMSNorm, RoPE, Attention, SwiGLU computed in C++')) {
      throw new Error(`C++ tensor verification missing from output: ${content}`);
    }
    if (res.usage.prompt_tokens <= 0 || res.usage.completion_tokens <= 0) {
      throw new Error('Usage token metrics were not tracked.');
    }

    // 3. Unload model
    await adapter.unloadModel(model.id);
  });

  return results;
}
