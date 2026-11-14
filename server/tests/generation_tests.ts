/**
 * NEXA AI Core Engine - Phase 3.3 Generation & Pipeline Integration Tests
 * Verifies real native autoregressive text generation, checkpoint loading,
 * temperature/top-k/top-p sampling, dataset pipeline, and evaluation metrics:
 * 1. Native C++ generation test suite (5/5 unit tests)
 * 2. .nexa checkpoint loading, tensor validation, and architecture verification
 * 3. Rejection of corrupted, truncated, or incompatible model checkpoints
 * 4. Autoregressive token-by-token text generation on CPU using trained weights
 * 5. Sampling reproducibility with fixed random seed
 * 6. Dataset preparation pipeline (UTF-8 validation, 80/20 train/val split)
 * 7. End-to-end dataset training loop (Loss decrease, Perplexity decrease, Checkpoint generation)
 */

import { NexaCppBridge } from '../native/cpp_bridge.js';
import { TestCaseResult } from './engine_tests.js';
import path from 'path';
import fs from 'fs';

export async function runGenerationIntegrationTests(): Promise<TestCaseResult[]> {
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
        details: 'Passed native generation verification.'
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
  const step40Checkpoint = path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_nano_step_40.nexa');
  const corpusDataset = path.resolve(process.cwd(), 'data', 'datasets', 'nexa_corpus.txt');

  // 1. Native C++ Unit Tests (All 5 Phase 3.3 tests in C++)
  await test('Generation Engine: 5 native C++ unit tests passed (validation, sampling, greedy, dataset, perplexity)', 'validation', async () => {
    if (!bridge.isAvailable()) {
      throw new Error('C++ binary not available.');
    }
    const res = await bridge.runGenerationTests();
    if (res.status !== 'ok' || res.tests_passed !== 5) {
      throw new Error(`Generation tests failed: status=${res.status}, passed=${res.tests_passed}/5`);
    }
  });

  // 2. Checkpoint Loading & Architecture Validation
  await test('Checkpoint Loader: Validate architecture, vocab, layers, heads and weight tensors', 'validation', async () => {
    if (!fs.existsSync(step40Checkpoint)) {
      throw new Error(`Step 40 checkpoint missing at ${step40Checkpoint}`);
    }
    const val = await bridge.validateCheckpoint(step40Checkpoint);
    if (!val.valid) {
      throw new Error(`Checkpoint validation failed: ${val.error}`);
    }
    if (val.version !== 2) {
      throw new Error(`Expected format version 2, got ${val.version}`);
    }
    if (val.vocab_size !== 256 || val.dim !== 64 || val.layers !== 2 || val.heads !== 2) {
      throw new Error(`Unexpected architecture in checkpoint: ${JSON.stringify(val)}`);
    }
    if (val.tensors !== 22) {
      throw new Error(`Expected 22 parameter tensors, got ${val.tensors}`);
    }
    if (val.weight_bytes <= 0) {
      throw new Error(`Invalid total weight bytes: ${val.weight_bytes}`);
    }
  });

  // 3. Rejection of Corrupted or Tampered Checkpoints
  await test('Checkpoint Security: Strictly reject corrupt magic bytes and truncated files', 'security', async () => {
    const corruptFile = path.resolve(process.cwd(), 'data', 'test_models', 'node_test_corrupt.nexa');
    fs.mkdirSync(path.dirname(corruptFile), { recursive: true });

    // Write bogus header
    fs.writeFileSync(corruptFile, Buffer.from([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66]));
    try {
      const val = await bridge.validateCheckpoint(corruptFile);
      if (val.valid) {
        throw new Error('Bridge failed to reject file with bogus magic bytes and truncated length!');
      }
    } finally {
      if (fs.existsSync(corruptFile)) fs.unlinkSync(corruptFile);
    }
  });

  // 4. Native Autoregressive Generation on CPU
  await test('Autoregressive Inference: Native C++ token-by-token generation from trained checkpoint', 'inference', async () => {
    const prompt = 'NEXA Engine';
    const gen = await bridge.generateText(step40Checkpoint, prompt, {
      maxTokens: 16,
      temperature: 0.7,
      topK: 30,
      topP: 0.9,
      repetitionPenalty: 1.15,
      seed: 42
    });

    if (gen.status !== 'ok') {
      throw new Error(`Generation failed: ${gen.error}`);
    }
    if (gen.tokens_generated !== 16) {
      throw new Error(`Expected 16 tokens generated, got ${gen.tokens_generated}`);
    }
    if (gen.duration_ms <= 0) {
      throw new Error(`Invalid duration: ${gen.duration_ms} ms`);
    }
    if (gen.tokens_per_second <= 10.0) {
      throw new Error(`Inference speed suspiciously low: ${gen.tokens_per_second} tokens/sec`);
    }
    if (!gen.full_text.startsWith(prompt)) {
      throw new Error(`Full text does not start with prompt: ${gen.full_text}`);
    }
  });

  // 5. Sampling Reproducibility with Fixed Seed
  await test('Sampling Reproducibility: Identical seeds produce bit-identical token sequences', 'validation', async () => {
    const prompt = 'TestSeed';
    const seed = 54321;
    const run1 = await bridge.generateText(step40Checkpoint, prompt, { maxTokens: 12, temperature: 0.8, seed });
    const run2 = await bridge.generateText(step40Checkpoint, prompt, { maxTokens: 12, temperature: 0.8, seed });

    if (run1.generated_text !== run2.generated_text) {
      throw new Error(`Non-deterministic output with seed ${seed}: "${run1.generated_text}" vs "${run2.generated_text}"`);
    }
  });

  // 6. Dataset Preparation & End-to-End Pipeline Training
  await test('Training Pipeline: End-to-end dataset loading, 80/20 train/val split, and perplexity reduction', 'lifecycle', async () => {
    if (!fs.existsSync(corpusDataset)) {
      throw new Error(`Training dataset missing at ${corpusDataset}`);
    }

    const trainedCkpt = path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_nano_test_pipeline.nexa');
    const trainRes = await bridge.trainDatasetPipeline(corpusDataset, {
      steps: 30,
      lr: 0.003,
      seqLen: 16,
      split: 0.8,
      checkpointPath: trainedCkpt
    });

    if (trainRes.status !== 'ok') {
      throw new Error(`Dataset training failed: ${trainRes.error}`);
    }
    if (trainRes.final_train_loss >= trainRes.initial_train_loss) {
      throw new Error(`Train loss did not decrease: ${trainRes.initial_train_loss} -> ${trainRes.final_train_loss}`);
    }
    if (trainRes.final_perplexity >= trainRes.initial_perplexity) {
      throw new Error(`Validation perplexity did not decrease: ${trainRes.initial_perplexity} -> ${trainRes.final_perplexity}`);
    }
    if (!trainRes.weights_updated || trainRes.weight_delta_norm <= 0.01) {
      throw new Error(`Weights were not updated during dataset training: delta=${trainRes.weight_delta_norm}`);
    }
    if (!fs.existsSync(trainedCkpt)) {
      throw new Error(`Checkpoint was not saved to ${trainedCkpt}`);
    }

    // Verify generated text from newly trained checkpoint
    const gen = await bridge.generateText(trainedCkpt, 'The NEXA', { maxTokens: 16, temperature: 0.0 });
    if (gen.status !== 'ok' || gen.tokens_generated <= 0) {
      throw new Error(`Failed generating from newly trained pipeline checkpoint`);
    }
  });

  return results;
}
