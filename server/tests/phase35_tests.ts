/**
 * NEXA AI Core Engine - Phase 3.5 Language Intelligence & Generalisation Tests
 * Verifies numerical gradients, checkpoint recovery, capacity benchmarking,
 * instruction-tuned training with prompt masking, and 50+ unseen question evaluation.
 */

import { NexaCppBridge } from '../native/cpp_bridge.js';
import { TestCaseResult } from './engine_tests.js';
import path from 'path';
import fs from 'fs';

export async function runPhase35IntegrationTests(): Promise<TestCaseResult[]> {
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
        details: 'Passed Phase 3.5 language intelligence & generalisation verification.'
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
  const phase35Dir = path.resolve(process.cwd(), 'data', 'datasets', 'phase35', 'train_val');
  const unseenFile = path.resolve(process.cwd(), 'data', 'datasets', 'phase35', 'unseen_test_qa.txt');
  const testBestCkpt = path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_phase35_test_best.nexa');
  const testNbt = path.resolve(process.cwd(), 'data', 'tokenizers', 'nexa_bpe_phase35_test.nbt');

  // 1. Native C++ Phase 3.5 Unit Tests (5/5)
  await test('Phase 3.5 C++ Unit Tests: Numerical gradients, checkpoint recovery, masking, unseen separation & best-val saving', 'validation', async () => {
    if (!bridge.isAvailable()) {
      throw new Error('C++ binary not available.');
    }
    const res = await bridge.runPhase35Tests();
    if (res.status !== 'ok' || res.tests_passed !== 5) {
      throw new Error(`Phase 3.5 C++ tests failed: status=${res.status}, passed=${res.tests_passed}/5`);
    }
  });

  // 2. Model Capacity & Hardware Memory Benchmark
  await test('Capacity Benchmark: Measure parameter counts, RAM estimates, and CPU step speed across 5 configs', 'inference', async () => {
    const bench = await bridge.benchmarkConfigs(768);
    if (bench.status !== 'ok' || !Array.isArray(bench.configs) || bench.configs.length < 4) {
      throw new Error(`Benchmark returned invalid payload: ${JSON.stringify(bench)}`);
    }
    for (const c of bench.configs) {
      if (c.params <= 0 || c.est_training_mb <= 0 || c.avg_step_ms <= 0) {
        throw new Error(`Invalid benchmark metrics for ${c.name}: ${JSON.stringify(c)}`);
      }
    }
  });

  // 3. Instruction-Tuned Training with Prompt Masking & Best Val Checkpoint
  await test('Instruction Tuning: Prompt-masked multi-domain training reduces train and validation loss', 'inference', async () => {
    if (!fs.existsSync(phase35Dir)) {
      throw new Error(`Phase 3.5 dataset directory missing at ${phase35Dir}`);
    }
    fs.mkdirSync(path.dirname(testBestCkpt), { recursive: true });
    fs.mkdirSync(path.dirname(testNbt), { recursive: true });

    const trainRes = await bridge.trainPhase35({
      datasetDir: phase35Dir,
      bestCheckpointPath: testBestCkpt,
      tokenizerPath: testNbt,
      steps: 25,
      dim: 64,
      layers: 2,
      heads: 2,
      vocabSize: 640,
      seqLen: 64,
      batchSize: 2,
      gradAccum: 1,
      lr: 0.003,
      seed: 42
    });

    if (trainRes.status !== 'ok') {
      throw new Error(`Phase 3.5 training failed: ${trainRes.error}`);
    }
    if (trainRes.final_train_loss >= trainRes.initial_train_loss) {
      throw new Error(`Train loss did not decrease: ${trainRes.initial_train_loss} -> ${trainRes.final_train_loss}`);
    }
    if (trainRes.best_val_loss >= trainRes.initial_val_loss) {
      throw new Error(`Validation loss did not improve: ${trainRes.initial_val_loss} -> ${trainRes.best_val_loss}`);
    }
    if (trainRes.masked_target_tokens <= 0 || trainRes.unmasked_target_tokens <= 0) {
      throw new Error('Prompt loss masking counts were invalid.');
    }
    if (!fs.existsSync(testBestCkpt) || !fs.existsSync(testNbt)) {
      throw new Error('Best checkpoint or BPE tokenizer file missing after Phase 3.5 training.');
    }
  });

  // 4. 50+ Unseen Test Question Evaluation & Disjoint Verification
  await test('Unseen Generalisation Eval: Evaluate 50+ disjoint unseen test questions cleanly in native C++', 'inference', async () => {
    if (!fs.existsSync(testBestCkpt)) {
      throw new Error(`Phase 3.5 checkpoint missing at ${testBestCkpt}`);
    }
    const evalRes = await bridge.evaluateUnseen({
      checkpointPath: testBestCkpt,
      tokenizerPath: testNbt,
      unseenQuestionsPath: unseenFile,
      trainValDir: phase35Dir,
      maxTokens: 24,
      temperature: 0.4,
      seed: 42
    });

    if (evalRes.status !== 'ok') {
      throw new Error(`Unseen evaluation failed: ${evalRes.error}`);
    }
    if (!evalRes.disjoint_verified) {
      throw new Error('Unseen test set overlapped with training/validation files!');
    }
    if (evalRes.total_questions < 50 || evalRes.samples.length !== evalRes.total_questions) {
      throw new Error(`Expected >= 50 unseen questions, got ${evalRes.total_questions}`);
    }
    if (evalRes.avg_latency_ms <= 0 || evalRes.avg_tokens_per_sec <= 0) {
      throw new Error('Invalid unseen inference timing metrics.');
    }
  });

  return results;
}
