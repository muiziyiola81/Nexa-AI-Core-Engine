/**
 * NEXA AI Core Engine - Phase 3.4 Language Understanding & QA Integration Tests
 * Verifies trainable BPE tokenization, multi-file dataset ingestion/deduplication,
 * warmup + cosine LR decay, mini-batch gradient accumulation, and real autoregressive QA inference.
 */

import { NexaCppBridge } from '../native/cpp_bridge.js';
import { TestCaseResult } from './engine_tests.js';
import path from 'path';
import fs from 'fs';

export async function runQaIntegrationTests(): Promise<TestCaseResult[]> {
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
        details: 'Passed Phase 3.4 language understanding verification.'
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
  const qaDatasetDir = path.resolve(process.cwd(), 'data', 'datasets', 'qa');
  const testBpePath = path.resolve(process.cwd(), 'data', 'test_models', 'node_test_qa.nbt');
  const qaCkptPath = path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_nano_test_qa.nexa');
  const qaNbtPath = path.resolve(process.cwd(), 'data', 'tokenizers', 'nexa_bpe_test_qa.nbt');

  // 1. Native C++ Phase 3.4 Unit Tests (5/5)
  await test('Phase 3.4 C++ Unit Tests: BPE training, .nbt persistence, multi-file deduplication, LR warmup/decay, gradient scaling', 'validation', async () => {
    if (!bridge.isAvailable()) {
      throw new Error('C++ binary not available.');
    }
    const res = await bridge.runPhase34Tests();
    if (res.status !== 'ok' || res.tests_passed !== 5) {
      throw new Error(`Phase 3.4 C++ tests failed: status=${res.status}, passed=${res.tests_passed}/5`);
    }
  });

  // 2. Trainable BPE Tokenizer Bridge Execution
  await test('BPE Tokenizer: Train BPE merges on QA dataset directory and persist .nbt vocabulary', 'lifecycle', async () => {
    if (!fs.existsSync(qaDatasetDir)) {
      throw new Error(`QA dataset directory missing at ${qaDatasetDir}`);
    }
    fs.mkdirSync(path.dirname(testBpePath), { recursive: true });
    if (fs.existsSync(testBpePath)) fs.unlinkSync(testBpePath);

    const bpeRes = await bridge.trainBpe(qaDatasetDir, testBpePath, 320, 2);
    if (bpeRes.status !== 'ok') {
      throw new Error(`BPE training failed: ${bpeRes.error}`);
    }
    if (bpeRes.vocab_size < 280 || bpeRes.merges_count <= 0) {
      throw new Error(`BPE did not expand vocab as expected: vocab=${bpeRes.vocab_size}, merges=${bpeRes.merges_count}`);
    }
    if (!fs.existsSync(testBpePath)) {
      throw new Error(`BPE .nbt file was not written to ${testBpePath}`);
    }
    fs.unlinkSync(testBpePath);
  });

  // 3. Multi-File QA Training with Mini-Batch Gradient Accumulation & Warmup/Cosine Decay
  await test('QA Training Engine: Multi-file QA training reduces loss and perplexity via BPE & AdamW', 'inference', async () => {
    fs.mkdirSync(path.dirname(qaCkptPath), { recursive: true });
    fs.mkdirSync(path.dirname(qaNbtPath), { recursive: true });

    const trainRes = await bridge.trainQa({
      datasetDir: qaDatasetDir,
      checkpointPath: qaCkptPath,
      tokenizerPath: qaNbtPath,
      steps: 40,
      batchSize: 2,
      gradAccum: 1,
      lr: 0.003,
      warmupSteps: 8,
      seqLen: 32
    });

    if (trainRes.status !== 'ok') {
      throw new Error(`QA training failed: ${trainRes.error}`);
    }
    if (trainRes.final_train_loss >= trainRes.initial_train_loss) {
      throw new Error(`QA train loss did not decrease: ${trainRes.initial_train_loss} -> ${trainRes.final_train_loss}`);
    }
    if (trainRes.final_perplexity >= trainRes.initial_perplexity) {
      throw new Error(`QA validation perplexity did not decrease: ${trainRes.initial_perplexity} -> ${trainRes.final_perplexity}`);
    }
    if (trainRes.weight_delta_norm <= 0.01) {
      throw new Error(`Weights did not update meaningfully: delta=${trainRes.weight_delta_norm}`);
    }
    if (!fs.existsSync(qaCkptPath) || !fs.existsSync(qaNbtPath)) {
      throw new Error('QA checkpoint or BPE tokenizer file missing after training.');
    }
  });

  // 4. Autoregressive QA Inference & Seed Reproducibility
  await test('QA Inference: Native C++ autoregressive question answering with reproducible fixed seed', 'inference', async () => {
    if (!fs.existsSync(qaCkptPath)) {
      throw new Error(`QA checkpoint missing at ${qaCkptPath}`);
    }

    const q = 'What is AI?';
    const ans1 = await bridge.answerQuestion(qaCkptPath, q, {
      tokenizerPath: qaNbtPath,
      maxTokens: 20,
      temperature: 0.5,
      seed: 42
    });
    const ans2 = await bridge.answerQuestion(qaCkptPath, q, {
      tokenizerPath: qaNbtPath,
      maxTokens: 20,
      temperature: 0.5,
      seed: 42
    });

    if (ans1.status !== 'ok') {
      throw new Error(`QA answer generation failed: ${ans1.error}`);
    }
    if (ans1.tokens_generated <= 0 || ans1.duration_ms <= 0) {
      throw new Error(`Invalid QA generation metrics: tokens=${ans1.tokens_generated}, ms=${ans1.duration_ms}`);
    }
    if (ans1.answer !== ans2.answer) {
      throw new Error(`Non-deterministic QA generation with fixed seed: "${ans1.answer}" vs "${ans2.answer}"`);
    }
  });

  return results;
}
