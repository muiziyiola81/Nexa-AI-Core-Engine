/**
 * NEXA AI Core Engine - Phase 3.6 Data Quality & Training Improvements Tests
 * Verifies managed dataset provenance & explicit license enforcement,
 * 8-way dataset quality checks, evaluation contamination rejection,
 * reproducible experiment logging, and expanded 100+ unseen evaluation.
 */

import { NexaCppBridge } from '../native/cpp_bridge.js';
import { TestCaseResult } from './engine_tests.js';
import path from 'path';
import fs from 'fs';

export async function runPhase36IntegrationTests(): Promise<TestCaseResult[]> {
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
        details: 'Passed Phase 3.6 data quality & training verification.'
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
  const tier1Dir = path.resolve(process.cwd(), 'data', 'datasets', 'phase36', 'tier1_small');
  const tier3Dir = path.resolve(process.cwd(), 'data', 'datasets', 'phase36', 'tier3_large');
  const expandedUnseen = path.resolve(process.cwd(), 'data', 'datasets', 'phase36', 'unseen_test_expanded.txt');

  // 1. Native C++ Phase 3.6 Unit Tests (6/6)
  await test('Phase 3.6 C++ Unit Tests: License safety, 8 quality checks, n-gram contamination, multi-turn masking & ExperimentManager', 'validation', async () => {
    if (!bridge.isAvailable()) {
      throw new Error('C++ binary not available.');
    }
    const res = await bridge.runPhase36Tests();
    if (res.status !== 'ok' || res.tests_passed !== 6) {
      throw new Error(`Phase 3.6 C++ tests failed: status=${res.status}, passed=${res.tests_passed}/6`);
    }
  });

  // 2. Dataset Audit & Unverified License Rejection via Node.js Bridge
  await test('Dataset Integrity & License Safety: Unverified datasets never labeled MIT; Tier 3 audit passes cleanly', 'validation', async () => {
    const tmpUnverified = path.resolve(process.cwd(), 'data', 'test_models', 'node_unverified_ds');
    fs.mkdirSync(tmpUnverified, { recursive: true });
    fs.writeFileSync(
      path.join(tmpUnverified, 'raw.txt'),
      'User: What is a queue?\nAssistant: A queue processes elements in first-in first-out order.\n\n' +
      'User: What is a stack?\nAssistant: A stack stores elements in last-in first-out order.\n',
      'utf-8'
    );

    const strictAudit = await bridge.auditDataset({
      datasetPath: tmpUnverified,
      requireVerifiedLicense: true
    });
    if (strictAudit.status === 'ok' || strictAudit.license_verified) {
      throw new Error('Audit allowed unverified dataset when requireVerifiedLicense=true!');
    }
    if (strictAudit.license !== 'UNVERIFIED') {
      throw new Error(`Expected license "UNVERIFIED", got "${strictAudit.license}"`);
    }

    if (!fs.existsSync(tier3Dir) || !fs.existsSync(expandedUnseen)) {
      throw new Error('Phase 3.6 Tier 3 dataset or expanded unseen file missing.');
    }
    const tier3Audit = await bridge.auditDataset({
      datasetPath: tier3Dir,
      unseenQuestionsPath: expandedUnseen,
      requireVerifiedLicense: true
    });
    if (tier3Audit.status !== 'ok' || !tier3Audit.license_verified || tier3Audit.contamination_count !== 0) {
      throw new Error(`Tier 3 audit failed: ${JSON.stringify(tier3Audit)}`);
    }
    if (tier3Audit.examples < 800 || tier3Audit.unique_tokens <= 0) {
      throw new Error(`Tier 3 scale below expectation: examples=${tier3Audit.examples}, unique_tokens=${tier3Audit.unique_tokens}`);
    }
  });

  // 3. Reproducible Managed Training & Experiment Registry JSON Persistence
  await test('Reproducible Phase 3.6 Training & Experiment Manager JSON Registry', 'inference', async () => {
    const ckpt = path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_phase36_test_best.nexa');
    const nbt = path.resolve(process.cwd(), 'data', 'tokenizers', 'nexa_bpe_phase36_test.nbt');
    const res = await bridge.trainPhase36({
      datasetPath: tier1Dir,
      bestCheckpointPath: ckpt,
      tokenizerPath: nbt,
      unseenQuestionsPath: expandedUnseen,
      experimentId: 'node_integration_phase36_test',
      steps: 20,
      dim: 64,
      layers: 2,
      heads: 2,
      vocabSize: 512,
      seqLen: 64,
      batchSize: 2,
      gradAccum: 1,
      lr: 0.003,
      seed: 42,
      requireVerifiedLicense: true
    });

    if (res.status !== 'ok') {
      throw new Error(`Phase 3.6 managed training failed: ${res.error}`);
    }
    if (!res.license_verified || res.contamination_count !== 0) {
      throw new Error('Managed training failed license verification or detected contamination.');
    }
    if (res.final_train_loss >= res.initial_train_loss || res.best_val_loss >= res.initial_val_loss) {
      throw new Error(`Loss did not improve: train ${res.initial_train_loss}->${res.final_train_loss}, val ${res.initial_val_loss}->${res.best_val_loss}`);
    }
    if (!fs.existsSync(res.experiment_path)) {
      throw new Error(`Experiment record file missing at ${res.experiment_path}`);
    }
    const expData = JSON.parse(fs.readFileSync(res.experiment_path, 'utf-8'));
    if (!expData.datasetInfo?.datasetVersion || !expData.evalSummary?.evaluated) {
      throw new Error('Experiment JSON missing datasetVersion or evalSummary.');
    }
  });

  // 4. Expanded 100+ Unseen Question Multi-Metric Evaluation
  await test('Expanded Unseen Eval (>=100 questions): Verify 0 contamination & separate multi-dimensional metrics', 'inference', async () => {
    const ckpt = path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_phase36_test_best.nexa');
    const nbt = path.resolve(process.cwd(), 'data', 'tokenizers', 'nexa_bpe_phase36_test.nbt');
    const evalRes = await bridge.evaluateUnseenV2({
      checkpointPath: ckpt,
      tokenizerPath: nbt,
      unseenQuestionsPath: expandedUnseen,
      trainValDir: path.join(tier3Dir, 'train_val'),
      maxTokens: 20,
      temperature: 0.4,
      seed: 42
    });

    if (evalRes.status !== 'ok' || !evalRes.disjoint_verified) {
      throw new Error(`Expanded unseen evaluation failed or detected overlap: ${evalRes.error}`);
    }
    if (evalRes.total_questions < 100 || evalRes.samples.length !== evalRes.total_questions) {
      throw new Error(`Expected >=100 unseen questions, got ${evalRes.total_questions}`);
    }
    if (
      typeof evalRes.avg_relevance_score !== 'number' ||
      typeof evalRes.avg_coherence_score !== 'number' ||
      typeof evalRes.avg_factual_accuracy_score !== 'number' ||
      typeof evalRes.avg_instruction_following_score !== 'number'
    ) {
      throw new Error('Missing separate multi-metric scores in unseen evaluation result.');
    }
  });

  return results;
}
