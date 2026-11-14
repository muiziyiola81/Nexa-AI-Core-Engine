/**
 * NEXA AI Core Engine - Native Training Engine Tests (Phase 3.2)
 * Verifies native C++20 autograd, AdamW optimizer, and NEXA-1 Nano training:
 * 1. Autograd unit tests (Matmul backward, RMSNorm backward, Softmax & Cross-entropy backward)
 * 2. Optimizer unit tests (AdamW first/second moment updates & weight descent)
 * 3. NEXA-1 Nano end-to-end training loop (random initialization, loss convergence)
 * 4. Weight update verification (||ΔW|| > 0)
 * 5. Checkpoint serialization into .nexa format with optimizer state
 * 6. Reload invariance verification (bit-identical logits upon reloading)
 * 7. Theoretical initial entropy validation (-ln(1/V) ≈ 5.545 for V=256)
 */

import { NexaCppBridge } from '../native/cpp_bridge.js';
import { TestCaseResult } from './engine_tests.js';
import fs from 'fs';

export async function runTrainingIntegrationTests(): Promise<TestCaseResult[]> {
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
        details: 'Passed native C++ training verification.'
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

  // 1. Native C++ Autograd & Optimizer Mathematical Verification
  await test('Training Engine: Native C++ autograd math verification (Matmul, RMSNorm, CrossEntropy, AdamW)', 'validation', async () => {
    if (!bridge.isAvailable()) {
      throw new Error('C++ binary not available. Build native first.');
    }

    const testRes = await bridge.runTrainingTests();
    if (testRes.status !== 'ok') {
      throw new Error(`Training unit tests failed with status: ${testRes.status}`);
    }
    if (testRes.tests_passed !== 4) {
      throw new Error(`Expected 4 autograd unit tests passed in C++, got ${testRes.tests_passed}`);
    }
  });

  // 2. NEXA-1 Nano Training Execution & Convergence
  let trainingRun: Awaited<ReturnType<typeof bridge.trainNano>> | null = null;
  await test('NEXA-1 Nano: End-to-end native training loop converges with loss reduction', 'inference', async () => {
    // Train for 40 steps on synthetic sequences
    trainingRun = await bridge.trainNano(40, 0.003);

    if (trainingRun.status !== 'ok') {
      throw new Error(`Training run failed with status: ${trainingRun.status}`);
    }
    if (trainingRun.total_steps !== 40) {
      throw new Error(`Expected 40 steps, got ${trainingRun.total_steps}`);
    }

    // Check that loss decreased significantly (at least 20%)
    if (trainingRun.final_loss >= trainingRun.initial_loss) {
      throw new Error(`Training failed to reduce loss: initial ${trainingRun.initial_loss}, final ${trainingRun.final_loss}`);
    }
    if (trainingRun.loss_reduction_pct < 20) {
      throw new Error(`Expected at least 20% loss reduction, got ${trainingRun.loss_reduction_pct.toFixed(2)}%`);
    }
  });

  // 3. Weight Delta & Parameter Mutation Verification
  await test('NEXA-1 Nano: Weights genuinely update from random initialization (||ΔW|| > 0)', 'validation', async () => {
    if (!trainingRun) {
      throw new Error('Training run was not executed.');
    }

    if (!trainingRun.weights_updated) {
      throw new Error('Weights were not updated during training.');
    }
    if (trainingRun.weight_delta_norm <= 0.01) {
      throw new Error(`Weight delta norm too small: ${trainingRun.weight_delta_norm}`);
    }
  });

  // 4. Theoretical Initial Cross-Entropy Loss Validation
  await test('NEXA-1 Nano: Initial loss matches theoretical random entropy for V=256 (~5.545)', 'validation', async () => {
    if (!trainingRun) {
      throw new Error('Training run was not executed.');
    }

    // Theoretical cross entropy for 256 uniform classes is -ln(1/256) ≈ 5.54518
    const theoreticalEntropy = -Math.log(1 / 256);
    const diff = Math.abs(trainingRun.initial_loss - theoreticalEntropy);
    if (diff > 0.5) {
      throw new Error(`Initial loss ${trainingRun.initial_loss} differs too much from theoretical entropy ${theoreticalEntropy.toFixed(4)}`);
    }
  });

  // 5. Checkpoint Persistence and Reload Invariance
  await test('NEXA-1 Nano: Checkpoint serialization & reloaded model bit-identical logits', 'lifecycle', async () => {
    if (!trainingRun) {
      throw new Error('Training run was not executed.');
    }

    if (!trainingRun.checkpoint_verified) {
      throw new Error('Checkpoint file could not be verified on disk.');
    }
    if (!trainingRun.reloaded_output_identical) {
      throw new Error('Reloaded model produced different logits than trained model!');
    }
    if (!fs.existsSync(trainingRun.checkpoint_path)) {
      throw new Error(`Checkpoint file missing from disk: ${trainingRun.checkpoint_path}`);
    }
    const stat = fs.statSync(trainingRun.checkpoint_path);
    if (stat.size < 1000) {
      throw new Error(`Checkpoint file too small (${stat.size} bytes)`);
    }
  });

  return results;
}
