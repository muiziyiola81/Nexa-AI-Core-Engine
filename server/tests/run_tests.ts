/**
 * NEXA AI Core Engine - CLI Test Runner
 * Run via: npm test
 */

import { EngineManager } from '../engine/engine_manager.js';
import { ModelRegistry } from '../store/model_registry.js';
import { KeyStore } from '../store/key_store.js';
import { ConfigStore } from '../store/config.js';
import { runAutomatedTests } from './engine_tests.js';

async function main() {
  console.log('====================================================');
  console.log('   NEXA AI CORE ENGINE - AUTOMATED TEST SUITE       ');
  console.log('====================================================\n');

  const registry = new ModelRegistry();
  const configStore = new ConfigStore();
  const keyStore = new KeyStore();
  const engineManager = new EngineManager(registry, configStore);

  try {
    const report = await runAutomatedTests({
      engineManager,
      registry,
      keyStore,
      configStore
    });

    console.log(`Executed ${report.total} tests in ${report.durationMs}ms\n`);

    for (const r of report.results) {
      const mark = r.passed ? '✓ PASS' : '✗ FAIL';
      console.log(`[${mark}] [${r.category.toUpperCase()}] ${r.name} (${r.durationMs}ms)`);
      if (!r.passed && r.error) {
        console.error(`       Error: ${r.error}`);
      }
    }

    console.log('\n----------------------------------------------------');
    console.log(`Summary: ${report.passed}/${report.total} Passed, ${report.failed} Failed`);
    console.log('----------------------------------------------------\n');

    if (report.failed > 0) {
      process.exit(1);
    }
  } catch (err) {
    console.error('Fatal test runner failure:', err);
    process.exit(1);
  }
}

main();
