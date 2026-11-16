/**
 * NEXA AI Core Engine - Cloud Run Deployment Readiness Tests
 * Verifies HTTP PORT/0.0.0.0 binding, /ready & /health probes, env-based API key auth,
 * RBAC scope enforcement, CORS origin allowlisting, startup auto-loading, and cold-start recovery.
 */

import http from 'http';
import { createNexaApp } from '../../server.js';
import { ConfigStore } from '../store/config.js';
import { KeyStore } from '../store/key_store.js';
import { ModelRegistry } from '../store/model_registry.js';
import { EngineManager } from '../engine/engine_manager.js';
import { TestCaseResult } from './engine_tests.js';

export async function runCloudRunDeploymentTests(): Promise<TestCaseResult[]> {
  const results: TestCaseResult[] = [];

  async function test(
    name: string,
    category: TestCaseResult['category'],
    fn: () => Promise<void>
  ) {
    const t0 = Date.now();
    try {
      await fn();
      results.push({
        name,
        category,
        passed: true,
        durationMs: Date.now() - t0,
        details: 'Passed Cloud Run deployment readiness verification.'
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

  const prevEnv = {
    NEXA_API_KEY: process.env.NEXA_API_KEY,
    REQUIRE_API_KEY: process.env.REQUIRE_API_KEY,
    ALLOWED_ORIGINS: process.env.ALLOWED_ORIGINS,
    ACTIVE_ADAPTER: process.env.ACTIVE_ADAPTER,
    AUTO_LOAD_MODELS: process.env.AUTO_LOAD_MODELS,
    AUTO_LOAD_ON_DEMAND: process.env.AUTO_LOAD_ON_DEMAND
  };

  process.env.NEXA_API_KEY = 'nexa_cloudrun_test_secret_token';
  process.env.REQUIRE_API_KEY = 'true';
  process.env.ALLOWED_ORIGINS = 'https://nexa-app.example.com,https://staging.nexa.local';
  process.env.ACTIVE_ADAPTER = 'nexa-native';
  process.env.AUTO_LOAD_MODELS = 'nexa-p36-tier3';
  process.env.AUTO_LOAD_ON_DEMAND = 'true';

  const registry = new ModelRegistry();
  const configStore = new ConfigStore();
  const keyStore = new KeyStore();
  const engineManager = new EngineManager(registry, configStore);
  const { app } = createNexaApp({ registry, configStore, keyStore, engineManager });

  const bootstrap = await engineManager.startupBootstrap();
  const server = http.createServer(app);

  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve());
  });

  const addr = server.address();
  const port = typeof addr === 'object' && addr ? addr.port : 0;
  const baseUrl = `http://127.0.0.1:${port}`;

  try {
    // 1. Startup checks & /ready probe
    await test('Cloud Run Probes: /ready & /health public accessibility and C++20 startup checks', 'api_compliance', async () => {
      if (!bootstrap.adapterAvailable || !bootstrap.dataDirWritable) {
        throw new Error(`Startup bootstrap failed checks: ${JSON.stringify(bootstrap)}`);
      }
      if (!bootstrap.autoLoadedModels.includes('nexa-p36-tier3')) {
        throw new Error(`Expected nexa-p36-tier3 to auto-load on startup: ${JSON.stringify(bootstrap)}`);
      }

      const readyRes = await fetch(`${baseUrl}/ready`);
      if (readyRes.status !== 200) {
        throw new Error(`Expected /ready HTTP 200, got ${readyRes.status}`);
      }
      const readyJson = await readyRes.json() as any;
      if (!readyJson.ready || readyJson.checks.loadedModelsCount < 1) {
        throw new Error(`/ready payload unexpected: ${JSON.stringify(readyJson)}`);
      }

      const healthRes = await fetch(`${baseUrl}/health`);
      if (healthRes.status !== 200) {
        throw new Error(`Expected /health HTTP 200, got ${healthRes.status}`);
      }
    });

    // 2. Strict Bearer Auth & Admin Route Lockdown
    await test('Cloud Run Auth: All /v1/* endpoints require env NEXA_API_KEY and enforce scopes', 'security', async () => {
      const unauthEndpoints = ['/v1/models', '/v1/engine/status', '/v1/engine/metrics', '/v1/keys'];
      for (const ep of unauthEndpoints) {
        const res = await fetch(`${baseUrl}${ep}`);
        if (res.status !== 401) {
          throw new Error(`Expected 401 unauthenticated for ${ep}, got ${res.status}`);
        }
      }

      const authModels = await fetch(`${baseUrl}/v1/models`, {
        headers: { Authorization: 'Bearer nexa_cloudrun_test_secret_token' }
      });
      if (authModels.status !== 200) {
        throw new Error(`Env NEXA_API_KEY rejected on /v1/models: HTTP ${authModels.status}`);
      }

      // Verify scoped key blocks admin endpoints with 403
      const { record, rawKey } = keyStore.createKey('Readonly Inference Client', ['chat:completions', 'models:read']);
      try {
        const allowedRes = await fetch(`${baseUrl}/v1/models`, {
          headers: { Authorization: `Bearer ${rawKey}` }
        });
        if (allowedRes.status !== 200) {
          throw new Error(`Scoped key should access /v1/models, got ${allowedRes.status}`);
        }

        const deniedAdmin = await fetch(`${baseUrl}/v1/keys`, {
          headers: { Authorization: `Bearer ${rawKey}` }
        });
        if (deniedAdmin.status !== 403) {
          throw new Error(`Scoped key without engine:admin should receive 403 on /v1/keys, got ${deniedAdmin.status}`);
        }
      } finally {
        keyStore.revokeKey(record.id);
      }
    });

    // 3. CORS Origin Allowlisting
    await test('Cloud Run CORS: Allowlisted NEXA AI origin permitted; untrusted origin denied', 'security', async () => {
      const allowedOpt = await fetch(`${baseUrl}/v1/models`, {
        method: 'OPTIONS',
        headers: { Origin: 'https://nexa-app.example.com' }
      });
      if (allowedOpt.status !== 204) {
        throw new Error(`Expected 204 for allowed CORS origin, got ${allowedOpt.status}`);
      }
      if (allowedOpt.headers.get('access-control-allow-origin') !== 'https://nexa-app.example.com') {
        throw new Error(`Expected matched CORS origin header, got ${allowedOpt.headers.get('access-control-allow-origin')}`);
      }

      const deniedOpt = await fetch(`${baseUrl}/v1/models`, {
        method: 'OPTIONS',
        headers: { Origin: 'https://untrusted-site.example.org' }
      });
      if (deniedOpt.status !== 403) {
        throw new Error(`Expected 403 for untrusted CORS origin, got ${deniedOpt.status}`);
      }
      if (deniedOpt.headers.get('access-control-allow-origin')) {
        throw new Error('Untrusted origin received Access-Control-Allow-Origin header!');
      }
    });

    // 4. Auto-loaded BPE Native Inference & Cold-Start Auto-Recovery
    await test('Cloud Run Recovery: Auto-loaded BPE completion + lazy cold-start load on nexa-nano-step-40', 'inference', async () => {
      const chatRes = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer nexa_cloudrun_test_secret_token'
        },
        body: JSON.stringify({
          model: 'nexa-p36-tier3',
          messages: [{ role: 'user', content: 'What is a binary search algorithm?' }],
          max_tokens: 16,
          temperature: 0.4
        })
      });
      if (chatRes.status !== 200) {
        throw new Error(`Expected 200 from auto-loaded nexa-p36-tier3, got ${chatRes.status}: ${await chatRes.text()}`);
      }
      const chatJson = await chatRes.json() as any;
      if (!chatJson.id?.startsWith('chatcmpl-nexa-cxx-')) {
        throw new Error(`Unexpected response ID: ${chatJson.id}`);
      }
      const replyContent = chatJson.choices?.[0]?.message?.content;
      if (typeof replyContent !== 'string' || replyContent.trim().length === 0) {
        throw new Error('Expected non-empty generated response from nexa-p36-tier3.');
      }
      if (replyContent.includes('C++20 COMPILED ENGINE')) {
        throw new Error('v2 checkpoint chat completion leaked internal telemetry notice.');
      }
      if (!chatJson.usage?.prompt_tokens || chatJson.usage.prompt_tokens <= 0) {
        throw new Error('Expected positive prompt_tokens usage metadata.');
      }

      // Test on-demand load recovery after cold container start on nexa-nano-step-40 (not in AUTO_LOAD_MODELS)
      const nanoRecoveryRes = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer nexa_cloudrun_test_secret_token'
        },
        body: JSON.stringify({
          model: 'nexa-nano-step-40',
          messages: [{ role: 'user', content: 'Hello NEXA' }],
          max_tokens: 12
        })
      });
      if (nanoRecoveryRes.status !== 200) {
        throw new Error(`Cold-start recovery failed for nexa-nano-step-40: HTTP ${nanoRecoveryRes.status}: ${await nanoRecoveryRes.text()}`);
      }
    });
  } finally {
    await engineManager.shutdown();
    await new Promise<void>((resolve) => server.close(() => resolve()));

    // Restore environment variables cleanly
    for (const [k, v] of Object.entries(prevEnv)) {
      if (v === undefined) {
        delete process.env[k];
      } else {
        process.env[k] = v;
      }
    }
  }

  return results;
}
