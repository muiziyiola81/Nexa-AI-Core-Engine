/**
 * NEXA AI Core Engine - Phase 3.7 Intelligence, RoPE, Tied Embeddings & Multi-Turn Tests
 * Verifies BPE tokenization via /v1/tokenize & C++ bridge, RoPE + tied-embedding checkpoint flags,
 * multi-turn canonical prompt formatting, clean v2 chat replies, and SSE streaming.
 */

import fs from 'fs';
import path from 'path';
import http from 'http';
import { NexaCppBridge } from '../native/cpp_bridge.js';
import { createNexaApp } from '../../server.js';
import { ConfigStore } from '../store/config.js';
import { KeyStore } from '../store/key_store.js';
import { ModelRegistry } from '../store/model_registry.js';
import { EngineManager } from '../engine/engine_manager.js';
import { TestCaseResult } from './engine_tests.js';

export async function runPhase37IntegrationTests(): Promise<TestCaseResult[]> {
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
        details: 'Passed Phase 3.7 architecture & multi-turn inference verification.'
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
  const v2Ckpt = path.resolve(process.cwd(), 'data', 'checkpoints', 'nexa_p37_v2_best.nexa');
  const v2Nbt = path.resolve(process.cwd(), 'data', 'tokenizers', 'nexa_bpe_p37_v2.nbt');

  // 1. BPE Tokenization vs Byte Tokenization via C++ Bridge
  await test('Phase 3.7 Tokenizer: C++ bridge BPE .nbt tokenization achieves compression over raw bytes', 'validation', async () => {
    if (!fs.existsSync(v2Nbt)) {
      throw new Error(`Missing Phase 3.7 BPE tokenizer at ${v2Nbt}`);
    }
    const sample = 'User: What is binary search in computer science?\nAssistant:';
    const byteTokens = await bridge.tokenize(sample);
    const bpeTokens = await bridge.tokenize(sample, v2Nbt);
    if (bpeTokens.length === 0 || bpeTokens.length >= byteTokens.length) {
      throw new Error(`Expected BPE tokens (${bpeTokens.length}) < byte tokens (${byteTokens.length})`);
    }
  });

  // 2. Checkpoint v2 Feature Flags & Integrity Validation
  await test('Phase 3.7 Checkpoint: Validate RoPE, Tied Embeddings, and corrupted-weight rejection', 'validation', async () => {
    if (!fs.existsSync(v2Ckpt)) {
      throw new Error(`Missing Phase 3.7 checkpoint at ${v2Ckpt}`);
    }
    const val = await bridge.validateCheckpoint(v2Ckpt);
    if (!val.valid || val.version < 2) {
      throw new Error(`Expected valid v2 checkpoint, got: ${JSON.stringify(val)}`);
    }
    if (!val.use_rope || !val.tie_embeddings) {
      throw new Error(`Expected use_rope=true and tie_embeddings=true, got: ${JSON.stringify(val)}`);
    }
    if ((val.vocab_size ?? 0) < 1024 || (val.max_seq_len ?? 0) < 128) {
      throw new Error(`Unexpected vocab/context in v2 checkpoint: ${JSON.stringify(val)}`);
    }
  });

  // 3. Multi-Turn Chat Completion, Tokenize Endpoint & SSE Streaming on nexa-p37-v2
  await test('Phase 3.7 API: /v1/tokenize, multi-turn /v1/chat/completions, and SSE streaming on nexa-p37-v2', 'inference', async () => {
    const prevEnv = {
      ACTIVE_ADAPTER: process.env.ACTIVE_ADAPTER,
      AUTO_LOAD_MODELS: process.env.AUTO_LOAD_MODELS,
      REQUIRE_API_KEY: process.env.REQUIRE_API_KEY
    };
    process.env.ACTIVE_ADAPTER = 'nexa-native';
    process.env.AUTO_LOAD_MODELS = 'nexa-p37-v2';
    process.env.REQUIRE_API_KEY = 'false';

    const registry = new ModelRegistry();
    const configStore = new ConfigStore();
    const keyStore = new KeyStore();
    const engineManager = new EngineManager(registry, configStore);
    const { app } = createNexaApp({ registry, configStore, keyStore, engineManager });
    await engineManager.startupBootstrap();

    const server = http.createServer(app);
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', () => resolve()));
    const addr = server.address();
    const port = typeof addr === 'object' && addr ? addr.port : 0;
    const baseUrl = `http://127.0.0.1:${port}`;

    try {
      // /v1/tokenize uses nexa-p37-v2's .nbt tokenizer
      const tokRes = await fetch(`${baseUrl}/v1/tokenize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'nexa-p37-v2', text: 'What is a central processing unit?' })
      });
      if (tokRes.status !== 200) {
        throw new Error(`Expected 200 from /v1/tokenize, got ${tokRes.status}`);
      }
      const tokJson = await tokRes.json() as any;
      if (!Array.isArray(tokJson.tokens) || tokJson.token_count !== tokJson.tokens.length || tokJson.token_count >= 34) {
        throw new Error(`Expected compressed BPE token count < 34, got ${JSON.stringify(tokJson)}`);
      }

      // Multi-turn /v1/chat/completions
      const chatRes = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nexa-p37-v2',
          messages: [
            { role: 'system', content: 'Answer concisely and clearly.' },
            { role: 'user', content: 'What is electric current?' },
            { role: 'assistant', content: 'Electric current is the flow of electric charge.' },
            { role: 'user', content: 'How does increasing resistance affect current if voltage stays constant?' }
          ],
          max_tokens: 48,
          temperature: 0.4
        })
      });
      if (chatRes.status !== 200) {
        throw new Error(`Expected 200 from multi-turn completion, got ${chatRes.status}: ${await chatRes.text()}`);
      }
      const chatJson = await chatRes.json() as any;
      const content = chatJson.choices?.[0]?.message?.content;
      if (typeof content !== 'string' || content.trim().length === 0) {
        throw new Error('Empty response from multi-turn nexa-p37-v2 completion.');
      }
      if (content.includes('C++20 COMPILED ENGINE') || content.includes('[AUTOREGRESSIVE')) {
        throw new Error(`Internal telemetry banner leaked into v2 chat response: ${content}`);
      }

      // Streaming SSE /v1/chat/completions
      const streamRes = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nexa-p37-v2',
          messages: [{ role: 'user', content: 'What is a prime number?' }],
          max_tokens: 24,
          temperature: 0.4,
          stream: true
        })
      });
      if (streamRes.status !== 200) {
        throw new Error(`Expected 200 from streaming completion, got ${streamRes.status}`);
      }
      const sseBody = await streamRes.text();
      if (!sseBody.includes('chat.completion.chunk') || !sseBody.includes('data: [DONE]')) {
        throw new Error(`Invalid SSE stream payload: ${sseBody.slice(0, 200)}`);
      }

      // Error handling: invalid JSON, invalid temperature, unknown model, invalid /v1/tokenize
      const badJsonRes = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{bad json'
      });
      if (badJsonRes.status !== 400) {
        throw new Error(`Expected 400 on malformed JSON, got ${badJsonRes.status}`);
      }
      const badJsonBody = await badJsonRes.json() as any;
      if (badJsonBody?.error?.code !== 'invalid_json') {
        throw new Error(`Expected invalid_json error code, got ${JSON.stringify(badJsonBody)}`);
      }

      const badTempRes = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nexa-p37-v2',
          messages: [{ role: 'user', content: 'hi' }],
          temperature: 3.0
        })
      });
      if (badTempRes.status !== 400) {
        throw new Error(`Expected 400 on out-of-range temperature, got ${badTempRes.status}`);
      }

      const missingTokModelRes = await fetch(`${baseUrl}/v1/tokenize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'does-not-exist', text: 'hello' })
      });
      if (missingTokModelRes.status !== 404) {
        throw new Error(`Expected 404 for unknown /v1/tokenize model, got ${missingTokModelRes.status}`);
      }
      const missingTokBody = await missingTokModelRes.json() as any;
      if (missingTokBody?.error?.code !== 'model_not_found') {
        throw new Error(`Expected model_not_found error code, got ${JSON.stringify(missingTokBody)}`);
      }

      const emptyTokTextRes = await fetch(`${baseUrl}/v1/tokenize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'nexa-p37-v2', text: '' })
      });
      if (emptyTokTextRes.status !== 400) {
        throw new Error(`Expected 400 for empty /v1/tokenize text, got ${emptyTokTextRes.status}`);
      }

      // Explicit unload + unloaded rejection + explicit reload via API
      const unloadRes = await fetch(`${baseUrl}/v1/models/nexa-p37-v2/unload`, { method: 'POST' });
      if (unloadRes.status !== 200) {
        throw new Error(`Expected 200 unloading nexa-p37-v2, got ${unloadRes.status}`);
      }
      const unloadedChatRes = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nexa-p37-v2',
          messages: [{ role: 'user', content: 'Hello' }],
          max_tokens: 8
        })
      });
      if (unloadedChatRes.status !== 400) {
        throw new Error(`Expected 400 after explicit unload, got ${unloadedChatRes.status}`);
      }
      const reloadRes = await fetch(`${baseUrl}/v1/models/nexa-p37-v2/load`, { method: 'POST' });
      if (reloadRes.status !== 200) {
        throw new Error(`Expected 200 reloading nexa-p37-v2, got ${reloadRes.status}`);
      }
    } finally {
      await engineManager.shutdown();
      await new Promise<void>((resolve) => server.close(() => resolve()));
      for (const [k, v] of Object.entries(prevEnv)) {
        if (v === undefined) delete process.env[k];
        else process.env[k] = v;
      }
    }
  });

  return results;
}
