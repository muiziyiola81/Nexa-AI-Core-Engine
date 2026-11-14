/**
 * NEXA AI Core Engine - API Routes
 * OpenAI-Compatible v1 Endpoints, Cloud Run Health/Readiness Probes,
 * & Scoped Infrastructure Management API.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { EngineManager } from '../engine/engine_manager.js';
import { ModelRegistry } from '../store/model_registry.js';
import { KeyStore } from '../store/key_store.js';
import { ConfigStore } from '../store/config.js';
import { createAuthMiddleware, AuthenticatedRequest } from '../middleware/auth.js';
import { createRateLimiter } from '../middleware/rate_limiter.js';
import { validateChatCompletionRequest, validateModelRegistration } from '../middleware/validator.js';
import { runAutomatedTests } from '../tests/engine_tests.js';

export function createApiRouter(
  engineManager: EngineManager,
  registry: ModelRegistry,
  keyStore: KeyStore,
  configStore: ConfigStore
): Router {
  const router = Router();

  const requireModelsRead = createAuthMiddleware(keyStore, configStore, 'models:read');
  const requireModelsManage = createAuthMiddleware(keyStore, configStore, 'models:manage');
  const requireChatCompletions = createAuthMiddleware(keyStore, configStore, 'chat:completions');
  const requireEngineAdmin = createAuthMiddleware(keyStore, configStore, 'engine:admin');
  const rateLimiter = createRateLimiter(configStore);

  // -------------------------------------------------------------
  // 1. HEALTH, READINESS & SYSTEM TELEMETRY (Public for Probes)
  // -------------------------------------------------------------
  const handleHealth = async (_req: Request, res: Response, next: NextFunction) => {
    try {
      const sysInfo = await engineManager.getSystemInfo();
      const metrics = engineManager.getMetrics();
      const loadedModels = engineManager.getAllModelStatuses().filter(m => m.state === 'loaded');
      const activeAdapter = engineManager.getActiveAdapter();
      const readiness = await engineManager.getReadinessStatus();

      res.json({
        status: readiness.ready ? 'healthy' : 'degraded',
        ready: readiness.ready,
        timestamp: new Date().toISOString(),
        engine: {
          name: 'NEXA AI Core Engine',
          version: '1.0.0',
          activeAdapter: {
            id: activeAdapter.id,
            name: activeAdapter.name,
            version: activeAdapter.version
          },
          loadedModelsCount: loadedModels.length,
          loadedModels: loadedModels.map(m => ({
            id: m.modelId,
            name: m.name,
            memoryMb: m.runtimeConfig?.memoryUsageMb,
            gpuLayers: m.runtimeConfig?.gpuLayers,
            contextSize: m.runtimeConfig?.contextSize
          }))
        },
        readinessChecks: readiness.checks,
        system: {
          platform: sysInfo.platform,
          arch: sysInfo.arch,
          cpu: sysInfo.cpuModel,
          cpuCores: sysInfo.cpuCores,
          totalMemoryMb: sysInfo.totalMemoryMb,
          freeMemoryMb: sysInfo.freeMemoryMb,
          gpuDetected: sysInfo.gpuDetected,
          gpuDescription: sysInfo.gpuDescription,
          executionMode: sysInfo.executionMode,
          llamaCppInstalled: sysInfo.llamaCppInstalled,
          llamaCppBinaryPath: sysInfo.llamaCppBinaryPath
        },
        metrics: {
          uptimeSeconds: metrics.uptimeSeconds,
          totalRequests: metrics.totalRequests,
          activeRequests: metrics.activeRequests,
          tokensPerSecond: metrics.tokensPerSecond
        }
      });
    } catch (err) {
      next(err);
    }
  };

  router.get('/health', handleHealth);
  router.get('/healthz', handleHealth);

  router.get('/ready', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      const readiness = await engineManager.getReadinessStatus();
      res.status(readiness.ready ? 200 : 503).json({
        status: readiness.ready ? 'ready' : 'not_ready',
        ...readiness
      });
    } catch (err) {
      next(err);
    }
  });

  // -------------------------------------------------------------
  // 2. OPENAI-COMPATIBLE: GET /v1/models
  // -------------------------------------------------------------
  router.get('/v1/models', requireModelsRead, rateLimiter, (_req: Request, res: Response) => {
    const models = registry.getAll();
    const statuses = engineManager.getAllModelStatuses();
    const statusMap = new Map(statuses.map(s => [s.modelId, s]));

    const response = {
      object: 'list',
      data: models.map(m => {
        const s = statusMap.get(m.id);
        return {
          id: m.id,
          object: 'model',
          created: 1710000000,
          owned_by: m.architecture,
          permission: [
            {
              id: `modelperm-${m.id}`,
              object: 'model_permission',
              created: 1710000000,
              allow_create_engine: false,
              allow_sampling: true,
              allow_logprobs: true,
              allow_search_indices: false,
              allow_view: true,
              allow_fine_tuning: false,
              organization: '*',
              group: null,
              is_blocking: false
            }
          ],
          root: m.id,
          parent: null,
          nexa_metadata: {
            name: m.name,
            architecture: m.architecture,
            format: m.format,
            parameterSize: m.parameterSize,
            quantization: m.quantization,
            contextWindow: m.contextWindow,
            weightPath: m.weightPath,
            tokenizerPath: m.tokenizerPath,
            fileSizeBytes: m.fileSizeBytes,
            isInstalled: m.isInstalled,
            actualDiskSizeBytes: m.actualDiskSizeBytes,
            state: s?.state || (m.isInstalled ? 'ready_to_load' : 'uninstalled'),
            runtimeConfig: s?.runtimeConfig,
            error: s?.error
          }
        };
      })
    };

    res.json(response);
  });

  // GET /v1/models/:modelId
  router.get('/v1/models/:modelId', requireModelsRead, rateLimiter, (req: Request, res: Response, next: NextFunction) => {
    try {
      const model = registry.get(req.params.modelId);
      if (!model) {
        res.status(404).json({
          error: {
            message: `The model '${req.params.modelId}' does not exist.`,
            type: 'invalid_request_error',
            param: 'model',
            code: 'model_not_found'
          }
        });
        return;
      }

      const s = engineManager.getModelState(model.id);

      res.json({
        id: model.id,
        object: 'model',
        created: 1710000000,
        owned_by: model.architecture,
        permission: [],
        root: model.id,
        parent: null,
        nexa_metadata: {
          ...model,
          state: s.state,
          runtimeConfig: s.runtimeConfig,
          error: s.error
        }
      });
    } catch (err) {
      next(err);
    }
  });

  // -------------------------------------------------------------
  // 3. OPENAI-COMPATIBLE: POST /v1/chat/completions
  // -------------------------------------------------------------
  router.post(
    '/v1/chat/completions',
    requireChatCompletions,
    rateLimiter,
    validateChatCompletionRequest,
    async (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
      try {
        const { model, messages, temperature, top_p, max_tokens, stream, stop } = req.body;

        if (stream) {
          res.setHeader('Content-Type', 'text/event-stream');
          res.setHeader('Cache-Control', 'no-cache');
          res.setHeader('Connection', 'keep-alive');
          res.setHeader('X-Accel-Buffering', 'no');
          res.flushHeaders?.();

          const chunkId = `chatcmpl-${Date.now()}`;
          const createdTime = Math.floor(Date.now() / 1000);

          const initialChunk = {
            id: chunkId,
            object: 'chat.completion.chunk',
            created: createdTime,
            model,
            choices: [
              {
                index: 0,
                delta: { role: 'assistant', content: '' },
                finish_reason: null
              }
            ]
          };
          res.write(`data: ${JSON.stringify(initialChunk)}\n\n`);

          await engineManager.generateChatCompletion(
            { model, messages, temperature, top_p, max_tokens, stream: true, stop },
            (tokenChunk: string) => {
              const chunk = {
                id: chunkId,
                object: 'chat.completion.chunk',
                created: createdTime,
                model,
                choices: [
                  {
                    index: 0,
                    delta: { content: tokenChunk },
                    finish_reason: null
                  }
                ]
              };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            }
          );

          const finalChunk = {
            id: chunkId,
            object: 'chat.completion.chunk',
            created: createdTime,
            model,
            choices: [
              {
                index: 0,
                delta: {},
                finish_reason: 'stop'
              }
            ]
          };
          res.write(`data: ${JSON.stringify(finalChunk)}\n\n`);
          res.write('data: [DONE]\n\n');
          res.end();
        } else {
          const completion = await engineManager.generateChatCompletion({
            model,
            messages,
            temperature,
            top_p,
            max_tokens,
            stream: false,
            stop
          });
          res.json(completion);
        }
      } catch (err) {
        if (res.headersSent) {
          res.write(`data: ${JSON.stringify({ error: { message: (err as Error).message } })}\n\n`);
          res.end();
        } else {
          next(err);
        }
      }
    }
  );

  // -------------------------------------------------------------
  // 4. MODEL MANAGEMENT: Register, Unregister, Load, Unload
  // -------------------------------------------------------------
  router.post(
    '/v1/models/register',
    requireModelsManage,
    rateLimiter,
    validateModelRegistration,
    (req: Request, res: Response, next: NextFunction) => {
      try {
        const registered = registry.register({
          id: req.body.id,
          name: req.body.name,
          architecture: req.body.architecture || 'custom',
          format: req.body.format || 'gguf',
          parameterSize: req.body.parameterSize || 'Unknown',
          quantization: req.body.quantization || 'Q4_K_M',
          contextWindow: req.body.contextWindow || 4096,
          weightPath: req.body.weightPath,
          tokenizerPath: req.body.tokenizerPath,
          fileSizeBytes: req.body.fileSizeBytes || 0,
          description: req.body.description || 'Custom registered model.',
          sourceUrl: req.body.sourceUrl,
          defaultParams: req.body.defaultParams
        });

        res.status(201).json({
          status: 'success',
          message: `Model "${registered.id}" registered successfully.`,
          model: registered
        });
      } catch (err) {
        next(err);
      }
    }
  );

  router.delete('/v1/models/:modelId', requireModelsManage, rateLimiter, (req: Request, res: Response, next: NextFunction) => {
    try {
      const removed = registry.unregister(req.params.modelId);
      if (!removed) {
        res.status(404).json({ error: { message: `Model "${req.params.modelId}" not found.` } });
        return;
      }
      res.json({ status: 'success', message: `Model "${req.params.modelId}" unregistered.` });
    } catch (err) {
      next(err);
    }
  });

  router.post('/v1/models/:modelId/load', requireModelsManage, rateLimiter, async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { contextSize, gpuLayers, threads, batchSize } = req.body || {};
      const status = await engineManager.loadModel(req.params.modelId, {
        contextSize,
        gpuLayers,
        threads,
        batchSize
      });
      res.json({
        status: 'success',
        message: `Model "${req.params.modelId}" loaded into memory successfully.`,
        data: status
      });
    } catch (err) {
      next(err);
    }
  });

  router.post('/v1/models/:modelId/unload', requireModelsManage, rateLimiter, async (req: Request, res: Response, next: NextFunction) => {
    try {
      const status = await engineManager.unloadModel(req.params.modelId);
      res.json({
        status: 'success',
        message: `Model "${req.params.modelId}" unloaded from memory.`,
        data: status
      });
    } catch (err) {
      next(err);
    }
  });

  // -------------------------------------------------------------
  // 5. ENGINE CONTROL & METRICS
  // -------------------------------------------------------------
  router.get('/v1/engine/status', requireEngineAdmin, rateLimiter, async (_req: Request, res: Response, next: NextFunction) => {
    try {
      const sysInfo = await engineManager.getSystemInfo();
      const metrics = engineManager.getMetrics();
      const config = configStore.get();
      const adapters = engineManager.getAvailableAdapters();
      const activeAdapter = engineManager.getActiveAdapter();

      res.json({
        engine: 'NEXA AI Core Engine',
        version: '1.0.0',
        activeAdapter: {
          id: activeAdapter.id,
          name: activeAdapter.name,
          version: activeAdapter.version
        },
        availableAdapters: adapters,
        config,
        system: sysInfo,
        metrics,
        startupReport: engineManager.getStartupReport()
      });
    } catch (err) {
      next(err);
    }
  });

  router.post('/v1/engine/config', requireEngineAdmin, rateLimiter, (req: Request, res: Response, next: NextFunction) => {
    try {
      if (req.body.activeAdapterId) {
        engineManager.setActiveAdapter(req.body.activeAdapterId);
      }
      const updated = configStore.update(req.body);
      res.json({ status: 'success', config: updated });
    } catch (err) {
      next(err);
    }
  });

  router.get('/v1/engine/metrics', requireEngineAdmin, rateLimiter, (_req: Request, res: Response) => {
    res.json(engineManager.getMetrics());
  });

  router.post('/v1/tokenize', requireChatCompletions, rateLimiter, async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { text, model } = req.body;
      if (!text || !model) {
        res.status(400).json({ error: { message: 'Missing "text" or "model" field.' } });
        return;
      }
      const tokens = await engineManager.tokenize(text, model);
      res.json({
        object: 'tokenize',
        model,
        token_count: tokens.length,
        tokens
      });
    } catch (err) {
      next(err);
    }
  });

  // -------------------------------------------------------------
  // 6. API KEYS MANAGEMENT
  // -------------------------------------------------------------
  router.get('/v1/keys', requireEngineAdmin, rateLimiter, (_req: Request, res: Response) => {
    res.json({ object: 'list', data: keyStore.getAll() });
  });

  router.post('/v1/keys', requireEngineAdmin, rateLimiter, (req: Request, res: Response) => {
    const { name, scopes, rateLimitPerMin } = req.body;
    if (!name) {
      res.status(400).json({ error: { message: 'Missing key "name".' } });
      return;
    }
    const created = keyStore.createKey(
      name,
      Array.isArray(scopes) && scopes.length > 0 ? scopes : ['chat:completions', 'models:read'],
      rateLimitPerMin
    );
    res.status(201).json({
      status: 'success',
      data: created.record,
      rawKey: created.rawKey,
      warning: 'Store this key safely. You will not be able to view the full secret again.'
    });
  });

  router.delete('/v1/keys/:keyId', requireEngineAdmin, rateLimiter, (req: Request, res: Response) => {
    const success = keyStore.revokeKey(req.params.keyId);
    if (!success) {
      res.status(404).json({ error: { message: 'Key not found.' } });
      return;
    }
    res.json({ status: 'success', message: `API Key "${req.params.keyId}" revoked.` });
  });

  // -------------------------------------------------------------
  // 7. AUTOMATED TESTS SUITE ENDPOINT
  // -------------------------------------------------------------
  router.post('/v1/tests/run', requireEngineAdmin, rateLimiter, async (_req: Request, res: Response) => {
    const results = await runAutomatedTests({
      engineManager,
      registry,
      keyStore,
      configStore
    });
    res.json(results);
  });

  return router;
}
