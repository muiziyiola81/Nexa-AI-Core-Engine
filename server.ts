/**
 * NEXA AI Core Engine - Server Entrypoint
 * Production & Cloud Run compatible Express server hosting OpenAI-compatible
 * API endpoints, native C++20 inference bridge, and optional Vite SPA UI.
 */

import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { ModelRegistry } from './server/store/model_registry.js';
import { ConfigStore } from './server/store/config.js';
import { KeyStore } from './server/store/key_store.js';
import { EngineManager } from './server/engine/engine_manager.js';
import { createApiRouter } from './server/api/routes.js';
import { errorHandler } from './server/middleware/error_handler.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export function createNexaApp(deps?: {
  registry?: ModelRegistry;
  configStore?: ConfigStore;
  keyStore?: KeyStore;
  engineManager?: EngineManager;
}) {
  const app = express();
  const registry = deps?.registry || new ModelRegistry();
  const configStore = deps?.configStore || new ConfigStore();
  const keyStore = deps?.keyStore || new KeyStore();
  const engineManager = deps?.engineManager || new EngineManager(registry, configStore);

  // Trust Cloud Run / reverse-proxy headers (X-Forwarded-For, X-Forwarded-Proto)
  app.set('trust proxy', 1);
  app.disable('x-powered-by');

  // Body parser with generous limit for prompt payloads
  app.use(express.json({ limit: '10mb' }));
  app.use(express.urlencoded({ extended: true, limit: '10mb' }));

  // Configurable CORS for separate NEXA AI client applications
  app.use((req, res, next) => {
    const allowedOrigins = configStore.get().allowedOrigins;
    const reqOrigin = req.headers.origin;
    const allowWildcard = allowedOrigins.includes('*');

    res.setHeader('Vary', 'Origin');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');
    res.setHeader('Access-Control-Max-Age', '86400');

    if (allowWildcard) {
      res.setHeader('Access-Control-Allow-Origin', '*');
    } else if (reqOrigin && allowedOrigins.includes(reqOrigin)) {
      res.setHeader('Access-Control-Allow-Origin', reqOrigin);
      res.setHeader('Access-Control-Allow-Credentials', 'true');
    }

    if (req.method === 'OPTIONS') {
      if (!allowWildcard && reqOrigin && !allowedOrigins.includes(reqOrigin)) {
        res.status(403).json({
          error: {
            message: `Origin "${reqOrigin}" is not permitted by CORS policy.`,
            type: 'invalid_request_error',
            param: null,
            code: 'cors_origin_denied'
          }
        });
        return;
      }
      res.sendStatus(204);
      return;
    }

    next();
  });

  // Mount API router and central JSON error handler
  const apiRouter = createApiRouter(engineManager, registry, keyStore, configStore);
  app.use(apiRouter);
  app.use(errorHandler);

  return { app, registry, configStore, keyStore, engineManager };
}

async function startServer() {
  const envPort =
    process.env.PORT && process.env.PORT !== process.env.NGINX_PORT
      ? process.env.PORT
      : process.env.DEFAULT_APP_PORT || '3000';
  let port = parseInt(envPort, 10);
  if (Number.isNaN(port) || port <= 0) {
    port = 3000;
  }
  const portIndex = process.argv.indexOf('--port');
  if (portIndex !== -1 && process.argv[portIndex + 1]) {
    port = parseInt(process.argv[portIndex + 1], 10);
  }

  let host = process.env.HOST || '0.0.0.0';
  const hostIndex = process.argv.indexOf('--host');
  if (hostIndex !== -1 && process.argv[hostIndex + 1]) {
    host = process.argv[hostIndex + 1];
  }

  const isProduction = process.env.NODE_ENV === 'production';
  const invokedViaStandaloneEntry =
    Boolean(process.argv[1] && process.argv[1].endsWith('standalone.ts')) ||
    process.argv.includes('--standalone');
  const inDevPreviewProxy = !isProduction && Boolean(process.env.CONTROL_PLANE_PORT);
  const isStandaloneApi =
    invokedViaStandaloneEntry ||
    (!inDevPreviewProxy && process.env.STANDALONE_API === 'true');

  const { app, registry, configStore, keyStore, engineManager } = createNexaApp();

  console.log('[NEXA ENGINE] Initializing subsystems...');
  const cfg = configStore.get();
  console.log(`[NEXA ENGINE] Active Adapter: ${cfg.activeAdapterId}`);
  console.log(`[NEXA ENGINE] Auth Required: ${cfg.requireApiKey ? 'YES (Bearer token required)' : 'NO (Local dev open mode)'}`);
  console.log(`[NEXA ENGINE] CORS Allowed Origins: ${cfg.allowedOrigins.join(', ')}`);
  console.log(`[NEXA ENGINE] Models registered: ${registry.getAll().length}`);

  if (cfg.requireApiKey && !keyStore.hasConfiguredKeys()) {
    console.warn('[NEXA ENGINE] WARNING: REQUIRE_API_KEY=true but no API keys are configured!');
  }

  // Run startup checks & pre-load configured/bundled .nexa checkpoints
  try {
    const startup = await engineManager.startupBootstrap();
    console.log(`[NEXA ENGINE] Native Adapter Available: ${startup.adapterAvailable ? 'YES (' + startup.adapterBinaryPath + ')' : 'NO (' + startup.adapterReason + ')'}`);
    console.log(`[NEXA ENGINE] Storage Writable: data=${startup.dataDirWritable}, models=${startup.modelsDirWritable}`);
    if (startup.autoLoadedModels.length > 0) {
      console.log(`[NEXA ENGINE] Auto-loaded models on boot: ${startup.autoLoadedModels.join(', ')}`);
    }
    if (startup.modelLoadErrors.length > 0) {
      console.warn(`[NEXA ENGINE] Model auto-load warnings: ${JSON.stringify(startup.modelLoadErrors)}`);
    }
  } catch (err) {
    console.warn('[NEXA ENGINE] Startup bootstrap warning:', err);
  }

  try {
    const sysInfo = await engineManager.getSystemInfo();
    console.log(`[NEXA ENGINE] Platform: ${sysInfo.platform} (${sysInfo.arch}), Cores: ${sysInfo.cpuCores}, Memory: ${sysInfo.totalMemoryMb} MB`);
  } catch (err) {
    console.warn('[NEXA ENGINE] System info probe warning:', err);
  }

  // Mount frontend SPA or standalone API landing route before central error handler
  if (isStandaloneApi) {
    console.log('[NEXA ENGINE] Running in Headless Standalone API Mode (Frontend disabled)');
    app.get('/', (_req, res) => {
      res.json({
        engine: 'NEXA AI Core Engine',
        version: '1.0.0',
        mode: 'standalone_api',
        status: 'healthy',
        endpoints: {
          health: '/health',
          ready: '/ready',
          models: '/v1/models',
          chat_completions: '/v1/chat/completions',
          engine_status: '/v1/engine/status'
        }
      });
    });
  } else if (!isProduction) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa'
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    if (fs.existsSync(distPath)) {
      app.use(express.static(distPath));
      app.get('*', (_req, res) => {
        res.sendFile(path.resolve(distPath, 'index.html'));
      });
    } else {
      app.get('/', (_req, res) => {
        res.json({
          engine: 'NEXA AI Core Engine',
          version: '1.0.0',
          mode: 'headless_production',
          note: 'Static frontend (dist/) not built. Serving pure API.',
          endpoints: {
            health: '/health',
            ready: '/ready',
            models: '/v1/models',
            chat_completions: '/v1/chat/completions'
          }
        });
      });
    }
  }

  // Mount central error handler last
  app.use(errorHandler);

  const server = app.listen(port, host, () => {
    console.log(`[NEXA ENGINE] Server listening on http://${host}:${port}`);
    console.log(`[NEXA ENGINE] Health check: http://localhost:${port}/health`);
    console.log(`[NEXA ENGINE] Readiness check: http://localhost:${port}/ready`);
    console.log(`[NEXA ENGINE] OpenAI Models: http://localhost:${port}/v1/models`);
    console.log(`[NEXA ENGINE] OpenAI Completions: http://localhost:${port}/v1/chat/completions`);
  });

  // Cloud Run graceful shutdown on SIGTERM / SIGINT (10s Cloud Run termination window)
  let shuttingDown = false;
  const gracefulShutdown = async (signal: string) => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`[NEXA ENGINE] Received ${signal}. Initiating graceful shutdown...`);

    const forceTimer = setTimeout(() => {
      console.error('[NEXA ENGINE] Graceful shutdown timed out; forcing exit.');
      process.exit(1);
    }, 8000);
    forceTimer.unref?.();

    server.close(async () => {
      await engineManager.shutdown();
      clearTimeout(forceTimer);
      console.log('[NEXA ENGINE] Shutdown complete.');
      process.exit(0);
    });
  };

  process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
  process.on('SIGINT', () => gracefulShutdown('SIGINT'));
}

const isMainModule =
  process.argv[1] &&
  (path.resolve(process.argv[1]) === __filename || process.argv[1].endsWith('standalone.ts'));

if (isMainModule) {
  startServer().catch(err => {
    console.error('[NEXA ENGINE] Fatal bootstrap error:', err);
    process.exit(1);
  });
}
