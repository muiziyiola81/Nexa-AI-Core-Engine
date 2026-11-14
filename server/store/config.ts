/**
 * NEXA AI Core Engine - Configuration Store
 * Global configuration management for inference limits, paths, CORS, and security.
 * Environment variables always take precedence over persisted disk configuration
 * on boot so Cloud Run and containerized deployments behave deterministically.
 */

import fs from 'fs';
import path from 'path';

export interface EngineConfig {
  activeAdapterId: 'nexa-native' | 'llama-cpp' | 'nexa-diagnostic';
  modelsDirectory: string;
  llamaCppServerUrl: string;
  llamaCppBinaryPath: string;
  maxConcurrentRequests: number;
  defaultContextWindow: number;
  defaultThreads: number;
  requireApiKey: boolean;
  rateLimitPerMinute: number;
  maxTokensPerRequest: number;
  executionBackend: 'auto' | 'cpu' | 'cuda' | 'metal';
  allowedOrigins: string[];
  autoLoadModels: string[];
  autoRegisterCheckpoints: boolean;
  autoLoadOnDemand: boolean;
}

export function getDataDir(): string {
  return process.env.DATA_DIR
    ? path.resolve(process.cwd(), process.env.DATA_DIR)
    : path.resolve(process.cwd(), 'data');
}

export function getModelsDir(): string {
  return process.env.MODELS_DIR
    ? path.resolve(process.cwd(), process.env.MODELS_DIR)
    : path.resolve(process.cwd(), 'models');
}

function parseOrigins(raw?: string): string[] {
  if (!raw || !raw.trim()) return ['*'];
  const list = raw
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);
  return list.length > 0 ? list : ['*'];
}

function parseModelList(raw?: string): string[] {
  if (raw === undefined) return [];
  return raw
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);
}

function buildDefaultConfig(): EngineConfig {
  const isProd = process.env.NODE_ENV === 'production';
  return {
    activeAdapterId: (process.env.ACTIVE_ADAPTER as EngineConfig['activeAdapterId']) || 'nexa-native',
    modelsDirectory: getModelsDir(),
    llamaCppServerUrl: process.env.LLAMA_CPP_SERVER_URL || 'http://127.0.0.1:8080',
    llamaCppBinaryPath: process.env.NEXA_LLAMA_CPP_PATH || '',
    maxConcurrentRequests: parseInt(process.env.MAX_CONCURRENT_REQUESTS || '4', 10),
    defaultContextWindow: parseInt(process.env.DEFAULT_CONTEXT_WINDOW || '4096', 10),
    defaultThreads: parseInt(process.env.DEFAULT_CPU_THREADS || '4', 10),
    requireApiKey: process.env.REQUIRE_API_KEY === 'true',
    rateLimitPerMinute: parseInt(process.env.RATE_LIMIT_PER_MINUTE || '120', 10),
    maxTokensPerRequest: parseInt(process.env.MAX_TOKENS_PER_REQUEST || '4096', 10),
    executionBackend: (process.env.EXECUTION_BACKEND as EngineConfig['executionBackend']) || 'auto',
    allowedOrigins: parseOrigins(process.env.ALLOWED_ORIGINS),
    autoLoadModels: parseModelList(process.env.AUTO_LOAD_MODELS),
    autoRegisterCheckpoints: process.env.AUTO_REGISTER_NEXA_CHECKPOINTS !== 'false',
    autoLoadOnDemand:
      process.env.AUTO_LOAD_ON_DEMAND === 'true' ||
      (isProd && process.env.AUTO_LOAD_ON_DEMAND !== 'false')
  };
}

export class ConfigStore {
  private config: EngineConfig;
  private configFile: string;

  constructor() {
    this.configFile = path.join(getDataDir(), 'engine_config.json');
    this.config = buildDefaultConfig();
    this.load();
  }

  private applyEnvOverrides() {
    if (process.env.ACTIVE_ADAPTER) {
      this.config.activeAdapterId = process.env.ACTIVE_ADAPTER as EngineConfig['activeAdapterId'];
    }
    if (process.env.MODELS_DIR) {
      this.config.modelsDirectory = getModelsDir();
    } else if (this.config.modelsDirectory && !path.isAbsolute(this.config.modelsDirectory)) {
      this.config.modelsDirectory = path.resolve(process.cwd(), this.config.modelsDirectory);
    }
    if (process.env.LLAMA_CPP_SERVER_URL) {
      this.config.llamaCppServerUrl = process.env.LLAMA_CPP_SERVER_URL;
    }
    if (process.env.NEXA_LLAMA_CPP_PATH) {
      this.config.llamaCppBinaryPath = process.env.NEXA_LLAMA_CPP_PATH;
    }
    if (process.env.MAX_CONCURRENT_REQUESTS) {
      this.config.maxConcurrentRequests = parseInt(process.env.MAX_CONCURRENT_REQUESTS, 10);
    }
    if (process.env.DEFAULT_CONTEXT_WINDOW) {
      this.config.defaultContextWindow = parseInt(process.env.DEFAULT_CONTEXT_WINDOW, 10);
    }
    if (process.env.DEFAULT_CPU_THREADS) {
      this.config.defaultThreads = parseInt(process.env.DEFAULT_CPU_THREADS, 10);
    }
    if (process.env.REQUIRE_API_KEY !== undefined) {
      this.config.requireApiKey = process.env.REQUIRE_API_KEY === 'true';
    }
    if (process.env.RATE_LIMIT_PER_MINUTE) {
      this.config.rateLimitPerMinute = parseInt(process.env.RATE_LIMIT_PER_MINUTE, 10);
    }
    if (process.env.MAX_TOKENS_PER_REQUEST) {
      this.config.maxTokensPerRequest = parseInt(process.env.MAX_TOKENS_PER_REQUEST, 10);
    }
    if (process.env.EXECUTION_BACKEND) {
      this.config.executionBackend = process.env.EXECUTION_BACKEND as EngineConfig['executionBackend'];
    }
    if (process.env.ALLOWED_ORIGINS !== undefined) {
      this.config.allowedOrigins = parseOrigins(process.env.ALLOWED_ORIGINS);
    }
    if (process.env.AUTO_LOAD_MODELS !== undefined) {
      this.config.autoLoadModels = parseModelList(process.env.AUTO_LOAD_MODELS);
    }
    if (process.env.AUTO_REGISTER_NEXA_CHECKPOINTS !== undefined) {
      this.config.autoRegisterCheckpoints = process.env.AUTO_REGISTER_NEXA_CHECKPOINTS !== 'false';
    }
    if (process.env.AUTO_LOAD_ON_DEMAND !== undefined) {
      this.config.autoLoadOnDemand = process.env.AUTO_LOAD_ON_DEMAND === 'true';
    }
  }

  private load() {
    const defaults = buildDefaultConfig();
    try {
      if (fs.existsSync(this.configFile)) {
        const raw = fs.readFileSync(this.configFile, 'utf-8');
        const parsed = JSON.parse(raw);
        this.config = { ...defaults, ...parsed };
      } else {
        this.config = defaults;
      }
    } catch {
      this.config = defaults;
    }
    this.applyEnvOverrides();
  }

  private save() {
    try {
      const dataDir = path.dirname(this.configFile);
      if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
      }
      fs.writeFileSync(this.configFile, JSON.stringify(this.config, null, 2), 'utf-8');
    } catch (err) {
      console.error('Failed to save engine config:', err);
    }
  }

  public get(): EngineConfig {
    return {
      ...this.config,
      allowedOrigins: [...this.config.allowedOrigins],
      autoLoadModels: [...this.config.autoLoadModels]
    };
  }

  public update(partial: Partial<EngineConfig>): EngineConfig {
    this.config = {
      ...this.config,
      ...partial
    };
    this.save();
    return this.get();
  }
}
