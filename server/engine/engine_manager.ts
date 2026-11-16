/**
 * NEXA AI Core Engine - Central Engine Manager
 * Orchestrates inference adapters, model state lifecycle (load/unload),
 * Cloud Run startup verification/auto-load, concurrency management, and hardware telemetry.
 */

import fs from 'fs';
import {
  InferenceAdapter,
  ModelState,
  ModelRuntimeConfig,
  ChatCompletionRequest,
  ChatCompletionResponse,
  SystemHardwareInfo,
  EngineMetrics
} from './types.js';
import { LlamaCppAdapter } from './llama_cpp_adapter.js';
import { DiagnosticAdapter } from './diagnostic_adapter.js';
import { NexaNativeAdapter } from '../native/native_adapter.js';
import { ModelRegistry } from '../store/model_registry.js';
import { ConfigStore, getDataDir, getModelsDir } from '../store/config.js';

export interface ModelLifecycleStatus {
  modelId: string;
  name: string;
  state: ModelState;
  runtimeConfig?: ModelRuntimeConfig;
  error?: string;
  isInstalled: boolean;
  actualDiskSizeBytes?: number;
}

export interface StartupBootstrapReport {
  completed: boolean;
  timestamp: string;
  activeAdapterId: string;
  adapterAvailable: boolean;
  adapterBinaryPath?: string;
  adapterReason?: string;
  dataDirWritable: boolean;
  modelsDirWritable: boolean;
  autoLoadedModels: string[];
  modelLoadErrors: Array<{ modelId: string; error: string }>;
}

export interface ReadinessStatus {
  ready: boolean;
  timestamp: string;
  activeAdapterId: string;
  checks: {
    adapterAvailable: boolean;
    adapterBinaryPath?: string;
    adapterReason?: string;
    dataDirWritable: boolean;
    modelsDirWritable: boolean;
    loadedModelsCount: number;
    loadedModels: string[];
    modelLoadErrors: Array<{ modelId: string; error: string }>;
  };
}

export class EngineManager {
  private adapters: Map<string, InferenceAdapter> = new Map();
  private activeAdapterId: string = 'nexa-native';
  private modelStates: Map<string, { state: ModelState; runtimeConfig?: ModelRuntimeConfig; error?: string }> = new Map();
  private explicitlyUnloaded: Set<string> = new Set();
  private inFlightLoads: Map<string, Promise<ModelLifecycleStatus>> = new Map();

  private registry: ModelRegistry;
  private configStore: ConfigStore;
  private startupReport: StartupBootstrapReport | null = null;

  // Metrics
  private startTime: number = Date.now();
  private totalRequests: number = 0;
  private successfulRequests: number = 0;
  private failedRequests: number = 0;
  private activeRequestsCount: number = 0;
  private totalPromptTokens: number = 0;
  private totalCompletionTokens: number = 0;
  private latencyHistory: number[] = [];

  constructor(registry: ModelRegistry, configStore: ConfigStore) {
    this.registry = registry;
    this.configStore = configStore;

    // Register adapters
    const nativeAdapter = new NexaNativeAdapter();
    const llamaAdapter = new LlamaCppAdapter();
    const diagAdapter = new DiagnosticAdapter();

    this.adapters.set(nativeAdapter.id, nativeAdapter);
    this.adapters.set(llamaAdapter.id, llamaAdapter);
    this.adapters.set(diagAdapter.id, diagAdapter);

    this.activeAdapterId = this.configStore.get().activeAdapterId || 'nexa-native';

    this.initModelStates();
  }

  private initModelStates() {
    const models = this.registry.getAll();
    for (const m of models) {
      this.modelStates.set(m.id, {
        state: m.isInstalled ? 'ready_to_load' : 'uninstalled'
      });
    }
  }

  private checkDirWritable(dirPath: string): boolean {
    try {
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
      }
      fs.accessSync(dirPath, fs.constants.R_OK | fs.constants.W_OK);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Executes startup checks and pre-loads configured or default .nexa models
   * so Cloud Run instances serve inference immediately on first request.
   */
  public async startupBootstrap(): Promise<StartupBootstrapReport> {
    const adapter = this.getActiveAdapter();
    const avail = await adapter.isAvailable();
    const dataDirWritable = this.checkDirWritable(getDataDir());
    const modelsDirWritable = this.checkDirWritable(getModelsDir());

    const config = this.configStore.get();
    const autoLoadedModels: string[] = [];
    const modelLoadErrors: Array<{ modelId: string; error: string }> = [];

    let targets: string[] = [];
    if (process.env.AUTO_LOAD_MODELS !== undefined) {
      targets = config.autoLoadModels;
    } else if (this.activeAdapterId === 'nexa-native' && avail.available) {
      // Auto-load the active production .nexa checkpoint on cold start (preserve nexa-p36-tier3 default)
      const installedNative = ['nexa-p36-tier3', 'nexa-p37-v2', 'nexa-nano-step-40'].filter(id => {
        const m = this.registry.get(id);
        return Boolean(m && m.isInstalled);
      });
      if (installedNative.length > 0) {
        targets = [installedNative[0]];
      }
    }

    for (const modelId of targets) {
      const m = this.registry.get(modelId);
      if (!m) {
        modelLoadErrors.push({ modelId, error: `Configured auto-load model "${modelId}" not found in registry.` });
        continue;
      }
      if (!m.isInstalled) {
        modelLoadErrors.push({ modelId, error: `Weights not found on disk at "${m.weightPath}".` });
        continue;
      }

      try {
        await this.loadModel(modelId);
        autoLoadedModels.push(modelId);
      } catch (err: unknown) {
        const message = (err as Error).message || 'Unknown load error';
        modelLoadErrors.push({ modelId, error: message });
      }
    }

    this.startupReport = {
      completed: true,
      timestamp: new Date().toISOString(),
      activeAdapterId: this.activeAdapterId,
      adapterAvailable: avail.available,
      adapterBinaryPath: avail.binaryPath,
      adapterReason: avail.reason,
      dataDirWritable,
      modelsDirWritable,
      autoLoadedModels,
      modelLoadErrors
    };

    return this.startupReport;
  }

  public getStartupReport(): StartupBootstrapReport | null {
    return this.startupReport;
  }

  public async getReadinessStatus(): Promise<ReadinessStatus> {
    const adapter = this.getActiveAdapter();
    const avail = await adapter.isAvailable();
    const dataDirWritable = this.checkDirWritable(getDataDir());
    const modelsDirWritable = this.checkDirWritable(getModelsDir());
    const loadedModels = this.getAllModelStatuses()
      .filter(m => m.state === 'loaded')
      .map(m => m.modelId);

    const requireLoadedModel = process.env.REQUIRE_LOADED_MODEL_FOR_READY === 'true';
    const ready =
      avail.available &&
      dataDirWritable &&
      (!requireLoadedModel || loadedModels.length > 0);

    return {
      ready,
      timestamp: new Date().toISOString(),
      activeAdapterId: this.activeAdapterId,
      checks: {
        adapterAvailable: avail.available,
        adapterBinaryPath: avail.binaryPath,
        adapterReason: avail.reason,
        dataDirWritable,
        modelsDirWritable,
        loadedModelsCount: loadedModels.length,
        loadedModels,
        modelLoadErrors: this.startupReport?.modelLoadErrors || []
      }
    };
  }

  public getActiveAdapter(): InferenceAdapter {
    const adapter = this.adapters.get(this.activeAdapterId);
    if (!adapter) {
      return this.adapters.get('nexa-native') || Array.from(this.adapters.values())[0];
    }
    return adapter;
  }

  public setActiveAdapter(adapterId: string): void {
    if (!this.adapters.has(adapterId)) {
      throw new Error(`Unknown inference adapter: "${adapterId}". Available: ${Array.from(this.adapters.keys()).join(', ')}`);
    }
    this.activeAdapterId = adapterId;
    this.configStore.update({ activeAdapterId: adapterId as 'nexa-native' | 'llama-cpp' | 'nexa-diagnostic' });
  }

  public getAvailableAdapters(): Array<{ id: string; name: string; version: string }> {
    return Array.from(this.adapters.values()).map(a => ({
      id: a.id,
      name: a.name,
      version: a.version
    }));
  }

  public getModelState(modelId: string): ModelLifecycleStatus {
    const model = this.registry.get(modelId);
    if (!model) {
      throw new Error(`Model "${modelId}" not found in registry.`);
    }

    const current = this.modelStates.get(modelId);
    let state = current?.state || (model.isInstalled ? 'ready_to_load' : 'uninstalled');
    if (state !== 'loaded' && state !== 'loading' && state !== 'error') {
      state = model.isInstalled ? 'ready_to_load' : 'uninstalled';
    }

    return {
      modelId: model.id,
      name: model.name,
      state,
      runtimeConfig: current?.runtimeConfig,
      error: current?.error,
      isInstalled: model.isInstalled,
      actualDiskSizeBytes: model.actualDiskSizeBytes
    };
  }

  public getAllModelStatuses(): ModelLifecycleStatus[] {
    const models = this.registry.getAll();
    return models.map(m => this.getModelState(m.id));
  }

  /**
   * Load model weights into memory (deduplicated for concurrent cold-start requests).
   */
  public async loadModel(modelId: string, options: Partial<ModelRuntimeConfig> = {}): Promise<ModelLifecycleStatus> {
    const existingInFlight = this.inFlightLoads.get(modelId);
    if (existingInFlight) {
      return existingInFlight;
    }

    const loadPromise = (async () => {
      const model = this.registry.get(modelId);
      if (!model) {
        throw new Error(`Cannot load model: "${modelId}" does not exist in the model registry.`);
      }

      const adapter = this.getActiveAdapter();
      this.explicitlyUnloaded.delete(modelId);
      this.modelStates.set(modelId, { state: 'loading' });

      try {
        const runtimeConfig = await adapter.loadModel(model, options);

        this.modelStates.set(modelId, {
          state: 'loaded',
          runtimeConfig
        });

        return this.getModelState(modelId);
      } catch (err: unknown) {
        const error = err as Error;
        this.modelStates.set(modelId, {
          state: 'error',
          error: error.message
        });
        throw error;
      } finally {
        this.inFlightLoads.delete(modelId);
      }
    })();

    this.inFlightLoads.set(modelId, loadPromise);
    return loadPromise;
  }

  /**
   * Unload model to release RAM and VRAM.
   */
  public async unloadModel(modelId: string): Promise<ModelLifecycleStatus> {
    const model = this.registry.get(modelId);
    if (!model) {
      throw new Error(`Model "${modelId}" does not exist in registry.`);
    }

    const current = this.modelStates.get(modelId);
    if (!current || current.state !== 'loaded') {
      throw new Error(`Model "${modelId}" is not currently loaded in memory.`);
    }

    this.modelStates.set(modelId, { state: 'unloading' });

    try {
      const adapter = this.getActiveAdapter();
      await adapter.unloadModel(modelId);

      this.explicitlyUnloaded.add(modelId);
      this.modelStates.set(modelId, {
        state: model.isInstalled ? 'ready_to_load' : 'uninstalled'
      });

      return this.getModelState(modelId);
    } catch (err: unknown) {
      const error = err as Error;
      this.modelStates.set(modelId, {
        state: 'error',
        error: error.message
      });
      throw error;
    }
  }

  public async shutdown(): Promise<void> {
    const loaded = this.getAllModelStatuses().filter(m => m.state === 'loaded');
    for (const m of loaded) {
      try {
        await this.unloadModel(m.modelId);
      } catch {
        // ignore shutdown cleanup failure
      }
    }
  }

  /**
   * Execute chat completion.
   * Rejects unloaded models unless AUTO_LOAD_ON_DEMAND is enabled for installed
   * models that were not explicitly unloaded (Cloud Run cold-start / restart recovery).
   */
  public async generateChatCompletion(
    request: ChatCompletionRequest,
    onToken?: (chunk: string) => void
  ): Promise<ChatCompletionResponse> {
    const model = this.registry.get(request.model);
    if (!model) {
      throw new Error(`[NEXA_ERR_MODEL_NOT_FOUND] Model "${request.model}" does not exist in the registry.`);
    }

    let state = this.getModelState(model.id);
    if (state.state !== 'loaded' || !state.runtimeConfig) {
      const config = this.configStore.get();
      if (
        config.autoLoadOnDemand &&
        model.isInstalled &&
        !this.explicitlyUnloaded.has(model.id)
      ) {
        try {
          state = await this.loadModel(model.id);
        } catch (loadErr: unknown) {
          throw new Error(
            `[NEXA_ERR_MODEL_LOAD_FAILED] Auto-load recovery failed for "${model.id}": ${(loadErr as Error).message}`
          );
        }
      }
    }

    if (state.state !== 'loaded' || !state.runtimeConfig) {
      throw new Error(
        `[NEXA_ERR_MODEL_NOT_LOADED] Model "${model.id}" is currently in state "${state.state}". ` +
        `Weights must be resident in memory before serving inference. Call POST /v1/models/${model.id}/load first.`
      );
    }

    // Check concurrency limit
    const maxConcurrent = this.configStore.get().maxConcurrentRequests;
    if (this.activeRequestsCount >= maxConcurrent) {
      throw new Error(`[NEXA_ERR_MAX_CONCURRENCY] Server reached maximum concurrent requests limit (${maxConcurrent}).`);
    }

    const adapter = this.getActiveAdapter();
    this.totalRequests++;
    this.activeRequestsCount++;
    state.runtimeConfig.activeRequests++;

    const startTime = Date.now();

    try {
      const response = await adapter.generateChatCompletion(request, model, state.runtimeConfig, onToken);

      const elapsed = Date.now() - startTime;
      this.successfulRequests++;
      this.totalPromptTokens += response.usage.prompt_tokens;
      this.totalCompletionTokens += response.usage.completion_tokens;

      this.latencyHistory.push(elapsed);
      if (this.latencyHistory.length > 100) this.latencyHistory.shift();

      return response;
    } catch (err: unknown) {
      this.failedRequests++;
      throw err;
    } finally {
      this.activeRequestsCount = Math.max(0, this.activeRequestsCount - 1);
      if (state.runtimeConfig) {
        state.runtimeConfig.activeRequests = Math.max(0, state.runtimeConfig.activeRequests - 1);
      }
    }
  }

  public async tokenize(text: string, modelId: string): Promise<number[]> {
    const model = this.registry.get(modelId);
    if (!model) throw new Error(`[NEXA_ERR_MODEL_NOT_FOUND] Model "${modelId}" does not exist in registry.`);
    const adapter = this.getActiveAdapter();
    return adapter.tokenize(text, model);
  }

  public async getSystemInfo(): Promise<SystemHardwareInfo> {
    const adapter = this.getActiveAdapter();
    return adapter.getSystemInfo();
  }

  public getMetrics(): EngineMetrics {
    const uptimeSeconds = Math.floor((Date.now() - this.startTime) / 1000);
    const avgLatency = this.latencyHistory.length > 0
      ? Math.round(this.latencyHistory.reduce((a, b) => a + b, 0) / this.latencyHistory.length)
      : 0;

    const tokensPerSec = uptimeSeconds > 0
      ? Number((this.totalCompletionTokens / uptimeSeconds).toFixed(2))
      : 0;

    return {
      totalRequests: this.totalRequests,
      successfulRequests: this.successfulRequests,
      failedRequests: this.failedRequests,
      activeRequests: this.activeRequestsCount,
      totalPromptTokens: this.totalPromptTokens,
      totalCompletionTokens: this.totalCompletionTokens,
      averageLatencyMs: avgLatency,
      tokensPerSecond: tokensPerSec,
      uptimeSeconds
    };
  }
}
