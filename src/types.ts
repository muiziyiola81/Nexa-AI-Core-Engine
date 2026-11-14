/**
 * NEXA Client Types
 */

export interface SystemTelemetry {
  status: string;
  timestamp: string;
  engine: {
    name: string;
    version: string;
    activeAdapter: {
      id: string;
      name: string;
      version: string;
    };
    loadedModelsCount: number;
    loadedModels: Array<{
      id: string;
      name: string;
      memoryMb?: number;
      gpuLayers?: number;
      contextSize?: number;
    }>;
  };
  system: {
    platform: string;
    arch: string;
    cpu: string;
    cpuCores: number;
    totalMemoryMb: number;
    freeMemoryMb: number;
    gpuDetected: boolean;
    gpuDescription?: string;
    executionMode: string;
    llamaCppInstalled: boolean;
    llamaCppBinaryPath?: string;
  };
  metrics: {
    uptimeSeconds: number;
    totalRequests: number;
    activeRequests: number;
    tokensPerSecond: number;
  };
}

export interface ModelItem {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  nexa_metadata: {
    name: string;
    architecture: string;
    format: string;
    parameterSize: string;
    quantization: string;
    contextWindow: number;
    weightPath: string;
    fileSizeBytes: number;
    isInstalled: boolean;
    actualDiskSizeBytes?: number;
    description: string;
    sourceUrl?: string;
    state: 'uninstalled' | 'ready_to_load' | 'loading' | 'loaded' | 'unloading' | 'error';
    runtimeConfig?: {
      modelId: string;
      contextSize: number;
      gpuLayers: number;
      threads: number;
      batchSize: number;
      loadedAt?: number;
      memoryUsageMb?: number;
      activeRequests: number;
    };
  };
}

export interface ApiKeyItem {
  id: string;
  name: string;
  keyPrefix: string;
  scopes: string[];
  createdAt: number;
  lastUsedAt?: number;
  totalRequests: number;
  rateLimitPerMin?: number;
}

export interface EngineStatusData {
  engine: string;
  version: string;
  activeAdapter: {
    id: string;
    name: string;
    version: string;
  };
  availableAdapters: Array<{ id: string; name: string; version: string }>;
  config: {
    activeAdapterId: string;
    modelsDirectory: string;
    llamaCppServerUrl: string;
    llamaCppBinaryPath: string;
    maxConcurrentRequests: number;
    defaultContextWindow: number;
    defaultThreads: number;
    requireApiKey: boolean;
    rateLimitPerMinute: number;
    maxTokensPerRequest: number;
    executionBackend: string;
  };
  system: SystemTelemetry['system'];
  metrics: {
    totalRequests: number;
    successfulRequests: number;
    failedRequests: number;
    activeRequests: number;
    totalPromptTokens: number;
    totalCompletionTokens: number;
    averageLatencyMs: number;
    tokensPerSecond: number;
    uptimeSeconds: number;
  };
}

export interface TestResultItem {
  name: string;
  category: string;
  passed: boolean;
  durationMs: number;
  details?: string;
  error?: string;
}

export interface TestSuiteReport {
  timestamp: string;
  total: number;
  passed: number;
  failed: number;
  durationMs: number;
  results: TestResultItem[];
}
