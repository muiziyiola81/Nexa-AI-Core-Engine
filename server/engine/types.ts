/**
 * NEXA AI Core Engine - Core Types and Interfaces
 * Modular inference engine architecture for open-weight models.
 */

export type ModelFormat = 'gguf' | 'safetensors' | 'onnx' | 'bin' | 'nexa';
export type ModelArchitecture = 'llama' | 'mistral' | 'qwen2' | 'phi3' | 'deepseek' | 'gemma' | 'custom' | 'transformer';

export type ModelState = 
  | 'uninstalled'     // Weights file (.gguf) not present on disk
  | 'ready_to_load'   // Weights verified on disk, ready to be loaded into memory
  | 'loading'         // Currently loading weights into RAM/VRAM
  | 'loaded'          // Weights resident in RAM/VRAM and ready for inference
  | 'unloading'       // Releasing memory
  | 'error';          // Failure during load/inference

export interface ModelMetadata {
  id: string;                         // Unique identifier, e.g. "llama-3-8b-instruct"
  name: string;                       // Display name, e.g. "Meta Llama 3 8B Instruct"
  architecture: ModelArchitecture;
  format: ModelFormat;
  parameterSize: string;              // e.g. "8B", "7B", "1.1B"
  quantization: string;               // e.g. "Q4_K_M", "Q8_0", "FP16"
  contextWindow: number;              // Max context tokens, e.g. 8192
  weightPath: string;                 // Path on disk, e.g. "models/llama-3-8b-instruct.Q4_K_M.gguf"
  fileSizeBytes: number;              // Estimated or actual file size in bytes
  isInstalled: boolean;               // True only if file exists on disk
  actualDiskSizeBytes?: number;       // Actual detected size if installed
  checksum?: string;                  // SHA-256 if computed
  description: string;
  sourceUrl?: string;                 // HuggingFace / open repository link
  tokenizerPath?: string;             // Optional .nbt BPE tokenizer path for native .nexa models
  defaultParams?: {
    temperature?: number;
    topP?: number;
    repeatPenalty?: number;
    defaultGpuLayers?: number;
  };
}

export interface ModelRuntimeConfig {
  modelId: string;
  contextSize: number;                // Active context window tokens
  gpuLayers: number;                  // Number of layers to offload to GPU (-1 or N)
  threads: number;                    // CPU thread allocation
  batchSize: number;                  // Inference batch size
  loadedAt?: number;                  // Timestamp when loaded
  memoryUsageMb?: number;             // Allocated RAM/VRAM
  activeRequests: number;             // Concurrency tracker
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  name?: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream?: boolean;
  stop?: string | string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  user?: string;
}

export interface ChatCompletionChoice {
  index: number;
  message: {
    role: 'assistant';
    content: string;
  };
  finish_reason: 'stop' | 'length' | 'content_filter' | null;
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: {
    role?: 'assistant';
    content?: string;
  };
  finish_reason: 'stop' | 'length' | 'content_filter' | null;
}

export interface UsageInfo {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: UsageInfo;
  system_fingerprint?: string;
}

export interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
}

export interface SystemHardwareInfo {
  platform: string;
  arch: string;
  cpuModel: string;
  cpuCores: number;
  totalMemoryMb: number;
  freeMemoryMb: number;
  gpuDetected: boolean;
  gpuDescription?: string;
  llamaCppInstalled: boolean;
  llamaCppBinaryPath?: string;
  llamaCppVersion?: string;
  executionMode: 'cpu_only' | 'cuda' | 'metal' | 'vulkan' | 'rocm' | 'mock_simulation';
}

export interface EngineMetrics {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  activeRequests: number;
  totalPromptTokens: number;
  totalCompletionTokens: number;
  averageLatencyMs: number;
  tokensPerSecond: number;
  uptimeSeconds: number;
}

export interface InferenceAdapter {
  readonly id: string;
  readonly name: string;
  readonly version: string;

  isAvailable(): Promise<{ available: boolean; binaryPath?: string; version?: string; reason?: string }>;
  loadModel(model: ModelMetadata, config: Partial<ModelRuntimeConfig>): Promise<ModelRuntimeConfig>;
  unloadModel(modelId: string): Promise<void>;
  generateChatCompletion(
    request: ChatCompletionRequest,
    model: ModelMetadata,
    config: ModelRuntimeConfig,
    onToken?: (chunk: string) => void
  ): Promise<ChatCompletionResponse>;
  tokenize(text: string, model: ModelMetadata): Promise<number[]>;
  getSystemInfo(): Promise<SystemHardwareInfo>;
}
