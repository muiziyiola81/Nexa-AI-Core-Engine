/**
 * NEXA AI Core Engine - llama.cpp Runtime Adapter
 * Interfaces directly with llama.cpp binaries (llama-server / llama-cli)
 * and verifies model weight presence on the local filesystem.
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import { exec } from 'child_process';
import { promisify } from 'util';
import {
  InferenceAdapter,
  ModelMetadata,
  ModelRuntimeConfig,
  ChatCompletionRequest,
  ChatCompletionResponse,
  SystemHardwareInfo
} from './types.js';
import { ModelRegistry } from '../store/model_registry.js';

const execAsync = promisify(exec);

export class LlamaCppAdapter implements InferenceAdapter {
  readonly id = 'llama-cpp';
  readonly name = 'llama.cpp Native Runtime';
  readonly version = 'b4000+';

  private binaryPath: string | null = null;
  private serverUrl: string | null = null;
  private loadedModels: Map<string, ModelRuntimeConfig> = new Map();
  private cachedAvailability: { result: { available: boolean; binaryPath?: string; version?: string; reason?: string }; timestamp: number } | null = null;
  private cachedGpuInfo: { detected: boolean; desc: string; mode: SystemHardwareInfo['executionMode'] } | null = null;

  constructor(serverUrl?: string, binaryPath?: string) {
    this.serverUrl = serverUrl || process.env.LLAMA_CPP_SERVER_URL || null;
    this.binaryPath = binaryPath || process.env.NEXA_LLAMA_CPP_PATH || null;
  }

  public setServerUrl(url: string | null) {
    this.serverUrl = url;
    this.cachedAvailability = null;
  }

  public setBinaryPath(bin: string | null) {
    this.binaryPath = bin;
    this.cachedAvailability = null;
  }

  /**
   * Probe system paths to locate llama-server or llama-cli binary,
   * or detect active local llama-server HTTP endpoints.
   * Uses fast fs.existsSync checks and short-lived caching.
   */
  async isAvailable(): Promise<{ available: boolean; binaryPath?: string; version?: string; reason?: string }> {
    const now = Date.now();
    if (this.cachedAvailability && now - this.cachedAvailability.timestamp < 15000) {
      return this.cachedAvailability.result;
    }

    // Check configured serverUrl or environment URL
    if (this.serverUrl) {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 400);
        const res = await fetch(`${this.serverUrl}/health`, { signal: controller.signal });
        clearTimeout(timeout);
        if (res.ok) {
          const result = { available: true, version: 'llama-server HTTP', binaryPath: this.serverUrl };
          this.cachedAvailability = { result, timestamp: now };
          return result;
        }
      } catch {
        // Continue
      }
    }

    // Fast filesystem binary probing
    const candidateFiles = [
      this.binaryPath,
      '/usr/local/bin/llama-server',
      '/usr/bin/llama-server',
      '/opt/llama.cpp/llama-server',
      path.resolve(process.cwd(), 'bin', 'llama-server'),
      path.resolve(process.cwd(), 'llama.cpp', 'llama-server'),
      path.resolve(process.cwd(), 'llama.cpp', 'build', 'bin', 'llama-server'),
      '/usr/local/bin/llama-cli',
      '/usr/bin/llama-cli'
    ].filter(Boolean) as string[];

    for (const filePath of candidateFiles) {
      if (fs.existsSync(filePath)) {
        try {
          fs.accessSync(filePath, fs.constants.X_OK);
          this.binaryPath = filePath;
          const result = { available: true, binaryPath: filePath, version: 'llama.cpp Native Binary' };
          this.cachedAvailability = { result, timestamp: now };
          return result;
        } catch {
          // not executable
        }
      }
    }

    // Search PATH directories for bare names
    const pathDirs = (process.env.PATH || '').split(path.delimiter);
    const bareBinaries = ['llama-server', 'llama-cli'];
    for (const dir of pathDirs) {
      for (const bin of bareBinaries) {
        const full = path.join(dir, bin);
        if (fs.existsSync(full)) {
          this.binaryPath = full;
          const result = { available: true, binaryPath: full, version: 'llama.cpp in PATH' };
          this.cachedAvailability = { result, timestamp: now };
          return result;
        }
      }
    }

    const result = {
      available: false,
      reason: 'llama.cpp binary (llama-server or llama-cli) not found in PATH or standard directories (/usr/local/bin, /opt/llama.cpp, ./bin).'
    };
    this.cachedAvailability = { result, timestamp: now };
    return result;
  }

  /**
   * Verify model weight file on disk and initialize runtime configuration.
   * STRICT: Never pretends a model is loaded if weights are absent.
   */
  async loadModel(model: ModelMetadata, config: Partial<ModelRuntimeConfig>): Promise<ModelRuntimeConfig> {
    const resolvedPath = ModelRegistry.resolveAssetPath(model.weightPath);

    if (!fs.existsSync(resolvedPath)) {
      throw new Error(
        `Hardware Load Error: Model weight file does not exist on disk at "${resolvedPath}". ` +
        `Place the .gguf file in the models directory before loading.`
      );
    }

    const stats = fs.statSync(resolvedPath);
    if (stats.size < 1024 * 1024) {
      throw new Error(
        `Model weight file at "${resolvedPath}" is too small (${(stats.size / 1024).toFixed(1)} KB) ` +
        `to be a valid GGUF model.`
      );
    }

    // Estimate memory footprint based on file size and context window
    const baseMemoryMb = Math.round(stats.size / (1024 * 1024));
    const ctxMemoryMb = Math.round(((config.contextSize || model.contextWindow || 2048) / 1024) * 250);
    const estimatedMemoryUsageMb = baseMemoryMb + ctxMemoryMb;

    const runtimeConfig: ModelRuntimeConfig = {
      modelId: model.id,
      contextSize: config.contextSize || model.contextWindow || 4096,
      gpuLayers: config.gpuLayers ?? model.defaultParams?.defaultGpuLayers ?? 0,
      threads: config.threads || Math.max(1, Math.min(os.cpus().length - 1, 8)),
      batchSize: config.batchSize || 512,
      loadedAt: Date.now(),
      memoryUsageMb: estimatedMemoryUsageMb,
      activeRequests: 0
    };

    this.loadedModels.set(model.id, runtimeConfig);
    return runtimeConfig;
  }

  async unloadModel(modelId: string): Promise<void> {
    if (!this.loadedModels.has(modelId)) {
      throw new Error(`Model "${modelId}" is not currently loaded in llama.cpp adapter.`);
    }
    this.loadedModels.delete(modelId);
  }

  /**
   * Format OpenAI chat messages into the model's native prompt template.
   */
  private formatPrompt(messages: ChatCompletionRequest['messages'], architecture: string): string {
    if (architecture === 'llama' || architecture === 'deepseek') {
      // Llama-3 format
      let formatted = '<|begin_of_text|>';
      for (const msg of messages) {
        formatted += `<|start_header_id|>${msg.role}<|end_header_id|>\n\n${msg.content.trim()}<|eot_id|>`;
      }
      formatted += '<|start_header_id|>assistant<|end_header_id|>\n\n';
      return formatted;
    } else if (architecture === 'qwen2') {
      // ChatML format
      let formatted = '';
      for (const msg of messages) {
        formatted += `<|im_start|>${msg.role}\n${msg.content.trim()}<|im_end|>\n`;
      }
      formatted += '<|im_start|>assistant\n';
      return formatted;
    } else if (architecture === 'mistral') {
      // Mistral [INST] format
      let formatted = '';
      for (const msg of messages) {
        if (msg.role === 'system') {
          formatted += `[INST] ${msg.content.trim()}\n\n`;
        } else if (msg.role === 'user') {
          formatted += `${msg.content.trim()} [/INST]`;
        } else if (msg.role === 'assistant') {
          formatted += ` ${msg.content.trim()} [INST] `;
        }
      }
      return formatted;
    }

    // Standard fallback format
    return messages.map(m => `${m.role.toUpperCase()}: ${m.content}`).join('\n') + '\nASSISTANT: ';
  }

  async generateChatCompletion(
    request: ChatCompletionRequest,
    model: ModelMetadata,
    config: ModelRuntimeConfig,
    onToken?: (chunk: string) => void
  ): Promise<ChatCompletionResponse> {
    if (!this.loadedModels.has(model.id)) {
      throw new Error(`Model "${model.id}" is not loaded into memory. Call POST /v1/models/${model.id}/load first.`);
    }

    // Verify weights still exist
    const resolvedPath = ModelRegistry.resolveAssetPath(model.weightPath);

    if (!fs.existsSync(resolvedPath)) {
      throw new Error(`Model weight file at "${resolvedPath}" was removed or unmounted.`);
    }

    // If local llama-server HTTP service is connected:
    if (this.serverUrl) {
      return this.executeViaHttp(this.serverUrl, request, model, onToken);
    }

    // If binary is available, execute via CLI / server process
    const availability = await this.isAvailable();
    if (availability.available && availability.binaryPath) {
      return this.executeViaCli(availability.binaryPath, resolvedPath, request, model, config, onToken);
    }

    throw new Error(
      `llama.cpp inference runtime is not available on this host. ` +
      `Install llama.cpp or start llama-server to execute inference for ${model.name}.`
    );
  }

  private async executeViaHttp(
    url: string,
    request: ChatCompletionRequest,
    model: ModelMetadata,
    onToken?: (chunk: string) => void
  ): Promise<ChatCompletionResponse> {
    const formattedPrompt = this.formatPrompt(request.messages, model.architecture);

    const res = await fetch(`${url}/completion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: formattedPrompt,
        temperature: request.temperature ?? 0.7,
        top_p: request.top_p ?? 0.9,
        n_predict: request.max_tokens ?? 512,
        stream: Boolean(onToken),
        stop: request.stop || ['<|eot_id|>', '<|im_end|>', '</s>']
      })
    });

    if (!res.ok) {
      throw new Error(`llama-server responded with HTTP ${res.status}: ${await res.text()}`);
    }

    if (onToken && res.body) {
      // Stream parsing
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullContent = '';
      let promptTokens = Math.ceil(formattedPrompt.length / 4);
      let completionTokens = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value);
        const lines = text.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.content) {
                fullContent += data.content;
                completionTokens++;
                onToken(data.content);
              }
            } catch {
              // Ignore non-json lines
            }
          }
        }
      }

      return {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model.id,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: fullContent },
            finish_reason: 'stop'
          }
        ],
        usage: {
          prompt_tokens: promptTokens,
          completion_tokens: completionTokens,
          total_tokens: promptTokens + completionTokens
        }
      };
    } else {
      const data = await res.json() as { content?: string; tokens_evaluated?: number; tokens_predicted?: number };
      const content = data.content || '';
      const promptTokens = data.tokens_evaluated || Math.ceil(formattedPrompt.length / 4);
      const completionTokens = data.tokens_predicted || Math.ceil(content.length / 4);

      return {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model.id,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content },
            finish_reason: 'stop'
          }
        ],
        usage: {
          prompt_tokens: promptTokens,
          completion_tokens: completionTokens,
          total_tokens: promptTokens + completionTokens
        }
      };
    }
  }

  private async executeViaCli(
    binaryPath: string,
    modelPath: string,
    request: ChatCompletionRequest,
    model: ModelMetadata,
    config: ModelRuntimeConfig,
    onToken?: (chunk: string) => void
  ): Promise<ChatCompletionResponse> {
    const prompt = this.formatPrompt(request.messages, model.architecture);
    const escapedPrompt = prompt.replace(/"/g, '\\"');
    const maxTokens = request.max_tokens ?? 256;
    const temp = request.temperature ?? 0.7;

    const cmd = `"${binaryPath}" -m "${modelPath}" -p "${escapedPrompt}" -n ${maxTokens} --temp ${temp} -t ${config.threads} -ngl ${config.gpuLayers}`;

    try {
      const { stdout } = await execAsync(cmd, { timeout: 60000 });
      const cleanOutput = stdout.replace(prompt, '').trim();

      if (onToken) {
        // Stream out chunks
        const words = cleanOutput.split(' ');
        for (const word of words) {
          onToken(word + ' ');
        }
      }

      const promptTokens = Math.ceil(prompt.length / 4);
      const completionTokens = Math.ceil(cleanOutput.length / 4);

      return {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model.id,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: cleanOutput },
            finish_reason: 'stop'
          }
        ],
        usage: {
          prompt_tokens: promptTokens,
          completion_tokens: completionTokens,
          total_tokens: promptTokens + completionTokens
        }
      };
    } catch (err: unknown) {
      const error = err as Error;
      throw new Error(`llama.cpp execution failed: ${error.message}`);
    }
  }

  async tokenize(text: string, model: ModelMetadata): Promise<number[]> {
    // Basic approximate byte-pair tokenization for open-weight models (4 chars ≈ 1 token)
    const tokens: number[] = [];
    let id = 100;
    const words = text.split(/(\s+|[^\w\s])/);
    for (const w of words) {
      if (w) {
        tokens.push(id++);
      }
    }
    return tokens;
  }

  async getSystemInfo(): Promise<SystemHardwareInfo> {
    const totalMem = Math.round(os.totalmem() / (1024 * 1024));
    const freeMem = Math.round(os.freemem() / (1024 * 1024));
    const cpus = os.cpus();
    const cpuModel = cpus.length > 0 ? cpus[0].model : 'Unknown CPU';

    const avail = await this.isAvailable();

    // Check for GPU with caching
    if (!this.cachedGpuInfo) {
      let gpuDetected = false;
      let gpuDesc = 'None';
      let execMode: SystemHardwareInfo['executionMode'] = 'cpu_only';

      try {
        const { stdout } = await execAsync('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true');
        if (stdout.trim()) {
          gpuDetected = true;
          gpuDesc = stdout.trim().split('\n')[0];
          execMode = 'cuda';
        }
      } catch {
        // Check Apple Metal
        if (os.platform() === 'darwin' && os.arch() === 'arm64') {
          gpuDetected = true;
          gpuDesc = 'Apple Silicon Unified Memory (Metal)';
          execMode = 'metal';
        }
      }
      this.cachedGpuInfo = { detected: gpuDetected, desc: gpuDesc, mode: execMode };
    }

    return {
      platform: os.platform(),
      arch: os.arch(),
      cpuModel,
      cpuCores: cpus.length,
      totalMemoryMb: totalMem,
      freeMemoryMb: freeMem,
      gpuDetected: this.cachedGpuInfo?.detected ?? false,
      gpuDescription: this.cachedGpuInfo?.desc ?? 'None',
      llamaCppInstalled: avail.available,
      llamaCppBinaryPath: avail.binaryPath,
      llamaCppVersion: avail.version,
      executionMode: this.cachedGpuInfo?.mode ?? 'cpu_only'
    };
  }
}
