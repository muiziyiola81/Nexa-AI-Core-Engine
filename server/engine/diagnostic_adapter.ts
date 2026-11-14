/**
 * NEXA AI Core Engine - Diagnostic & Test Runtime Adapter
 * A transparent diagnostic runtime used for API compatibility validation,
 * automated unit tests, and pipeline latency profiling.
 * 
 * STRICT COMPLIANCE: Clearly marks all responses as synthetic diagnostics.
 * Never claims to have trained models or to be a proprietary cloud LLM.
 */

import os from 'os';
import {
  InferenceAdapter,
  ModelMetadata,
  ModelRuntimeConfig,
  ChatCompletionRequest,
  ChatCompletionResponse,
  SystemHardwareInfo
} from './types.js';

export class DiagnosticAdapter implements InferenceAdapter {
  readonly id = 'nexa-diagnostic';
  readonly name = 'NEXA Diagnostic & Automated Test Engine';
  readonly version = '1.0.0-core';

  private loadedModels: Map<string, ModelRuntimeConfig> = new Map();

  async isAvailable(): Promise<{ available: boolean; version: string; binaryPath: string }> {
    return {
      available: true,
      version: this.version,
      binaryPath: 'internal://nexa-core-diagnostic-runtime'
    };
  }

  async loadModel(model: ModelMetadata, config: Partial<ModelRuntimeConfig>): Promise<ModelRuntimeConfig> {
    const runtimeConfig: ModelRuntimeConfig = {
      modelId: model.id,
      contextSize: config.contextSize || model.contextWindow || 2048,
      gpuLayers: config.gpuLayers ?? 0,
      threads: config.threads || 4,
      batchSize: config.batchSize || 128,
      loadedAt: Date.now(),
      memoryUsageMb: 128, // Diagnostic runtime memory baseline
      activeRequests: 0
    };

    this.loadedModels.set(model.id, runtimeConfig);
    return runtimeConfig;
  }

  async unloadModel(modelId: string): Promise<void> {
    if (!this.loadedModels.has(modelId)) {
      throw new Error(`Model "${modelId}" is not loaded in Diagnostic Engine.`);
    }
    this.loadedModels.delete(modelId);
  }

  async generateChatCompletion(
    request: ChatCompletionRequest,
    model: ModelMetadata,
    _config: ModelRuntimeConfig,
    onToken?: (chunk: string) => void
  ): Promise<ChatCompletionResponse> {
    if (!this.loadedModels.has(model.id)) {
      throw new Error(
        `[NEXA_ERR_MODEL_UNLOADED] Model "${model.id}" is registered but not loaded into memory. ` +
        `Send a POST /v1/models/${model.id}/load request to allocate runtime memory first.`
      );
    }

    const lastMessage = request.messages[request.messages.length - 1];
    const userPrompt = lastMessage ? lastMessage.content : '';

    // Formulate a transparent diagnostic payload
    const syntheticResponse = 
      `[NEXA CORE INFRASTRUCTURE RUNTIME - DIAGNOSTIC RESPONSE]\n` +
      `Target Model ID: ${model.id} (${model.architecture.toUpperCase()}, Quant: ${model.quantization})\n` +
      `System Status: Engine pipeline verified. OpenAI-compatible request structure accepted.\n` +
      `Input Prompt Length: ${userPrompt.length} chars (${Math.ceil(userPrompt.length / 4)} tokens)\n` +
      `Inference Runtime: Diagnostic loopback active. To run neural inference, configure local llama.cpp binary and mount .gguf model weights.`;

    const promptTokens = Math.ceil(
      request.messages.reduce((acc, m) => acc + m.content.length, 0) / 4
    );
    const completionTokens = Math.ceil(syntheticResponse.length / 4);

    if (onToken) {
      // Simulate realistic token streaming with slight delay for testing SSE
      const tokens = syntheticResponse.split(/(\s+)/);
      for (const token of tokens) {
        if (token) {
          onToken(token);
          // brief yield
          await new Promise(r => setTimeout(r, 8));
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
          message: {
            role: 'assistant',
            content: syntheticResponse
          },
          finish_reason: 'stop'
        }
      ],
      usage: {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: promptTokens + completionTokens
      },
      system_fingerprint: 'fp_nexa_core_diag_v1'
    };
  }

  async tokenize(text: string, _model: ModelMetadata): Promise<number[]> {
    const tokens: number[] = [];
    for (let i = 0; i < text.length; i += 4) {
      tokens.push(1000 + i);
    }
    return tokens;
  }

  async getSystemInfo(): Promise<SystemHardwareInfo> {
    return {
      platform: os.platform(),
      arch: os.arch(),
      cpuModel: os.cpus()[0]?.model || 'Generic Host CPU',
      cpuCores: os.cpus().length,
      totalMemoryMb: Math.round(os.totalmem() / (1024 * 1024)),
      freeMemoryMb: Math.round(os.freemem() / (1024 * 1024)),
      gpuDetected: false,
      gpuDescription: 'Hardware acceleration disabled in diagnostic mode',
      llamaCppInstalled: false,
      executionMode: 'mock_simulation'
    };
  }
}
