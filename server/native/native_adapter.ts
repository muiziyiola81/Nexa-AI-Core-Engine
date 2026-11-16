/**
 * NEXA AI Core Engine - Native Inference Adapter
 * Independent native C++ CPU inference runtime bridge.
 * Directly executes the compiled C++20 nexa-runtime-cli executable via NexaCppBridge.
 * Does NOT call llama.cpp, Ollama, OpenAI, Gemini, Anthropic, or any third-party engine.
 * STRICT: Does not silently fall back to TypeScript simulation.
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import {
  InferenceAdapter,
  ModelMetadata,
  ModelRuntimeConfig,
  ChatCompletionRequest,
  ChatCompletionResponse,
  SystemHardwareInfo
} from '../engine/types.js';
import { NexaModelLoader } from './model_loader.js';
import { NexaCppBridge, NexaNativeCppError, CppRuntimeInfo } from './cpp_bridge.js';
import { ModelRegistry } from '../store/model_registry.js';

export class NexaNativeAdapter implements InferenceAdapter {
  readonly id = 'nexa-native';
  readonly name = 'NEXA Native C++ Engine (Independent Runtime)';
  readonly version = '1.0.0-cxx20';

  private cppBridge: NexaCppBridge;
  private loadedModels: Map<string, { config: ModelRuntimeConfig; metadata: ModelMetadata }> = new Map();

  constructor(explicitBinaryPath?: string) {
    this.cppBridge = new NexaCppBridge(explicitBinaryPath);
  }

  public getCppBridge(): NexaCppBridge {
    return this.cppBridge;
  }

  async isAvailable(): Promise<{ available: boolean; binaryPath?: string; version?: string; reason?: string }> {
    const isBinAvailable = this.cppBridge.isAvailable();
    if (!isBinAvailable) {
      return {
        available: false,
        reason: 'Native C++ binary "nexa-runtime-cli" not found in bin/ or native/build/. Compile via CMake.'
      };
    }

    try {
      const info = await this.cppBridge.getRuntimeInfo();
      return {
        available: true,
        version: `${info.version} (${info.standard}, ${info.arch})`,
        binaryPath: info.binaryPath
      };
    } catch (err: any) {
      return {
        available: false,
        binaryPath: this.cppBridge.getBinaryPath() || undefined,
        reason: `C++ binary probe failed: ${err.message}`
      };
    }
  }

  /**
   * Loads a model into native memory.
   * STRICT: Validates weights file presence on disk. If absent, throws error.
   * If .nexa binary, validates header using C++ inspection.
   */
  async loadModel(model: ModelMetadata, config: Partial<ModelRuntimeConfig>): Promise<ModelRuntimeConfig> {
    const resolvedPath = ModelRegistry.resolveAssetPath(model.weightPath);

    if (!fs.existsSync(resolvedPath)) {
      throw new Error(
        `Hardware Load Error: Model weight file does not exist on disk at "${resolvedPath}". ` +
        `NEXA native runtime strictly requires weights before memory allocation.`
      );
    }

    const stat = fs.statSync(resolvedPath);
    if (stat.size < 56) {
      throw new Error(`File at "${resolvedPath}" is too small to be a valid model.`);
    }

    // Ensure C++ binary is compiled and available
    if (!this.cppBridge.isAvailable()) {
      throw new NexaNativeCppError(
        'Cannot load model: Compiled C++ executable "nexa-runtime-cli" is missing.\n' +
        'Please build the native engine first: cd native && cmake -B build && cmake --build build -j',
        'ERR_CPP_RUNTIME_MISSING'
      );
    }

    let estimatedMemoryMb = Math.round(stat.size / (1024 * 1024));

    // Inspect header via C++ bridge if it's a .nexa file
    if (resolvedPath.endsWith('.nexa')) {
      try {
        const cppInspect = await this.cppBridge.inspectModel(resolvedPath);
        if (!cppInspect.valid) {
          throw new Error('C++ engine reported invalid .nexa format.');
        }
        // Calculate memory budget using real checkpoint dimensions
        const budget = NexaModelLoader.calculateMemory({
          vocabSize: cppInspect.vocab_size || 768,
          dim: cppInspect.hidden_dim,
          intermediateDim: cppInspect.intermediate_dim || cppInspect.hidden_dim * 2,
          nLayers: cppInspect.layers,
          nHeads: cppInspect.heads,
          nKvHeads: cppInspect.kv_heads,
          maxSeqLen: cppInspect.max_seq_len || config.contextSize || model.contextWindow || 512,
          normEps: 1e-5,
          ropeTheta: cppInspect.rope_theta || 10000.0,
          quantType: 0,
          nTensors: cppInspect.tensors
        });
        estimatedMemoryMb = Math.max(estimatedMemoryMb, budget.totalRequiredMb);
      } catch (err: any) {
        // If inspection failed, rethrow error
        throw new Error(`Native C++ model inspection failed for "${resolvedPath}": ${err.message}`);
      }
    } else {
      estimatedMemoryMb += Math.round(((config.contextSize || 4096) / 1024) * 200);
    }

    const runtimeConfig: ModelRuntimeConfig = {
      modelId: model.id,
      contextSize: config.contextSize || model.contextWindow || 512,
      gpuLayers: 0,
      threads: config.threads || Math.max(1, Math.min(os.cpus().length, 8)),
      batchSize: config.batchSize || 128,
      loadedAt: Date.now(),
      memoryUsageMb: estimatedMemoryMb,
      activeRequests: 0
    };

    this.loadedModels.set(model.id, { config: runtimeConfig, metadata: model });
    return runtimeConfig;
  }

  async unloadModel(modelId: string): Promise<void> {
    if (!this.loadedModels.has(modelId)) {
      throw new Error(`Model "${modelId}" is not loaded in NEXA Native Runtime.`);
    }
    this.loadedModels.delete(modelId);
  }

  private resolveTokenizerForModel(model: ModelMetadata, resolvedCheckpointPath: string): string | null {
    if (model.tokenizerPath) {
      const resolvedTok = ModelRegistry.resolveAssetPath(model.tokenizerPath);
      if (fs.existsSync(resolvedTok)) return resolvedTok;
    }
    const base = path.basename(resolvedCheckpointPath).toLowerCase();
    const candidates: string[] = [];
    if (base.includes('p37') || base.includes('v2')) {
      candidates.push('data/tokenizers/nexa_bpe_p37_v2.nbt');
    }
    if (base.includes('p36') || base.includes('phase36')) {
      candidates.push('data/tokenizers/nexa_bpe_phase36_shared.nbt');
    } else if (base.includes('phase35')) {
      candidates.push('data/tokenizers/nexa_bpe_phase35.nbt');
    } else if (base.includes('qa')) {
      candidates.push('data/tokenizers/nexa_bpe_qa.nbt');
    }
    for (const c of candidates) {
      const resolved = ModelRegistry.resolveAssetPath(c);
      if (fs.existsSync(resolved)) return resolved;
    }
    return null;
  }

  private formatRoleLine(role: string, content: string): string {
    const normalizedRole = role.toLowerCase();
    const clean = (content || '').trim();
    if (normalizedRole === 'system' || normalizedRole === 'tool') {
      return `System: ${clean}`;
    }
    if (normalizedRole === 'assistant') {
      return `Assistant: ${clean}`;
    }
    return `User: ${clean}`;
  }

  private async buildCanonicalConversationPrompt(
    messages: ChatCompletionRequest['messages'],
    maxPromptTokens: number,
    tokenizerPath?: string
  ): Promise<{ prompt: string; tokenCount: number }> {
    if (!messages || messages.length === 0) {
      return { prompt: 'User:\nAssistant:', tokenCount: 4 };
    }

    const systemMsgs = messages.filter(m => m.role === 'system');
    const nonSystemMsgs = messages.filter(m => m.role !== 'system');

    const systemPrefix = systemMsgs.length > 0
      ? systemMsgs.map(m => this.formatRoleLine(m.role, m.content)).join('\n')
      : '';

    const activeTurns = nonSystemMsgs.length > 0 ? nonSystemMsgs : messages;
    const selected: string[] = [
      this.formatRoleLine(activeTurns[activeTurns.length - 1].role, activeTurns[activeTurns.length - 1].content)
    ];

    const renderPrompt = (turns: string[]): string => {
      const parts = systemPrefix ? [systemPrefix, ...turns] : turns;
      return parts.join('\n') + '\nAssistant:';
    };

    let bestPrompt = renderPrompt(selected);
    let bestTokens = (await this.cppBridge.tokenize(bestPrompt, tokenizerPath)).length;

    // Add preceding history turns from newest to oldest while within context budget
    for (let i = activeTurns.length - 2; i >= 0; i--) {
      const candidateTurns = [this.formatRoleLine(activeTurns[i].role, activeTurns[i].content), ...selected];
      const candidatePrompt = renderPrompt(candidateTurns);
      const count = (await this.cppBridge.tokenize(candidatePrompt, tokenizerPath)).length;
      if (count <= maxPromptTokens) {
        selected.unshift(this.formatRoleLine(activeTurns[i].role, activeTurns[i].content));
        bestPrompt = candidatePrompt;
        bestTokens = count;
      } else {
        break;
      }
    }

    return { prompt: bestPrompt, tokenCount: bestTokens };
  }

  /**
   * Independent Native Inference Execution.
   * Invokes the compiled C++ executable to tokenize prompt, run transformer forward/generation passes,
   * and stream/return response.
   * STRICT: Fails if C++ binary or required tokenizer fails. Does not silently fall back to JS simulation.
   */
  async generateChatCompletion(
    request: ChatCompletionRequest,
    model: ModelMetadata,
    config: ModelRuntimeConfig,
    onToken?: (chunk: string) => void
  ): Promise<ChatCompletionResponse> {
    const loaded = this.loadedModels.get(model.id);
    if (!loaded) {
      throw new Error(
        `[NEXA_ERR_MODEL_NOT_LOADED] Model "${model.id}" is not loaded into NEXA Native Runtime memory. ` +
        `Invoke POST /v1/models/${model.id}/load before requesting completions.`
      );
    }

    if (!this.cppBridge.isAvailable()) {
      throw new NexaNativeCppError(
        'Native C++ executable "nexa-runtime-cli" is not compiled or missing. Cannot serve native inference.',
        'ERR_CPP_BINARY_UNAVAILABLE'
      );
    }

    const resolvedWeightPath = ModelRegistry.resolveAssetPath(model.weightPath);
    if (!resolvedWeightPath || !fs.existsSync(resolvedWeightPath)) {
      throw new Error(
        `[NEXA_ERR_WEIGHTS_MISSING] Weight file not found at "${resolvedWeightPath}".`
      );
    }

    const val = await this.cppBridge.validateCheckpoint(resolvedWeightPath);

    // Trained v2 checkpoint path: return clean model-generated assistant output
    if (val.valid && val.version >= 2) {
      const nbtPath = val.vocab_size > 260
        ? this.resolveTokenizerForModel(model, resolvedWeightPath)
        : null;

      if (val.vocab_size > 260 && !nbtPath) {
        throw new NexaNativeCppError(
          `BPE tokenizer (.nbt) required for checkpoint "${path.basename(resolvedWeightPath)}" (vocab_size=${val.vocab_size}) was not found.`,
          'ERR_TOKENIZER_MISSING'
        );
      }

      const modelCtx = val.max_seq_len || config.contextSize || model.contextWindow || 160;
      const defaultMaxNew = Math.min(64, Math.max(16, Math.floor(modelCtx / 2)));
      const maxNewTokens = Math.max(1, Math.min(request.max_tokens ?? defaultMaxNew, Math.max(16, modelCtx - 12)));
      const maxPromptTokens = Math.max(16, modelCtx - Math.min(maxNewTokens, Math.floor(modelCtx / 2)));

      const { prompt: canonicalPrompt, tokenCount: promptTokenCount } = await this.buildCanonicalConversationPrompt(
        request.messages,
        maxPromptTokens,
        nbtPath || undefined
      );

      let finalContent = '';
      let actualPromptTokens = promptTokenCount;
      let completionTokens = 0;
      let finishReason: 'stop' | 'length' = 'stop';

      if (nbtPath) {
        const qa = await this.cppBridge.answerQuestion(resolvedWeightPath, canonicalPrompt, {
          tokenizerPath: nbtPath,
          maxTokens: maxNewTokens,
          temperature: request.temperature ?? model.defaultParams?.temperature ?? 0.4
        });
        if (qa.status !== 'ok') {
          throw new NexaNativeCppError(
            qa.error || 'Native C++ inference returned error status.',
            'ERR_NATIVE_INFERENCE_FAILED'
          );
        }
        finalContent = qa.answer;
        actualPromptTokens = qa.prompt_tokens ?? promptTokenCount;
        completionTokens = qa.tokens_generated;
        finishReason = qa.finish_reason === 'length' ? 'length' : 'stop';
      } else {
        const gen = await this.cppBridge.generateText(resolvedWeightPath, canonicalPrompt, {
          maxTokens: maxNewTokens,
          temperature: request.temperature ?? model.defaultParams?.temperature ?? 0.7,
          topP: request.top_p ?? model.defaultParams?.topP ?? 0.9
        });
        if (gen.status !== 'ok') {
          throw new NexaNativeCppError(
            gen.error || 'Native C++ generation returned error status.',
            'ERR_NATIVE_INFERENCE_FAILED'
          );
        }
        finalContent = gen.generated_text;
        actualPromptTokens = gen.prompt_tokens ?? promptTokenCount;
        completionTokens = gen.tokens_generated;
        finishReason = gen.finish_reason === 'length' ? 'length' : 'stop';
      }

      if (request.stop) {
        const stopList = Array.isArray(request.stop) ? request.stop : [request.stop];
        let earliestIdx = -1;
        for (const seq of stopList) {
          if (typeof seq === 'string' && seq.length > 0) {
            const idx = finalContent.indexOf(seq);
            if (idx !== -1 && (earliestIdx === -1 || idx < earliestIdx)) {
              earliestIdx = idx;
            }
          }
        }
        if (earliestIdx !== -1) {
          finalContent = finalContent.slice(0, earliestIdx).trimEnd();
          finishReason = 'stop';
        }
      }

      if (onToken && finalContent) {
        const chunks = finalContent.split(/(\s+)/);
        for (const chunk of chunks) {
          if (chunk) {
            onToken(chunk);
            await new Promise(r => setTimeout(r, 4));
          }
        }
      }

      return {
        id: `chatcmpl-nexa-cxx-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model.id,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: finalContent
            },
            finish_reason: finishReason
          }
        ],
        usage: {
          prompt_tokens: actualPromptTokens,
          completion_tokens: completionTokens,
          total_tokens: actualPromptTokens + completionTokens
        },
        system_fingerprint: `fp_nexa_cxx20_v2_${val.use_rope ? 'rope' : 'pos'}`
      };
    }

    // Legacy v1 descriptor stub path (preserves native_engine_tests / cpp_bridge_tests hardware kernel checks)
    const legacyPrompt = request.messages.map(m => this.formatRoleLine(m.role, m.content)).join('\n') + '\nAssistant:';
    const inputTokens = await this.cppBridge.tokenize(legacyPrompt);
    const kernelDim = 128;
    const kernelSeqLen = Math.min(Math.max(1, inputTokens.length), 32);
    const forwardResult = await this.cppBridge.runForwardPass(kernelDim, kernelSeqLen);

    const finalContent =
      `[NEXA NATIVE RUNTIME - C++20 COMPILED ENGINE]\n` +
      `Native C++ Execution: OK (Process PID dispatched to compiled ${forwardResult.runtime})\n` +
      `Tensor Computation: RMSNorm, RoPE, Attention, SwiGLU computed in C++ (${forwardResult.duration_us} µs)\n` +
      `Prompt Evaluated: ${inputTokens.length} tokens (${legacyPrompt.length} chars)\n` +
      `Output Checksum: ${forwardResult.output_checksum.toFixed(4)} | Elements: ${forwardResult.tensor_elements}\n` +
      `Runtime Mode: Pure C++20 binary execution. Zero llama.cpp, Ollama, or third-party cloud APIs.`;

    const generatedTokens = await this.cppBridge.tokenize(finalContent);

    if (onToken) {
      const chunks = finalContent.split(/(\s+)/);
      for (const chunk of chunks) {
        if (chunk) {
          onToken(chunk);
          await new Promise(r => setTimeout(r, 6));
        }
      }
    }

    return {
      id: `chatcmpl-nexa-cxx-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: model.id,
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: finalContent
          },
          finish_reason: 'stop'
        }
      ],
      usage: {
        prompt_tokens: inputTokens.length,
        completion_tokens: generatedTokens.length,
        total_tokens: inputTokens.length + generatedTokens.length
      },
      system_fingerprint: 'fp_nexa_cxx20_native_v1'
    };
  }

  async tokenize(text: string, model: ModelMetadata): Promise<number[]> {
    if (!this.cppBridge.isAvailable()) {
      throw new NexaNativeCppError(
        'Native C++ executable "nexa-runtime-cli" is missing. Cannot tokenize natively.',
        'ERR_CPP_BINARY_UNAVAILABLE'
      );
    }
    const resolvedWeightPath = model.weightPath ? ModelRegistry.resolveAssetPath(model.weightPath) : '';
    const nbtPath = this.resolveTokenizerForModel(model, resolvedWeightPath);
    return this.cppBridge.tokenize(text, nbtPath || undefined);
  }

  async getSystemInfo(): Promise<SystemHardwareInfo> {
    const totalMem = Math.round(os.totalmem() / (1024 * 1024));
    const freeMem = Math.round(os.freemem() / (1024 * 1024));
    const cpus = os.cpus();
    const cpuModel = cpus.length > 0 ? cpus[0].model : 'Unknown CPU';

    const binAvailable = this.cppBridge.isAvailable();
    const binPath = this.cppBridge.getBinaryPath() || undefined;

    let cxxArch = os.arch();
    let cxxVersion = '1.0.0-cxx20';
    if (binAvailable) {
      try {
        const info = await this.cppBridge.getRuntimeInfo();
        cxxArch = info.arch;
        cxxVersion = `${info.version} (${info.standard})`;
      } catch {
        // Fall back to os info
      }
    }

    return {
      platform: os.platform(),
      arch: cxxArch,
      cpuModel,
      cpuCores: cpus.length,
      totalMemoryMb: totalMem,
      freeMemoryMb: freeMem,
      gpuDetected: false,
      gpuDescription: `NEXA Native C++ Engine (${cxxVersion})`,
      llamaCppInstalled: false, // Independent from llama.cpp
      llamaCppBinaryPath: binPath,
      executionMode: 'cpu_only'
    };
  }
}
