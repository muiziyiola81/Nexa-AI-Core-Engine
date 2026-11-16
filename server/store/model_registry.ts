/**
 * NEXA AI Core Engine - Model Registry
 * Stores metadata, disk paths, quantization parameters, and installation status
 * for locally hosted open-weight models (.gguf and native .nexa checkpoints).
 */

import fs from 'fs';
import path from 'path';
import { ModelMetadata } from '../engine/types.js';
import { getDataDir, getModelsDir } from './config.js';

export class ModelRegistry {
  private models: Map<string, ModelMetadata> = new Map();
  private storageFile: string;

  constructor() {
    this.storageFile = path.join(getDataDir(), 'models_registry.json');
    this.ensureDirectories();
    this.initDefaultModels();
    if (process.env.AUTO_REGISTER_NEXA_CHECKPOINTS !== 'false') {
      this.autoRegisterNativeCheckpoints();
    }
    this.loadPersistedModels();
    this.refreshInstallationStatus();
  }

  public static resolveAssetPath(assetPath: string): string {
    if (!assetPath) return assetPath;
    if (path.isAbsolute(assetPath)) {
      return assetPath;
    }

    const cwdResolved = path.resolve(process.cwd(), assetPath);
    if (fs.existsSync(cwdResolved)) {
      return cwdResolved;
    }

    const modelsDir = getModelsDir();
    if (assetPath.startsWith('models/') || assetPath.startsWith('models\\')) {
      const rel = assetPath.slice('models/'.length);
      const inModelsDir = path.join(modelsDir, rel);
      if (fs.existsSync(inModelsDir)) {
        return inModelsDir;
      }
    }

    const byBaseInModels = path.join(modelsDir, path.basename(assetPath));
    if (fs.existsSync(byBaseInModels)) {
      return byBaseInModels;
    }

    const dataDir = getDataDir();
    if (assetPath.startsWith('data/') || assetPath.startsWith('data\\')) {
      const rel = assetPath.slice('data/'.length);
      const inDataDir = path.join(dataDir, rel);
      if (fs.existsSync(inDataDir)) {
        return inDataDir;
      }
    }

    return cwdResolved;
  }

  private ensureDirectories() {
    const modelsDir = getModelsDir();
    if (!fs.existsSync(modelsDir)) {
      try {
        fs.mkdirSync(modelsDir, { recursive: true });
      } catch {
        // ignore if read-only volume
      }
    }
    const dataDir = getDataDir();
    if (!fs.existsSync(dataDir)) {
      try {
        fs.mkdirSync(dataDir, { recursive: true });
      } catch {
        // ignore
      }
    }
  }

  private autoRegisterNativeCheckpoints() {
    const bundledCheckpoints: Array<{
      id: string;
      name: string;
      weightPath: string;
      tokenizerPath?: string;
      parameterSize: string;
      contextWindow: number;
      description: string;
    }> = [
      {
        id: 'nexa-p37-v2',
        name: 'NEXA-1 Phase 3.7 RoPE Tied BPE (Native C++20)',
        weightPath: 'data/checkpoints/nexa_p37_v2_best.nexa',
        tokenizerPath: 'data/tokenizers/nexa_bpe_p37_v2.nbt',
        parameterSize: '1.05M',
        contextWindow: 160,
        description:
          'Native C++20 RoPE + Tied-Embeddings BPE transformer checkpoint (4L, 4H, 128D, FFN 384, vocab 1536, ctx 160) for multi-turn NEXA CPU inference.'
      },
      {
        id: 'nexa-p36-tier3',
        name: 'NEXA-1 Phase 3.6 Conversational BPE (Native C++20)',
        weightPath: 'data/checkpoints/nexa_p36_tier3_conv_best.nexa',
        tokenizerPath: 'data/tokenizers/nexa_bpe_phase36_shared.nbt',
        parameterSize: '340K',
        contextWindow: 512,
        description:
          'Native C++20 trained BPE transformer checkpoint (2L, 4H, 96D, vocab 768) for independent NEXA CPU inference.'
      },
      {
        id: 'nexa-nano-step-40',
        name: 'NEXA-1 Nano v1.0 Byte-Level (Native C++20)',
        weightPath: 'data/checkpoints/nexa_nano_step_40.nexa',
        parameterSize: '117K',
        contextWindow: 512,
        description:
          'Compact native C++20 byte-level transformer checkpoint (2L, 2H, 64D, vocab 256) bundled for instant cold starts.'
      },
      {
        id: 'nexa-p35-micro',
        name: 'NEXA-1 Micro 4L BPE (Native C++20)',
        weightPath: 'data/checkpoints/nexa_phase35_micro_best.nexa',
        tokenizerPath: 'data/tokenizers/nexa_bpe_phase35.nbt',
        parameterSize: '860K',
        contextWindow: 512,
        description:
          '4-layer native C++20 BPE transformer checkpoint (4L, 4H, 128D, vocab 768) for multi-domain CPU inference.'
      }
    ];

    for (const item of bundledCheckpoints) {
      const resolved = ModelRegistry.resolveAssetPath(item.weightPath);
      if (fs.existsSync(resolved)) {
        try {
          const stat = fs.statSync(resolved);
          if (stat.size >= 56) {
            this.models.set(item.id, {
              id: item.id,
              name: item.name,
              architecture: 'transformer',
              format: 'nexa',
              parameterSize: item.parameterSize,
              quantization: 'FP32',
              contextWindow: item.contextWindow,
              weightPath: item.weightPath,
              tokenizerPath: item.tokenizerPath,
              fileSizeBytes: stat.size,
              actualDiskSizeBytes: stat.size,
              isInstalled: true,
              description: item.description,
              defaultParams: {
                temperature: 0.5,
                topP: 0.9,
                defaultGpuLayers: 0
              }
            });
          }
        } catch {
          // ignore unreadable checkpoint
        }
      }
    }
  }

  private initDefaultModels() {
    const defaults: ModelMetadata[] = [
      {
        id: 'llama-3-8b-instruct',
        name: 'Meta Llama 3 8B Instruct',
        architecture: 'llama',
        format: 'gguf',
        parameterSize: '8B',
        quantization: 'Q4_K_M',
        contextWindow: 8192,
        weightPath: 'models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf',
        fileSizeBytes: 4920000000,
        isInstalled: false,
        description: 'Meta\'s premier open-weights general reasoning and dialogue model optimized for llama.cpp.',
        sourceUrl: 'https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF',
        defaultParams: {
          temperature: 0.7,
          topP: 0.9,
          defaultGpuLayers: 33
        }
      },
      {
        id: 'mistral-7b-instruct',
        name: 'Mistral 7B Instruct v0.3',
        architecture: 'mistral',
        format: 'gguf',
        parameterSize: '7.3B',
        quantization: 'Q4_K_M',
        contextWindow: 32768,
        weightPath: 'models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf',
        fileSizeBytes: 4370000000,
        isInstalled: false,
        description: 'Fast, high-efficiency open weights with native function calling support and 32k context.',
        sourceUrl: 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3',
        defaultParams: {
          temperature: 0.6,
          topP: 0.95,
          defaultGpuLayers: 33
        }
      },
      {
        id: 'qwen-2.5-7b-instruct',
        name: 'Qwen 2.5 7B Instruct',
        architecture: 'qwen2',
        format: 'gguf',
        parameterSize: '7.6B',
        quantization: 'Q4_K_M',
        contextWindow: 32768,
        weightPath: 'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf',
        fileSizeBytes: 4680000000,
        isInstalled: false,
        description: 'Alibaba Cloud\'s state-of-the-art multilingual and coding open-weight model.',
        sourceUrl: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF',
        defaultParams: {
          temperature: 0.7,
          topP: 0.8,
          defaultGpuLayers: 28
        }
      },
      {
        id: 'phi-3-mini-4k-instruct',
        name: 'Microsoft Phi-3 Mini 4K',
        architecture: 'phi3',
        format: 'gguf',
        parameterSize: '3.8B',
        quantization: 'Q4_K_M',
        contextWindow: 4096,
        weightPath: 'models/Phi-3-mini-4k-instruct-q4.gguf',
        fileSizeBytes: 2390000000,
        isInstalled: false,
        description: 'Ultra-compact model with strong reasoning, ideal for CPU-only and edge servers.',
        sourceUrl: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf',
        defaultParams: {
          temperature: 0.5,
          topP: 0.9,
          defaultGpuLayers: 0
        }
      },
      {
        id: 'deepseek-r1-distill-qwen-7b',
        name: 'DeepSeek-R1 Distill Qwen 7B',
        architecture: 'deepseek',
        format: 'gguf',
        parameterSize: '7B',
        quantization: 'Q4_K_M',
        contextWindow: 32768,
        weightPath: 'models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf',
        fileSizeBytes: 4680000000,
        isInstalled: false,
        description: 'Open-weights reasoning model distilled from DeepSeek-R1 reinforcement learning.',
        sourceUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        defaultParams: {
          temperature: 0.6,
          topP: 0.95,
          defaultGpuLayers: 33
        }
      },
      {
        id: 'tinyllama-1.1b-chat',
        name: 'TinyLlama 1.1B Chat v1.0',
        architecture: 'llama',
        format: 'gguf',
        parameterSize: '1.1B',
        quantization: 'Q4_K_M',
        contextWindow: 2048,
        weightPath: 'models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
        fileSizeBytes: 669000000,
        isInstalled: false,
        description: 'Compact 1.1B parameter Llama architecture model for instant lightweight deployment.',
        sourceUrl: 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
        defaultParams: {
          temperature: 0.7,
          topP: 0.9,
          defaultGpuLayers: 0
        }
      }
    ];

    for (const m of defaults) {
      this.models.set(m.id, m);
    }
  }

  private loadPersistedModels() {
    try {
      if (fs.existsSync(this.storageFile)) {
        const raw = fs.readFileSync(this.storageFile, 'utf-8');
        const list = JSON.parse(raw) as ModelMetadata[];
        for (const item of list) {
          this.models.set(item.id, item);
        }
      }
    } catch (err) {
      console.warn('Could not load persisted models registry:', err);
    }
  }

  private persist() {
    try {
      const list = Array.from(this.models.values());
      fs.writeFileSync(this.storageFile, JSON.stringify(list, null, 2), 'utf-8');
    } catch (err) {
      console.error('Failed to persist model registry:', err);
    }
  }

  public refreshInstallationStatus(): void {
    for (const [id, model] of this.models.entries()) {
      const resolved = ModelRegistry.resolveAssetPath(model.weightPath);

      if (fs.existsSync(resolved)) {
        try {
          const stats = fs.statSync(resolved);
          const minSize = model.format === 'nexa' ? 56 : 1024 * 1024;
          model.isInstalled = stats.size >= minSize;
          model.actualDiskSizeBytes = stats.size;
        } catch {
          model.isInstalled = false;
        }
      } else {
        model.isInstalled = false;
        model.actualDiskSizeBytes = undefined;
      }
      this.models.set(id, model);
    }
  }

  public getAll(): ModelMetadata[] {
    this.refreshInstallationStatus();
    return Array.from(this.models.values());
  }

  public get(id: string): ModelMetadata | undefined {
    this.refreshInstallationStatus();
    return this.models.get(id);
  }

  public register(model: Omit<ModelMetadata, 'isInstalled'>): ModelMetadata {
    if (this.models.has(model.id)) {
      throw new Error(`Model with ID "${model.id}" already exists in registry.`);
    }

    const newModel: ModelMetadata = {
      ...model,
      isInstalled: false
    };

    const resolved = ModelRegistry.resolveAssetPath(newModel.weightPath);

    if (fs.existsSync(resolved)) {
      const stat = fs.statSync(resolved);
      const minSize = newModel.format === 'nexa' ? 56 : 1024 * 1024;
      newModel.isInstalled = stat.size >= minSize;
      newModel.actualDiskSizeBytes = stat.size;
    }

    this.models.set(newModel.id, newModel);
    this.persist();
    return newModel;
  }

  public unregister(id: string): boolean {
    if (!this.models.has(id)) {
      return false;
    }
    this.models.delete(id);
    this.persist();
    return true;
  }
}
