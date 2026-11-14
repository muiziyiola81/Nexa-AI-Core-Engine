/**
 * NEXA AI Core Engine - Models Management
 * Register, inspect, load into memory, and unload open-weight models.
 */

import React, { useState } from 'react';
import { ModelItem } from '../types';
import { HardDrive, Play, Square, Plus, Trash2, CheckCircle2, AlertCircle, RefreshCw, X, Sliders } from 'lucide-react';

interface ModelsTabProps {
  models: ModelItem[];
  onLoadModel: (modelId: string, options: { contextSize?: number; gpuLayers?: number; threads?: number }) => Promise<void>;
  onUnloadModel: (modelId: string) => Promise<void>;
  onRegisterModel: (modelData: any) => Promise<void>;
  onUnregisterModel: (modelId: string) => Promise<void>;
  onRefresh: () => void;
  activeAdapterId: string;
}

export const ModelsTab: React.FC<ModelsTabProps> = ({
  models,
  onLoadModel,
  onUnloadModel,
  onRegisterModel,
  onUnregisterModel,
  onRefresh,
  activeAdapterId
}) => {
  const [loadModalModel, setLoadModalModel] = useState<ModelItem | null>(null);
  const [loadingModelId, setLoadingModelId] = useState<string | null>(null);
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  // Load config form state
  const [contextSize, setContextSize] = useState<number>(4096);
  const [gpuLayers, setGpuLayers] = useState<number>(33);
  const [threads, setThreads] = useState<number>(4);

  // Register form state
  const [regId, setRegId] = useState('');
  const [regName, setRegName] = useState('');
  const [regArch, setRegArch] = useState('llama');
  const [regParam, setRegParam] = useState('7B');
  const [regQuant, setRegQuant] = useState('Q4_K_M');
  const [regPath, setRegPath] = useState('models/custom-model.Q4_K_M.gguf');
  const [regCtx, setRegCtx] = useState(4096);
  const [regDesc, setRegDesc] = useState('');

  const handleOpenLoad = (model: ModelItem) => {
    setLoadModalModel(model);
    setContextSize(model.nexa_metadata.contextWindow || 4096);
    setGpuLayers(model.nexa_metadata.architecture === 'phi3' ? 0 : 33);
    setActionError(null);
  };

  const executeLoad = async () => {
    if (!loadModalModel) return;
    setLoadingModelId(loadModalModel.id);
    setActionError(null);
    try {
      await onLoadModel(loadModalModel.id, { contextSize, gpuLayers, threads });
      setLoadModalModel(null);
    } catch (err: unknown) {
      setActionError((err as Error).message);
    } finally {
      setLoadingModelId(null);
    }
  };

  const executeUnload = async (modelId: string) => {
    setLoadingModelId(modelId);
    setActionError(null);
    try {
      await onUnloadModel(modelId);
    } catch (err: unknown) {
      setActionError((err as Error).message);
    } finally {
      setLoadingModelId(null);
    }
  };

  const handleRegisterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setActionError(null);
    try {
      await onRegisterModel({
        id: regId.trim(),
        name: regName.trim(),
        architecture: regArch,
        parameterSize: regParam.trim(),
        quantization: regQuant.trim(),
        weightPath: regPath.trim(),
        contextWindow: Number(regCtx),
        description: regDesc.trim() || 'Custom registered GGUF model.'
      });
      setShowRegisterModal(false);
      // Reset
      setRegId('');
      setRegName('');
    } catch (err: unknown) {
      setActionError((err as Error).message);
    }
  };

  const formatBytes = (bytes: number) => {
    if (!bytes) return '0 B';
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) return `${gb.toFixed(2)} GB`;
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(0)} MB`;
  };

  return (
    <div className="space-y-6">
      {/* Editorial Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">Model Management & Registry</h1>
          <div className="flex items-center gap-2 mt-1 text-xs text-neutral-400">
            <span>Open-Weight Catalog</span>
            <span aria-hidden="true">·</span>
            <span>GGUF Format</span>
            <span aria-hidden="true">·</span>
            <span>Local Filesystem Verification</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowRegisterModal(true)}
            className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-white bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 rounded-md transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Register Model</span>
          </button>
        </div>
      </div>

      {actionError && (
        <div className="p-3 bg-red-950/50 border border-red-800 rounded-lg text-xs text-red-300 flex items-start gap-2">
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold text-red-200">Execution Error: </span>
            {actionError}
          </div>
          <button onClick={() => setActionError(null)} className="text-red-400 hover:text-white">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Notice on installation and adapter */}
      <div className="p-4 bg-neutral-900 border border-neutral-800 rounded-lg text-xs text-neutral-400 space-y-1">
        <p className="font-medium text-neutral-200">Strict Model Execution Guarantee</p>
        <p>
          Inference requires weights to exist on disk at the configured <code className="text-neutral-300">weightPath</code>.
          If weights are missing, the engine will refuse to load them with a hardware error. To run synthetic compatibility tests without weights, switch to the Diagnostic Engine on the Overview tab.
        </p>
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {models.map(model => {
          const meta = model.nexa_metadata;
          const isLoaded = meta.state === 'loaded';
          const isInstalled = meta.isInstalled;
          const isLoading = loadingModelId === model.id || meta.state === 'loading';

          return (
            <div
              key={model.id}
              className="p-5 bg-neutral-900 border border-neutral-800 rounded-lg flex flex-col justify-between space-y-4"
            >
              <div>
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h2 className="text-sm font-semibold text-white">{meta.name}</h2>
                    <div className="text-xs font-mono text-neutral-500 mt-0.5">{model.id}</div>
                  </div>

                  {/* Clean text status */}
                  <div className="text-right shrink-0">
                    {isLoaded ? (
                      <span className="text-xs font-mono text-emerald-400">LOADED IN RAM</span>
                    ) : isInstalled ? (
                      <span className="text-xs font-mono text-neutral-300">READY ON DISK</span>
                    ) : (
                      <span className="text-xs font-mono text-neutral-500">NOT INSTALLED</span>
                    )}
                  </div>
                </div>

                <p className="mt-2 text-xs text-neutral-400 line-clamp-2">{meta.description}</p>

                {/* Metadata list */}
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 mt-4 text-xs text-neutral-400 font-mono">
                  <span>{meta.architecture.toUpperCase()}</span>
                  <span aria-hidden="true" className="text-neutral-600">·</span>
                  <span>{meta.parameterSize}</span>
                  <span aria-hidden="true" className="text-neutral-600">·</span>
                  <span>{meta.quantization}</span>
                  <span aria-hidden="true" className="text-neutral-600">·</span>
                  <span className="tabular-nums">{meta.contextWindow.toLocaleString()} ctx</span>
                  <span aria-hidden="true" className="text-neutral-600">·</span>
                  <span className="tabular-nums">{formatBytes(meta.actualDiskSizeBytes || meta.fileSizeBytes)}</span>
                </div>

                {/* File path note */}
                <div className="mt-3 pt-3 border-t border-neutral-800/60 text-[11px] text-neutral-500 flex justify-between">
                  <span className="truncate max-w-[240px]">Path: {meta.weightPath}</span>
                  <span className="shrink-0">{meta.isInstalled ? 'File verified' : 'File missing'}</span>
                </div>
              </div>

              {/* Action Bar */}
              <div className="flex items-center justify-between pt-3 border-t border-neutral-800">
                <div className="text-xs text-neutral-500 font-mono">
                  {isLoaded && meta.runtimeConfig ? (
                    <span className="tabular-nums">
                      {meta.runtimeConfig.memoryUsageMb ?? 0} MB · {meta.runtimeConfig.gpuLayers} GPU layers
                    </span>
                  ) : null}
                </div>

                <div className="flex items-center gap-2">
                  {isLoaded ? (
                    <button
                      onClick={() => executeUnload(model.id)}
                      disabled={isLoading}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-amber-300 bg-amber-950/40 hover:bg-amber-900/50 border border-amber-800/60 rounded-md transition-colors disabled:opacity-50"
                    >
                      <Square className="w-3.5 h-3.5" />
                      <span>{isLoading ? 'Unloading...' : 'Unload'}</span>
                    </button>
                  ) : (
                    <button
                      onClick={() => handleOpenLoad(model)}
                      disabled={isLoading}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 rounded-md transition-colors disabled:opacity-50"
                    >
                      <Play className="w-3.5 h-3.5 text-neutral-400" />
                      <span>{isLoading ? 'Loading...' : 'Load Model'}</span>
                    </button>
                  )}

                  <button
                    onClick={() => onUnregisterModel(model.id)}
                    title="Unregister model"
                    className="p-1.5 text-neutral-500 hover:text-red-400 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Model Loading Modal */}
      {loadModalModel && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-neutral-900 border border-neutral-800 rounded-xl max-w-lg w-full p-6 space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
              <div>
                <h3 className="text-sm font-semibold text-white">Load Model into Hardware Memory</h3>
                <p className="text-xs text-neutral-400 mt-0.5">{loadModalModel.nexa_metadata.name}</p>
              </div>
              <button onClick={() => setLoadModalModel(null)} className="text-neutral-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-4 text-xs">
              <div>
                <label className="text-neutral-300 block mb-1">Context Window Size (Tokens)</label>
                <select
                  value={contextSize}
                  onChange={(e) => setContextSize(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white"
                >
                  <option value={2048}>2,048 tokens (Low memory)</option>
                  <option value={4096}>4,096 tokens (Standard)</option>
                  <option value={8192}>8,192 tokens (Extended)</option>
                  <option value={16384}>16,384 tokens (High RAM)</option>
                  <option value={32768}>32,768 tokens (Full context)</option>
                </select>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-neutral-300">GPU Layer Offload (-ngl)</label>
                  <span className="font-mono text-neutral-400">{gpuLayers} layers</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={40}
                  value={gpuLayers}
                  onChange={(e) => setGpuLayers(Number(e.target.value))}
                  className="w-full accent-neutral-400"
                />
                <div className="flex justify-between text-[11px] text-neutral-500 mt-1">
                  <span>0 (CPU Only)</span>
                  <span>20 (Hybrid)</span>
                  <span>33+ (Full VRAM)</span>
                </div>
              </div>

              <div>
                <label className="text-neutral-300 block mb-1">CPU Thread Allocation</label>
                <input
                  type="number"
                  min={1}
                  max={32}
                  value={threads}
                  onChange={(e) => setThreads(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white font-mono"
                />
              </div>

              <div className="p-3 bg-neutral-950 border border-neutral-800 rounded-md space-y-1 text-[11px] text-neutral-400">
                <div className="flex justify-between">
                  <span>Active Adapter:</span>
                  <span className="font-mono text-neutral-200">{activeAdapterId}</span>
                </div>
                <div className="flex justify-between">
                  <span>Disk Presence:</span>
                  <span className="font-mono text-neutral-200">
                    {loadModalModel.nexa_metadata.isInstalled ? 'File Present' : 'File Missing on Disk'}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 pt-3 border-t border-neutral-800">
              <button
                onClick={() => setLoadModalModel(null)}
                className="px-4 py-2 text-xs text-neutral-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                onClick={executeLoad}
                disabled={loadingModelId !== null}
                className="px-4 py-2 text-xs font-medium text-white bg-neutral-700 hover:bg-neutral-600 rounded-md transition-colors disabled:opacity-50"
              >
                {loadingModelId ? 'Loading Weights...' : 'Confirm Load'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Register Custom Model Modal */}
      {showRegisterModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <form onSubmit={handleRegisterSubmit} className="bg-neutral-900 border border-neutral-800 rounded-xl max-w-lg w-full p-6 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
              <h3 className="text-sm font-semibold text-white">Register Open-Weight GGUF Model</h3>
              <button type="button" onClick={() => setShowRegisterModal(false)} className="text-neutral-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="text-neutral-300 block mb-1">Model Slug / ID</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. custom-llama-3-8b"
                  value={regId}
                  onChange={(e) => setRegId(e.target.value)}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white font-mono"
                />
              </div>

              <div>
                <label className="text-neutral-300 block mb-1">Display Name</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Custom Llama 3 8B Fine-tune"
                  value={regName}
                  onChange={(e) => setRegName(e.target.value)}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-neutral-300 block mb-1">Architecture</label>
                  <select
                    value={regArch}
                    onChange={(e) => setRegArch(e.target.value)}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white"
                  >
                    <option value="llama">Llama (Llama 3/2)</option>
                    <option value="mistral">Mistral</option>
                    <option value="qwen2">Qwen 2 / 2.5</option>
                    <option value="phi3">Phi-3</option>
                    <option value="deepseek">DeepSeek</option>
                    <option value="custom">Generic / Custom</option>
                  </select>
                </div>
                <div>
                  <label className="text-neutral-300 block mb-1">Quantization</label>
                  <input
                    type="text"
                    value={regQuant}
                    onChange={(e) => setRegQuant(e.target.value)}
                    placeholder="Q4_K_M"
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white font-mono"
                  />
                </div>
              </div>

              <div>
                <label className="text-neutral-300 block mb-1">Local Weight File Path (.gguf)</label>
                <input
                  type="text"
                  required
                  value={regPath}
                  onChange={(e) => setRegPath(e.target.value)}
                  placeholder="models/my-model.Q4_K_M.gguf"
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white font-mono"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-neutral-300 block mb-1">Context Window</label>
                  <input
                    type="number"
                    value={regCtx}
                    onChange={(e) => setRegCtx(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white font-mono"
                  />
                </div>
                <div>
                  <label className="text-neutral-300 block mb-1">Parameter Count</label>
                  <input
                    type="text"
                    value={regParam}
                    onChange={(e) => setRegParam(e.target.value)}
                    placeholder="7B"
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white font-mono"
                  />
                </div>
              </div>

              <div>
                <label className="text-neutral-300 block mb-1">Description</label>
                <textarea
                  rows={2}
                  value={regDesc}
                  onChange={(e) => setRegDesc(e.target.value)}
                  placeholder="Model details, training source, or target tasks..."
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white"
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 pt-3 border-t border-neutral-800">
              <button
                type="button"
                onClick={() => setShowRegisterModal(false)}
                className="px-4 py-2 text-xs text-neutral-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 text-xs font-medium text-white bg-neutral-700 hover:bg-neutral-600 rounded-md transition-colors"
              >
                Register Model
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};
