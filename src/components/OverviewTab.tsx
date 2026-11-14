/**
 * NEXA AI Core Engine - Overview & Telemetry Dashboard
 * Real hardware telemetry, active engine runtime state, memory footprint, and llama.cpp probe.
 */

import React, { useState } from 'react';
import { SystemTelemetry, EngineStatusData, ModelItem } from '../types';
import { Cpu, Server, HardDrive, Terminal, CheckCircle2, AlertCircle, RefreshCw, Layers } from 'lucide-react';

interface OverviewTabProps {
  telemetry: SystemTelemetry | null;
  engineStatus: EngineStatusData | null;
  models: ModelItem[];
  onSwitchAdapter: (adapterId: string) => Promise<void>;
  onNavigateToModels: () => void;
  onRefresh: () => void;
}

export const OverviewTab: React.FC<OverviewTabProps> = ({
  telemetry,
  engineStatus,
  models,
  onSwitchAdapter,
  onNavigateToModels,
  onRefresh
}) => {
  const [switching, setSwitching] = useState(false);

  const formatUptime = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  };

  const handleAdapterChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSwitching(true);
    try {
      await onSwitchAdapter(e.target.value);
    } finally {
      setSwitching(false);
    }
  };

  const activeAdapter = engineStatus?.activeAdapter || { id: 'llama-cpp', name: 'llama.cpp Native Runtime', version: 'b4000+' };
  const loadedModels = models.filter(m => m.nexa_metadata.state === 'loaded');
  const installedModels = models.filter(m => m.nexa_metadata.isInstalled);

  return (
    <div className="space-y-8">
      {/* Editorial Section Header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-white">System Architecture & Engine Telemetry</h1>
        <div className="flex items-center gap-2 mt-1 text-xs text-neutral-400">
          <span>Inference Control Plane</span>
          <span aria-hidden="true">·</span>
          <span>OpenAI-Compatible Service</span>
          <span aria-hidden="true">·</span>
          <span>Port 3000</span>
        </div>
      </div>

      {/* Architectural Notice */}
      <div className="p-4 bg-neutral-900/60 border border-neutral-800 rounded-lg text-xs leading-relaxed text-neutral-300">
        <p className="font-medium text-neutral-200">Independent AI Infrastructure Engine</p>
        <p className="mt-1 text-neutral-400">
          NEXA AI Core Engine executes locally hosted open-weight models without forwarding requests to third-party proprietary APIs.
          Models are resident in host RAM and GPU VRAM only when explicitly loaded into memory.
        </p>
      </div>

      {/* Key Telemetry Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-neutral-900 border border-neutral-800 rounded-lg">
          <div className="text-xs text-neutral-400">Engine Uptime</div>
          <div className="mt-2 text-xl font-medium font-mono tabular-nums text-white">
            {formatUptime(engineStatus?.metrics.uptimeSeconds || telemetry?.metrics.uptimeSeconds || 0)}
          </div>
          <div className="mt-1 text-xs text-neutral-500">Continuous Service</div>
        </div>

        <div className="p-4 bg-neutral-900 border border-neutral-800 rounded-lg">
          <div className="text-xs text-neutral-400">Requests Processed</div>
          <div className="mt-2 text-xl font-medium font-mono tabular-nums text-white">
            {engineStatus?.metrics.totalRequests ?? 0}
          </div>
          <div className="mt-1 text-xs text-neutral-500">
            {engineStatus?.metrics.activeRequests ?? 0} active currently
          </div>
        </div>

        <div className="p-4 bg-neutral-900 border border-neutral-800 rounded-lg">
          <div className="text-xs text-neutral-400">Tokens Generated</div>
          <div className="mt-2 text-xl font-medium font-mono tabular-nums text-white">
            {engineStatus?.metrics.totalCompletionTokens ?? 0}
          </div>
          <div className="mt-1 text-xs text-neutral-500">
            {engineStatus?.metrics.tokensPerSecond ?? 0} tok/sec throughput
          </div>
        </div>

        <div className="p-4 bg-neutral-900 border border-neutral-800 rounded-lg">
          <div className="text-xs text-neutral-400">Average Latency</div>
          <div className="mt-2 text-xl font-medium font-mono tabular-nums text-white">
            {engineStatus?.metrics.averageLatencyMs ?? 0} ms
          </div>
          <div className="mt-1 text-xs text-neutral-500">Sliding 100-request avg</div>
        </div>
      </div>

      {/* Hardware & Runtime Configuration Strip */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Host Compute Hardware */}
        <div className="p-5 bg-neutral-900 border border-neutral-800 rounded-lg space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Cpu className="w-4 h-4 text-neutral-400" />
              <span>Host Compute Hardware</span>
            </h2>
            <span className="text-xs font-mono text-neutral-400 uppercase">
              {telemetry?.system.executionMode || 'cpu_only'}
            </span>
          </div>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between items-center py-1">
              <span className="text-neutral-400">Processor / Architecture:</span>
              <span className="font-mono text-neutral-200">
                {telemetry?.system.cpu || 'Detecting...'} ({telemetry?.system.arch || 'x64'}, {telemetry?.system.cpuCores || 4} cores)
              </span>
            </div>

            <div className="flex justify-between items-center py-1">
              <span className="text-neutral-400">Host Memory (RAM):</span>
              <span className="font-mono text-neutral-200 tabular-nums">
                {telemetry?.system.freeMemoryMb ?? 0} MB free / {telemetry?.system.totalMemoryMb ?? 0} MB total
              </span>
            </div>

            <div className="flex justify-between items-center py-1">
              <span className="text-neutral-400">Hardware Acceleration:</span>
              <span className="font-mono text-neutral-200">
                {telemetry?.system.gpuDetected ? telemetry.system.gpuDescription : 'CPU Compute Only'}
              </span>
            </div>

            <div className="flex justify-between items-center py-1">
              <span className="text-neutral-400">Operating System:</span>
              <span className="font-mono text-neutral-200">{telemetry?.system.platform || 'linux'}</span>
            </div>
          </div>
        </div>

        {/* Inference Runtime & Adapter Control */}
        <div className="p-5 bg-neutral-900 border border-neutral-800 rounded-lg space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Server className="w-4 h-4 text-neutral-400" />
              <span>Active Inference Runtime</span>
            </h2>
            <span className="text-xs font-mono text-neutral-400">{activeAdapter.id}</span>
          </div>

          <div className="space-y-3 text-xs">
            <div>
              <label className="text-neutral-400 block mb-1.5">Runtime Adapter Engine</label>
              <select
                value={activeAdapter.id}
                onChange={handleAdapterChange}
                disabled={switching}
                className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-xs text-white focus:outline-none focus:border-neutral-500"
              >
                <option value="nexa-native">NEXA Native CPU Engine (In-House C++/TS Tensor Runtime)</option>
                <option value="llama-cpp">llama.cpp Native Runtime (Open-Weights GGUF Engine - Isolated)</option>
                <option value="nexa-diagnostic">NEXA Diagnostic & Automated Test Engine (Synthetic Loopback)</option>
              </select>
              <p className="mt-1 text-neutral-500 text-[11px]">
                {activeAdapter.id === 'nexa-native'
                  ? 'Independent in-house CPU inference engine executing custom tensor math, attention, and .nexa models with zero 3rd-party dependencies.'
                  : activeAdapter.id === 'llama-cpp'
                  ? 'Executes GGUF weights via local llama-server or llama-cli binaries with hardware acceleration (isolated comparison adapter).'
                  : 'Diagnostic pipeline testing engine for validating OpenAI spec compatibility without mounting weights.'}
              </p>
            </div>

            <div className="pt-2 border-t border-neutral-800/60 flex justify-between items-center">
              <span className="text-neutral-400">llama.cpp Binary Detection:</span>
              <span className="flex items-center gap-1.5 font-mono">
                {telemetry?.system.llamaCppInstalled ? (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400">Installed ({telemetry.system.llamaCppBinaryPath})</span>
                  </>
                ) : (
                  <>
                    <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
                    <span className="text-amber-400">Binary Not Found in PATH</span>
                  </>
                )}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-neutral-400">Max Inflight Concurrency:</span>
              <span className="font-mono text-neutral-200">
                {engineStatus?.config.maxConcurrentRequests || 4} requests
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Resident Models in Memory */}
      <div className="p-5 bg-neutral-900 border border-neutral-800 rounded-lg space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
          <div>
            <h2 className="text-sm font-semibold text-white">Models Resident in Memory</h2>
            <div className="text-xs text-neutral-400 mt-0.5">
              <span>{loadedModels.length} loaded</span>
              <span aria-hidden="true" className="mx-1">·</span>
              <span>{installedModels.length} installed on disk</span>
              <span aria-hidden="true" className="mx-1">·</span>
              <span>{models.length} total in catalog</span>
            </div>
          </div>
          <button
            onClick={onNavigateToModels}
            className="px-3 py-1.5 text-xs text-neutral-200 bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 rounded-md transition-colors"
          >
            Manage Models
          </button>
        </div>

        {loadedModels.length === 0 ? (
          <div className="py-8 text-center text-xs text-neutral-400 space-y-2">
            <HardDrive className="w-8 h-8 text-neutral-600 mx-auto mb-2" />
            <p className="font-medium text-neutral-300">No Model Resident in Memory</p>
            <p className="max-w-md mx-auto text-neutral-500">
              In accordance with technical architecture rule 1, the engine does not pretend a model is running.
              Go to the Models tab to verify disk weights and load a model into memory.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-neutral-800 text-neutral-400">
                  <th className="py-2.5 px-3 font-medium">Model ID</th>
                  <th className="py-2.5 px-3 font-medium">Architecture</th>
                  <th className="py-2.5 px-3 font-medium">Quantization</th>
                  <th className="py-2.5 px-3 font-medium text-right">Context Size</th>
                  <th className="py-2.5 px-3 font-medium text-right">GPU Offload</th>
                  <th className="py-2.5 px-3 font-medium text-right">Memory Footprint</th>
                  <th className="py-2.5 px-3 font-medium text-right">Active Inflight</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/60 font-mono">
                {loadedModels.map(m => (
                  <tr key={m.id} className="hover:bg-neutral-800/40">
                    <td className="py-3 px-3 font-sans font-medium text-white">{m.id}</td>
                    <td className="py-3 px-3 text-neutral-300">{m.nexa_metadata.architecture}</td>
                    <td className="py-3 px-3 text-neutral-300">{m.nexa_metadata.quantization}</td>
                    <td className="py-3 px-3 text-right tabular-nums text-neutral-200">
                      {m.nexa_metadata.runtimeConfig?.contextSize ?? m.nexa_metadata.contextWindow} tok
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums text-neutral-200">
                      {m.nexa_metadata.runtimeConfig?.gpuLayers ?? 0} layers
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums text-neutral-200">
                      {m.nexa_metadata.runtimeConfig?.memoryUsageMb ?? 0} MB
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums text-neutral-200">
                      {m.nexa_metadata.runtimeConfig?.activeRequests ?? 0}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* NEXA Native Runtime Info Card */}
      <div className="p-5 bg-neutral-900 border border-neutral-800 rounded-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-neutral-800">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-400" />
            <span>NEXA Native Engine (Independent C++ & Node Runtime)</span>
          </h3>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/60">
            Phase 3 Native Architecture
          </span>
        </div>
        <p className="text-xs text-neutral-400 leading-relaxed">
          NEXA Native Runtime is our self-hosted inference engine featuring custom tensor operations, memory-aligned buffers,
          attention blocks, and native <code className="text-neutral-200 font-mono">.nexa</code> model loading. It operates completely free of external AI APIs or third-party runtimes.
        </p>
        <div className="p-3 bg-neutral-950 border border-neutral-800 rounded-md font-mono text-xs text-neutral-300 space-y-2 overflow-x-auto">
          <div><span className="text-neutral-500"># Native C++ Compilation (Linux x86_64, ARM64, and Android Termux)</span></div>
          <div>cd native && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j</div>
          <div className="pt-1"><span className="text-neutral-500"># Run C++ tensor verification benchmark</span></div>
          <div>./build/bin/nexa-runtime-cli --test-tensor</div>
        </div>
      </div>

      {/* Host llama.cpp Setup Guide Card (Isolated / Optional) */}
      <div className="p-5 bg-neutral-900 border border-neutral-800 rounded-lg space-y-3">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Terminal className="w-4 h-4 text-neutral-400" />
          <span>llama.cpp Comparison Runtime (Isolated / Optional)</span>
        </h3>
        <p className="text-xs text-neutral-400 leading-relaxed">
          NEXA connects directly to native <code className="text-neutral-200 font-mono">llama-server</code> or executes via subprocess. To compile or run llama.cpp on your local workstation:
        </p>

        <div className="p-3 bg-neutral-950 border border-neutral-800 rounded-md font-mono text-xs text-neutral-300 space-y-2 overflow-x-auto">
          <div><span className="text-neutral-500"># 1. Clone and compile llama.cpp with CUDA or Metal</span></div>
          <div>git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp</div>
          <div>cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j</div>
          <div className="pt-2"><span className="text-neutral-500"># 2. Launch OpenAI-compatible server on standard port</span></div>
          <div>./build/bin/llama-server -m models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --port 8080 --host 0.0.0.0 -ngl 33</div>
        </div>
      </div>
    </div>
  );
};
