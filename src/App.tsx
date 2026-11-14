/**
 * NEXA AI Core Engine - Main Application Shell
 * Workbench for local open-weight model serving, monitoring, and testing.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Header, ActiveTab } from './components/Header';
import { OverviewTab } from './components/OverviewTab';
import { ModelsTab } from './components/ModelsTab';
import { PlaygroundTab } from './components/PlaygroundTab';
import { KeysTab } from './components/KeysTab';
import { TestsTab } from './components/TestsTab';
import { DocsTab } from './components/DocsTab';
import { SystemTelemetry, EngineStatusData, ModelItem, ApiKeyItem } from './types';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('overview');
  const [telemetry, setTelemetry] = useState<SystemTelemetry | null>(null);
  const [engineStatus, setEngineStatus] = useState<EngineStatusData | null>(null);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [keys, setKeys] = useState<ApiKeyItem[]>([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [engineOnline, setEngineOnline] = useState(true);

  // Active client key stored in browser session for playground
  const [activeApiKey, setActiveApiKey] = useState<string>(() => {
    return localStorage.getItem('nexa_client_api_key') || 'nexa_live_dev_master_key_local';
  });

  const handleSetActiveApiKey = (k: string) => {
    setActiveApiKey(k);
    localStorage.setItem('nexa_client_api_key', k);
  };

  const fetchGlobalData = useCallback(async () => {
    setIsRefreshing(true);
    try {
      // 1. Health
      const healthRes = await fetch('/health').catch(() => null);
      if (healthRes && healthRes.ok) {
        const hData = await healthRes.json();
        setTelemetry(hData);
        setEngineOnline(true);
      } else {
        setEngineOnline(false);
      }

      // 2. Engine status
      const statusRes = await fetch('/v1/engine/status').catch(() => null);
      if (statusRes && statusRes.ok) {
        const sData = await statusRes.json();
        setEngineStatus(sData);
      }

      // 3. Models
      const modelsRes = await fetch('/v1/models').catch(() => null);
      if (modelsRes && modelsRes.ok) {
        const mData = await modelsRes.json();
        setModels(mData.data || []);
      }

      // 4. Keys
      const keysRes = await fetch('/v1/keys').catch(() => null);
      if (keysRes && keysRes.ok) {
        const kData = await keysRes.json();
        setKeys(kData.data || []);
      }
    } catch (err) {
      console.error('Failed to refresh engine state:', err);
      setEngineOnline(false);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchGlobalData();
    const interval = setInterval(fetchGlobalData, 10000);
    return () => clearInterval(interval);
  }, [fetchGlobalData]);

  // Actions
  const handleSwitchAdapter = async (adapterId: string) => {
    const res = await fetch('/v1/engine/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ activeAdapterId: adapterId })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error?.message || 'Failed to switch adapter.');
    }
    await fetchGlobalData();
  };

  const handleLoadModel = async (modelId: string, options: any) => {
    const res = await fetch(`/v1/models/${modelId}/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options)
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error?.message || `Failed to load model "${modelId}".`);
    }
    await fetchGlobalData();
  };

  const handleUnloadModel = async (modelId: string) => {
    const res = await fetch(`/v1/models/${modelId}/unload`, {
      method: 'POST'
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error?.message || `Failed to unload model "${modelId}".`);
    }
    await fetchGlobalData();
  };

  const handleRegisterModel = async (modelData: any) => {
    const res = await fetch('/v1/models/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(modelData)
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error?.message || 'Failed to register model.');
    }
    await fetchGlobalData();
  };

  const handleUnregisterModel = async (modelId: string) => {
    if (!confirm(`Are you sure you want to unregister model "${modelId}"?`)) return;
    const res = await fetch(`/v1/models/${modelId}`, {
      method: 'DELETE'
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error?.message || 'Failed to unregister model.');
    }
    await fetchGlobalData();
  };

  const handleToggleRequireAuth = async (require: boolean) => {
    const res = await fetch('/v1/engine/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ requireApiKey: require })
    });
    if (!res.ok) throw new Error('Failed to update authentication requirement.');
    await fetchGlobalData();
  };

  const handleCreateKey = async (name: string, scopes: string[], rateLimitPerMin?: number) => {
    const res = await fetch('/v1/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, scopes, rateLimitPerMin })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error?.message || 'Failed to create key.');
    await fetchGlobalData();
    return { record: data.data, rawKey: data.rawKey };
  };

  const handleRevokeKey = async (keyId: string) => {
    if (!confirm('Are you sure you want to revoke this API key?')) return;
    const res = await fetch(`/v1/keys/${keyId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to revoke API key.');
    await fetchGlobalData();
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col font-sans">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onRefresh={fetchGlobalData}
        isRefreshing={isRefreshing}
        engineOnline={engineOnline}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8">
        {activeTab === 'overview' && (
          <OverviewTab
            telemetry={telemetry}
            engineStatus={engineStatus}
            models={models}
            onSwitchAdapter={handleSwitchAdapter}
            onNavigateToModels={() => setActiveTab('models')}
            onRefresh={fetchGlobalData}
          />
        )}

        {activeTab === 'models' && (
          <ModelsTab
            models={models}
            onLoadModel={handleLoadModel}
            onUnloadModel={handleUnloadModel}
            onRegisterModel={handleRegisterModel}
            onUnregisterModel={handleUnregisterModel}
            onRefresh={fetchGlobalData}
            activeAdapterId={engineStatus?.activeAdapter.id || 'llama-cpp'}
          />
        )}

        {activeTab === 'playground' && (
          <PlaygroundTab
            models={models}
            onLoadModel={handleLoadModel}
            apiKey={activeApiKey}
          />
        )}

        {activeTab === 'keys' && (
          <KeysTab
            keys={keys}
            requireApiKey={engineStatus?.config.requireApiKey || false}
            onToggleRequireAuth={handleToggleRequireAuth}
            onCreateKey={handleCreateKey}
            onRevokeKey={handleRevokeKey}
            activeApiKey={activeApiKey}
            onSetActiveApiKey={handleSetActiveApiKey}
          />
        )}

        {activeTab === 'tests' && <TestsTab />}

        {activeTab === 'docs' && <DocsTab />}
      </main>

      <footer className="border-t border-neutral-800/80 px-6 py-4 text-xs text-neutral-500 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <span>NEXA AI Core Engine</span>
          <span aria-hidden="true">·</span>
          <span>Open-Weight Inference Infrastructure</span>
          <span aria-hidden="true">·</span>
          <span className="font-mono">OpenAI v1 Compatible</span>
        </div>
        <div className="font-mono text-[11px] text-neutral-600">
          Independent Model Serving Layer · No Cloud API Dependencies
        </div>
      </footer>
    </div>
  );
}
