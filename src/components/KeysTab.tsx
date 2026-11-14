/**
 * NEXA AI Core Engine - API Keys & Security Management
 * Server-side API key generation, scope assignment, rate limiting, and auth enforcement.
 */

import React, { useState } from 'react';
import { ApiKeyItem } from '../types';
import { Key, Shield, Plus, Trash2, Copy, Check, AlertCircle, X } from 'lucide-react';

interface KeysTabProps {
  keys: ApiKeyItem[];
  requireApiKey: boolean;
  onToggleRequireAuth: (require: boolean) => Promise<void>;
  onCreateKey: (name: string, scopes: string[], rateLimitPerMin?: number) => Promise<{ record: ApiKeyItem; rawKey: string }>;
  onRevokeKey: (keyId: string) => Promise<void>;
  activeApiKey: string;
  onSetActiveApiKey: (key: string) => void;
}

export const KeysTab: React.FC<KeysTabProps> = ({
  keys,
  requireApiKey,
  onToggleRequireAuth,
  onCreateKey,
  onRevokeKey,
  activeApiKey,
  onSetActiveApiKey
}) => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [keyName, setKeyName] = useState('');
  const [scopes, setScopes] = useState<string[]>(['chat:completions', 'models:read']);
  const [rateLimit, setRateLimit] = useState<number>(120);

  const [newlyCreatedKey, setNewlyCreatedKey] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleToggleScope = (scope: string) => {
    if (scopes.includes(scope)) {
      setScopes(scopes.filter(s => s !== scope));
    } else {
      setScopes([...scopes, scope]);
    }
  };

  const handleCreateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!keyName.trim()) return;
    setErrorMsg(null);
    try {
      const res = await onCreateKey(keyName.trim(), scopes, rateLimit > 0 ? rateLimit : undefined);
      setNewlyCreatedKey(res.rawKey);
      setKeyName('');
    } catch (err: unknown) {
      setErrorMsg((err as Error).message);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(true);
    setTimeout(() => setCopiedKey(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Editorial Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">Security & API Key Management</h1>
          <div className="flex items-center gap-2 mt-1 text-xs text-neutral-400">
            <span>Bearer Authentication</span>
            <span aria-hidden="true">·</span>
            <span>Granular Scopes</span>
            <span aria-hidden="true">·</span>
            <span>Rate Limiting</span>
          </div>
        </div>

        <button
          onClick={() => { setShowCreateModal(true); setNewlyCreatedKey(null); }}
          className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-white bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 rounded-md transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Create API Key</span>
        </button>
      </div>

      {errorMsg && (
        <div className="p-3 bg-red-950/50 border border-red-800 rounded-lg text-xs text-red-300 flex items-center justify-between">
          <span>{errorMsg}</span>
          <button onClick={() => setErrorMsg(null)}><X className="w-3.5 h-3.5" /></button>
        </div>
      )}

      {/* Global Enforcement Toggle Card */}
      <div className="p-5 bg-neutral-900 border border-neutral-800 rounded-lg flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="text-sm font-semibold text-white flex items-center gap-2">
            <Shield className="w-4 h-4 text-neutral-400" />
            <span>Enforce Bearer Token Authentication</span>
          </div>
          <p className="text-xs text-neutral-400 max-w-xl">
            When enabled, all incoming requests to <code className="text-neutral-300 font-mono">/v1/*</code> must present a valid <code className="text-neutral-300 font-mono">Authorization: Bearer nexa_live_...</code> header. Requests without a valid key will receive HTTP 401 Unauthorized.
          </p>
        </div>

        <button
          onClick={() => onToggleRequireAuth(!requireApiKey)}
          className={`px-4 py-2 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
            requireApiKey
              ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-800 hover:bg-emerald-900/60'
              : 'bg-neutral-800 text-neutral-300 border border-neutral-700 hover:bg-neutral-700'
          }`}
        >
          {requireApiKey ? 'Enforced (Secure Mode)' : 'Permissive (Dev Mode)'}
        </button>
      </div>

      {/* Active API Key for Console Playground */}
      <div className="p-4 bg-neutral-900 border border-neutral-800 rounded-lg text-xs space-y-2">
        <label className="text-neutral-300 block font-medium">Console Active Bearer Key (Used in Playground & Tests)</label>
        <div className="flex gap-2">
          <input
            type="text"
            value={activeApiKey}
            onChange={(e) => onSetActiveApiKey(e.target.value)}
            placeholder="Paste your nexa_live_... key here"
            className="flex-1 px-3 py-2 bg-neutral-950 border border-neutral-800 rounded-md text-white font-mono text-xs"
          />
        </div>
        <p className="text-neutral-500 text-[11px]">
          This key is stored locally in your browser session for authorizing playground requests.
        </p>
      </div>

      {/* Keys Table */}
      <div className="p-5 bg-neutral-900 border border-neutral-800 rounded-lg space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
          <h2 className="text-sm font-semibold text-white">Registered API Keys ({keys.length})</h2>
          <span className="text-xs text-neutral-500 font-mono">SHA-256 Hashed on Server</span>
        </div>

        {keys.length === 0 ? (
          <div className="py-8 text-center text-xs text-neutral-500">
            No API keys created yet. Click "Create API Key" to provision one.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-neutral-800 text-neutral-400">
                  <th className="py-2.5 px-3 font-medium">Key Name</th>
                  <th className="py-2.5 px-3 font-medium">Prefix</th>
                  <th className="py-2.5 px-3 font-medium">Scopes</th>
                  <th className="py-2.5 px-3 font-medium text-right">RPM Limit</th>
                  <th className="py-2.5 px-3 font-medium text-right">Total Invocations</th>
                  <th className="py-2.5 px-3 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/60 font-mono">
                {keys.map(k => (
                  <tr key={k.id} className="hover:bg-neutral-800/30">
                    <td className="py-3 px-3 font-sans font-medium text-white">{k.name}</td>
                    <td className="py-3 px-3 text-neutral-400">{k.keyPrefix}</td>
                    <td className="py-3 px-3 font-sans text-neutral-300">
                      <div className="flex flex-wrap gap-1">
                        {k.scopes.map(s => (
                          <span key={s} className="text-[11px] text-neutral-400">
                            {s}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums text-neutral-300">
                      {k.rateLimitPerMin ? `${k.rateLimitPerMin} req/m` : 'Default'}
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums text-neutral-300">
                      {k.totalRequests}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={() => onRevokeKey(k.id)}
                        className="p-1 text-neutral-500 hover:text-red-400 transition-colors"
                        title="Revoke Key"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Create Key Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-neutral-900 border border-neutral-800 rounded-xl max-w-md w-full p-6 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
              <h3 className="text-sm font-semibold text-white">Create Server-Side API Key</h3>
              <button onClick={() => setShowCreateModal(false)} className="text-neutral-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>

            {newlyCreatedKey ? (
              <div className="space-y-4 text-xs">
                <div className="p-3 bg-emerald-950/40 border border-emerald-800 rounded-lg text-emerald-300 space-y-1">
                  <div className="font-semibold text-emerald-200">API Key Generated Successfully</div>
                  <div>Store this key now. The plaintext secret cannot be retrieved after closing this window.</div>
                </div>

                <div className="space-y-1">
                  <label className="text-neutral-400 font-medium">Secret Key</label>
                  <div className="flex items-center gap-2 p-2 bg-neutral-950 border border-neutral-800 rounded font-mono text-white break-all">
                    <span className="flex-1">{newlyCreatedKey}</span>
                    <button
                      onClick={() => copyToClipboard(newlyCreatedKey)}
                      className="p-1.5 bg-neutral-800 hover:bg-neutral-700 rounded text-neutral-200"
                    >
                      {copiedKey ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <button
                  onClick={() => {
                    onSetActiveApiKey(newlyCreatedKey);
                    setShowCreateModal(false);
                  }}
                  className="w-full py-2 bg-neutral-800 hover:bg-neutral-700 text-white rounded text-xs font-medium"
                >
                  Use this key in Console & Close
                </button>
              </div>
            ) : (
              <form onSubmit={handleCreateSubmit} className="space-y-4 text-xs">
                <div>
                  <label className="text-neutral-300 block mb-1">Key Name / Client ID</label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Production Backend Worker"
                    value={keyName}
                    onChange={(e) => setKeyName(e.target.value)}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white"
                  />
                </div>

                <div>
                  <label className="text-neutral-300 block mb-1.5">Permitted Scopes</label>
                  <div className="space-y-2">
                    {[
                      { id: 'chat:completions', label: 'chat:completions (Inference Execution)' },
                      { id: 'models:read', label: 'models:read (List and inspect models)' },
                      { id: 'models:manage', label: 'models:manage (Load/Unload/Register)' },
                      { id: 'engine:admin', label: 'engine:admin (Full administrative access)' }
                    ].map(s => (
                      <label key={s.id} className="flex items-center gap-2 text-neutral-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={scopes.includes(s.id)}
                          onChange={() => handleToggleScope(s.id)}
                          className="rounded bg-neutral-800 border-neutral-700"
                        />
                        <span>{s.label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-neutral-300 block mb-1">Rate Limit (Requests per Minute)</label>
                  <input
                    type="number"
                    min={1}
                    max={10000}
                    value={rateLimit}
                    onChange={(e) => setRateLimit(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white font-mono"
                  />
                </div>

                <div className="flex items-center justify-end gap-3 pt-3 border-t border-neutral-800">
                  <button
                    type="button"
                    onClick={() => setShowCreateModal(false)}
                    className="px-4 py-2 text-xs text-neutral-400 hover:text-white"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 text-xs font-medium text-white bg-neutral-700 hover:bg-neutral-600 rounded-md transition-colors"
                  >
                    Generate Key
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
