/**
 * NEXA AI Core Engine - Inference Playground & OpenAI API Inspector
 * Interactive testing for POST /v1/chat/completions with SSE streaming and SDK exporters.
 */

import React, { useState, useRef } from 'react';
import { ModelItem } from '../types';
import { Play, Copy, Check, Terminal, Code, Sliders, AlertCircle, RefreshCw } from 'lucide-react';

interface PlaygroundTabProps {
  models: ModelItem[];
  onLoadModel: (modelId: string, options: any) => Promise<void>;
  apiKey: string;
}

export const PlaygroundTab: React.FC<PlaygroundTabProps> = ({
  models,
  onLoadModel,
  apiKey
}) => {
  const loadedModels = models.filter(m => m.nexa_metadata.state === 'loaded');
  const defaultModelId = loadedModels[0]?.id || models[0]?.id || 'llama-3-8b-instruct';

  const [selectedModelId, setSelectedModelId] = useState<string>(defaultModelId);
  const [systemPrompt, setSystemPrompt] = useState<string>('You are a helpful, concise AI assistant powered by the NEXA self-hosted core engine.');
  const [userPrompt, setUserPrompt] = useState<string>('Explain the difference between GGUF quantization formats and full-precision weights in two sentences.');
  const [temperature, setTemperature] = useState<number>(0.7);
  const [topP, setTopP] = useState<number>(0.9);
  const [maxTokens, setMaxTokens] = useState<number>(512);
  const [stream, setStream] = useState<boolean>(true);

  // Response state
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [responseContent, setResponseContent] = useState<string>('');
  const [responseStats, setResponseStats] = useState<{ latencyMs: number; promptTokens: number; completionTokens: number } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [rawJsonResponse, setRawJsonResponse] = useState<string | null>(null);

  // Exporter tab
  const [exportLanguage, setExportLanguage] = useState<'curl' | 'python' | 'node' | 'json'>('curl');
  const [copied, setCopied] = useState<boolean>(false);

  const selectedModel = models.find(m => m.id === selectedModelId);
  const isSelectedModelLoaded = selectedModel?.nexa_metadata.state === 'loaded';

  const handleQuickLoad = async () => {
    if (!selectedModel) return;
    setErrorMsg(null);
    try {
      await onLoadModel(selectedModel.id, { contextSize: 4096, gpuLayers: 33, threads: 4 });
    } catch (err: unknown) {
      setErrorMsg((err as Error).message);
    }
  };

  const handleRunCompletion = async () => {
    setIsGenerating(true);
    setResponseContent('');
    setRawJsonResponse(null);
    setResponseStats(null);
    setErrorMsg(null);

    const startTime = Date.now();
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    const payload = {
      model: selectedModelId,
      messages: [
        ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
        { role: 'user', content: userPrompt }
      ],
      temperature,
      top_p: topP,
      max_tokens: maxTokens,
      stream
    };

    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers,
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ error: { message: `HTTP ${res.status}` } }));
        throw new Error(errorData.error?.message || `Request failed with status ${res.status}`);
      }

      if (stream && res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let accumulated = '';
        let chunkCount = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const text = decoder.decode(value);
          const lines = text.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6).trim();
              if (dataStr === '[DONE]') break;
              try {
                const data = JSON.parse(dataStr);
                const delta = data.choices?.[0]?.delta?.content;
                if (delta) {
                  accumulated += delta;
                  chunkCount++;
                  setResponseContent(accumulated);
                }
              } catch {
                // ignore
              }
            }
          }
        }

        const elapsed = Date.now() - startTime;
        setResponseStats({
          latencyMs: elapsed,
          promptTokens: Math.ceil((systemPrompt.length + userPrompt.length) / 4),
          completionTokens: Math.ceil(accumulated.length / 4)
        });
      } else {
        const data = await res.json();
        const elapsed = Date.now() - startTime;
        setRawJsonResponse(JSON.stringify(data, null, 2));
        setResponseContent(data.choices?.[0]?.message?.content || '');
        setResponseStats({
          latencyMs: elapsed,
          promptTokens: data.usage?.prompt_tokens || 0,
          completionTokens: data.usage?.completion_tokens || 0
        });
      }
    } catch (err: unknown) {
      setErrorMsg((err as Error).message);
    } finally {
      setIsGenerating(false);
    }
  };

  const getCodeSnippet = () => {
    const origin = typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000';
    const authHeader = apiKey ? ` -H "Authorization: Bearer ${apiKey}"` : '';

    if (exportLanguage === 'curl') {
      return `curl -X POST ${origin}/v1/chat/completions \\
  -H "Content-Type: application/json"${authHeader ? ` \\\n ${authHeader.trim()}` : ''} \\
  -d '{
    "model": "${selectedModelId}",
    "messages": [
      {"role": "system", "content": ${JSON.stringify(systemPrompt)}},
      {"role": "user", "content": ${JSON.stringify(userPrompt)}}
    ],
    "temperature": ${temperature},
    "stream": ${stream}
  }'`;
    }

    if (exportLanguage === 'python') {
      return `from openai import OpenAI

# Connect standard OpenAI Python SDK to NEXA AI Core Engine
client = OpenAI(
    base_url="${origin}/v1",
    api_key="${apiKey || 'nexa_live_dev_master_key_local'}"
)

response = client.chat.completions.create(
    model="${selectedModelId}",
    messages=[
        {"role": "system", "content": ${JSON.stringify(systemPrompt)}},
        {"role": "user", "content": ${JSON.stringify(userPrompt)}}
    ],
    temperature=${temperature},
    stream=${stream}
)

if ${stream}:
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
else:
    print(response.choices[0].message.content)`;
    }

    if (exportLanguage === 'node') {
      return `import OpenAI from 'openai';

// Connect official OpenAI Node.js SDK to NEXA Core
const openai = new OpenAI({
  baseURL: '${origin}/v1',
  apiKey: '${apiKey || 'nexa_live_dev_master_key_local'}'
});

const completion = await openai.chat.completions.create({
  model: '${selectedModelId}',
  messages: [
    { role: 'system', content: ${JSON.stringify(systemPrompt)} },
    { role: 'user', content: ${JSON.stringify(userPrompt)} }
  ],
  temperature: ${temperature},
  stream: ${stream}
});

${stream ? `for await (const chunk of completion) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}` : `console.log(completion.choices[0].message.content);`}`;
    }

    return JSON.stringify(
      {
        model: selectedModelId,
        messages: [
          ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
          { role: 'user', content: userPrompt }
        ],
        temperature,
        top_p: topP,
        max_tokens: maxTokens,
        stream
      },
      null,
      2
    );
  };

  const copyCode = () => {
    navigator.clipboard.writeText(getCodeSnippet());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Editorial Header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-white">OpenAI-Compatible Inference Playground</h1>
        <div className="flex items-center gap-2 mt-1 text-xs text-neutral-400">
          <span>POST /v1/chat/completions</span>
          <span aria-hidden="true">·</span>
          <span>SSE Streaming</span>
          <span aria-hidden="true">·</span>
          <span>Strict Hardware State Enforcement</span>
        </div>
      </div>

      {/* Unloaded Warning Banner */}
      {!isSelectedModelLoaded && (
        <div className="p-4 bg-amber-950/40 border border-amber-800/80 rounded-lg text-xs text-amber-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-white">Model Not Resident in Memory: </span>
              "{selectedModel?.nexa_metadata.name || selectedModelId}" is in state "{selectedModel?.nexa_metadata.state || 'uninstalled'}".
              In accordance with technical architecture rule 1, the engine will strictly reject completions until weights are resident in RAM/VRAM.
            </div>
          </div>
          <button
            onClick={handleQuickLoad}
            className="px-3 py-1.5 bg-amber-900/60 hover:bg-amber-800/80 border border-amber-700 rounded-md font-medium text-white transition-colors whitespace-nowrap shrink-0"
          >
            Attempt Load Now
          </button>
        </div>
      )}

      {errorMsg && (
        <div className="p-4 bg-red-950/50 border border-red-800 rounded-lg text-xs text-red-300 space-y-1">
          <div className="font-semibold text-red-200">OpenAI API Error Response:</div>
          <div className="font-mono">{errorMsg}</div>
        </div>
      )}

      {/* Main Workspace Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Request Configuration (5 cols) */}
        <div className="lg:col-span-5 space-y-5 p-5 bg-neutral-900 border border-neutral-800 rounded-lg text-xs">
          <div>
            <label className="text-neutral-300 block mb-1 font-medium">Target Model</label>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white font-mono"
            >
              {models.map(m => (
                <option key={m.id} value={m.id}>
                  {m.nexa_metadata.name} ({m.nexa_metadata.state.toUpperCase()})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-neutral-300 block mb-1 font-medium">System Instructions</label>
            <textarea
              rows={3}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="Optional system context..."
              className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white leading-relaxed"
            />
          </div>

          <div>
            <label className="text-neutral-300 block mb-1 font-medium">User Prompt</label>
            <textarea
              rows={4}
              value={userPrompt}
              onChange={(e) => setUserPrompt(e.target.value)}
              placeholder="Enter message for chat completion..."
              className="w-full px-3 py-2 bg-neutral-800 border border-neutral-700 rounded-md text-white leading-relaxed"
            />
          </div>

          {/* Hyperparameters */}
          <div className="pt-4 border-t border-neutral-800 space-y-4">
            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-neutral-400">Temperature</span>
                <span className="font-mono text-neutral-200">{temperature.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-neutral-400"
              />
            </div>

            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-neutral-400">Max Tokens</span>
                <span className="font-mono text-neutral-200">{maxTokens}</span>
              </div>
              <input
                type="range"
                min={16}
                max={2048}
                step={16}
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
                className="w-full accent-neutral-400"
              />
            </div>

            <div className="flex items-center justify-between pt-2">
              <span className="text-neutral-400">Server-Sent Events (SSE Stream)</span>
              <input
                type="checkbox"
                checked={stream}
                onChange={(e) => setStream(e.target.checked)}
                className="rounded bg-neutral-800 border-neutral-700"
              />
            </div>
          </div>

          <div className="pt-2">
            <button
              onClick={handleRunCompletion}
              disabled={isGenerating}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-white text-neutral-900 font-semibold rounded-md hover:bg-neutral-200 transition-colors disabled:opacity-50"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>{isGenerating ? 'Generating Response...' : 'Run Chat Completion'}</span>
            </button>
          </div>
        </div>

        {/* Right Column: Live Output & Code Exporter (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          {/* Output Viewer Card */}
          <div className="p-5 bg-neutral-900 border border-neutral-800 rounded-lg space-y-3 min-h-[300px] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
                <span className="text-xs font-semibold text-white">Inference Stream Output</span>
                {responseStats && (
                  <div className="flex items-center gap-2 text-[11px] font-mono text-neutral-400">
                    <span className="tabular-nums">{responseStats.latencyMs} ms</span>
                    <span aria-hidden="true">·</span>
                    <span className="tabular-nums">{responseStats.promptTokens} prompt tok</span>
                    <span aria-hidden="true">·</span>
                    <span className="tabular-nums">{responseStats.completionTokens} comp tok</span>
                  </div>
                )}
              </div>

              <div className="mt-4 text-xs font-mono text-neutral-200 whitespace-pre-wrap leading-relaxed">
                {responseContent ? (
                  responseContent
                ) : isGenerating ? (
                  <span className="text-neutral-500 animate-pulse">Waiting for first token chunk from engine...</span>
                ) : (
                  <span className="text-neutral-600">Response tokens will stream here in real time.</span>
                )}
              </div>
            </div>

            {rawJsonResponse && (
              <div className="mt-4 pt-3 border-t border-neutral-800">
                <span className="text-[11px] font-medium text-neutral-400 block mb-1">Raw JSON Payload</span>
                <pre className="p-3 bg-neutral-950 rounded border border-neutral-800 text-[11px] font-mono text-neutral-400 overflow-x-auto max-h-48">
                  {rawJsonResponse}
                </pre>
              </div>
            )}
          </div>

          {/* Interactive Code Exporter Card */}
          <div className="p-5 bg-neutral-900 border border-neutral-800 rounded-lg space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-neutral-800">
              <div className="flex items-center gap-1 p-1 bg-neutral-950 rounded-lg">
                <button
                  onClick={() => setExportLanguage('curl')}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    exportLanguage === 'curl' ? 'bg-neutral-800 text-white' : 'text-neutral-400 hover:text-white'
                  }`}
                >
                  cURL
                </button>
                <button
                  onClick={() => setExportLanguage('python')}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    exportLanguage === 'python' ? 'bg-neutral-800 text-white' : 'text-neutral-400 hover:text-white'
                  }`}
                >
                  Python (OpenAI)
                </button>
                <button
                  onClick={() => setExportLanguage('node')}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    exportLanguage === 'node' ? 'bg-neutral-800 text-white' : 'text-neutral-400 hover:text-white'
                  }`}
                >
                  Node.js (OpenAI)
                </button>
                <button
                  onClick={() => setExportLanguage('json')}
                  className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                    exportLanguage === 'json' ? 'bg-neutral-800 text-white' : 'text-neutral-400 hover:text-white'
                  }`}
                >
                  Raw HTTP JSON
                </button>
              </div>

              <button
                onClick={copyCode}
                className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-neutral-300 hover:text-white bg-neutral-800 hover:bg-neutral-700 rounded border border-neutral-700 transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>

            <pre className="p-3 bg-neutral-950 rounded border border-neutral-800 text-xs font-mono text-neutral-300 overflow-x-auto leading-relaxed">
              {getCodeSnippet()}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
