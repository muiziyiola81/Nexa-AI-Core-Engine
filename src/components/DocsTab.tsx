/**
 * NEXA AI Core Engine - Architecture & Integration Documentation
 * Technical design, OpenAI SDK configuration, and local llama.cpp setup.
 */

import React from 'react';
import { BookOpen, Terminal, Layers, Shield, Cpu, HardDrive } from 'lucide-react';

export const DocsTab: React.FC = () => {
  return (
    <div className="space-y-8 max-w-4xl">
      {/* Editorial Header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-white">System Architecture & Technical Reference</h1>
        <div className="flex items-center gap-2 mt-1 text-xs text-neutral-400">
          <span>Inference Control Plane</span>
          <span aria-hidden="true">·</span>
          <span>OpenAI API Compatibility</span>
          <span aria-hidden="true">·</span>
          <span>Hardware Runtime Contracts</span>
        </div>
      </div>

      {/* 1. Core Architecture Principles */}
      <div className="p-6 bg-neutral-900 border border-neutral-800 rounded-lg space-y-4">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <Layers className="w-4 h-4 text-neutral-400" />
          <span>Three-Tier Modular Architecture</span>
        </h2>
        <p className="text-xs text-neutral-300 leading-relaxed">
          NEXA AI Core Engine is structured to decouple model execution completely from the application interface.
          It operates across three distinct boundaries:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 text-xs">
          <div className="p-4 bg-neutral-950 border border-neutral-800 rounded-md space-y-2">
            <div className="font-semibold text-white">1. Client / Consumer</div>
            <p className="text-neutral-400 leading-relaxed">
              Standard OpenAI SDKs (Python, Node.js), LangChain, cURL, or frontends communicating over standard HTTP & SSE endpoints.
            </p>
          </div>

          <div className="p-4 bg-neutral-950 border border-neutral-800 rounded-md space-y-2">
            <div className="font-semibold text-white">2. NEXA API Gateway</div>
            <p className="text-neutral-400 leading-relaxed">
              Express router handling Bearer auth, token-bucket rate limiting, JSON schema validation, and OpenAI request serialization.
            </p>
          </div>

          <div className="p-4 bg-neutral-950 border border-neutral-800 rounded-md space-y-2">
            <div className="font-semibold text-white">3. Inference Engine Layer</div>
            <p className="text-neutral-400 leading-relaxed">
              Subsystem managing model weights, RAM/VRAM residency, concurrency queues, and native adapters (llama.cpp).
            </p>
          </div>
        </div>
      </div>

      {/* 2. Technical Invariants & Transparency */}
      <div className="p-6 bg-neutral-900 border border-neutral-800 rounded-lg space-y-4">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <Shield className="w-4 h-4 text-neutral-400" />
          <span>Operational Invariants & Guarantees</span>
        </h2>

        <div className="space-y-3 text-xs text-neutral-300 leading-relaxed">
          <div className="flex items-start gap-2">
            <span className="font-mono text-neutral-500">01.</span>
            <div>
              <strong className="text-white">Zero Third-Party Inference Forwarding: </strong>
              The engine does not proxy or forward user requests to Gemini, OpenAI, Anthropic, or any hosted cloud AI. All inference is intended to run locally on your host machine.
            </div>
          </div>

          <div className="flex items-start gap-2">
            <span className="font-mono text-neutral-500">02.</span>
            <div>
              <strong className="text-white">Strict Hardware State Machine: </strong>
              The engine will never pretend a model is running if its GGUF weights are not verified on the filesystem and loaded into memory. Requests to unloaded models return an explicit HTTP 400 error.
            </div>
          </div>

          <div className="flex items-start gap-2">
            <span className="font-mono text-neutral-500">03.</span>
            <div>
              <strong className="text-white">Transparent Provenance: </strong>
              NEXA is an inference engine and model-serving software layer for open-weight foundation models (Llama 3, Mistral, Qwen, Phi-3). It does not claim to have trained its own base weights.
            </div>
          </div>
        </div>
      </div>

      {/* 3. OpenAI Drop-In SDK Integration */}
      <div className="p-6 bg-neutral-900 border border-neutral-800 rounded-lg space-y-4">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <Terminal className="w-4 h-4 text-neutral-400" />
          <span>Connecting Client Applications with OpenAI SDK</span>
        </h2>
        <p className="text-xs text-neutral-300">
          Any application built for OpenAI can be pointed to your NEXA instance by changing <code className="text-neutral-200 font-mono">base_url</code>:
        </p>

        {/* Python Snippet */}
        <div className="space-y-2">
          <span className="text-xs font-medium text-neutral-400">Python:</span>
          <pre className="p-4 bg-neutral-950 border border-neutral-800 rounded-md text-xs font-mono text-neutral-300 overflow-x-auto leading-relaxed">
{`from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="nexa_live_dev_master_key_local"
)

response = client.chat.completions.create(
    model="llama-3-8b-instruct",
    messages=[
        {"role": "user", "content": "How does vector caching optimize inference?"}
    ],
    temperature=0.7,
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)`}
          </pre>
        </div>

        {/* Node.js Snippet */}
        <div className="space-y-2 pt-2">
          <span className="text-xs font-medium text-neutral-400">Node.js / TypeScript:</span>
          <pre className="p-4 bg-neutral-950 border border-neutral-800 rounded-md text-xs font-mono text-neutral-300 overflow-x-auto leading-relaxed">
{`import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:3000/v1',
  apiKey: 'nexa_live_dev_master_key_local'
});

const response = await openai.chat.completions.create({
  model: 'llama-3-8b-instruct',
  messages: [{ role: 'user', content: 'What are the main components of a tensor?' }],
  stream: false
});

console.log(response.choices[0].message.content);`}
          </pre>
        </div>
      </div>

      {/* 4. Quantization Reference Table */}
      <div className="p-6 bg-neutral-900 border border-neutral-800 rounded-lg space-y-4">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <HardDrive className="w-4 h-4 text-neutral-400" />
          <span>GGUF Quantization Matrix & Hardware Requirements</span>
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400">
                <th className="py-2.5 px-3 font-medium">Model</th>
                <th className="py-2.5 px-3 font-medium">Quantization</th>
                <th className="py-2.5 px-3 font-medium">File Size</th>
                <th className="py-2.5 px-3 font-medium">Minimum RAM</th>
                <th className="py-2.5 px-3 font-medium">Recommended VRAM</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800 font-mono text-neutral-300">
              <tr>
                <td className="py-2.5 px-3 font-sans font-medium text-white">TinyLlama 1.1B</td>
                <td className="py-2.5 px-3">Q4_K_M</td>
                <td className="py-2.5 px-3">~669 MB</td>
                <td className="py-2.5 px-3">2 GB</td>
                <td className="py-2.5 px-3">1 GB (Full offload)</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-sans font-medium text-white">Phi-3 Mini 4K (3.8B)</td>
                <td className="py-2.5 px-3">Q4_K_M</td>
                <td className="py-2.5 px-3">~2.39 GB</td>
                <td className="py-2.5 px-3">4 GB</td>
                <td className="py-2.5 px-3">3 GB (Full offload)</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-sans font-medium text-white">Mistral 7B v0.3</td>
                <td className="py-2.5 px-3">Q4_K_M</td>
                <td className="py-2.5 px-3">~4.37 GB</td>
                <td className="py-2.5 px-3">8 GB</td>
                <td className="py-2.5 px-3">6 GB (Full offload)</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-sans font-medium text-white">Meta Llama 3 8B</td>
                <td className="py-2.5 px-3">Q4_K_M</td>
                <td className="py-2.5 px-3">~4.92 GB</td>
                <td className="py-2.5 px-3">8 GB</td>
                <td className="py-2.5 px-3">6 GB (Full offload)</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-sans font-medium text-white">Qwen 2.5 7B</td>
                <td className="py-2.5 px-3">Q4_K_M</td>
                <td className="py-2.5 px-3">~4.68 GB</td>
                <td className="py-2.5 px-3">8 GB</td>
                <td className="py-2.5 px-3">6 GB (Full offload)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 5. Phase 2 Independent Deployment Topologies */}
      <div className="p-6 bg-neutral-900 border border-neutral-800 rounded-lg space-y-4">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <Cpu className="w-4 h-4 text-neutral-400" />
          <span>Independent Deployment Topologies</span>
        </h2>
        <p className="text-xs text-neutral-300 leading-relaxed">
          NEXA AI Core Engine is engineered to run outside hosted studio environments. You can run the backend headless, separate frontend and backend, or serve unified full-stack:
        </p>

        <div className="space-y-3 text-xs">
          <div className="p-3 bg-neutral-950 border border-neutral-800 rounded-md space-y-1">
            <span className="font-semibold text-white">Option A: Headless Backend Alone (Docker / Termux / Server)</span>
            <pre className="text-[11px] font-mono text-neutral-400">npm run dev:server    # or in production: npm run start:server</pre>
            <p className="text-[11px] text-neutral-500">Starts pure Express API on port 3000 without loading Vite or browser assets.</p>
          </div>

          <div className="p-3 bg-neutral-950 border border-neutral-800 rounded-md space-y-1">
            <span className="font-semibold text-white">Option B: Separated Frontend & Backend</span>
            <pre className="text-[11px] font-mono text-neutral-400">Terminal 1: npm run dev:server    # Port 3000 (API)&#10;Terminal 2: npm run dev:client    # Port 5173 (Vite proxying to 3000)</pre>
            <p className="text-[11px] text-neutral-500">Enables frontend development against a standalone remote or local API.</p>
          </div>

          <div className="p-3 bg-neutral-950 border border-neutral-800 rounded-md space-y-1">
            <span className="font-semibold text-white">Option C: Run Deployment Verification Audit</span>
            <pre className="text-[11px] font-mono text-neutral-400">npm run verify:deployment</pre>
            <p className="text-[11px] text-neutral-500">Audits CPU architecture, RAM tiers, storage read/write permissions, and llama.cpp binary presence.</p>
          </div>
        </div>
      </div>

      {/* 6. Android / Termux Deployment Path */}
      <div className="p-6 bg-neutral-900 border border-neutral-800 rounded-lg space-y-4">
        <h2 className="text-base font-semibold text-white flex items-center gap-2">
          <Terminal className="w-4 h-4 text-neutral-400" />
          <span>Android Termux Deployment (ARM64)</span>
        </h2>
        <p className="text-xs text-neutral-300 leading-relaxed">
          Android devices differ in architecture and thermal profiles. Modern 64-bit ARM devices (Snapdragon 8, Dimensity, Tensor) support native NEON SIMD inference.
        </p>

        <div className="p-4 bg-neutral-950 border border-neutral-800 rounded-md space-y-2 text-xs font-mono text-neutral-300 overflow-x-auto leading-relaxed">
          <div><span className="text-neutral-500"># 1. Acquire wake lock in Termux to prevent CPU sleep</span></div>
          <div>termux-wake-lock</div>
          <div className="pt-2"><span className="text-neutral-500"># 2. Install required packages</span></div>
          <div>pkg update -y && pkg install -y nodejs-lts git cmake clang</div>
          <div className="pt-2"><span className="text-neutral-500"># 3. Compile llama.cpp natively for ARM64</span></div>
          <div>git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp</div>
          <div>cmake -B build -DGGML_OPENBLAS=OFF && cmake --build build --config Release -j$(nproc)</div>
          <div className="pt-2"><span className="text-neutral-500"># 4. Start standalone NEXA engine</span></div>
          <div>cd ~/nexa-engine && npm run start:server</div>
        </div>

        <div className="text-[11px] text-neutral-400 space-y-1">
          <p><strong className="text-neutral-300">Android Limitation Note:</strong> 32-bit ARM (armv7l) has a ~2 GB per-process address limit and cannot run models $\ge$ 2 GB. Android Low Memory Killer (LMK) will terminate processes if RAM exceeds physical limits. For detailed architecture matrix, review <code className="text-neutral-200">docs/TERMUX_DEPLOYMENT.md</code>.</p>
        </div>
      </div>
    </div>
  );
};
