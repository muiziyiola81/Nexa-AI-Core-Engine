/**
 * NEXA AI Core Engine - Deployment Verification Script
 * Validates host environment, architecture compatibility, filesystem permissions,
 * llama.cpp availability, memory capacity, and zero-cloud-dependency compliance.
 * 
 * Run via: npm run verify:deployment
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import http from 'http';
import { ModelRegistry } from '../server/store/model_registry.js';
import { ConfigStore } from '../server/store/config.js';
import { KeyStore } from '../server/store/key_store.js';
import { LlamaCppAdapter } from '../server/engine/llama_cpp_adapter.js';
import { NexaNativeAdapter } from '../server/native/native_adapter.js';

interface CheckItem {
  id: string;
  name: string;
  status: 'PASS' | 'WARN' | 'FAIL' | 'INFO';
  message: string;
  recommendation?: string;
}

async function runDeploymentVerification() {
  console.log('================================================================');
  console.log('       NEXA AI CORE ENGINE - DEPLOYMENT VERIFICATION AUDIT       ');
  console.log('================================================================\n');

  const checks: CheckItem[] = [];

  // 1. Node.js Runtime Check
  const nodeVersion = process.version;
  const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0], 10);
  if (majorVersion >= 18) {
    checks.push({
      id: 'node_runtime',
      name: 'Node.js Runtime Version',
      status: 'PASS',
      message: `Node.js ${nodeVersion} detected (supported >= v18.0.0).`
    });
  } else {
    checks.push({
      id: 'node_runtime',
      name: 'Node.js Runtime Version',
      status: 'FAIL',
      message: `Node.js ${nodeVersion} is unsupported. Please upgrade to Node.js v18.0.0 or higher.`
    });
  }

  // 2. Hardware Architecture & Platform
  const arch = os.arch();
  const platform = os.platform();
  const isAndroidTermux = fs.existsSync('/data/data/com.termux');

  let archCompatibility: CheckItem['status'] = 'PASS';
  let archMessage = `Architecture: ${arch}, OS Platform: ${platform}${isAndroidTermux ? ' (Termux Environment)' : ''}.`;
  let archRec: string | undefined;

  if (arch === 'arm64' || arch === 'aarch64') {
    archMessage += ' Native 64-bit ARM with NEON SIMD acceleration supported.';
  } else if (arch === 'x64') {
    archMessage += ' 64-bit x86 with AVX/AVX2 vector instructions supported.';
  } else if (arch === 'arm') {
    archCompatibility = 'WARN';
    archMessage += ' 32-bit ARM detected. Virtual address space is capped at ~2-3 GB.';
    archRec = 'Only sub-1B models (e.g. TinyLlama quantized) can fit in 32-bit memory. Upgrade to 64-bit ARM64 device for 7B/8B models.';
  } else if (arch === 'ia32') {
    archCompatibility = 'WARN';
    archMessage += ' 32-bit x86 detected. Memory address space limited to < 4 GB.';
    archRec = '32-bit architectures cannot load large open-weight foundation models.';
  }

  checks.push({
    id: 'cpu_architecture',
    name: 'CPU Architecture & Instruction Set',
    status: archCompatibility,
    message: archMessage,
    recommendation: archRec
  });

  // 3. Host Memory Capacity
  const totalMemMb = Math.round(os.totalmem() / (1024 * 1024));
  const freeMemMb = Math.round(os.freemem() / (1024 * 1024));
  let memStatus: CheckItem['status'] = 'PASS';
  let memMsg = `Total RAM: ${totalMemMb} MB, Free RAM: ${freeMemMb} MB.`;
  let memRec: string | undefined;

  if (totalMemMb < 2048) {
    memStatus = 'WARN';
    memMsg += ' Device has less than 2 GB RAM.';
    memRec = 'Insufficient memory for standard neural inference. Requires low-memory edge quantized weights (< 1B).';
  } else if (totalMemMb < 6000) {
    memStatus = 'INFO';
    memMsg += ' Suitable for native .nexa models and compact GGUF models (TinyLlama 1.1B, Phi-3 Mini 4K Q4).';
  } else {
    memMsg += ' Adequate memory for native .nexa models and 7B/8B open-weight models.';
  }

  checks.push({
    id: 'memory_capacity',
    name: 'Host Memory (RAM) Allocation',
    status: memStatus,
    message: memMsg,
    recommendation: memRec
  });

  // 4. Filesystem & Storage Directories
  const modelsDir = path.resolve(process.cwd(), process.env.MODELS_DIR || 'models');
  const dataDir = path.resolve(process.cwd(), process.env.DATA_DIR || 'data');

  let fsStatus: CheckItem['status'] = 'PASS';
  let fsMsg = 'Storage directories verified.';

  try {
    if (!fs.existsSync(modelsDir)) fs.mkdirSync(modelsDir, { recursive: true });
    fs.accessSync(modelsDir, fs.constants.R_OK | fs.constants.W_OK);

    if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });
    fs.accessSync(dataDir, fs.constants.R_OK | fs.constants.W_OK);

    fsMsg = `Models dir: "${modelsDir}" (read/write OK), Data dir: "${dataDir}" (read/write OK).`;
  } catch (err: unknown) {
    fsStatus = 'FAIL';
    fsMsg = `Filesystem permissions failure: ${(err as Error).message}`;
  }

  checks.push({
    id: 'storage_directories',
    name: 'Filesystem Permissions & Directories',
    status: fsStatus,
    message: fsMsg
  });

  // 5. NEXA Native C++20 Runtime Binary Detection
  const nativeAdapter = new NexaNativeAdapter();
  const nativeAvail = await nativeAdapter.isAvailable();
  checks.push({
    id: 'nexa_native_cpp_runtime',
    name: 'NEXA Native C++20 Runtime (nexa-runtime-cli)',
    status: nativeAvail.available ? 'PASS' : 'FAIL',
    message: nativeAvail.available
      ? `NEXA C++20 binary ready: ${nativeAvail.version} (${nativeAvail.binaryPath}).`
      : `NEXA C++20 binary missing or unexecutable: ${nativeAvail.reason}`,
    recommendation: nativeAvail.available
      ? undefined
      : 'Compile native C++ runtime via: cd native && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && cp build/nexa-runtime-cli ../bin/'
  });

  // 6. Optional llama.cpp Binary & Server Detection
  const llamaAdapter = new LlamaCppAdapter();
  const llamaAvail = await llamaAdapter.isAvailable();
  let llamaStatus: CheckItem['status'] = 'INFO';
  let llamaMsg = '';
  let llamaRec: string | undefined;

  if (llamaAvail.available) {
    llamaStatus = 'PASS';
    llamaMsg = `Optional llama.cpp runtime detected: ${llamaAvail.version} (${llamaAvail.binaryPath}).`;
  } else {
    llamaStatus = 'INFO';
    llamaMsg = 'Optional llama.cpp binary not detected (not required when using nexa-native C++20 adapter).';
    llamaRec = isAndroidTermux
      ? 'Optional in Termux: Compile via "pkg install git cmake clang && git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j".'
      : 'Optional: Mount llama-server or compile llama.cpp if serving external .gguf models.';
  }

  checks.push({
    id: 'llama_cpp_runtime',
    name: 'Optional llama.cpp Runtime Availability',
    status: llamaStatus,
    message: llamaMsg,
    recommendation: llamaRec
  });

  // 7. Installed Model Weights Check (.nexa & .gguf)
  const registry = new ModelRegistry();
  const models = registry.getAll();
  const installedModels = models.filter(m => m.isInstalled);
  let weightsStatus: CheckItem['status'] = 'INFO';
  let weightsMsg = '';
  let weightsRec: string | undefined;

  if (installedModels.length > 0) {
    weightsStatus = 'PASS';
    weightsMsg = `${installedModels.length} model checkpoint(s) verified on disk: ${installedModels.map(m => `${m.id} (${m.format})`).join(', ')}.`;
  } else {
    weightsStatus = 'WARN';
    weightsMsg = 'No .nexa or .gguf weight files found on disk.';
    weightsRec = 'Verify data/checkpoints/*.nexa or mount a GCS bucket to MODELS_DIR for .gguf weights.';
  }

  checks.push({
    id: 'model_weights_presence',
    name: 'Model Weights Inventory (.nexa / .gguf)',
    status: weightsStatus,
    message: weightsMsg,
    recommendation: weightsRec
  });

  // 7. Cloud AI Exclusion & Security Audit
  const serverFiles = [
    'server/engine/llama_cpp_adapter.ts',
    'server/engine/diagnostic_adapter.ts',
    'server/engine/engine_manager.ts',
    'server/api/routes.ts',
    'server/middleware/auth.ts',
    'server/middleware/rate_limiter.ts',
    'server.ts'
  ];
  const forbidden = ['api.openai.com', 'api.anthropic.com', 'generativelanguage.googleapis.com', '@google/genai'];
  let cloudAuditPass = true;
  let cloudViolation = '';

  for (const f of serverFiles) {
    const full = path.resolve(process.cwd(), f);
    if (fs.existsSync(full)) {
      const src = fs.readFileSync(full, 'utf-8');
      for (const forb of forbidden) {
        if (src.includes(forb)) {
          cloudAuditPass = false;
          cloudViolation = `Found forbidden token "${forb}" in ${f}`;
          break;
        }
      }
    }
  }

  checks.push({
    id: 'cloud_exclusion_audit',
    name: 'Independence Audit (Zero Cloud AI APIs)',
    status: cloudAuditPass ? 'PASS' : 'FAIL',
    message: cloudAuditPass
      ? 'Clean: Zero Gemini, OpenAI, or Anthropic hosted inference endpoints used in engine codebase.'
      : cloudViolation
  });

  // 8. Output Report
  for (const c of checks) {
    const badge = `[${c.status}]`.padEnd(8);
    console.log(`${badge} ${c.name}`);
    console.log(`         ${c.message}`);
    if (c.recommendation) {
      console.log(`         💡 Action: ${c.recommendation}`);
    }
    console.log('');
  }

  const failedCount = checks.filter(c => c.status === 'FAIL').length;
  const warnCount = checks.filter(c => c.status === 'WARN').length;
  const passCount = checks.filter(c => c.status === 'PASS').length;

  console.log('----------------------------------------------------------------');
  console.log(`Audit Summary: ${passCount} Passed, ${warnCount} Warnings, ${failedCount} Failures`);
  console.log('----------------------------------------------------------------\n');

  if (failedCount > 0) {
    process.exit(1);
  }
}

runDeploymentVerification().catch(err => {
  console.error('Fatal deployment verification error:', err);
  process.exit(1);
});
