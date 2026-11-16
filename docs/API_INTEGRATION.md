# NEXA AI Core Engine — API Integration Specification

This specification documents the external HTTP API exposed by the **NEXA AI Core Engine** (independent C++20 inference runtime `bin/nexa-runtime-cli` orchestrated by the Node.js/Express backend).

---

## 1. Startup, Hosting & Environment Variables

### Server Entry Points
- **Headless Standalone API (Docker / Cloud Run / Render):**
  ```bash
  npm run start:server   # runs tsx server/standalone.ts
  ```
- **Unified Full-Stack Server (API + Vite SPA):**
  ```bash
  npm run dev            # runs tsx server.ts on PORT (default 3000)
  ```

### Required & Key Environment Variables
| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` (Docker/Cloud Run: `8080`) | HTTP listen port (`HOST=0.0.0.0`) |
| `STANDALONE_API` | `false` (`true` in `standalone.ts`) | Disables Vite/SPA serving for pure API deployments |
| `ACTIVE_ADAPTER` | `nexa-native` | Active runtime (`nexa-native` \| `llama-cpp` \| `nexa-diagnostic`) |
| `NEXA_CPP_BINARY_PATH` | auto (`bin/nexa-runtime-cli`) | Explicit path to compiled C++20 `nexa-runtime-cli` binary |
| `DATA_DIR` | `./data` | Directory holding `engine_config.json`, `api_keys.json`, checkpoints, tokenizers |
| `AUTO_REGISTER_NEXA_CHECKPOINTS` | `true` | Auto-registers bundled `.nexa` checkpoints & `.nbt` tokenizers |
| `AUTO_LOAD_MODELS` | unset (Docker/Cloud Run: `nexa-p36-tier3`) | Comma-separated model IDs to pre-load on startup |
| `AUTO_LOAD_ON_DEMAND` | `false` (`true` in `NODE_ENV=production`) | Lazily loads installed models on cold-start requests unless explicitly unloaded |
| `REQUIRE_LOADED_MODEL_FOR_READY` | `false` | If `true`, `/ready` returns `503` until at least one model is loaded |
| `REQUIRE_API_KEY` | `false` (`true` in production) | Enforces `Authorization: Bearer <token>` on all `/v1/*` routes |
| `NEXA_API_KEY` | unset | Primary environment Bearer API key (`engine:admin` + all scopes) |
| `NEXA_API_KEYS` | unset | Comma-separated additional Bearer API keys |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS allowlist (e.g. `https://app.example.com`) |

---

## 2. Authentication & CORS

- **Public Endpoints (no auth required for container health probes):**
  - `GET /health`, `GET /healthz`, `GET /ready`
- **Protected Endpoints (`/v1/*` when `REQUIRE_API_KEY=true`):**
  - Send header: `Authorization: Bearer <YOUR_API_KEY>`
  - Scopes:
    - `models:read`: `GET /v1/models`, `GET /v1/models/:modelId`
    - `chat:completions`: `POST /v1/chat/completions`, `POST /v1/tokenize`
    - `models:manage`: `POST /v1/models/register`, `DELETE /v1/models/:modelId`, `POST /v1/models/:modelId/load`, `POST /v1/models/:modelId/unload`
    - `engine:admin`: `/v1/engine/*`, `/v1/keys*`, `/v1/tests/run`
- **CORS:**
  - Exact allowlisted origins receive `Access-Control-Allow-Origin: <origin>` and `Access-Control-Allow-Credentials: true`.
  - Disallowed `OPTIONS` preflight origins receive HTTP `403` (`code: "cors_origin_denied"`).

---

## 3. Registered Native Checkpoints

| Model ID | Architecture | Vocab / Ctx | Checkpoint & Tokenizer | Notes |
|---|---|---|---|---|
| `nexa-p36-tier3` | 2L, 4H, 96D (`340K`) | BPE `768` / `512` | `nexa_p36_tier3_conv_best.nexa` + `nexa_bpe_phase36_shared.nbt` | Active production default |
| `nexa-p37-v2` | 4L, 4H, 128D, FFN `384`, RoPE + Tied Embeddings (`1.05M`) | BPE `1536` / `160` | `nexa_p37_v2_best.nexa` + `nexa_bpe_p37_v2.nbt` | Candidate Phase 3.7 model |
| `nexa-nano-step-40` | 2L, 2H, 64D (`117K`) | Byte `256` / `512` | `nexa_nano_step_40.nexa` | Compact cold-start model |

---

## 4. Endpoints & Real Request/Response Examples

### 4.1 Health & Readiness
`GET /health` (alias `GET /healthz`) and `GET /ready`

```bash
curl -s http://localhost:3000/ready
```
```json
{
  "status": "ready",
  "ready": true,
  "timestamp": "2026-11-16T10:15:00.000Z",
  "activeAdapterId": "nexa-native",
  "checks": {
    "adapterAvailable": true,
    "adapterBinaryPath": "/app/bin/nexa-runtime-cli",
    "dataDirWritable": true,
    "modelsDirWritable": true,
    "loadedModelsCount": 1,
    "loadedModels": ["nexa-p36-tier3"],
    "modelLoadErrors": []
  }
}
```

### 4.2 Discover Models
`GET /v1/models` and `GET /v1/models/:modelId`

```bash
curl -s http://localhost:3000/v1/models \
  -H "Authorization: Bearer $NEXA_API_KEY"
```

### 4.3 Load / Unload Model
`POST /v1/models/:modelId/load` and `POST /v1/models/:modelId/unload`

```bash
curl -s -X POST http://localhost:3000/v1/models/nexa-p37-v2/load \
  -H "Authorization: Bearer $NEXA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'
```
```json
{
  "status": "success",
  "message": "Model \"nexa-p37-v2\" loaded into memory successfully.",
  "data": {
    "modelId": "nexa-p37-v2",
    "state": "loaded",
    "isInstalled": true
  }
}
```

### 4.4 Chat Completions (Single-Turn, Multi-Turn & Streaming SSE)
`POST /v1/chat/completions`

- Converts OpenAI `messages[]` (`system`, `user`, `assistant`, `tool`) into canonical `System:/User:/Assistant:` turns truncated cleanly from oldest turns upward to fit the checkpoint's token budget (`max_seq_len`).
- Supports `temperature` (`0.0–2.0`), `top_p` (`0.0–1.0`), `max_tokens` (`>= 1`), `stop` (`string | string[]`), and `stream` (`boolean`).

**Non-streaming Multi-Turn Example:**
```bash
curl -s -X POST http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer $NEXA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nexa-p37-v2",
    "messages": [
      {"role": "system", "content": "Answer concisely and clearly."},
      {"role": "user", "content": "What is electric current?"},
      {"role": "assistant", "content": "Electric current is the flow of electric charge."},
      {"role": "user", "content": "How does increasing resistance affect current if voltage stays constant?"}
    ],
    "max_tokens": 48,
    "temperature": 0.4,
    "stream": false
  }'
```
```json
{
  "id": "chatcmpl-nexa-cxx-1794824000000",
  "object": "chat.completion",
  "created": 1794824000,
  "model": "nexa-p37-v2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 49,
    "completion_tokens": 13,
    "total_tokens": 62
  },
  "system_fingerprint": "fp_nexa_cxx20_v2_rope"
}
```

**Streaming SSE Example (`stream: true`):**
```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1794824000,"model":"nexa-p37-v2","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1794824000,"model":"nexa-p37-v2","choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1794824000,"model":"nexa-p37-v2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### 4.5 Tokenization
`POST /v1/tokenize`

Automatically selects the matching `.nbt` BPE tokenizer for `nexa-p37-v2` / `nexa-p36-tier3` or byte-level tokenizer for `nexa-nano-step-40`.

```bash
curl -s -X POST http://localhost:3000/v1/tokenize \
  -H "Authorization: Bearer $NEXA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"nexa-p37-v2","text":"What is binary search?"}'
```
```json
{
  "object": "tokenize",
  "model": "nexa-p37-v2",
  "token_count": 7,
  "tokens": [91, 108, 268, 306, 868, 938, 67]
}
```

### 4.6 Standard OpenAI-Compatible Error Envelope
All validation, auth, CORS, lifecycle, and runtime errors return structured JSON:
```json
{
  "error": {
    "message": "Model \"unknown\" does not exist in the registry.",
    "type": "invalid_request_error",
    "param": null,
    "code": "model_not_found"
  }
}
```
Common HTTP status codes: `400` (`invalid_json`, `param_required`, `invalid_value`, `model_not_loaded`), `401` (`invalid_api_key`), `403` (`insufficient_scope`, `cors_origin_denied`), `404` (`model_not_found`), `429` (`rate_limit_exceeded`), `503` (`server_overloaded`, `model_load_failed`), `504` (`inference_timeout`).
