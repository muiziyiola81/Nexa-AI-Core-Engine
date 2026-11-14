# NEXA AI Core Engine — Production Deployment Guide

NEXA AI Core Engine is an independent, self-contained AI inference server and model-serving layer. It does not depend on Google AI Studio runtime services or proprietary cloud AI APIs (Gemini, OpenAI, Anthropic).

---

## 1. Architecture Overview & Decoupling

The codebase supports three execution topologies:

| Topology | Command | Port(s) | Description |
| :--- | :--- | :--- | :--- |
| **Unified Full-Stack** | `npm run dev` / `npm start` | `3000` | Express API hosting both OpenAI `/v1/*` endpoints and React SPA console on a single port. |
| **Headless Backend Alone** | `npm run dev:server` / `npm run start:server` | `3000` | Standalone Node.js Express API. Zero browser or frontend overhead. Ideal for servers, Docker, and Termux. |
| **Separated Frontend & Backend** | Terminal 1: `npm run dev:server`<br>Terminal 2: `npm run dev:client` | `3000` (API)<br>`5173` (UI) | Dedicated Vite frontend server proxying `/v1` and `/health` requests to the standalone backend. |

---

## 2. Running Backend and Frontend Separately

### Step 1: Start the Standalone Backend

In your first terminal:

```bash
# Set environment variables (or copy .env.example to .env)
cp .env.example .env

# Start standalone headless Node.js API
npm run dev:server
# or in production:
# npm run start:server
```

The backend boots without loading Vite:
```
[NEXA ENGINE] Initializing subsystems...
[NEXA ENGINE] Models registered: 6
[NEXA ENGINE] Running in Headless Standalone API Mode (Frontend disabled)
[NEXA ENGINE] Server listening on http://0.0.0.0:3000
[NEXA ENGINE] Health check: http://localhost:3000/health
[NEXA ENGINE] OpenAI Models: http://localhost:3000/v1/models
[NEXA ENGINE] OpenAI Completions: http://localhost:3000/v1/chat/completions
```

Verify backend health independently:
```bash
curl http://localhost:3000/health
```

### Step 2: Start the Standalone React Frontend

In your second terminal:

```bash
# Start Vite development server
npm run dev:client
```

Vite will serve the console on `http://localhost:5173` and automatically proxy all `/v1/*` and `/health` calls to `http://localhost:3000`.

### Step 3: Production Frontend Build

To build the static frontend bundle for deployment behind Nginx, Caddy, or Cloudflare Pages:

```bash
npm run build:client
```
The compiled assets will be placed in `./dist/`.

---

## 3. Environment Configuration (`.env`)

Create a `.env` file from the provided `.env.example`:

```bash
cp .env.example .env
```

Key environment variables:

```ini
# Networking
PORT=3000
HOST=0.0.0.0
NODE_ENV=production

# Headless mode (true skips Vite middleware completely)
STANDALONE_API=true

# Storage directory for .gguf model weights
MODELS_DIR=./models

# Local llama.cpp service or binary location
LLAMA_CPP_SERVER_URL=http://127.0.0.1:8080
NEXA_LLAMA_CPP_PATH=/usr/local/bin/llama-server

# Security
REQUIRE_API_KEY=true
MASTER_API_KEY=nexa_live_my_secure_secret_key_here
RATE_LIMIT_PER_MINUTE=120
MAX_CONCURRENT_REQUESTS=4
```

---

## 4. Deployment Verification

Run the automated deployment verification script before launching in production:

```bash
npm run verify:deployment
```

This verifies:
1. Node.js engine compatibility (>= v18.0.0).
2. Host CPU architecture (ARM64, x64, ARM32 warnings).
3. Memory capacity and inference tier sizing.
4. Filesystem read/write permissions for `./models` and `./data`.
5. Detection of `llama-server` or `llama-cli`.
6. Zero cloud AI dependency audit.

---

## 5. Docker Deployment (Headless API)

Create a `Dockerfile` for headless server deployment:

```dockerfile
FROM node:22-bookworm-slim

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev

COPY . .
ENV NODE_ENV=production
ENV STANDALONE_API=true
ENV PORT=3000
ENV MODELS_DIR=/models

VOLUME ["/models", "/app/data"]
EXPOSE 3000

CMD ["npm", "run", "start:server"]
```

Build and run:
```bash
docker build -t nexa-core-engine .
docker run -d -p 3000:3000 -v /path/to/my/models:/models nexa-core-engine
```
