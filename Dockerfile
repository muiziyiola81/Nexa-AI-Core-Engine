# ==============================================================================
# NEXA AI Core Engine - Cloud Run Production Dockerfile (Multi-Stage)
# Preserves independent C++20 nexa-runtime-cli & Node.js Express API server.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Compile Native C++20 Inference Engine (Portable Cloud Run x86_64/ARM64)
# Default ENABLE_AVX2=OFF ensures compatibility across heterogeneous Cloud Run CPUs.
# Override with --build-arg ENABLE_AVX2=ON if targeting dedicated AVX2 hosts.
# ------------------------------------------------------------------------------
FROM debian:bookworm-slim AS cpp-builder

ARG ENABLE_AVX2=OFF
ARG ENABLE_ARM_NEON=ON

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /build/native
COPY native/CMakeLists.txt ./
COPY native/include ./include
COPY native/src ./src

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_AVX2=${ENABLE_AVX2} \
      -DENABLE_ARM_NEON=${ENABLE_ARM_NEON} \
  && cmake --build build -j"$(nproc)" \
  && ./build/nexa-runtime-cli --info-json

# ------------------------------------------------------------------------------
# Stage 2: Install Production Node.js Dependencies
# ------------------------------------------------------------------------------
FROM node:22-bookworm-slim AS node-deps

WORKDIR /app
COPY package.json package-lock.json* bun.lock* ./
RUN npm install --omit=dev --no-audit --no-fund

# ------------------------------------------------------------------------------
# Stage 3: Minimal Cloud Run Runtime Image
# ------------------------------------------------------------------------------
FROM node:22-bookworm-slim AS runtime

RUN groupadd -r nexa && useradd -r -g nexa -m -d /home/nexa nexa

WORKDIR /app

# Copy compiled C++20 native engine binary
COPY --from=cpp-builder /build/native/build/nexa-runtime-cli /app/bin/nexa-runtime-cli
RUN chmod +x /app/bin/nexa-runtime-cli

# Copy production Node modules & backend server sources
COPY --from=node-deps /app/node_modules /app/node_modules
COPY package.json server.ts tsconfig.json ./
COPY server ./server

# Copy engine defaults, tokenizers, managed datasets, and production .nexa checkpoints
COPY data/engine_config.json ./data/engine_config.json
COPY data/tokenizers ./data/tokenizers
COPY data/datasets ./data/datasets
COPY data/checkpoints ./data/checkpoints

# Create writable runtime directories (for ephemeral cache or GCS volume mount)
RUN mkdir -p /app/models /app/data/experiments /app/data/test_models \
  && chown -R nexa:nexa /app

USER nexa

# Cloud Run defaults (override via gcloud run deploy --set-env-vars / secrets)
ENV NODE_ENV=production \
    PORT=8080 \
    HOST=0.0.0.0 \
    STANDALONE_API=true \
    ACTIVE_ADAPTER=nexa-native \
    REQUIRE_API_KEY=true \
    ALLOWED_ORIGINS=* \
    MODELS_DIR=/app/models \
    DATA_DIR=/app/data \
    AUTO_REGISTER_NEXA_CHECKPOINTS=true \
    AUTO_LOAD_MODELS=nexa-p36-tier3 \
    AUTO_LOAD_ON_DEMAND=true \
    DEFAULT_CPU_THREADS=2 \
    MAX_CONCURRENT_REQUESTS=4

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:' + (process.env.PORT || 8080) + '/health').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"

# Direct PID 1 execution so SIGTERM reaches Node.js cleanly without shell wrapper
CMD ["node", "--import", "tsx", "server/standalone.ts"]
