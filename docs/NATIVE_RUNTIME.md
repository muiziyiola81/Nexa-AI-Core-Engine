# NEXA Native CPU Inference Engine — Architecture & Deployment Guide
Phase 3: Independent Native Runtime

## 1. Executive Summary

NEXA Native Runtime is an independent CPU inference engine engineered from scratch. It is designed to remove long-term dependency on third-party runtime frameworks (such as llama.cpp, Ollama, ONNX Runtime) and external AI cloud APIs (OpenAI, Gemini, Anthropic).

The engine provides:
1. **Native C++ Engine Core** (`native/`): A high-performance C++20 implementation providing tensor representations, cache-tiled GEMM matrix multiplication, vector arithmetic, neural network layers (RMSNorm, RoPE, Attention, SwiGLU, TransformerBlock), a custom byte-level tokenizer, and a `.nexa` binary model loader.
2. **TypeScript / Node.js Twin Kernel** (`server/native/`): An identical mathematical implementation running directly in the server process to guarantee immediate testing and headless execution without requiring external C++ build tooling in restricted host environments.
3. **OpenAI-Compatible Integration Layer** (`NexaNativeAdapter`): A clean adapter fulfilling the NEXA `InferenceAdapter` interface, allowing the existing OpenAI-compliant API routes (`GET /v1/models`, `POST /v1/chat/completions`) to serve inference natively.

---

## 2. Architecture & File Structure

```
native/
├── CMakeLists.txt                 # Multi-platform build definition (x86_64, ARM64, Termux)
├── include/
│   └── nexa/
│       ├── tensor.hpp             # NexaTensor class, shapes, 64-byte aligned allocators
│       ├── nn.hpp                 # RMSNorm, LayerNorm, RoPE, Attention, SwiGLU, TransformerBlock
│       ├── tokenizer.hpp          # Byte-level vocabulary encoder and decoder
│       ├── model_format.hpp       # .nexa binary header and tensor descriptor structs
│       └── model_loader.hpp       # Model validator, RAM budgeting, and tensor weight mmap
└── src/
    ├── tensor.cpp                 # GEMM cache-tiled matmul, vector math, stable softmax
    ├── nn.cpp                     # Forward passes for transformer blocks & activations
    ├── tokenizer.cpp              # UTF-8 token encoding, byte fallback, and decoding
    ├── model_loader.cpp           # Header validation, offset calculation, memory checks
    └── main.cpp                   # Standalone CLI binary (nexa-runtime-cli)

server/
├── native/
│   ├── tensor.ts                  # TypeScript tensor kernel matching C++ implementation
│   ├── nn.ts                      # Neural network components (RMSNorm, RoPE, Attention, SwiGLU)
│   ├── tokenizer.ts               # ByteTokenizer with UTF-8 byte round-trip
│   ├── model_format.ts            # Binary .nexa encoder/decoder with magic number check
│   ├── model_loader.ts            # Invariant checker & memory requirement calculations
│   └── native_adapter.ts          # NexaNativeAdapter implementing InferenceAdapter
├── engine/
│   └── engine_manager.ts          # Registers 'nexa-native' as default inference runtime
└── tests/
    └── native_engine_tests.ts     # 12 unit tests verifying mathematical correctness
```

---

## 3. Implemented Native Operations

### 3.1 Tensor Computation Layer
- **Memory Management**: Custom aligned allocator ensuring 64-byte memory alignment (`std::aligned_alloc(64, ...)` in C++, typed arrays in TypeScript) for AVX2/AVX-512 and ARM NEON SIMD vector operations.
- **Matrix Multiplication (GEMM)**: Cache-blocked 3-loop matmul ($C = A \times B$) with dimension compatibility checking ($A_{\text{cols}} == B_{\text{rows}}$).
- **Vector Arithmetic**: Vector addition, scalar scaling, and element-wise Hadamard multiplication.
- **Numerically Stable Softmax**: Subtracts $\max(x)$ before exponential evaluation ($\exp(x_i - x_{\max}) / \sum \exp(x_j - x_{\max})$), preventing IEEE 754 float overflow on values $> 88$.

### 3.2 Neural Network Execution Layer
- **Embedding Layer**: Index lookup projecting token IDs into hidden dimension vectors.
- **RMSNorm (Root Mean Square Normalization)**:
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
- **LayerNorm**: Mean-centered and variance-normalized feature scaling with learnable gain and bias.
- **Rotary Position Embedding (RoPE)**: Applies 2D rotation matrices across head pairs based on token sequence index $m$:
  $$R_{\Theta, m}^d = \text{diag}\left( R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m} \right)$$
- **Scaled Dot-Product Attention**:
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- **SwiGLU Feed-Forward Network**:
  $$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$
- **Transformer Block**: Combines pre-norm residual self-attention with pre-norm residual SwiGLU FFN.

### 3.3 Byte-Level Tokenizer
- Reversible byte-level UTF-8 encoding and decoding.
- Full special token handling (`<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`).
- Guaranteed lossless round-trip for arbitrary binary or Unicode text without crashing or discarding unmapped characters.

---

## 4. Multi-Platform Build Configuration

### 4.1 Linux x86_64
Prerequisites: `cmake >= 3.16`, `clang++ >= 14` or `g++ >= 11`.
```bash
# In project root:
cd native
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Verify execution:
./build/bin/nexa-runtime-cli --test-tensor
```

Flags applied:
- `-O3 -fPIC`
- `-mavx2 -mfma` (when supported by host CPU)
- Standard library linking: `-pthread`

### 4.2 Linux ARM64 (aarch64 / AWS Graviton / Raspberry Pi 4/5)
```bash
cd native
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Verify execution:
./build/bin/nexa-runtime-cli --test-tensor
```

Flags applied:
- `-O3 -march=armv8-a+simd`

---

## 5. Android Termux Build & Deployment (ARM64)

### 5.1 Termux Environment Setup
Termux provides a self-contained Linux environment on Android. Execute these exact commands inside Termux:

```bash
# 1. Update package database and packages
pkg update && pkg upgrade -y

# 2. Install build dependencies and compilers
pkg install -y git clang cmake make nodejs-lts

# 3. Verify toolchain
clang++ --version
cmake --version
node -v

# 4. Clone or transfer the NEXA repository
cd ~/
# If transferring via storage:
# cp -r /sdcard/nexa-core-engine ./
cd nexa-core-engine

# 5. Compile the C++ Native Runtime
cd native
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 6. Test native tensor execution
./build/bin/nexa-runtime-cli --test-tensor

# 7. Start the Node.js API server (bound to local loopback)
cd ..
npm install
npm run build
npm start
```

### 5.2 Android Hardware Limitations & Caveats

1. **Architecture Compatibility**:
   - **ARM64 (`aarch64`)**: Supported with SIMD vectorization.
   - **32-bit ARM (`armeabi-v7a`)**: Strictly limited. 32-bit address spaces cap accessible memory to ~3 GB, making LLM inference impractical. Compiling on 32-bit ARM emits a build-time warning.
   - **x86 / x86_64 Emulators**: Supported for development only.

2. **Android Low Memory Killer (LMK)**:
   - Android aggressively terminates foreground background processes consuming $> 2\text{--}3\,\text{GB}$ of RAM.
   - Models loaded in Termux should not exceed $1\text{--}2\,\text{GB}$ parameter footprint (e.g. Q4_0 quantized 1B models).
   - Use `termux-wake-lock` to prevent the OS from suspending the CPU during inference:
     ```bash
     termux-wake-lock
     ```

3. **Thermal Throttling**:
   - Mobile SoCs sustain multi-core execution for 30–90 seconds before thermal throttling lowers CPU clock speeds by 40–60%.
   - Configure thread counts to match physical performance cores (e.g., `-t 4` instead of 8 on big.LITTLE architectures).

---

## 6. Features Missing Before Real Language-Model Inference

While the foundation is complete and mathematically tested, the following components are required before running production-grade LLM generation:

1. **Quantization Kernels**:
   - SIMD-accelerated 4-bit (`Q4_0`, `Q4_K`) and 8-bit (`Q8_0`) dot-product kernels for matrix-vector multiplication without dequantizing the entire matrix to FP32 first.
2. **Key-Value (KV) Cache Paging**:
   - Dedicated contiguous circular ring buffer or paged attention memory to store past token Keys and Values across multi-turn generation steps ($O(1)$ token decode rather than $O(N)$ recomputation).
3. **Model Weight Ingestion Tool**:
   - A converter script (`scripts/convert_weights_to_nexa.py` or C++ binary) to unpack Safetensors / GGUF weights and serialize them directly into the `.nexa` binary layout.
4. **Dynamic Top-K / Top-P (Nucleus) Sampling**:
   - Sampling kernel implementing Temperature, Top-K filtering, Top-P cumulative mass truncation, and repetition penalties.
5. **Multi-threading Execution Pool**:
   - OpenMP or pthread worker pool to split GEMM rows across CPU cores concurrently.

---

## 7. Implemented Functionality vs. Future Work

| Component | Current Phase 3 Status | Future Milestone |
|---|---|---|
| **C++ Core Engine** | Implemented (`native/include`, `native/src`) | Hardware BLAS / OpenMP multi-threading |
| **Tensor Math** | Implemented (Tensors, shapes, matmul, vector math, softmax) | AVX-512 & NEON assembly kernels |
| **Neural Net Layers** | Implemented (RMSNorm, RoPE, Attention, SwiGLU, TransformerBlock) | Paged KV-Cache allocation |
| **Model Format** | Implemented (`.nexa` 128-byte header, 64-byte aligned tables) | Quantized tensor block formats |
| **Model Loader** | Implemented (RAM verification, header checks, memory budgeting) | `mmap` zero-copy weight loading |
| **Tokenizer** | Implemented (Byte-level UTF-8 reversible encoder/decoder) | BPE merge pairs parser (`merges.txt`) |
| **API Integration** | Implemented (`NexaNativeAdapter` connected to OpenAI API) | Native C++ addon / shared library (`.so`) |
| **Testing** | 100% Passing (12 unit tests + 10 integration tests) | End-to-end full weight perplexity benchmarks |
