# NEXA AI Core Engine — Android Termux Deployment Guide

This guide details the procedure for deploying the **NEXA AI Core Engine** on Android devices using **Termux**.

---

## 1. Device Architecture Compatibility Matrix

Android devices span several CPU architectures. Do not assume all devices have the same capabilities.

| Architecture | Identification (`uname -m`) | Hardware Examples | Compatibility Status | Notes & Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **ARM64 (aarch64)** | `aarch64`, `arm64` | Snapdragon 8 Gen 1/2/3, Dimensity 9000+, Tensor G1/G2/G3 | **Fully Supported (Recommended)** | 64-bit address space. Supports ARM NEON SIMD vectorization and FP16. Can run 7B/8B models if device has $\ge$ 8GB RAM. |
| **ARM 32-bit (armv7l)** | `armv7l`, `armv8l`, `armhf` | Older budget phones (Cortex-A7, A53 in 32-bit mode) | **Severely Constrained** | Virtual memory space is limited to **~2–3 GB per process**. Models $\ge$ 2 GB (including 7B/8B Q4) will immediately crash with `std::bad_alloc` or OOM. Only sub-1B models (e.g. TinyLlama 1.1B Q4) can theoretically load. |
| **x86_64** | `x86_64` | Android emulators, Intel-based Chromebooks | **Fully Supported** | Standard 64-bit AVX/AVX2 vector acceleration. |
| **x86 (32-bit)** | `i686`, `x86` | Legacy Android-x86 emulators | **Constrained** | 32-bit memory address cap (< 4 GB). |

---

## 2. Step-by-Step Termux Commands (Copy & Paste)

### Step 1: Install Termux Environment
Download and install **Termux** from [F-Droid](https://f-droid.org/packages/com.termux/) (do not use the deprecated Google Play Store build).

### Step 2: Prevent Android Battery Doze (Wake Lock)
Run this command in Termux to keep the CPU active when the screen turns off:
```bash
termux-wake-lock
```

### Step 3: Update Packages & Install Node.js
```bash
pkg update -y
pkg install -y nodejs-lts git cmake clang
```

### Step 4: Clone / Copy NEXA Core Engine & Install
```bash
# Navigate to desired directory
cd ~

# If cloning from git repository:
# git clone <your-repo-url> nexa-engine && cd nexa-engine
# Or navigate to existing directory:
cd nexa-engine

# Install Node.js dependencies
npm install --omit=dev
```

### Step 5: Run Automated Deployment Verification
```bash
npm run verify:deployment
```
This inspects the CPU architecture, checks RAM, and verifies directory permissions.

### Step 6: Native llama.cpp Compilation for Android ARM64
To enable native hardware-accelerated local inference:

```bash
# Clone llama.cpp repository
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Configure and compile natively for ARM64 with NEON SIMD
cmake -B build -DGGML_OPENBLAS=OFF
cmake --build build --config Release -j$(nproc)

# Verify llama-server binary
./build/bin/llama-server --version

# Symlink or copy to NEXA bin directory or system PATH
mkdir -p ~/nexa-engine/bin
cp build/bin/llama-server ~/nexa-engine/bin/
cp build/bin/llama-cli ~/nexa-engine/bin/
cd ~/nexa-engine
```

### Step 7: Place GGUF Model Weights
*(Requirement 13: Models must be acquired deliberately; they are never downloaded automatically in background).*

For a high-efficiency model suitable for mobile RAM:
```bash
mkdir -p models

# Example: Download TinyLlama 1.1B Q4_K_M (~669 MB)
curl -L -o models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### Step 8: Start Headless NEXA Engine in Termux
```bash
# Run headless backend API server
npm run start:server
```

You can now query the engine locally on your Android device:
```bash
# Health check:
curl http://127.0.0.1:3000/health

# OpenAI model catalog:
curl http://127.0.0.1:3000/v1/models
```

---

## 3. Real Android Compatibility Limitations & Warnings

Even on modern ARM64 devices, Android introduces specific operating system constraints that must be accounted for:

1. **Android Low Memory Killer (OOM Daemon)**:
   * Android OS prioritizes foreground user interfaces over background terminal processes.
   * If memory pressure rises (e.g., switching to Camera or heavy app while generating tokens), Android will instantly kill the `node` or `llama-server` process with `SIGKILL` (`Killed`).
   * **Mitigation**: Do not allocate more than 60–70% of physical device RAM to model context + weights.

2. **Android 12+ Phantom Process Killer**:
   * Android 12 and 13 introduced a background daemon that monitors processes spawned by child shells. If more than 32 child processes are active or excessive CPU is consumed in the background, Android terminates them.
   * **Mitigation**: In Termux, run `termux-wake-lock`, or use `adb shell "/system/bin/device_config put activity_manager max_phantom_processes 2147483647"`.

3. **Passive Thermal Throttling**:
   * Smartphones lack active cooling fans. Continuous multi-threaded token generation on 8 CPU cores will cause the SoC temperature to reach 45°C–50°C within 3–5 minutes.
   * Android kernel thermal governors will aggressively throttle CPU frequencies (e.g. from 3.0 GHz down to 1.1 GHz), reducing generation speed by 50–70%.
   * **Mitigation**: Limit thread count to 4 cores (`DEFAULT_CPU_THREADS=4`) to avoid pegging all performance cores simultaneously.

4. **Lack of Vulkan/OpenCL Native Offloading in Termux without Custom Driver Bridges**:
   * Out-of-the-box Termux does not have direct access to Qualcomm Adreno or ARM Mali OpenCL/Vulkan proprietary libraries without custom wrapper packages (`freedreno` or termux-x11 opencl).
   * Inference will initially run via **CPU NEON SIMD**, not the GPU, unless custom OpenCL libraries are configured.
