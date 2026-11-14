#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# NEXA AI Core Engine - Termux (Android) Automated Setup Script
# Supports: ARM64 (aarch64), ARM 32-bit (armv7l), x86_64
# ==============================================================================

set -e

echo "================================================================"
echo "    NEXA AI CORE ENGINE - TERMUX (ANDROID) DEPLOYMENT SETUP     "
echo "================================================================"
echo ""

# 1. Architecture Detection
ARCH=$(uname -m)
echo "[1/6] Detecting CPU Architecture..."
echo "      Detected: $ARCH"

case "$ARCH" in
  aarch64|arm64)
    echo "      Status: Native 64-bit ARM detected. Fully supported for neural inference."
    ARCH_SUPPORT="ARM64"
    ;;
  armv7l|armv8l|armhf)
    echo "      WARNING: 32-bit ARM detected ($ARCH)."
    echo "      Android 32-bit kernels limit single-process address space to ~2-3 GB."
    echo "      7B/8B models CANNOT be loaded on 32-bit. Only sub-1B models (TinyLlama Q4) supported."
    ARCH_SUPPORT="ARM32"
    ;;
  x86_64)
    echo "      Status: 64-bit x86 detected (Emulator / Chromebook). Supported."
    ARCH_SUPPORT="X86_64"
    ;;
  i686|x86)
    echo "      WARNING: 32-bit x86 detected. Memory limited to < 4 GB."
    ARCH_SUPPORT="X86_32"
    ;;
  *)
    echo "      Unknown architecture: $ARCH. Proceeding with caution."
    ARCH_SUPPORT="UNKNOWN"
    ;;
esac

# 2. RAM & Hardware Resource Assessment
echo ""
echo "[2/6] Inspecting Available Device Memory..."
if [ -f /proc/meminfo ]; then
  TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
  TOTAL_RAM_MB=$((TOTAL_RAM_KB / 1024))
  echo "      Total System RAM: ${TOTAL_RAM_MB} MB"

  if [ "$TOTAL_RAM_MB" -lt 3000 ]; then
    echo "      WARNING: Under 3 GB RAM. Use compact models only (TinyLlama 1.1B Q4_K_M)."
  elif [ "$TOTAL_RAM_MB" -lt 6000 ]; then
    echo "      Status: 4-6 GB RAM. Recommended for Phi-3 Mini (3.8B) or TinyLlama."
  else
    echo "      Status: 8+ GB RAM. Sufficient for 7B/8B models (Llama 3 8B Q4_K_M, Mistral 7B)."
  fi
fi

# 3. Android Battery & Process Lock
echo ""
echo "[3/6] Android Runtime Environment Checks..."
if command -v termux-wake-lock >/dev/null 2>&1; then
  echo "      Acquiring Termux Wake Lock to prevent Android CPU sleep..."
  termux-wake-lock
  echo "      Wake lock active."
else
  echo "      Notice: 'termux-wake-lock' command not found. To prevent CPU sleep,"
  echo "      install Termux:API and run: termux-wake-lock"
fi

# 4. Install Dependencies
echo ""
echo "[4/6] Updating Termux repositories and installing build packages..."
pkg update -y
pkg install -y nodejs-lts git cmake clang

# 5. Node.js Dependency Verification
echo ""
echo "[5/6] Installing NEXA Project Dependencies..."
cd "$(dirname "$0")/.."
npm install --omit=dev || npm install

# Create storage directories
mkdir -p models data

# 6. llama.cpp Setup Guidance
echo ""
echo "[6/6] Checking for local llama.cpp binary in Termux..."
if command -v llama-server >/dev/null 2>&1 || [ -f "$PREFIX/bin/llama-server" ] || [ -f "./bin/llama-server" ]; then
  echo "      llama-server is already installed."
else
  echo "      llama-server not found in PATH."
  echo "      To build llama.cpp natively in Termux for $ARCH:"
  echo ""
  echo "      git clone https://github.com/ggerganov/llama.cpp.git"
  echo "      cd llama.cpp"
  echo "      cmake -B build -DGGML_OPENBLAS=OFF"
  echo "      cmake --build build --config Release -j\$(nproc)"
  echo "      cp build/bin/llama-server ../bin/ || cp build/bin/llama-cli ../bin/"
  echo "      cd .."
fi

echo ""
echo "================================================================"
echo "                   SETUP COMPLETE                               "
echo "================================================================"
echo "To run the headless NEXA engine in Termux:"
echo "  npm run start:server"
echo ""
echo "To verify deployment readiness:"
echo "  npm run verify:deployment"
echo "================================================================"
