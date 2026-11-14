# NEXA AI — Phase 3.2: Native Training Engine Foundation

## 1. Architectural Overview

NEXA AI's Phase 3.2 establishes an independent, self-contained C++20 training foundation capable of training neural language models from randomly initialized weights on CPU without any third-party framework dependencies (no PyTorch, TensorFlow, libtorch, llama.cpp, or Ollama).

```
                      +---------------------------------------+
                      |       NEXA Native Training Engine     |
                      |        (C++20 Native Core Engine)     |
                      +---------------------------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                 |                                 |
+---------------+                 +---------------+                 +---------------+
|  Autograd Ops |                 | AdamW Engine  |                 |  NEXA-1 Nano  |
|  & Gradients  |                 |  (Optimizer)  |                 | (Transformer) |
+---------------+                 +---------------+                 +---------------+
| • Matmul Bwd  |                 | • 1st Moment  |                 | • V=256, D=64 |
| • RMSNorm Bwd |                 | • 2nd Moment  |                 | • 2 Layers    |
| • SwiGLU Bwd  |                 | • Decoupled   |                 | • Self-Attn   |
| • Softmax/CE  |                 |   WeightDecay |                 | • SwiGLU FFN  |
| • Grad Clip   |                 | • Save/Load   |                 | • Checkpoint  |
+---------------+                 +---------------+                 +---------------+
```

---

## 2. Mathematical Components

### 2.1 Autograd Operations (`native/src/autograd.cpp`)
1. **Matrix Multiplication Backward ($Y = A \cdot B$):**
   $$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial Y} B^T, \quad \frac{\partial \mathcal{L}}{\partial B} = A^T \frac{\partial \mathcal{L}}{\partial Y}$$
   Optimized with cache-friendly loop ordering and zero-allocation accumulation.

2. **RMSNorm Backward ($y = \frac{x}{\text{RMS}(x)} \odot \gamma$):**
   $$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum \frac{\partial \mathcal{L}}{\partial y} \odot \frac{x}{\text{RMS}(x)}$$
   $$\frac{\partial \mathcal{L}}{\partial x} = \frac{\gamma}{\text{RMS}(x)} \odot \left[ \frac{\partial \mathcal{L}}{\partial y} - \frac{x}{d \cdot \text{RMS}(x)^2} \sum \left( \frac{\partial \mathcal{L}}{\partial y} \odot \gamma \odot x \right) \right]$$
   Ensures that gradient components along the radial direction satisfy the RMS normalization constraint.

3. **SwiGLU & SiLU Backward ($y = \text{silu}(x_1) \odot x_2$):**
   $$\text{silu}'(z) = \sigma(z) + z \sigma(z)(1 - \sigma(z))$$
   $$\frac{\partial \mathcal{L}}{\partial x_1} = \frac{\partial \mathcal{L}}{\partial y} \odot x_2 \odot \text{silu}'(x_1), \quad \frac{\partial \mathcal{L}}{\partial x_2} = \frac{\partial \mathcal{L}}{\partial y} \odot \text{silu}(x_1)$$

4. **Softmax & Cross-Entropy Loss Backward:**
   $$\mathcal{L} = -\log(p_{\text{target}}), \quad \frac{\partial \mathcal{L}}{\partial z_i} = p_i - \mathbb{I}(i = \text{target})$$
   Gradient conservation strictly verified: $\sum_i \frac{\partial \mathcal{L}}{\partial z_i} = 0$.

5. **L2 Gradient Norm Clipping:**
   $$g \leftarrow g \cdot \frac{\text{max\_norm}}{\max(\|g\|_2, \text{max\_norm})}$$
   Prevents gradient explosion during initial training steps.

### 2.2 AdamW Optimizer (`native/src/optimizer.cpp`)
- **First moment update:** $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
- **Second moment update:** $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
- **Bias correction:** $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
- **Parameter step with decoupled weight decay:**
  $$\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
- State persistence: Full $m$ and $v$ serialization support.

---

## 3. NEXA-1 Nano Model Architecture

A compact, decoder-only causal Transformer implemented from scratch:
- **Vocabulary ($V$):** 256 (Byte-level UTF-8)
- **Hidden Dimension ($D$):** 64
- **Attention Heads ($H$):** 2 (head dimension 32)
- **Layers ($L$):** 2
- **Intermediate Dimension ($FFN$):** 172 (SwiGLU)
- **Max Context Length ($T$):** 64
- **Parameter Count:** ~110,000 parameters (CPU-optimized for microsecond execution)
- **Weight Initialization:** Standard Gaussian $\mathcal{N}(0, 0.02)$ with Xavier scaled projections

---

## 4. Verification Results

### 4.1 C++ Unit Tests (`./bin/nexa-runtime-cli --test-training`)
```
[TEST 1] Testing Matmul Autograd Backward... PASS
[TEST 2] Testing RMSNorm Autograd Backward... PASS
[TEST 3] Testing Cross-Entropy Loss & Softmax Backward... PASS
[TEST 4] Testing AdamW Optimizer Step... PASS
ALL AUTOGRAD & OPTIMIZER TESTS PASSED (4/4).
```

### 4.2 Synthetic Training Run (`./bin/nexa-runtime-cli --train-nano 40 0.003`)
- **Total Training Steps:** 40
- **Duration:** 0.12 seconds on CPU
- **Initial Loss:** 5.4030 (theoretical entropy for uniform $V=256$: $-\ln(1/256) \approx 5.545$)
- **Final Loss:** 2.7114
- **Loss Reduction:** **-49.8%**
- **Initial Weight Norm:** 41.7483
- **Final Weight Norm:** 42.9463
- **Weight Delta Norm ($\|\Delta W\|$):** 11.0264 (**Weights genuinely updated**)
- **Checkpoint Persistence:** Saved to `data/checkpoints/nexa_nano_step_40.nexa`
- **Reload Invariance:** Bit-identical logits on reloaded model.

### 4.3 Node.js Full Test Suite (`npm test`)
- Total tests executed: 37
- Passing tests: 37/37 (100%)
- Failing tests: 0
