# NEXA Model Format Specification (`.nexa`)
Version: 1.0.0
Author: NEXA AI Core Engine Team

## 1. Overview
The `.nexa` format is an independent, single-file binary model representation engineered specifically for the NEXA Native CPU Inference Engine. It stores model architecture metadata, hyper-parameters, tensor symbol tables, and raw aligned tensor weights without dependency on GGUF, GGML, SafeTensors, or ONNX formats.

Key design principles:
- **Zero Third-Party Tooling**: Native serialization and deserialization in both C++ and TypeScript.
- **Hardware Memory Alignment**: Tensor offsets are aligned to 64-byte boundaries (`alignas(64)`) to support AVX-512, AVX2, and ARM NEON SIMD vector loads.
- **Header Safety**: Magic bytes `0x4E455841` ("NEXA" in ASCII) prevent corrupted or invalid model mounting.
- **Pre-execution Memory Budgeting**: Stores complete dimension tables allowing the runtime to compute exact host RAM requirements before allocating memory.

---

## 2. Binary Layout

A `.nexa` file consists of three contiguous sections:

```
+-------------------------------------------------------------+
| Header (128 bytes fixed size)                               |
+-------------------------------------------------------------+
| Tensor Descriptor Table (128 bytes * tensor_count)          |
+-------------------------------------------------------------+
| Tensor Data Payloads (Aligned to 64-byte boundaries)        |
| Tensor 0: [W_embed]                                         |
| Tensor 1: [W_q]                                             |
| ...                                                         |
| Tensor N-1: [W_out]                                         |
+-------------------------------------------------------------+
```

---

## 3. Section Specifications

### 3.1 Header (128 Bytes Fixed)

All fields are Little-Endian.

| Offset (Bytes) | Field Name             | Type      | Description                                      |
|----------------|------------------------|-----------|--------------------------------------------------|
| `0x00 - 0x03`  | `magic`                | `uint32`  | Fixed magic number `0x4E455841` (`NEXA`)         |
| `0x04 - 0x07`  | `version`              | `uint32`  | Format version (currently `1`)                   |
| `0x08 - 0x27`  | `architecture`         | `char[32]`| Null-terminated ASCII string (e.g. `llama`, `gpt`)|
| `0x28 - 0x2B`  | `vocab_size`           | `uint32`  | Vocabulary size (e.g. 32000)                     |
| `0x2C - 0x2F`  | `hidden_dim`           | `uint32`  | Hidden dimension size $d_{model}$ (e.g. 4096)    |
| `0x30 - 0x33`  | `num_layers`           | `uint32`  | Number of transformer layers $L$ (e.g. 32)       |
| `0x34 - 0x37`  | `num_heads`            | `uint32`  | Number of query attention heads $H$ (e.g. 32)    |
| `0x38 - 0x3B`  | `intermediate_dim`     | `uint32`  | FFN/SwiGLU intermediate dimension (e.g. 11008)  |
| `0x3C - 0x3F`  | `context_window`       | `uint32`  | Maximum sequence length $T_{max}$ (e.g. 4096)    |
| `0x40 - 0x43`  | `quantization`         | `uint32`  | Quantization enum code (0: fp32, 1: fp16, 2: q4_0, 3: q8_0) |
| `0x44 - 0x47`  | `tensor_count`         | `uint32`  | Number of tensors stored in file                 |
| `0x48 - 0x4F`  | `tensor_table_offset`  | `uint64`  | Byte offset from file start to descriptor table  |
| `0x50 - 0x7F`  | `reserved`             | `byte[48]`| Reserved for future extension (zeroed)           |

---

### 3.2 Tensor Descriptor Entry (128 Bytes Each)

The descriptor table directly follows the header at `tensor_table_offset` (typically offset `128`):

| Offset (Bytes) | Field Name             | Type       | Description                                     |
|----------------|------------------------|------------|-------------------------------------------------|
| `0x00 - 0x3F`  | `name`                 | `char[64]` | UTF-8 tensor identifier (e.g. `layers.0.q_proj`)|
| `0x40 - 0x43`  | `num_dims`             | `uint32`   | Number of dimensions (1 to 4)                   |
| `0x44 - 0x53`  | `dims`                 | `uint32[4]`| Array of dimension sizes [dim0, dim1, dim2, dim3]|
| `0x54 - 0x57`  | `data_type`            | `uint32`   | Data type enum (0: FP32, 1: FP16, 2: Q4_0, 3: Q8_0)|
| `0x58 - 0x5F`  | `offset`               | `uint64`   | Absolute byte offset from start of file         |
| `0x60 - 0x67`  | `size_bytes`           | `uint64`   | Byte length of raw tensor buffer                |
| `0x68 - 0x7F`  | `reserved`             | `byte[24]` | Reserved padding for 64-byte alignment           |

---

### 3.3 Data Types and Element Sizes

| Data Type Enum | Type Code | Element Size | Storage Format                                  |
|----------------|-----------|--------------|-------------------------------------------------|
| `FP32`         | `0`       | 4 bytes      | IEEE 754 Single-Precision Float                 |
| `FP16`         | `1`       | 2 bytes      | IEEE 754 Half-Precision Float                   |
| `Q4_0`         | `2`       | 0.5625 bytes | 32-value blocks (1x FP16 scale + 16 bytes nibbles)|
| `Q8_0`         | `3`       | 1.0625 bytes | 32-value blocks (1x FP16 scale + 32 bytes int8)  |

---

## 4. Tensor Naming Conventions

Tensors in `.nexa` follow standard hierarchical naming:
- `model.embed_tokens.weight` — Token embedding table: `[vocab_size, hidden_dim]`
- `model.layers.{i}.input_layernorm.weight` — Input RMSNorm scale: `[hidden_dim]`
- `model.layers.{i}.self_attn.q_proj.weight` — Query projection: `[hidden_dim, hidden_dim]`
- `model.layers.{i}.self_attn.k_proj.weight` — Key projection: `[hidden_dim, hidden_dim]`
- `model.layers.{i}.self_attn.v_proj.weight` — Value projection: `[hidden_dim, hidden_dim]`
- `model.layers.{i}.self_attn.o_proj.weight` — Attention output projection: `[hidden_dim, hidden_dim]`
- `model.layers.{i}.post_attention_layernorm.weight` — Post-attention RMSNorm: `[hidden_dim]`
- `model.layers.{i}.mlp.gate_proj.weight` — SwiGLU Gate projection: `[intermediate_dim, hidden_dim]`
- `model.layers.{i}.mlp.up_proj.weight` — SwiGLU Up projection: `[intermediate_dim, hidden_dim]`
- `model.layers.{i}.mlp.down_proj.weight` — SwiGLU Down projection: `[hidden_dim, intermediate_dim]`
- `model.norm.weight` — Final output RMSNorm scale: `[hidden_dim]`
- `lm_head.weight` — Unembedding logits projection: `[vocab_size, hidden_dim]`

---

## 5. Verification & Memory Calculation Formula

Before allocating heap memory, the loader verifies:
1. `magic == 0x4E455841`
2. `version == 1`
3. `file_size >= header.tensor_table_offset + header.tensor_count * 128`
4. For every descriptor $i$:
   - `offset % 64 == 0` (64-byte alignment check)
   - `offset + size_bytes <= file_size`
5. Host RAM check:
   $$\text{RAM}_{\text{req}} = \text{WeightsBytes} + \left( 2 \times L \times H \times d_{\text{head}} \times T_{\text{ctx}} \times 2 \right) + 64\,\text{MB working buffer}$$
   If $\text{RAM}_{\text{req}} > \text{HostFreeRAM}$, loading fails early with an informative diagnostic error.
