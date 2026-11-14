/**
 * NEXA Native Runtime - Neural Network Operations Header
 * Defines Embedding, RMSNorm, LayerNorm, Rotary Positional Embeddings (RoPE),
 * Scaled Dot-Product Attention, SwiGLU MLP, and Transformer Block.
 */

#ifndef NEXA_NN_HPP
#define NEXA_NN_HPP

#include "tensor.hpp"
#include <vector>

namespace nexa {

struct TransformerHyperparams {
    uint32_t vocabSize = 32000;
    uint32_t dim = 2048;             // Hidden dimension
    uint32_t intermediateDim = 5632; // MLP hidden dim
    uint32_t nLayers = 16;
    uint32_t nHeads = 16;            // Query heads
    uint32_t nKvHeads = 8;           // Key/Value heads (GQA)
    uint32_t maxSeqLen = 2048;
    float normEps = 1e-5f;
    float ropeTheta = 10000.0f;
};

class RMSNorm {
public:
    static void forward(const Tensor& input, const Tensor& weight, Tensor& output, float eps = 1e-5f);
};

class LayerNorm {
public:
    static void forward(const Tensor& input, const Tensor& weight, const Tensor* bias, Tensor& output, float eps = 1e-5f);
};

class RoPE {
public:
    /**
     * In-place or output application of Rotary Positional Embeddings.
     * Computes (x[2*i] * cos - x[2*i+1] * sin) and (x[2*i] * sin + x[2*i+1] * cos).
     */
    static void forward(Tensor& qOrK, int64_t pos, int64_t headDim, float theta = 10000.0f);
};

class Embedding {
public:
    static void forward(const std::vector<int32_t>& tokens, const Tensor& embedTable, Tensor& output);
};

class FeedForwardSwiGLU {
public:
    /**
     * SwiGLU: output = (SiLU(x * W_gate) * (x * W_up)) * W_down
     */
    static void forward(
        const Tensor& x,
        const Tensor& wGate,
        const Tensor& wUp,
        const Tensor& wDown,
        Tensor& output
    );
};

class Attention {
public:
    /**
     * Multi-Head / Grouped-Query Causal Attention forward pass.
     */
    static void forward(
        const Tensor& q,          // [seq_len, n_heads, head_dim]
        const Tensor& k,          // [seq_len, n_kv_heads, head_dim]
        const Tensor& v,          // [seq_len, n_kv_heads, head_dim]
        const Tensor& wOut,       // [dim, dim]
        Tensor& output,           // [seq_len, dim]
        bool isCausal = true
    );
};

class TransformerBlock {
public:
    /**
     * Full Pre-Norm Residual Transformer Block forward pass:
     * 1. x_norm = RMSNorm(x, attn_norm_weight)
     * 2. x_attn = Attention(q, k, v)
     * 3. x = x + x_attn (Residual)
     * 4. x_ffn_norm = RMSNorm(x, ffn_norm_weight)
     * 5. x_ffn = SwiGLU(x_ffn_norm)
     * 6. x = x + x_ffn (Residual)
     */
    static void forward(
        const Tensor& input,
        const Tensor& attnNormWeight,
        const Tensor& wQ,
        const Tensor& wK,
        const Tensor& wV,
        const Tensor& wOut,
        const Tensor& ffnNormWeight,
        const Tensor& wGate,
        const Tensor& wUp,
        const Tensor& wDown,
        Tensor& output,
        const TransformerHyperparams& hp,
        int64_t startPos = 0
    );
};

} // namespace nexa

#endif // NEXA_NN_HPP
