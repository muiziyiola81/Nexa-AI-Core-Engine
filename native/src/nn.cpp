/**
 * NEXA Native Runtime - Neural Network Operations Implementation
 */

#include "../include/nexa/nn.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace nexa {

void RMSNorm::forward(const Tensor& input, const Tensor& weight, Tensor& output, float eps) {
    if (input.shape() != output.shape()) {
        throw std::invalid_argument("Shape mismatch between input and output in RMSNorm");
    }
    int64_t dim = input.shape()[input.rank() - 1];
    if (weight.numel() != dim) {
        throw std::invalid_argument("Weight dimension does not match last dimension of input in RMSNorm");
    }

    const float* in = input.dataFloat();
    const float* w = weight.dataFloat();
    float* out = output.dataFloat();
    int64_t totalTokens = input.numel() / dim;

    for (int64_t t = 0; t < totalTokens; ++t) {
        const float* inVec = in + t * dim;
        float* outVec = out + t * dim;

        // Calculate Root Mean Square: sqrt(1/d * sum(x_i^2) + eps)
        float sumSquares = 0.0f;
        for (int64_t i = 0; i < dim; ++i) {
            sumSquares += inVec[i] * inVec[i];
        }
        float invRms = 1.0f / std::sqrt((sumSquares / static_cast<float>(dim)) + eps);

        for (int64_t i = 0; i < dim; ++i) {
            outVec[i] = inVec[i] * invRms * w[i];
        }
    }
}

void LayerNorm::forward(const Tensor& input, const Tensor& weight, const Tensor* bias, Tensor& output, float eps) {
    int64_t dim = input.shape()[input.rank() - 1];
    const float* in = input.dataFloat();
    const float* w = weight.dataFloat();
    const float* b = bias ? bias->dataFloat() : nullptr;
    float* out = output.dataFloat();
    int64_t totalTokens = input.numel() / dim;

    for (int64_t t = 0; t < totalTokens; ++t) {
        const float* inVec = in + t * dim;
        float* outVec = out + t * dim;

        float mean = 0.0f;
        for (int64_t i = 0; i < dim; ++i) mean += inVec[i];
        mean /= static_cast<float>(dim);

        float variance = 0.0f;
        for (int64_t i = 0; i < dim; ++i) {
            float diff = inVec[i] - mean;
            variance += diff * diff;
        }
        float invStd = 1.0f / std::sqrt((variance / static_cast<float>(dim)) + eps);

        for (int64_t i = 0; i < dim; ++i) {
            float val = (inVec[i] - mean) * invStd * w[i];
            if (b) val += b[i];
            outVec[i] = val;
        }
    }
}

void RoPE::forward(Tensor& qOrK, int64_t pos, int64_t headDim, float theta) {
    float* data = qOrK.dataFloat();
    int64_t totalElements = qOrK.numel();
    int64_t nHeads = totalElements / headDim;

    for (int64_t h = 0; h < nHeads; ++h) {
        float* headVec = data + h * headDim;
        for (int64_t i = 0; i < headDim / 2; ++i) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(headDim));
            float val = static_cast<float>(pos) * freq;
            float cosVal = std::cos(val);
            float sinVal = std::sin(val);

            float v0 = headVec[2 * i];
            float v1 = headVec[2 * i + 1];

            // Rotary transformation
            headVec[2 * i]     = v0 * cosVal - v1 * sinVal;
            headVec[2 * i + 1] = v0 * sinVal + v1 * cosVal;
        }
    }
}

void Embedding::forward(const std::vector<int32_t>& tokens, const Tensor& embedTable, Tensor& output) {
    int64_t dim = embedTable.shape()[1];
    const float* table = embedTable.dataFloat();
    float* out = output.dataFloat();

    for (size_t t = 0; t < tokens.size(); ++t) {
        int32_t tok = tokens[t];
        if (tok < 0 || tok >= embedTable.shape()[0]) {
            tok = 0; // fallback to unk
        }
        std::memcpy(out + t * dim, table + tok * dim, static_cast<size_t>(dim) * sizeof(float));
    }
}

void FeedForwardSwiGLU::forward(
    const Tensor& x,
    const Tensor& wGate,
    const Tensor& wUp,
    const Tensor& wDown,
    Tensor& output
) {
    int64_t seqLen = x.shape()[0];
    int64_t dim = x.shape()[1];
    int64_t interDim = wGate.shape()[1];

    Tensor gate(Shape({seqLen, interDim}));
    Tensor up(Shape({seqLen, interDim}));

    Tensor::matmul(x, wGate, gate);
    Tensor::matmul(x, wUp, up);

    float* gData = gate.dataFloat();
    const float* uData = up.dataFloat();
    int64_t total = seqLen * interDim;

    // Apply SiLU activation to gate: SiLU(x) = x / (1 + exp(-x))
    // and elementwise multiply with up: (SiLU(gate) * up)
    for (int64_t i = 0; i < total; ++i) {
        float val = gData[i];
        float silu = val / (1.0f + std::exp(-val));
        gData[i] = silu * uData[i];
    }

    // Down projection back to hidden dimension
    Tensor::matmul(gate, wDown, output);
}

void Attention::forward(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& wOut,
    Tensor& output,
    bool isCausal
) {
    int64_t seqLen = q.shape()[0];
    int64_t dim = q.shape()[1];
    int64_t headDim = 64;
    int64_t nHeads = dim / headDim;

    Tensor context(Shape({seqLen, dim}));
    context.fillZero();

    float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

    // Scaled Dot-Product Attention for each head
    for (int64_t h = 0; h < nHeads; ++h) {
        for (int64_t i = 0; i < seqLen; ++i) {
            // Attention weights for token i across all tokens j <= i
            std::vector<float> scores(seqLen, -1e9f);
            float maxScore = -1e9f;

            int64_t jLimit = isCausal ? (i + 1) : seqLen;
            for (int64_t j = 0; j < jLimit; ++j) {
                float dot = 0.0f;
                const float* qPtr = q.dataFloat() + i * dim + h * headDim;
                const float* kPtr = k.dataFloat() + j * dim + h * headDim;
                for (int64_t d = 0; d < headDim; ++d) {
                    dot += qPtr[d] * kPtr[d];
                }
                float score = dot * scale;
                scores[j] = score;
                if (score > maxScore) maxScore = score;
            }

            // Softmax
            float sumExp = 0.0f;
            for (int64_t j = 0; j < jLimit; ++j) {
                scores[j] = std::exp(scores[j] - maxScore);
                sumExp += scores[j];
            }
            float invSum = sumExp > 0.0f ? (1.0f / sumExp) : 0.0f;
            for (int64_t j = 0; j < jLimit; ++j) {
                scores[j] *= invSum;
            }

            // Weighted sum of values V
            float* ctxPtr = context.dataFloat() + i * dim + h * headDim;
            for (int64_t j = 0; j < jLimit; ++j) {
                const float* vPtr = v.dataFloat() + j * dim + h * headDim;
                float weight = scores[j];
                for (int64_t d = 0; d < headDim; ++d) {
                    ctxPtr[d] += weight * vPtr[d];
                }
            }
        }
    }

    // Output projection
    Tensor::matmul(context, wOut, output);
}

void TransformerBlock::forward(
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
    int64_t startPos
) {
    int64_t seqLen = input.shape()[0];
    int64_t dim = input.shape()[1];

    // 1. Attention Pre-Normalization
    Tensor xNorm(Shape({seqLen, dim}));
    RMSNorm::forward(input, attnNormWeight, xNorm, hp.normEps);

    // 2. Linear Projections for Q, K, V
    Tensor q(Shape({seqLen, dim}));
    Tensor k(Shape({seqLen, dim}));
    Tensor v(Shape({seqLen, dim}));
    Tensor::matmul(xNorm, wQ, q);
    Tensor::matmul(xNorm, wK, k);
    Tensor::matmul(xNorm, wV, v);

    // 3. Apply RoPE to Query and Key
    int64_t headDim = dim / hp.nHeads;
    for (int64_t i = 0; i < seqLen; ++i) {
        Tensor qSlice = Tensor::createView(q, Shape({1, dim}), i * dim);
        Tensor kSlice = Tensor::createView(k, Shape({1, dim}), i * dim);
        RoPE::forward(qSlice, startPos + i, headDim, hp.ropeTheta);
        RoPE::forward(kSlice, startPos + i, headDim, hp.ropeTheta);
    }

    // 4. Scaled Dot-Product Attention with Output Projection
    Tensor attnOut(Shape({seqLen, dim}));
    Attention::forward(q, k, v, wOut, attnOut, true);

    // 5. First Residual Connection: x1 = input + attnOut
    Tensor x1(Shape({seqLen, dim}));
    Tensor::vecAdd(input, attnOut, x1);

    // 6. FFN Pre-Normalization
    Tensor x1Norm(Shape({seqLen, dim}));
    RMSNorm::forward(x1, ffnNormWeight, x1Norm, hp.normEps);

    // 7. SwiGLU Feed-Forward Network
    Tensor ffnOut(Shape({seqLen, dim}));
    FeedForwardSwiGLU::forward(x1Norm, wGate, wUp, wDown, ffnOut);

    // 8. Second Residual Connection: output = x1 + ffnOut
    Tensor::vecAdd(x1, ffnOut, output);
}

} // namespace nexa
