/**
 * NEXA Native Training Engine - Autograd and Backward Operations Header
 * Defines gradient tensor storage, forward/backward differentiation primitives,
 * matrix multiplication gradients, RMSNorm gradients, cross-entropy loss, and gradient clipping.
 */

#ifndef NEXA_AUTOGRAD_HPP
#define NEXA_AUTOGRAD_HPP

#include "tensor.hpp"
#include <vector>
#include <string>
#include <memory>
#include <cmath>

namespace nexa {

class GradTensor {
public:
    std::string name;
    Tensor value;
    Tensor grad;
    bool requiresGrad = true;

    GradTensor() = default;
    GradTensor(const std::string& tensorName, const Shape& shape, bool reqGrad = true);

    void zeroGrad();
    void fillRandom(float stddev, uint64_t seed = 42);
    float l2NormGrad() const;
    float l2NormValue() const;
    void clipGrad(float scale);
};

class AutogradOps {
public:
    /**
     * Matrix multiplication forward & backward:
     * Forward:  Y = A * B   (A: [M, K], B: [K, N], Y: [M, N])
     * Backward: dA += dY * B^T
     *           dB += A^T * dY
     */
    static void matmulForward(const Tensor& A, const Tensor& B, Tensor& Y);
    
    static void matmulBackward(
        const Tensor& A,
        const Tensor& B,
        const Tensor& gradY,
        Tensor* gradA,
        Tensor* gradB,
        bool accumulateA = true,
        bool accumulateB = true
    );

    /**
     * RMSNorm forward & backward:
     * Y = (X / RMS(X)) * W
     * RMS(X) = sqrt(mean(X^2) + eps)
     */
    static void rmsnormForward(
        const Tensor& X,
        const Tensor& W,
        Tensor& Y,
        Tensor& invRmsOut,
        float eps = 1e-5f
    );

    static void rmsnormBackward(
        const Tensor& X,
        const Tensor& W,
        const Tensor& invRms,
        const Tensor& gradY,
        Tensor* gradX,
        Tensor* gradW,
        bool accumulate = true
    );

    /**
     * SiLU (Swish) non-linearity forward & backward:
     * Y = X * sigmoid(X)
     * dY/dX = sigmoid(X) * (1 + X * (1 - sigmoid(X)))
     */
    static void siluForward(const Tensor& X, Tensor& Y);

    static void siluBackward(
        const Tensor& X,
        const Tensor& gradY,
        Tensor& gradX,
        bool accumulate = false
    );

    /**
     * Element-wise multiplication backward:
     * Y = A * B
     * dA = gradY * B, dB = gradY * A
     */
    static void elemwiseMulBackward(
        const Tensor& A,
        const Tensor& B,
        const Tensor& gradY,
        Tensor* gradA,
        Tensor* gradB,
        bool accumulate = true
    );

    /**
     * Embedding lookup forward & backward:
     * Forward:  Y[t] = EmbedTable[tokens[t]]
     * Backward: gradEmbedTable[tokens[t]] += gradY[t]
     */
    static void embeddingForward(
        const std::vector<int32_t>& tokens,
        const Tensor& embedTable,
        Tensor& output
    );

    static void embeddingBackward(
        const std::vector<int32_t>& tokens,
        const Tensor& gradY,
        Tensor& gradEmbedTable
    );

    /**
     * Softmax and Cross-Entropy Loss with fused backward pass:
     * Forward:  Loss = -log(softmax(logits)[target])
     * Backward: dLogits[i] = (softmax(logits)[i] - (i == target ? 1 : 0)) / validCount
     * Positions with target == ignoreIndex are masked out (loss = 0, grad = 0).
     */
    static float crossEntropyForwardAndBackward(
        const Tensor& logits,              // [N, V]
        const std::vector<int32_t>& targets,// [N]
        Tensor& gradLogits,                // [N, V]
        Tensor& probsOut,                  // [N, V]
        int32_t ignoreIndex = -1
    );

    static double crossEntropyForwardAndBackwardSum(
        const Tensor& logits,              // [N, V]
        const std::vector<int32_t>& targets,// [N]
        Tensor& gradLogits,                // [N, V] (unnormalized gradients)
        Tensor& probsOut,                  // [N, V]
        int32_t ignoreIndex = -1,
        uint64_t* outValidTokens = nullptr
    );

    /**
     * Global gradient norm computation, scaling, and clipping:
     */
    static float computeGlobalGradNorm(const std::vector<GradTensor*>& params);
    static void clipGradients(const std::vector<GradTensor*>& params, float maxNorm);
    static void scaleGradients(const std::vector<GradTensor*>& params, float factor);
};

} // namespace nexa

#endif // NEXA_AUTOGRAD_HPP
