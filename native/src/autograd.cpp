/**
 * NEXA Native Training Engine - Autograd and Backward Operations Implementation
 */

#include "../include/nexa/autograd.hpp"
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace nexa {

GradTensor::GradTensor(const std::string& tensorName, const Shape& shape, bool reqGrad)
    : name(tensorName), value(shape, DataType::FLOAT32), requiresGrad(reqGrad) {
    if (requiresGrad) {
        grad = Tensor(shape, DataType::FLOAT32);
        grad.fillZero();
    }
}

void GradTensor::zeroGrad() {
    if (requiresGrad) {
        grad.fillZero();
    }
}

void GradTensor::fillRandom(float stddev, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);

    float* data = value.dataFloat();
    int64_t n = value.numel();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

float GradTensor::l2NormGrad() const {
    if (!requiresGrad) return 0.0f;
    const float* g = grad.dataFloat();
    int64_t n = grad.numel();
    double sumSq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sumSq += static_cast<double>(g[i]) * static_cast<double>(g[i]);
    }
    return static_cast<float>(std::sqrt(sumSq));
}

float GradTensor::l2NormValue() const {
    const float* v = value.dataFloat();
    int64_t n = value.numel();
    double sumSq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sumSq += static_cast<double>(v[i]) * static_cast<double>(v[i]);
    }
    return static_cast<float>(std::sqrt(sumSq));
}

void GradTensor::clipGrad(float scale) {
    if (!requiresGrad) return;
    float* g = grad.dataFloat();
    int64_t n = grad.numel();
    for (int64_t i = 0; i < n; ++i) {
        g[i] *= scale;
    }
}

// ----------------------------------------------------------------------------
// Autograd Operations Forward & Backward
// ----------------------------------------------------------------------------

void AutogradOps::matmulForward(const Tensor& A, const Tensor& B, Tensor& Y) {
    Tensor::matmul(A, B, Y);
}

void AutogradOps::matmulBackward(
    const Tensor& A,
    const Tensor& B,
    const Tensor& gradY,
    Tensor* gradA,
    Tensor* gradB,
    bool accumulateA,
    bool accumulateB
) {
    // A: [M, K], B: [K, N], gradY: [M, N]
    const int64_t M = A.shape()[0];
    const int64_t K = A.shape()[1];
    const int64_t N = B.shape()[1];

    const float* aData = A.dataFloat();
    const float* bData = B.dataFloat();
    const float* gyData = gradY.dataFloat();

    // 1. dA = gradY * B^T -> [M, K]
    // dA[i, k] = sum_j (gradY[i, j] * B[k, j])
    if (gradA) {
        float* gaData = gradA->dataFloat();
        if (!accumulateA) {
            gradA->fillZero();
        }

        constexpr int64_t BLOCK = 32;
        for (int64_t ii = 0; ii < M; ii += BLOCK) {
            int64_t iMax = std::min(M, ii + BLOCK);
            for (int64_t kk = 0; kk < K; kk += BLOCK) {
                int64_t kMax = std::min(K, kk + BLOCK);
                for (int64_t jj = 0; jj < N; jj += BLOCK) {
                    int64_t jMax = std::min(N, jj + BLOCK);

                    for (int64_t i = ii; i < iMax; ++i) {
                        const float* gyRow = gyData + (i * N);
                        float* gaRow = gaData + (i * K);

                        for (int64_t k = kk; k < kMax; ++k) {
                            const float* bRow = bData + (k * N);
                            float sum = 0.0f;
                            for (int64_t j = jj; j < jMax; ++j) {
                                sum += gyRow[j] * bRow[j];
                            }
                            gaRow[k] += sum;
                        }
                    }
                }
            }
        }
    }

    // 2. dB = A^T * gradY -> [K, N]
    // dB[k, j] = sum_i (A[i, k] * gradY[i, j])
    if (gradB) {
        float* gbData = gradB->dataFloat();
        if (!accumulateB) {
            gradB->fillZero();
        }

        constexpr int64_t BLOCK = 32;
        for (int64_t kk = 0; kk < K; kk += BLOCK) {
            int64_t kMax = std::min(K, kk + BLOCK);
            for (int64_t ii = 0; ii < M; ii += BLOCK) {
                int64_t iMax = std::min(M, ii + BLOCK);
                for (int64_t jj = 0; jj < N; jj += BLOCK) {
                    int64_t jMax = std::min(N, jj + BLOCK);

                    for (int64_t i = ii; i < iMax; ++i) {
                        const float* aRow = aData + (i * K);
                        const float* gyRow = gyData + (i * N);

                        for (int64_t k = kk; k < kMax; ++k) {
                            float aVal = aRow[k];
                            float* gbRow = gbData + (k * N);
                            for (int64_t j = jj; j < jMax; ++j) {
                                gbRow[j] += aVal * gyRow[j];
                            }
                        }
                    }
                }
            }
        }
    }
}

void AutogradOps::rmsnormForward(
    const Tensor& X,
    const Tensor& W,
    Tensor& Y,
    Tensor& invRmsOut,
    float eps
) {
    const int64_t dim = W.shape()[0];
    const int64_t nTokens = X.numel() / dim;

    const float* xData = X.dataFloat();
    const float* wData = W.dataFloat();
    float* yData = Y.dataFloat();
    float* invRmsData = invRmsOut.dataFloat();

    for (int64_t t = 0; t < nTokens; ++t) {
        const float* xRow = xData + (t * dim);
        float* yRow = yData + (t * dim);

        double sumSq = 0.0;
        for (int64_t i = 0; i < dim; ++i) {
            float val = xRow[i];
            sumSq += static_cast<double>(val) * static_cast<double>(val);
        }

        float meanSq = static_cast<float>(sumSq / dim);
        float invRms = 1.0f / std::sqrt(meanSq + eps);
        invRmsData[t] = invRms;

        for (int64_t i = 0; i < dim; ++i) {
            yRow[i] = xRow[i] * invRms * wData[i];
        }
    }
}

void AutogradOps::rmsnormBackward(
    const Tensor& X,
    const Tensor& W,
    const Tensor& invRms,
    const Tensor& gradY,
    Tensor* gradX,
    Tensor* gradW,
    bool accumulate
) {
    const int64_t dim = W.shape()[0];
    const int64_t nTokens = X.numel() / dim;

    const float* xData = X.dataFloat();
    const float* wData = W.dataFloat();
    const float* invRmsData = invRms.dataFloat();
    const float* gyData = gradY.dataFloat();

    if (gradW && !accumulate) {
        gradW->fillZero();
    }
    if (gradX && !accumulate) {
        gradX->fillZero();
    }

    float* gwData = gradW ? gradW->dataFloat() : nullptr;
    float* gxData = gradX ? gradX->dataFloat() : nullptr;

    for (int64_t t = 0; t < nTokens; ++t) {
        const float* xRow = xData + (t * dim);
        const float* gyRow = gyData + (t * dim);
        float invRmsVal = invRmsData[t];

        // 1. gradW += gy * (X * invRms)
        if (gwData) {
            for (int64_t i = 0; i < dim; ++i) {
                gwData[i] += gyRow[i] * (xRow[i] * invRmsVal);
            }
        }

        // 2. gradX:
        // sum(gy * w * x)
        if (gxData) {
            float* gxRow = gxData + (t * dim);
            double sumGyWx = 0.0;
            for (int64_t i = 0; i < dim; ++i) {
                sumGyWx += static_cast<double>(gyRow[i]) * static_cast<double>(wData[i]) * static_cast<double>(xRow[i]);
            }

            float invRms3 = invRmsVal * invRmsVal * invRmsVal;
            float factor = static_cast<float>(sumGyWx / dim) * invRms3;

            for (int64_t i = 0; i < dim; ++i) {
                float dx = (gyRow[i] * wData[i] * invRmsVal) - (xRow[i] * factor);
                gxRow[i] += dx;
            }
        }
    }
}

void AutogradOps::siluForward(const Tensor& X, Tensor& Y) {
    const float* xData = X.dataFloat();
    float* yData = Y.dataFloat();
    int64_t n = X.numel();

    for (int64_t i = 0; i < n; ++i) {
        float x = xData[i];
        float sig;
        if (x >= 0.0f) {
            sig = 1.0f / (1.0f + std::exp(-x));
        } else {
            float e = std::exp(x);
            sig = e / (1.0f + e);
        }
        yData[i] = x * sig;
    }
}

void AutogradOps::siluBackward(const Tensor& X, const Tensor& gradY, Tensor& gradX, bool accumulate) {
    const float* xData = X.dataFloat();
    const float* gyData = gradY.dataFloat();
    float* gxData = gradX.dataFloat();
    int64_t n = X.numel();

    for (int64_t i = 0; i < n; ++i) {
        float x = xData[i];
        float sig;
        if (x >= 0.0f) {
            sig = 1.0f / (1.0f + std::exp(-x));
        } else {
            float e = std::exp(x);
            sig = e / (1.0f + e);
        }
        float dSilu = sig * (1.0f + x * (1.0f - sig));
        float dx = gyData[i] * dSilu;
        if (accumulate) {
            gxData[i] += dx;
        } else {
            gxData[i] = dx;
        }
    }
}

void AutogradOps::elemwiseMulBackward(
    const Tensor& A,
    const Tensor& B,
    const Tensor& gradY,
    Tensor* gradA,
    Tensor* gradB,
    bool accumulate
) {
    const float* a = A.dataFloat();
    const float* b = B.dataFloat();
    const float* gy = gradY.dataFloat();
    int64_t n = A.numel();

    if (gradA) {
        float* ga = gradA->dataFloat();
        for (int64_t i = 0; i < n; ++i) {
            float val = gy[i] * b[i];
            if (accumulate) ga[i] += val; else ga[i] = val;
        }
    }

    if (gradB) {
        float* gb = gradB->dataFloat();
        for (int64_t i = 0; i < n; ++i) {
            float val = gy[i] * a[i];
            if (accumulate) gb[i] += val; else gb[i] = val;
        }
    }
}

void AutogradOps::embeddingForward(
    const std::vector<int32_t>& tokens,
    const Tensor& embedTable,
    Tensor& output
) {
    const int64_t vocabSize = embedTable.shape()[0];
    const int64_t dim = embedTable.shape()[1];
    const float* table = embedTable.dataFloat();
    float* out = output.dataFloat();

    for (size_t t = 0; t < tokens.size(); ++t) {
        int32_t tok = tokens[t];
        if (tok < 0 || tok >= vocabSize) tok = 0;
        const float* src = table + (static_cast<int64_t>(tok) * dim);
        float* dst = out + (static_cast<int64_t>(t) * dim);
        std::memcpy(dst, src, dim * sizeof(float));
    }
}

void AutogradOps::embeddingBackward(
    const std::vector<int32_t>& tokens,
    const Tensor& gradY,
    Tensor& gradEmbedTable
) {
    const int64_t vocabSize = gradEmbedTable.shape()[0];
    const int64_t dim = gradEmbedTable.shape()[1];
    const float* gy = gradY.dataFloat();
    float* gTable = gradEmbedTable.dataFloat();

    for (size_t t = 0; t < tokens.size(); ++t) {
        int32_t tok = tokens[t];
        if (tok < 0 || tok >= vocabSize) continue;
        const float* src = gy + (static_cast<int64_t>(t) * dim);
        float* dst = gTable + (static_cast<int64_t>(tok) * dim);
        for (int64_t d = 0; d < dim; ++d) {
            dst[d] += src[d];
        }
    }
}

float AutogradOps::crossEntropyForwardAndBackward(
    const Tensor& logits,
    const std::vector<int32_t>& targets,
    Tensor& gradLogits,
    Tensor& probsOut,
    int32_t ignoreIndex
) {
    // logits: [N, V]
    const int64_t N = static_cast<int64_t>(targets.size());
    const int64_t V = logits.shape()[1];

    const float* lData = logits.dataFloat();
    float* glData = gradLogits.dataFloat();
    float* pData = probsOut.dataFloat();

    gradLogits.fillZero();

    int64_t validCount = 0;
    for (int64_t t = 0; t < N; ++t) {
        if (targets[t] != ignoreIndex && targets[t] >= 0 && targets[t] < V) {
            validCount++;
        }
    }

    double totalLoss = 0.0;
    const float invValid = validCount > 0 ? (1.0f / static_cast<float>(validCount)) : 0.0f;

    for (int64_t t = 0; t < N; ++t) {
        const float* lRow = lData + (t * V);
        float* pRow = pData + (t * V);
        float* glRow = glData + (t * V);
        int32_t target = targets[t];

        // 1. Max for stability
        float maxVal = lRow[0];
        for (int64_t v = 1; v < V; ++v) {
            if (lRow[v] > maxVal) maxVal = lRow[v];
        }

        // 2. Exp and sum
        double sumExp = 0.0;
        for (int64_t v = 0; v < V; ++v) {
            float expVal = std::exp(lRow[v] - maxVal);
            pRow[v] = expVal;
            sumExp += expVal;
        }

        // 3. Normalize to get probabilities
        float invSum = static_cast<float>(1.0 / sumExp);
        for (int64_t v = 0; v < V; ++v) {
            pRow[v] *= invSum;
        }

        if (target == ignoreIndex || target < 0 || target >= V) {
            // Masked position: zero loss and zero gradient
            continue;
        }

        // 4. Cross-entropy loss for target: -log(p_target)
        float pTarget = pRow[target];
        if (pTarget < 1e-12f) pTarget = 1e-12f;
        totalLoss += -std::log(static_cast<double>(pTarget));

        // 5. Gradient: (p[v] - delta(v, target)) / validCount
        for (int64_t v = 0; v < V; ++v) {
            float indicator = (v == target) ? 1.0f : 0.0f;
            glRow[v] = (pRow[v] - indicator) * invValid;
        }
    }

    return validCount > 0 ? static_cast<float>(totalLoss * invValid) : 0.0f;
}

float AutogradOps::computeGlobalGradNorm(const std::vector<GradTensor*>& params) {
    double sumSq = 0.0;
    for (const auto* p : params) {
        if (!p->requiresGrad) continue;
        const float* g = p->grad.dataFloat();
        int64_t n = p->grad.numel();
        for (int64_t i = 0; i < n; ++i) {
            sumSq += static_cast<double>(g[i]) * static_cast<double>(g[i]);
        }
    }
    return static_cast<float>(std::sqrt(sumSq));
}

void AutogradOps::clipGradients(const std::vector<GradTensor*>& params, float maxNorm) {
    float norm = computeGlobalGradNorm(params);
    if (norm > maxNorm && norm > 1e-6f) {
        float scale = maxNorm / norm;
        for (auto* p : params) {
            p->clipGrad(scale);
        }
    }
}

void AutogradOps::scaleGradients(const std::vector<GradTensor*>& params, float factor) {
    for (auto* p : params) {
        if (!p->requiresGrad) continue;
        float* g = p->grad.dataFloat();
        const int64_t n = p->grad.numel();
        for (int64_t i = 0; i < n; ++i) {
            g[i] *= factor;
        }
    }
}

} // namespace nexa
