/**
 * NEXA Native Training Engine - Autograd and Backward Operations Implementation
 */

#include "../include/nexa/autograd.hpp"
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

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

namespace {

inline float dotContig(const float* a, const float* b, int64_t len) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 8 <= len; j += 8) {
        __m256 va = _mm256_loadu_ps(a + j);
        __m256 vb = _mm256_loadu_ps(b + j);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_hadd_ps(s4, s4);
    __m128 s1 = _mm_hadd_ps(s2, s2);
    float sum = _mm_cvtss_f32(s1);
    for (; j < len; ++j) {
        sum += a[j] * b[j];
    }
    return sum;
#else
    float sum = 0.0f;
    for (int64_t j = 0; j < len; ++j) {
        sum += a[j] * b[j];
    }
    return sum;
#endif
}

inline void saxpyAccum(int64_t len, float alpha, const float* src, float* dst) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 aVec = _mm256_set1_ps(alpha);
    int64_t j = 0;
    for (; j + 8 <= len; j += 8) {
        __m256 dVec = _mm256_loadu_ps(dst + j);
        __m256 sVec = _mm256_loadu_ps(src + j);
        dVec = _mm256_fmadd_ps(aVec, sVec, dVec);
        _mm256_storeu_ps(dst + j, dVec);
    }
    for (; j < len; ++j) {
        dst[j] += alpha * src[j];
    }
#else
    for (int64_t j = 0; j < len; ++j) {
        dst[j] += alpha * src[j];
    }
#endif
}

void matmulBackwardARange(
    const float* gyData,
    const float* bData,
    float* gaData,
    int64_t iBegin,
    int64_t iEnd,
    int64_t K,
    int64_t N
) {
    for (int64_t i = iBegin; i < iEnd; ++i) {
        const float* gyRow = gyData + i * N;
        float* gaRow = gaData + i * K;
        for (int64_t k = 0; k < K; ++k) {
            gaRow[k] += dotContig(gyRow, bData + k * N, N);
        }
    }
}

void matmulBackwardBRange(
    const float* aData,
    const float* gyData,
    float* gbData,
    int64_t kBegin,
    int64_t kEnd,
    int64_t M,
    int64_t K,
    int64_t N
) {
    for (int64_t i = 0; i < M; ++i) {
        const float* aRow = aData + i * K;
        const float* gyRow = gyData + i * N;
        for (int64_t k = kBegin; k < kEnd; ++k) {
            saxpyAccum(N, aRow[k], gyRow, gbData + k * N);
        }
    }
}

} // namespace

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

    unsigned int hwThreads = std::thread::hardware_concurrency();
    int64_t maxThreads = hwThreads > 0 ? static_cast<int64_t>(std::min<unsigned int>(hwThreads, 4)) : 1;

    // 1. dA = gradY * B^T -> [M, K]
    if (gradA) {
        float* gaData = gradA->dataFloat();
        if (!accumulateA) {
            gradA->fillZero();
        }
        int64_t numThreads = (M >= 4 && (M * K * N) >= 65536 && maxThreads > 1) ? std::min(maxThreads, M) : 1;
        if (numThreads <= 1) {
            matmulBackwardARange(gyData, bData, gaData, 0, M, K, N);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(numThreads - 1));
            int64_t rowsPerThread = (M + numThreads - 1) / numThreads;
            for (int64_t t = 1; t < numThreads; ++t) {
                int64_t rStart = t * rowsPerThread;
                int64_t rEnd = std::min(M, rStart + rowsPerThread);
                if (rStart < rEnd) {
                    workers.emplace_back(matmulBackwardARange, gyData, bData, gaData, rStart, rEnd, K, N);
                }
            }
            matmulBackwardARange(gyData, bData, gaData, 0, std::min(M, rowsPerThread), K, N);
            for (auto& w : workers) w.join();
        }
    }

    // 2. dB = A^T * gradY -> [K, N]
    if (gradB) {
        float* gbData = gradB->dataFloat();
        if (!accumulateB) {
            gradB->fillZero();
        }
        int64_t numThreads = (K >= 4 && (M * K * N) >= 65536 && maxThreads > 1) ? std::min(maxThreads, K) : 1;
        if (numThreads <= 1) {
            matmulBackwardBRange(aData, gyData, gbData, 0, K, M, K, N);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(numThreads - 1));
            int64_t rowsPerThread = (K + numThreads - 1) / numThreads;
            for (int64_t t = 1; t < numThreads; ++t) {
                int64_t kStart = t * rowsPerThread;
                int64_t kEnd = std::min(K, kStart + rowsPerThread);
                if (kStart < kEnd) {
                    workers.emplace_back(matmulBackwardBRange, aData, gyData, gbData, kStart, kEnd, M, K, N);
                }
            }
            matmulBackwardBRange(aData, gyData, gbData, 0, std::min(K, rowsPerThread), M, K, N);
            for (auto& w : workers) w.join();
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

double AutogradOps::crossEntropyForwardAndBackwardSum(
    const Tensor& logits,
    const std::vector<int32_t>& targets,
    Tensor& gradLogits,
    Tensor& probsOut,
    int32_t ignoreIndex,
    uint64_t* outValidTokens
) {
    // logits: [N, V]
    const int64_t N = static_cast<int64_t>(targets.size());
    const int64_t V = logits.shape()[1];

    const float* lData = logits.dataFloat();
    float* glData = gradLogits.dataFloat();
    float* pData = probsOut.dataFloat();

    gradLogits.fillZero();

    uint64_t validCount = 0;
    double totalLoss = 0.0;

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

        validCount++;

        // 4. Cross-entropy loss for target: -log(p_target)
        float pTarget = pRow[target];
        if (pTarget < 1e-12f) pTarget = 1e-12f;
        totalLoss += -std::log(static_cast<double>(pTarget));

        // 5. Unnormalized gradient: (p[v] - delta(v, target))
        for (int64_t v = 0; v < V; ++v) {
            float indicator = (v == target) ? 1.0f : 0.0f;
            glRow[v] = pRow[v] - indicator;
        }
    }

    if (outValidTokens) {
        *outValidTokens = validCount;
    }
    return totalLoss;
}

float AutogradOps::crossEntropyForwardAndBackward(
    const Tensor& logits,
    const std::vector<int32_t>& targets,
    Tensor& gradLogits,
    Tensor& probsOut,
    int32_t ignoreIndex
) {
    uint64_t validCount = 0;
    double totalLoss = crossEntropyForwardAndBackwardSum(
        logits, targets, gradLogits, probsOut, ignoreIndex, &validCount
    );
    if (validCount == 0) {
        return 0.0f;
    }
    const float invValid = 1.0f / static_cast<float>(validCount);
    float* glData = gradLogits.dataFloat();
    const int64_t totalElems = gradLogits.numel();
    for (int64_t i = 0; i < totalElems; ++i) {
        glData[i] *= invValid;
    }
    return static_cast<float>(totalLoss * static_cast<double>(invValid));
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
