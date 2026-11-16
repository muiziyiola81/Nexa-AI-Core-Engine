/**
 * NEXA Native Training Engine - NEXA-1 Nano Model Implementation
 */

#include "../include/nexa/nano_model.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace nexa {

NexaNanoModel::NexaNanoModel(const NanoConfig& cfg, uint64_t initSeed)
    : config(cfg) {
    initWeights(initSeed);
}

void NexaNanoModel::initWeights(uint64_t seed) {
    allParams.clear();
    const float dScale = 1.0f / std::sqrt(static_cast<float>(config.dim));
    const float iScale = 1.0f / std::sqrt(static_cast<float>(config.intermediateDim));
    const float resScale = 1.0f / std::sqrt(2.0f * static_cast<float>(std::max<uint32_t>(1, config.nLayers)));

    // 1. Embeddings
    wEmbed = GradTensor("model.embed_tokens.weight", Shape({static_cast<int64_t>(config.vocabSize), static_cast<int64_t>(config.dim)}));
    wEmbed.fillRandom(config.initStddev, seed++);
    allParams.push_back(&wEmbed);

    if (!config.useRope) {
        wPos = GradTensor("model.embed_positions.weight", Shape({static_cast<int64_t>(config.maxSeqLen), static_cast<int64_t>(config.dim)}));
        wPos.fillRandom(config.initStddev, seed++);
        allParams.push_back(&wPos);
    } else {
        wPos = GradTensor();
    }

    // 2. Transformer Layers
    layers.resize(config.nLayers);
    for (uint32_t l = 0; l < config.nLayers; ++l) {
        std::string pfx = "model.layers." + std::to_string(l) + ".";

        LayerParams& lp = layers[l];
        lp.attnNorm = GradTensor(pfx + "attn_norm.weight", Shape({static_cast<int64_t>(config.dim)}));
        lp.attnNorm.value.fillConstant(1.0f);
        allParams.push_back(&lp.attnNorm);

        lp.wQ = GradTensor(pfx + "self_attn.q_proj.weight", Shape({static_cast<int64_t>(config.dim), static_cast<int64_t>(config.dim)}));
        lp.wQ.fillRandom(dScale, seed++);
        allParams.push_back(&lp.wQ);

        lp.wK = GradTensor(pfx + "self_attn.k_proj.weight", Shape({static_cast<int64_t>(config.dim), static_cast<int64_t>(config.dim)}));
        lp.wK.fillRandom(dScale, seed++);
        allParams.push_back(&lp.wK);

        lp.wV = GradTensor(pfx + "self_attn.v_proj.weight", Shape({static_cast<int64_t>(config.dim), static_cast<int64_t>(config.dim)}));
        lp.wV.fillRandom(dScale, seed++);
        allParams.push_back(&lp.wV);

        lp.wOut = GradTensor(pfx + "self_attn.o_proj.weight", Shape({static_cast<int64_t>(config.dim), static_cast<int64_t>(config.dim)}));
        lp.wOut.fillRandom(dScale * resScale, seed++);
        allParams.push_back(&lp.wOut);

        lp.ffnNorm = GradTensor(pfx + "ffn_norm.weight", Shape({static_cast<int64_t>(config.dim)}));
        lp.ffnNorm.value.fillConstant(1.0f);
        allParams.push_back(&lp.ffnNorm);

        lp.wGate = GradTensor(pfx + "mlp.gate_proj.weight", Shape({static_cast<int64_t>(config.dim), static_cast<int64_t>(config.intermediateDim)}));
        lp.wGate.fillRandom(dScale, seed++);
        allParams.push_back(&lp.wGate);

        lp.wUp = GradTensor(pfx + "mlp.up_proj.weight", Shape({static_cast<int64_t>(config.dim), static_cast<int64_t>(config.intermediateDim)}));
        lp.wUp.fillRandom(dScale, seed++);
        allParams.push_back(&lp.wUp);

        lp.wDown = GradTensor(pfx + "mlp.down_proj.weight", Shape({static_cast<int64_t>(config.intermediateDim), static_cast<int64_t>(config.dim)}));
        lp.wDown.fillRandom(iScale * resScale, seed++);
        allParams.push_back(&lp.wDown);
    }

    // 3. Final Norm and Output Projection
    finalNorm = GradTensor("model.norm.weight", Shape({static_cast<int64_t>(config.dim)}));
    finalNorm.value.fillConstant(1.0f);
    allParams.push_back(&finalNorm);

    if (!config.tieEmbeddings) {
        wHead = GradTensor("lm_head.weight", Shape({static_cast<int64_t>(config.dim), static_cast<int64_t>(config.vocabSize)}));
        wHead.fillRandom(dScale, seed++);
        allParams.push_back(&wHead);
    } else {
        wHead = GradTensor();
    }
}

void NexaNanoModel::zeroGrad() {
    for (auto* p : allParams) {
        p->zeroGrad();
    }
}

int64_t NexaNanoModel::totalParameters() const {
    int64_t total = 0;
    for (const auto* p : allParams) {
        total += p->value.numel();
    }
    return total;
}

ModelMemoryEstimate NexaNanoModel::estimateMemory(const NanoConfig& cfg) {
    ModelMemoryEstimate est;
    const int64_t V = cfg.vocabSize;
    const int64_t D = cfg.dim;
    const int64_t I = cfg.intermediateDim;
    const int64_t L = cfg.nLayers;
    const int64_t T = cfg.maxSeqLen;
    const int64_t H = cfg.nHeads;

    int64_t embedParams = V * D + (cfg.useRope ? 0 : T * D);
    int64_t perLayerParams = 2 * D + 4 * D * D + 3 * D * I;
    int64_t headParams = D + (cfg.tieEmbeddings ? 0 : D * V);

    est.totalParameters = embedParams + L * perLayerParams + headParams;
    est.weightsBytes = static_cast<uint64_t>(est.totalParameters) * sizeof(float);
    est.gradientsBytes = est.weightsBytes;
    est.optimizerBytes = est.weightsBytes * 2ULL; // AdamW m and v

    // Per-layer cached activations for training sequence length T
    int64_t perLayerActFloats = (7 * T * D) + (H * T * T) + (4 * T * I) + (2 * T);
    int64_t totalActFloats = (3 * T * D) + L * perLayerActFloats + (T * D + T) + (2 * T * V);
    est.activationsBytes = static_cast<uint64_t>(totalActFloats) * sizeof(float);
    est.totalInferenceBytes = est.weightsBytes + static_cast<uint64_t>(4 * T * D + T * V) * sizeof(float);
    est.totalTrainingBytes = est.weightsBytes + est.gradientsBytes + est.optimizerBytes + est.activationsBytes;
    return est;
}

ModelMemoryEstimate NexaNanoModel::estimateMemoryWithOptions(
    const NanoConfig& cfg,
    bool simulateTiedEmbeddings,
    bool simulateRopeNoWPos,
    uint32_t overrideMaxSeqLen
) {
    NanoConfig effective = cfg;
    if (overrideMaxSeqLen > 0) {
        effective.maxSeqLen = overrideMaxSeqLen;
    }
    ModelMemoryEstimate est = estimateMemory(effective);
    int64_t savedParams = 0;
    if (simulateTiedEmbeddings) {
        savedParams += static_cast<int64_t>(effective.dim) * static_cast<int64_t>(effective.vocabSize);
    }
    if (simulateRopeNoWPos) {
        savedParams += static_cast<int64_t>(effective.maxSeqLen) * static_cast<int64_t>(effective.dim);
    }
    if (savedParams > 0 && savedParams < est.totalParameters) {
        est.totalParameters -= savedParams;
        est.weightsBytes = static_cast<uint64_t>(est.totalParameters) * sizeof(float);
        est.gradientsBytes = est.weightsBytes;
        est.optimizerBytes = est.weightsBytes * 2ULL;
        est.totalInferenceBytes = est.weightsBytes +
            static_cast<uint64_t>(4 * static_cast<int64_t>(effective.maxSeqLen) * static_cast<int64_t>(effective.dim) +
                                  static_cast<int64_t>(effective.maxSeqLen) * static_cast<int64_t>(effective.vocabSize)) * sizeof(float);
        est.totalTrainingBytes = est.weightsBytes + est.gradientsBytes + est.optimizerBytes + est.activationsBytes;
    }
    return est;
}

ArchitecturePrepSpec NexaNanoModel::getArchitecturePrepSpec() {
    ArchitecturePrepSpec spec;
    return spec;
}

namespace {

void applyRopeInPlace(
    Tensor& qk,
    int64_t seqLen,
    int64_t dim,
    int64_t nHeads,
    float ropeTheta,
    bool inverse
) {
    const int64_t headDim = dim / nHeads;
    const int64_t half = headDim / 2;
    if (half <= 0) return;

    float* data = qk.dataFloat();
    const float theta = (std::isfinite(ropeTheta) && ropeTheta > 1.0f) ? ropeTheta : 10000.0f;

    for (int64_t t = 0; t < seqLen; ++t) {
        for (int64_t p = 0; p < half; ++p) {
            float freq = 1.0f / std::pow(theta, (2.0f * static_cast<float>(p)) / static_cast<float>(headDim));
            float angle = static_cast<float>(t) * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            for (int64_t h = 0; h < nHeads; ++h) {
                int64_t idx0 = t * dim + h * headDim + 2 * p;
                int64_t idx1 = idx0 + 1;
                float x0 = data[idx0];
                float x1 = data[idx1];
                if (!inverse) {
                    data[idx0] = x0 * c - x1 * s;
                    data[idx1] = x0 * s + x1 * c;
                } else {
                    data[idx0] =  x0 * c + x1 * s;
                    data[idx1] = -x0 * s + x1 * c;
                }
            }
        }
    }
}

void tiedLmHeadForwardRange(
    const float* hData,
    const float* eData,
    float* lData,
    int64_t T,
    int64_t D,
    int64_t V,
    int64_t vBegin,
    int64_t vEnd
) {
    for (int64_t t = 0; t < T; ++t) {
        const float* hRow = hData + t * D;
        float* lRow = lData + t * V;
        for (int64_t v = vBegin; v < vEnd; ++v) {
            const float* eRow = eData + v * D;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 acc = _mm256_setzero_ps();
            int64_t d = 0;
            for (; d + 8 <= D; d += 8) {
                __m256 vh = _mm256_loadu_ps(hRow + d);
                __m256 ve = _mm256_loadu_ps(eRow + d);
                acc = _mm256_fmadd_ps(vh, ve, acc);
            }
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 s4 = _mm_add_ps(lo, hi);
            __m128 s2 = _mm_hadd_ps(s4, s4);
            __m128 s1 = _mm_hadd_ps(s2, s2);
            float dot = _mm_cvtss_f32(s1);
            for (; d < D; ++d) {
                dot += hRow[d] * eRow[d];
            }
            lRow[v] = dot;
#else
            float dot = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                dot += hRow[d] * eRow[d];
            }
            lRow[v] = dot;
#endif
        }
    }
}

void tiedLmHeadForward(
    const Tensor& finalNormOut,
    const Tensor& wEmbedVal,
    Tensor& logitsOut
) {
    // finalNormOut: [T, D], wEmbedVal: [V, D] -> logitsOut: [T, V]
    const int64_t T = finalNormOut.shape()[0];
    const int64_t D = finalNormOut.shape()[1];
    const int64_t V = wEmbedVal.shape()[0];
    const float* hData = finalNormOut.dataFloat();
    const float* eData = wEmbedVal.dataFloat();
    float* lData = logitsOut.dataFloat();

    unsigned int hwThreads = std::thread::hardware_concurrency();
    int64_t numThreads = hwThreads > 0 ? static_cast<int64_t>(std::min<unsigned int>(hwThreads, 4)) : 1;
    if (V < 8 || (T * D * V) < 65536 || numThreads <= 1) {
        tiedLmHeadForwardRange(hData, eData, lData, T, D, V, 0, V);
        return;
    }

    numThreads = std::min(numThreads, V);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(numThreads - 1));
    int64_t colsPerThread = (V + numThreads - 1) / numThreads;
    for (int64_t th = 1; th < numThreads; ++th) {
        int64_t vStart = th * colsPerThread;
        int64_t vEnd = std::min(V, vStart + colsPerThread);
        if (vStart < vEnd) {
            workers.emplace_back(tiedLmHeadForwardRange, hData, eData, lData, T, D, V, vStart, vEnd);
        }
    }
    tiedLmHeadForwardRange(hData, eData, lData, T, D, V, 0, std::min(V, colsPerThread));
    for (auto& w : workers) w.join();
}

double forwardAndLossCore(
    NexaNanoModel& self,
    const std::vector<int32_t>& inputTokens,
    const std::vector<int32_t>& targetTokens,
    ModelActivations& act,
    int32_t ignoreIndex,
    bool normalizeByTokens,
    uint64_t* outValidTokens
) {
    const int64_t seqLen = static_cast<int64_t>(inputTokens.size());
    if (seqLen <= 0 || seqLen > static_cast<int64_t>(self.config.maxSeqLen)) {
        throw std::invalid_argument("forwardAndLoss: inputTokens length exceeds config.maxSeqLen");
    }
    const int64_t dim = self.config.dim;
    const int64_t vocab = self.config.vocabSize;
    act.tokens = inputTokens;

    // ------------------------------------------------------------------------
    // FORWARD PASS (Caching intermediate activations for backprop)
    // ------------------------------------------------------------------------
    act.embedOut = Tensor(Shape({seqLen, dim}));
    AutogradOps::embeddingForward(inputTokens, self.wEmbed.value, act.embedOut);

    act.initialX = Tensor(Shape({seqLen, dim}));
    if (!self.config.useRope) {
        act.posOut = Tensor(Shape({seqLen, dim}));
        std::memcpy(act.posOut.dataFloat(), self.wPos.value.dataFloat(), seqLen * dim * sizeof(float));
        Tensor::vecAdd(act.embedOut, act.posOut, act.initialX);
    } else {
        std::memcpy(act.initialX.dataFloat(), act.embedOut.dataFloat(), seqLen * dim * sizeof(float));
    }

    act.layers.resize(self.config.nLayers);
    Tensor x = act.initialX;

    const int64_t nHeads = self.config.nHeads;
    const int64_t headDim = dim / nHeads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

    for (uint32_t l = 0; l < self.config.nLayers; ++l) {
        LayerParams& lp = self.layers[l];
        LayerActivations& la = act.layers[l];

        // 1. Attn pre-norm
        la.xNormAttn = Tensor(Shape({seqLen, dim}));
        la.invRmsAttn = Tensor(Shape({seqLen}));
        AutogradOps::rmsnormForward(x, lp.attnNorm.value, la.xNormAttn, la.invRmsAttn, self.config.normEps);

        // 2. Q, K, V
        la.q = Tensor(Shape({seqLen, dim}));
        la.k = Tensor(Shape({seqLen, dim}));
        la.v = Tensor(Shape({seqLen, dim}));
        AutogradOps::matmulForward(la.xNormAttn, lp.wQ.value, la.q);
        AutogradOps::matmulForward(la.xNormAttn, lp.wK.value, la.k);
        AutogradOps::matmulForward(la.xNormAttn, lp.wV.value, la.v);

        if (self.config.useRope) {
            applyRopeInPlace(la.q, seqLen, dim, nHeads, self.config.ropeTheta, false);
            applyRopeInPlace(la.k, seqLen, dim, nHeads, self.config.ropeTheta, false);
        }

        // 3. Multi-Head Attention
        la.attnProbs = Tensor(Shape({nHeads, seqLen, seqLen}));
        la.attnProbs.fillZero();
        la.attnContext = Tensor(Shape({seqLen, dim}));
        la.attnContext.fillZero();

        const float* qD = la.q.dataFloat();
        const float* kD = la.k.dataFloat();
        const float* vD = la.v.dataFloat();
        float* pD = la.attnProbs.dataFloat();
        float* ctxD = la.attnContext.dataFloat();

        for (int64_t h = 0; h < nHeads; ++h) {
            for (int64_t i = 0; i < seqLen; ++i) {
                int64_t pRow = (h * seqLen + i) * seqLen;
                float maxS = -1e9f;

                for (int64_t j = 0; j <= i; ++j) {
                    float dot = 0.0f;
                    int64_t qOff = i * dim + h * headDim;
                    int64_t kOff = j * dim + h * headDim;
                    for (int64_t d = 0; d < headDim; ++d) {
                        dot += qD[qOff + d] * kD[kOff + d];
                    }
                    float s = dot * scale;
                    pD[pRow + j] = s;
                    if (s > maxS) maxS = s;
                }

                float sumExp = 0.0f;
                for (int64_t j = 0; j <= i; ++j) {
                    float expVal = std::exp(pD[pRow + j] - maxS);
                    pD[pRow + j] = expVal;
                    sumExp += expVal;
                }
                float invSum = sumExp > 0.0f ? (1.0f / sumExp) : 0.0f;
                for (int64_t j = 0; j <= i; ++j) {
                    pD[pRow + j] *= invSum;
                }

                int64_t ctxOff = i * dim + h * headDim;
                for (int64_t j = 0; j <= i; ++j) {
                    float weight = pD[pRow + j];
                    int64_t vOff = j * dim + h * headDim;
                    for (int64_t d = 0; d < headDim; ++d) {
                        ctxD[ctxOff + d] += weight * vD[vOff + d];
                    }
                }
            }
        }

        // Out projection & residual 1
        la.attnOut = Tensor(Shape({seqLen, dim}));
        AutogradOps::matmulForward(la.attnContext, lp.wOut.value, la.attnOut);

        la.x1 = Tensor(Shape({seqLen, dim}));
        Tensor::vecAdd(x, la.attnOut, la.x1);

        // FFN pre-norm
        la.xNormFfn = Tensor(Shape({seqLen, dim}));
        la.invRmsFfn = Tensor(Shape({seqLen}));
        AutogradOps::rmsnormForward(la.x1, lp.ffnNorm.value, la.xNormFfn, la.invRmsFfn, self.config.normEps);

        // SwiGLU
        la.ffnGate = Tensor(Shape({seqLen, static_cast<int64_t>(self.config.intermediateDim)}));
        la.ffnUp = Tensor(Shape({seqLen, static_cast<int64_t>(self.config.intermediateDim)}));
        AutogradOps::matmulForward(la.xNormFfn, lp.wGate.value, la.ffnGate);
        AutogradOps::matmulForward(la.xNormFfn, lp.wUp.value, la.ffnUp);

        la.ffnSilu = Tensor(Shape({seqLen, static_cast<int64_t>(self.config.intermediateDim)}));
        AutogradOps::siluForward(la.ffnGate, la.ffnSilu);

        la.ffnHidden = Tensor(Shape({seqLen, static_cast<int64_t>(self.config.intermediateDim)}));
        Tensor::vecMul(la.ffnSilu, la.ffnUp, la.ffnHidden);

        la.ffnOut = Tensor(Shape({seqLen, dim}));
        AutogradOps::matmulForward(la.ffnHidden, lp.wDown.value, la.ffnOut);

        la.x2 = Tensor(Shape({seqLen, dim}));
        Tensor::vecAdd(la.x1, la.ffnOut, la.x2);

        x = la.x2;
    }

    // Final RMSNorm
    act.finalNormOut = Tensor(Shape({seqLen, dim}));
    act.finalInvRms = Tensor(Shape({seqLen}));
    AutogradOps::rmsnormForward(x, self.finalNorm.value, act.finalNormOut, act.finalInvRms, self.config.normEps);

    // LM Head
    act.logits = Tensor(Shape({seqLen, vocab}));
    if (self.config.tieEmbeddings) {
        tiedLmHeadForward(act.finalNormOut, self.wEmbed.value, act.logits);
    } else {
        AutogradOps::matmulForward(act.finalNormOut, self.wHead.value, act.logits);
    }

    // ------------------------------------------------------------------------
    // LOSS CALCULATION & INITIAL BACKWARD GRADIENT
    // ------------------------------------------------------------------------
    Tensor gradLogits(Shape({seqLen, vocab}));
    act.probs = Tensor(Shape({seqLen, vocab}));
    double lossVal = 0.0;
    if (normalizeByTokens) {
        lossVal = static_cast<double>(AutogradOps::crossEntropyForwardAndBackward(
            act.logits, targetTokens, gradLogits, act.probs, ignoreIndex
        ));
    } else {
        lossVal = AutogradOps::crossEntropyForwardAndBackwardSum(
            act.logits, targetTokens, gradLogits, act.probs, ignoreIndex, outValidTokens
        );
    }

    // ------------------------------------------------------------------------
    // BACKPROPAGATION THROUGH ALL TRANSFORMER LAYERS
    // ------------------------------------------------------------------------
    // 1. Through LM Head
    Tensor gradFinalNormOut(Shape({seqLen, dim}));
    if (self.config.tieEmbeddings) {
        AutogradOps::matmulForward(gradLogits, self.wEmbed.value, gradFinalNormOut);
        AutogradOps::matmulBackward(gradLogits, self.wEmbed.value, act.finalNormOut, nullptr, &self.wEmbed.grad, false, true);
    } else {
        AutogradOps::matmulBackward(act.finalNormOut, self.wHead.value, gradLogits, &gradFinalNormOut, &self.wHead.grad, false, true);
    }

    // 2. Through Final RMSNorm
    Tensor dx(Shape({seqLen, dim}));
    const Tensor& lastX = self.config.nLayers > 0 ? act.layers[self.config.nLayers - 1].x2 : act.initialX;
    AutogradOps::rmsnormBackward(lastX, self.finalNorm.value, act.finalInvRms, gradFinalNormOut, &dx, &self.finalNorm.grad, true);

    // 3. Through Layers in reverse order
    for (int32_t l = static_cast<int32_t>(self.config.nLayers) - 1; l >= 0; --l) {
        LayerParams& lp = self.layers[l];
        LayerActivations& la = act.layers[l];

        // Residual 2: x2 = x1 + ffnOut
        Tensor dFfnOut = dx;
        Tensor dx1(Shape({seqLen, dim}));
        std::memcpy(dx1.dataFloat(), dx.dataFloat(), seqLen * dim * sizeof(float));

        Tensor dFfnHidden(Shape({seqLen, static_cast<int64_t>(self.config.intermediateDim)}));
        AutogradOps::matmulBackward(la.ffnHidden, lp.wDown.value, dFfnOut, &dFfnHidden, &lp.wDown.grad, false, true);

        Tensor dFfnSilu(Shape({seqLen, static_cast<int64_t>(self.config.intermediateDim)}));
        Tensor dFfnUp(Shape({seqLen, static_cast<int64_t>(self.config.intermediateDim)}));
        AutogradOps::elemwiseMulBackward(la.ffnSilu, la.ffnUp, dFfnHidden, &dFfnSilu, &dFfnUp, false);

        Tensor dxNormFfn(Shape({seqLen, dim}));
        AutogradOps::matmulBackward(la.xNormFfn, lp.wUp.value, dFfnUp, &dxNormFfn, &lp.wUp.grad, false, true);

        Tensor dFfnGate(Shape({seqLen, static_cast<int64_t>(self.config.intermediateDim)}));
        AutogradOps::siluBackward(la.ffnGate, dFfnSilu, dFfnGate, false);

        AutogradOps::matmulBackward(la.xNormFfn, lp.wGate.value, dFfnGate, &dxNormFfn, &lp.wGate.grad, true, true);

        Tensor dx1FromFfn(Shape({seqLen, dim}));
        AutogradOps::rmsnormBackward(la.x1, lp.ffnNorm.value, la.invRmsFfn, dxNormFfn, &dx1FromFfn, &lp.ffnNorm.grad, true);
        Tensor::vecAdd(dx1, dx1FromFfn, dx1);

        // Residual 1: x1 = xPrev + attnOut
        Tensor dxPrev(Shape({seqLen, dim}));
        std::memcpy(dxPrev.dataFloat(), dx1.dataFloat(), seqLen * dim * sizeof(float));
        Tensor dAttnOut = dx1;

        Tensor dAttnContext(Shape({seqLen, dim}));
        AutogradOps::matmulBackward(la.attnContext, lp.wOut.value, dAttnOut, &dAttnContext, &lp.wOut.grad, false, true);

        Tensor dQ(Shape({seqLen, dim}));
        Tensor dK(Shape({seqLen, dim}));
        Tensor dV(Shape({seqLen, dim}));
        dQ.fillZero();
        dK.fillZero();
        dV.fillZero();

        const float* dCtx = dAttnContext.dataFloat();
        const float* pD = la.attnProbs.dataFloat();
        const float* qD = la.q.dataFloat();
        const float* kD = la.k.dataFloat();
        const float* vD = la.v.dataFloat();
        float* dqD = dQ.dataFloat();
        float* dkD = dK.dataFloat();
        float* dvD = dV.dataFloat();

        for (int64_t h = 0; h < nHeads; ++h) {
            for (int64_t i = 0; i < seqLen; ++i) {
                int64_t pRow = (h * seqLen + i) * seqLen;
                int64_t ctxOff = i * dim + h * headDim;

                std::vector<float> dProbs(i + 1, 0.0f);
                for (int64_t j = 0; j <= i; ++j) {
                    float weight = pD[pRow + j];
                    int64_t vOff = j * dim + h * headDim;
                    float dotCtxV = 0.0f;
                    for (int64_t d = 0; d < headDim; ++d) {
                        dvD[vOff + d] += weight * dCtx[ctxOff + d];
                        dotCtxV += dCtx[ctxOff + d] * vD[vOff + d];
                    }
                    dProbs[j] = dotCtxV;
                }

                double sumDProbsProbs = 0.0;
                for (int64_t j = 0; j <= i; ++j) {
                    sumDProbsProbs += static_cast<double>(dProbs[j]) * static_cast<double>(pD[pRow + j]);
                }
                float sumVal = static_cast<float>(sumDProbsProbs);

                for (int64_t j = 0; j <= i; ++j) {
                    float dScore = pD[pRow + j] * (dProbs[j] - sumVal);
                    float dDot = dScore * scale;
                    int64_t qOff = i * dim + h * headDim;
                    int64_t kOff = j * dim + h * headDim;

                    for (int64_t d = 0; d < headDim; ++d) {
                        dqD[qOff + d] += dDot * kD[kOff + d];
                        dkD[kOff + d] += dDot * qD[qOff + d];
                    }
                }
            }
        }

        if (self.config.useRope) {
            applyRopeInPlace(dQ, seqLen, dim, nHeads, self.config.ropeTheta, true);
            applyRopeInPlace(dK, seqLen, dim, nHeads, self.config.ropeTheta, true);
        }

        Tensor dxNormAttn(Shape({seqLen, dim}));
        AutogradOps::matmulBackward(la.xNormAttn, lp.wQ.value, dQ, &dxNormAttn, &lp.wQ.grad, false, true);
        AutogradOps::matmulBackward(la.xNormAttn, lp.wK.value, dK, &dxNormAttn, &lp.wK.grad, true, true);
        AutogradOps::matmulBackward(la.xNormAttn, lp.wV.value, dV, &dxNormAttn, &lp.wV.grad, true, true);

        const Tensor& prevLayerX = (l > 0) ? act.layers[l - 1].x2 : act.initialX;
        Tensor dxFromAttn(Shape({seqLen, dim}));
        AutogradOps::rmsnormBackward(prevLayerX, lp.attnNorm.value, la.invRmsAttn, dxNormAttn, &dxFromAttn, &lp.attnNorm.grad, true);
        Tensor::vecAdd(dxPrev, dxFromAttn, dx);
    }

    // 4. Through Initial Embeddings & Positional Embeddings
    AutogradOps::embeddingBackward(inputTokens, dx, self.wEmbed.grad);

    if (!self.config.useRope) {
        float* gPos = self.wPos.grad.dataFloat();
        const float* gX = dx.dataFloat();
        for (int64_t t = 0; t < seqLen; ++t) {
            for (int64_t d = 0; d < dim; ++d) {
                gPos[t * dim + d] += gX[t * dim + d];
            }
        }
    }

    return lossVal;
}

} // namespace

void NexaNanoModel::forward(const std::vector<int32_t>& inputTokens, Tensor& logitsOut) {
    ModelActivations act;
    const int64_t seqLen = static_cast<int64_t>(inputTokens.size());
    if (seqLen <= 0 || seqLen > static_cast<int64_t>(config.maxSeqLen)) {
        throw std::invalid_argument("forward: inputTokens length exceeds config.maxSeqLen");
    }
    const int64_t dim = config.dim;
    const int64_t vocab = config.vocabSize;

    // 1. Embedding lookup
    act.embedOut = Tensor(Shape({seqLen, dim}));
    AutogradOps::embeddingForward(inputTokens, wEmbed.value, act.embedOut);

    // 2. Add learned positional embedding (when RoPE is disabled)
    act.initialX = Tensor(Shape({seqLen, dim}));
    if (!config.useRope) {
        const float* emb = act.embedOut.dataFloat();
        const float* pos = wPos.value.dataFloat();
        float* xCur = act.initialX.dataFloat();

        for (int64_t t = 0; t < seqLen; ++t) {
            for (int64_t d = 0; d < dim; ++d) {
                xCur[t * dim + d] = emb[t * dim + d] + pos[t * dim + d];
            }
        }
    } else {
        std::memcpy(act.initialX.dataFloat(), act.embedOut.dataFloat(), seqLen * dim * sizeof(float));
    }

    Tensor x = act.initialX;
    const int64_t nHeads = config.nHeads;
    const int64_t headDim = dim / nHeads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

    for (uint32_t l = 0; l < config.nLayers; ++l) {
        LayerParams& lp = layers[l];

        // Attention pre-norm
        Tensor xNormAttn(Shape({seqLen, dim}));
        Tensor invRmsAttn(Shape({seqLen}));
        AutogradOps::rmsnormForward(x, lp.attnNorm.value, xNormAttn, invRmsAttn, config.normEps);

        // Q, K, V
        Tensor q(Shape({seqLen, dim}));
        Tensor k(Shape({seqLen, dim}));
        Tensor v(Shape({seqLen, dim}));
        AutogradOps::matmulForward(xNormAttn, lp.wQ.value, q);
        AutogradOps::matmulForward(xNormAttn, lp.wK.value, k);
        AutogradOps::matmulForward(xNormAttn, lp.wV.value, v);

        if (config.useRope) {
            applyRopeInPlace(q, seqLen, dim, nHeads, config.ropeTheta, false);
            applyRopeInPlace(k, seqLen, dim, nHeads, config.ropeTheta, false);
        }

        // Causal multi-head self-attention
        Tensor attnContext(Shape({seqLen, dim}));
        attnContext.fillZero();

        const float* qD = q.dataFloat();
        const float* kD = k.dataFloat();
        const float* vD = v.dataFloat();
        float* ctxD = attnContext.dataFloat();

        for (int64_t h = 0; h < nHeads; ++h) {
            for (int64_t i = 0; i < seqLen; ++i) {
                std::vector<float> scores(i + 1);
                float maxS = -1e9f;

                for (int64_t j = 0; j <= i; ++j) {
                    float dot = 0.0f;
                    int64_t qOff = i * dim + h * headDim;
                    int64_t kOff = j * dim + h * headDim;
                    for (int64_t d = 0; d < headDim; ++d) {
                        dot += qD[qOff + d] * kD[kOff + d];
                    }
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > maxS) maxS = s;
                }

                float sumExp = 0.0f;
                for (int64_t j = 0; j <= i; ++j) {
                    scores[j] = std::exp(scores[j] - maxS);
                    sumExp += scores[j];
                }
                float invSum = sumExp > 0.0f ? (1.0f / sumExp) : 0.0f;
                for (int64_t j = 0; j <= i; ++j) {
                    scores[j] *= invSum;
                }

                int64_t ctxOff = i * dim + h * headDim;
                for (int64_t j = 0; j <= i; ++j) {
                    float w = scores[j];
                    int64_t vOff = j * dim + h * headDim;
                    for (int64_t d = 0; d < headDim; ++d) {
                        ctxD[ctxOff + d] += w * vD[vOff + d];
                    }
                }
            }
        }

        // Out projection & residual
        Tensor attnOut(Shape({seqLen, dim}));
        AutogradOps::matmulForward(attnContext, lp.wOut.value, attnOut);

        Tensor x1(Shape({seqLen, dim}));
        Tensor::vecAdd(x, attnOut, x1);

        // FFN pre-norm
        Tensor xNormFfn(Shape({seqLen, dim}));
        Tensor invRmsFfn(Shape({seqLen}));
        AutogradOps::rmsnormForward(x1, lp.ffnNorm.value, xNormFfn, invRmsFfn, config.normEps);

        // SwiGLU: (SiLU(x * wGate) * (x * wUp)) * wDown
        Tensor ffnGate(Shape({seqLen, static_cast<int64_t>(config.intermediateDim)}));
        Tensor ffnUp(Shape({seqLen, static_cast<int64_t>(config.intermediateDim)}));
        AutogradOps::matmulForward(xNormFfn, lp.wGate.value, ffnGate);
        AutogradOps::matmulForward(xNormFfn, lp.wUp.value, ffnUp);

        Tensor ffnSilu(Shape({seqLen, static_cast<int64_t>(config.intermediateDim)}));
        AutogradOps::siluForward(ffnGate, ffnSilu);

        Tensor ffnHidden(Shape({seqLen, static_cast<int64_t>(config.intermediateDim)}));
        Tensor::vecMul(ffnSilu, ffnUp, ffnHidden);

        Tensor ffnOut(Shape({seqLen, dim}));
        AutogradOps::matmulForward(ffnHidden, lp.wDown.value, ffnOut);

        // Second residual
        Tensor x2(Shape({seqLen, dim}));
        Tensor::vecAdd(x1, ffnOut, x2);
        x = x2;
    }

    // Final RMSNorm
    Tensor finalNormOut(Shape({seqLen, dim}));
    Tensor finalInvRms(Shape({seqLen}));
    AutogradOps::rmsnormForward(x, finalNorm.value, finalNormOut, finalInvRms, config.normEps);

    // LM Head
    if (logitsOut.rank() != 2 || logitsOut.shape()[0] != seqLen || logitsOut.shape()[1] != vocab) {
        logitsOut = Tensor(Shape({seqLen, vocab}));
    }
    if (config.tieEmbeddings) {
        tiedLmHeadForward(finalNormOut, wEmbed.value, logitsOut);
    } else {
        AutogradOps::matmulForward(finalNormOut, wHead.value, logitsOut);
    }
}

float NexaNanoModel::forwardAndLoss(
    const std::vector<int32_t>& inputTokens,
    const std::vector<int32_t>& targetTokens,
    ModelActivations& act,
    int32_t ignoreIndex
) {
    return static_cast<float>(forwardAndLossCore(*this, inputTokens, targetTokens, act, ignoreIndex, true, nullptr));
}

double NexaNanoModel::forwardAndLossSum(
    const std::vector<int32_t>& inputTokens,
    const std::vector<int32_t>& targetTokens,
    ModelActivations& act,
    int32_t ignoreIndex,
    uint64_t* outValidTokens
) {
    return forwardAndLossCore(*this, inputTokens, targetTokens, act, ignoreIndex, false, outValidTokens);
}

int32_t NexaNanoModel::predictNextToken(const std::vector<int32_t>& inputTokens) {
    if (inputTokens.empty()) return 0;
    Tensor logits;
    forward(inputTokens, logits);
    int64_t lastRow = inputTokens.size() - 1;
    const float* lData = logits.dataFloat() + (lastRow * config.vocabSize);

    int32_t bestTok = 0;
    float maxLogit = lData[0];
    for (uint32_t v = 1; v < config.vocabSize; ++v) {
        if (lData[v] > maxLogit) {
            maxLogit = lData[v];
            bestTok = static_cast<int32_t>(v);
        }
    }
    return bestTok;
}

int32_t NexaNanoModel::sampleNextToken(
    const std::vector<float>& logits,
    const std::vector<int32_t>& contextTokens,
    const GenerationConfig& cfg,
    std::mt19937_64& rng
) {
    const size_t vocabSize = logits.size();
    if (vocabSize == 0) return 0;

    std::vector<float> filtered = logits;

    // 0. Never sample <unk> (0), <bos> (1), or <pad> (3) during decoding (keep <eos> (2) active)
    if (vocabSize > 4) {
        filtered[0] = -1e9f;
        filtered[1] = -1e9f;
        filtered[3] = -1e9f;
    }

    // 1. Repetition penalty (deduplicated + frequency-aware over recent generated tokens)
    if (cfg.repetitionPenalty > 1.0f && !contextTokens.empty()) {
        std::unordered_map<int32_t, int> freq;
        for (int32_t tok : contextTokens) {
            if (tok >= 0 && static_cast<size_t>(tok) < vocabSize) {
                freq[tok]++;
            }
        }
        for (const auto& kv : freq) {
            int32_t tok = kv.first;
            float penalty = cfg.repetitionPenalty;
            if (kv.second >= 2) {
                penalty += 0.10f * static_cast<float>(std::min(kv.second - 1, 4));
            }
            if (filtered[tok] > 0.0f) {
                filtered[tok] /= penalty;
            } else {
                filtered[tok] *= penalty;
            }
        }
    }

    // 2. Greedy if temperature near 0
    if (cfg.temperature <= 1e-4f) {
        int32_t best = 0;
        float maxVal = filtered[0];
        for (size_t i = 1; i < vocabSize; ++i) {
            if (filtered[i] > maxVal) {
                maxVal = filtered[i];
                best = static_cast<int32_t>(i);
            }
        }
        return best;
    }

    // 3. Temperature scaling
    for (size_t i = 0; i < vocabSize; ++i) {
        filtered[i] /= cfg.temperature;
    }

    // 4. Stable Softmax
    float maxL = *std::max_element(filtered.begin(), filtered.end());
    double sumExp = 0.0;
    std::vector<float> probs(vocabSize);
    for (size_t i = 0; i < vocabSize; ++i) {
        probs[i] = std::exp(filtered[i] - maxL);
        sumExp += probs[i];
    }
    float invSum = sumExp > 0.0 ? static_cast<float>(1.0 / sumExp) : 0.0f;
    for (size_t i = 0; i < vocabSize; ++i) {
        probs[i] *= invSum;
    }

    // 5. Build index-probability pairs for Top-K / Top-P
    struct ProbIndex {
        float prob;
        int32_t index;
    };
    std::vector<ProbIndex> candidates(vocabSize);
    for (size_t i = 0; i < vocabSize; ++i) {
        candidates[i] = {probs[i], static_cast<int32_t>(i)};
    }
    std::sort(candidates.begin(), candidates.end(), [](const ProbIndex& a, const ProbIndex& b) {
        return a.prob > b.prob;
    });

    // 6. Top-K truncation
    if (cfg.topK > 0 && static_cast<size_t>(cfg.topK) < candidates.size()) {
        candidates.resize(cfg.topK);
    }

    // 7. Top-P (Nucleus) truncation
    if (cfg.topP > 0.0f && cfg.topP < 1.0f) {
        double cumSum = 0.0;
        size_t cutoff = candidates.size();
        for (size_t i = 0; i < candidates.size(); ++i) {
            cumSum += candidates[i].prob;
            if (cumSum >= cfg.topP && i > 0) {
                cutoff = i + 1;
                break;
            }
        }
        if (cutoff < candidates.size()) {
            candidates.resize(cutoff);
        }
    }

    // Re-normalize probabilities of remaining candidates
    double totalProb = 0.0;
    for (const auto& c : candidates) {
        totalProb += c.prob;
    }
    if (totalProb <= 0.0) {
        return candidates[0].index;
    }

    std::uniform_real_distribution<double> dist(0.0, totalProb);
    double r = dist(rng);
    double accum = 0.0;
    for (const auto& c : candidates) {
        accum += c.prob;
        if (accum >= r) {
            return c.index;
        }
    }

    return candidates.back().index;
}

GenerationResult NexaNanoModel::generateText(
    const std::string& prompt,
    Tokenizer& tokenizer,
    const GenerationConfig& genCfg
) {
    GenerationResult res;
    res.prompt = prompt;
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Encode prompt
    std::vector<int32_t> currentTokens = tokenizer.encode(prompt, true, false);
    if (currentTokens.empty()) {
        currentTokens.push_back(tokenizer.specialTokens().bosId);
    }
    res.promptTokens = currentTokens;

    std::mt19937_64 rng(genCfg.seed);
    res.finishReason = "length";

    Tensor logits;
    for (uint32_t step = 0; step < genCfg.maxNewTokens; ++step) {
        // Enforce context window bounds
        std::vector<int32_t> windowTokens = currentTokens;
        if (windowTokens.size() > config.maxSeqLen) {
            windowTokens.erase(windowTokens.begin(), windowTokens.begin() + (windowTokens.size() - config.maxSeqLen));
        }

        // Forward pass
        forward(windowTokens, logits);

        int64_t lastRow = windowTokens.size() - 1;
        const float* lData = logits.dataFloat() + (lastRow * config.vocabSize);
        std::vector<float> lastLogits(lData, lData + config.vocabSize);

        // Build sliding window over GENERATED tokens only (never penalize user prompt tokens)
        const size_t genCount = res.generatedTokens.size();
        const size_t repWindowStart = genCount > 48 ? (genCount - 48) : 0;
        std::vector<int32_t> recentGenerated(res.generatedTokens.begin() + repWindowStart, res.generatedTokens.end());

        // Suppress exact repeated 3-gram cycles in generated tokens (prevents phrase/token loops)
        if (genCount >= 3) {
            int32_t p2 = res.generatedTokens[genCount - 2];
            int32_t p1 = res.generatedTokens[genCount - 1];
            for (size_t i = 0; i + 2 < genCount; ++i) {
                if (res.generatedTokens[i] == p2 && res.generatedTokens[i + 1] == p1) {
                    int32_t banned = res.generatedTokens[i + 2];
                    if (banned != tokenizer.specialTokens().eosId &&
                        banned >= 0 && static_cast<size_t>(banned) < lastLogits.size()) {
                        lastLogits[banned] -= 12.0f;
                    }
                }
            }
        }

        // Sample next token using only generated history for repetition penalty
        int32_t nextTok = sampleNextToken(lastLogits, recentGenerated, genCfg, rng);

        // Check stop token
        if (genCfg.stopToken >= 0 && nextTok == genCfg.stopToken) {
            res.finishReason = "stop_token";
            break;
        }
        if (nextTok == tokenizer.specialTokens().eosId) {
            res.finishReason = "stop_token";
            break;
        }

        currentTokens.push_back(nextTok);
        res.generatedTokens.push_back(nextTok);

        // Check stop sequences
        if (!genCfg.stopSequences.empty()) {
            std::string currentDecoded = tokenizer.decode(res.generatedTokens, true);
            size_t earliestPos = std::string::npos;
            for (const auto& stopSeq : genCfg.stopSequences) {
                if (!stopSeq.empty()) {
                    size_t pos = currentDecoded.find(stopSeq);
                    if (pos != std::string::npos && (earliestPos == std::string::npos || pos < earliestPos)) {
                        earliestPos = pos;
                    }
                }
            }
            if (earliestPos != std::string::npos) {
                res.finishReason = "stop_sequence";
                res.generatedText = currentDecoded.substr(0, earliestPos);
                break;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    res.durationMs = std::chrono::duration<double, std::milli>(end - start).count();
    res.tokensGenerated = static_cast<uint32_t>(res.generatedTokens.size());
    res.tokensPerSecond = res.durationMs > 0.0 ? (res.tokensGenerated / (res.durationMs / 1000.0)) : 0.0;

    if (res.finishReason != "stop_sequence") {
        res.generatedText = tokenizer.decode(res.generatedTokens, true);
    }
    res.fullText = prompt + res.generatedText;
    res.success = true;

    return res;
}

bool NexaNanoModel::saveCheckpoint(
    const std::string& filePath,
    uint64_t step,
    float loss,
    const AdamW* opt
) const {
    std::ofstream out(filePath, std::ios::binary);
    if (!out.is_open()) return false;

    // Header (128 bytes fixed)
    uint32_t magic = 0x4E455841; // 'NEXA'
    uint32_t version = 2;         // Format version 2: includes training checkpoint state
    uint32_t archType = 1;        // Transformer Decoder
    uint32_t hasOpt = (opt != nullptr) ? 1 : 0;
    uint32_t nTensors = static_cast<uint32_t>(allParams.size());

    out.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&archType), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&config.vocabSize), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&config.dim), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&config.intermediateDim), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&config.nLayers), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&config.nHeads), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&config.nHeads), sizeof(uint32_t)); // nKvHeads
    out.write(reinterpret_cast<const char*>(&config.maxSeqLen), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&config.normEps), sizeof(float));
    float ropeTheta = config.ropeTheta;
    out.write(reinterpret_cast<const char*>(&ropeTheta), sizeof(float));
    uint32_t quantType = 0; // FP32
    out.write(reinterpret_cast<const char*>(&quantType), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&nTensors), sizeof(uint32_t));

    // Training checkpoint extension fields:
    out.write(reinterpret_cast<const char*>(&step), sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(&loss), sizeof(float));
    out.write(reinterpret_cast<const char*>(&hasOpt), sizeof(uint32_t));

    // Pad header to exactly 128 bytes (encoding featureFlags + optional tokenizerHash in bytes 0..11)
    char pad[56] = {0};
    uint32_t featureFlags = (config.useRope ? 1u : 0u) |
                            (config.tieEmbeddings ? 2u : 0u) |
                            (config.tokenizerHash != 0ULL ? 4u : 0u);
    std::memcpy(pad, &featureFlags, sizeof(uint32_t));
    std::memcpy(pad + sizeof(uint32_t), &config.tokenizerHash, sizeof(uint64_t));
    out.write(pad, sizeof(pad));

    // Write parameter tensors
    for (const auto* p : allParams) {
        uint32_t nameLen = static_cast<uint32_t>(p->name.size());
        out.write(reinterpret_cast<const char*>(&nameLen), sizeof(uint32_t));
        out.write(p->name.data(), nameLen);

        uint32_t rank = static_cast<uint32_t>(p->value.rank());
        out.write(reinterpret_cast<const char*>(&rank), sizeof(uint32_t));
        for (size_t r = 0; r < rank; ++r) {
            int64_t d = p->value.shape()[r];
            out.write(reinterpret_cast<const char*>(&d), sizeof(int64_t));
        }

        int64_t numel = p->value.numel();
        out.write(reinterpret_cast<const char*>(&numel), sizeof(int64_t));
        out.write(reinterpret_cast<const char*>(p->value.dataFloat()), numel * sizeof(float));
    }

    // Optional optimizer state
    if (opt && hasOpt) {
        opt->saveState(out);
    }

    return out.good();
}

CheckpointValidationResult NexaNanoModel::validateCheckpoint(const std::string& filePath) {
    CheckpointValidationResult res;
    std::ifstream in(filePath, std::ios::binary);
    if (!in.is_open()) {
        res.valid = false;
        res.error = "Failed to open checkpoint file: " + filePath;
        return res;
    }

    in.seekg(0, std::ios::end);
    std::streamsize fileSize = in.tellg();
    in.seekg(0, std::ios::beg);

    if (fileSize < 128) {
        res.valid = false;
        res.error = "File is too small for .nexa header (less than 128 bytes)";
        return res;
    }

    uint32_t magic = 0, version = 0, archType = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&archType), sizeof(uint32_t));

    if (magic != 0x4E455841) {
        res.valid = false;
        res.error = "Invalid magic bytes: expected 0x4E455841 ('NEXA')";
        return res;
    }

    if (version != 1 && version != 2) {
        res.valid = false;
        res.error = "Unsupported format version: " + std::to_string(version);
        return res;
    }

    res.version = version;
    res.archType = archType;

    in.read(reinterpret_cast<char*>(&res.config.vocabSize), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&res.config.dim), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&res.config.intermediateDim), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&res.config.nLayers), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&res.config.nHeads), sizeof(uint32_t));
    uint32_t dummyKv = 0;
    in.read(reinterpret_cast<char*>(&dummyKv), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&res.config.maxSeqLen), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&res.config.normEps), sizeof(float));
    float ropeTheta = 10000.0f;
    in.read(reinterpret_cast<char*>(&ropeTheta), sizeof(float));
    uint32_t quantType = 0, nTensors = 0;
    in.read(reinterpret_cast<char*>(&quantType), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&nTensors), sizeof(uint32_t));
    res.tensorCount = nTensors;
    res.config.ropeTheta = (std::isfinite(ropeTheta) && ropeTheta > 1.0f) ? ropeTheta : 10000.0f;

    if (res.config.vocabSize == 0 || res.config.dim == 0 || res.config.intermediateDim == 0 ||
        res.config.nLayers == 0 || res.config.nHeads == 0 || res.config.maxSeqLen == 0 ||
        (res.config.dim % res.config.nHeads) != 0 ||
        !std::isfinite(res.config.normEps) || res.config.normEps <= 0.0f) {
        res.valid = false;
        res.error = "Invalid model hyperparameters in .nexa header";
        return res;
    }

    if (version >= 2) {
        uint32_t hasOpt = 0;
        in.read(reinterpret_cast<char*>(&res.step), sizeof(uint64_t));
        in.read(reinterpret_cast<char*>(&res.loss), sizeof(float));
        in.read(reinterpret_cast<char*>(&hasOpt), sizeof(uint32_t));
        res.hasOptimizerState = (hasOpt != 0);
        if (!std::isfinite(res.loss)) {
            res.valid = false;
            res.error = "Corrupted checkpoint header: non-finite loss";
            return res;
        }
        char pad[56] = {0};
        in.read(pad, sizeof(pad));
        uint32_t featureFlags = 0;
        uint64_t tokHash = 0;
        std::memcpy(&featureFlags, pad, sizeof(uint32_t));
        std::memcpy(&tokHash, pad + sizeof(uint32_t), sizeof(uint64_t));
        res.config.useRope = (featureFlags & 1u) != 0;
        res.config.tieEmbeddings = (featureFlags & 2u) != 0;
        res.config.tokenizerHash = (featureFlags & 4u) ? tokHash : 0ULL;
    }

    in.seekg(128, std::ios::beg);

    uint64_t weightBytes = 0;
    if (version == 1) {
        for (uint32_t t = 0; t < nTensors; ++t) {
            if (in.eof() || in.fail()) {
                res.valid = false;
                res.error = "Corrupted v1 descriptor: unexpected EOF reading tensor " + std::to_string(t);
                return res;
            }
            uint32_t nameLen = 0;
            in.read(reinterpret_cast<char*>(&nameLen), sizeof(uint32_t));
            if (nameLen > 256) {
                res.valid = false;
                res.error = "Corrupted tensor name length: " + std::to_string(nameLen);
                return res;
            }
            std::string name(nameLen, '\0');
            in.read(&name[0], nameLen);

            uint32_t dtypeVal = 0, nDims = 0;
            in.read(reinterpret_cast<char*>(&dtypeVal), sizeof(uint32_t));
            in.read(reinterpret_cast<char*>(&nDims), sizeof(uint32_t));
            if (nDims > 4) {
                res.valid = false;
                res.error = "Invalid tensor rank: " + std::to_string(nDims);
                return res;
            }
            in.seekg(static_cast<std::streamoff>(nDims * sizeof(uint64_t)), std::ios::cur);
            uint64_t offset = 0, sizeBytes = 0;
            in.read(reinterpret_cast<char*>(&offset), sizeof(uint64_t));
            in.read(reinterpret_cast<char*>(&sizeBytes), sizeof(uint64_t));
            weightBytes += sizeBytes;
        }
    } else {
        std::vector<float> buf(4096);
        for (uint32_t t = 0; t < nTensors; ++t) {
            if (in.eof() || in.fail()) {
                res.valid = false;
                res.error = "Corrupted checkpoint: unexpected EOF reading tensor " + std::to_string(t);
                return res;
            }
            uint32_t nameLen = 0;
            in.read(reinterpret_cast<char*>(&nameLen), sizeof(uint32_t));
            if (!in.good() || nameLen == 0 || nameLen > 256) {
                res.valid = false;
                res.error = "Corrupted tensor name length: " + std::to_string(nameLen);
                return res;
            }
            std::string name(nameLen, '\0');
            in.read(&name[0], nameLen);

            uint32_t rank = 0;
            in.read(reinterpret_cast<char*>(&rank), sizeof(uint32_t));
            if (!in.good() || rank == 0 || rank > 4) {
                res.valid = false;
                res.error = "Invalid tensor rank: " + std::to_string(rank);
                return res;
            }
            int64_t expectedProd = 1;
            for (uint32_t r = 0; r < rank; ++r) {
                int64_t d = 0;
                in.read(reinterpret_cast<char*>(&d), sizeof(int64_t));
                if (!in.good() || d <= 0 || d > 100000000LL) {
                    res.valid = false;
                    res.error = "Invalid dimension in tensor " + name;
                    return res;
                }
                expectedProd *= d;
            }

            int64_t numel = 0;
            in.read(reinterpret_cast<char*>(&numel), sizeof(int64_t));
            if (!in.good() || numel <= 0 || numel > 100000000LL || numel != expectedProd) {
                res.valid = false;
                res.error = "Invalid numel in tensor " + name + ": " + std::to_string(numel);
                return res;
            }
            uint64_t tensorBytes = static_cast<uint64_t>(numel) * sizeof(float);
            std::streamoff curPos = in.tellg();
            if (curPos < 0 || static_cast<uint64_t>(curPos) + tensorBytes > static_cast<uint64_t>(fileSize)) {
                res.valid = false;
                res.error = "Truncated checkpoint: insufficient bytes for tensor " + name;
                return res;
            }
            // Stream-check finite weights to detect NaN/Inf corruption
            int64_t remaining = numel;
            while (remaining > 0) {
                int64_t chunk = std::min<int64_t>(remaining, static_cast<int64_t>(buf.size()));
                in.read(reinterpret_cast<char*>(buf.data()), chunk * static_cast<std::streamsize>(sizeof(float)));
                if (!in.good()) {
                    res.valid = false;
                    res.error = "Failed reading tensor weights for " + name;
                    return res;
                }
                for (int64_t i = 0; i < chunk; ++i) {
                    if (!std::isfinite(buf[i])) {
                        res.valid = false;
                        res.error = "Corrupted weights (NaN/Inf) detected in tensor " + name;
                        return res;
                    }
                }
                remaining -= chunk;
            }
            weightBytes += tensorBytes;
        }
    }

    res.totalWeightBytes = weightBytes;
    res.valid = true;
    return res;
}

bool NexaNanoModel::loadCheckpoint(
    const std::string& filePath,
    uint64_t* outStep,
    float* outLoss,
    AdamW* opt,
    std::string* outError
) {
    auto val = validateCheckpoint(filePath);
    if (!val.valid) {
        if (outError) *outError = val.error;
        return false;
    }

    // Strict architecture compatibility check
    if (val.version < 2) {
        if (outError) *outError = "Checkpoint version 1 descriptor cannot be loaded as v2 weights";
        return false;
    }
    if (val.config.vocabSize != config.vocabSize) {
        std::string err = "Vocabulary mismatch: checkpoint has " + std::to_string(val.config.vocabSize) +
                          ", model has " + std::to_string(config.vocabSize);
        if (outError) *outError = err;
        return false;
    }
    if (val.config.dim != config.dim) {
        std::string err = "Hidden dim mismatch: checkpoint has " + std::to_string(val.config.dim) +
                          ", model has " + std::to_string(config.dim);
        if (outError) *outError = err;
        return false;
    }
    if (val.config.intermediateDim != config.intermediateDim) {
        std::string err = "Intermediate dim mismatch: checkpoint has " + std::to_string(val.config.intermediateDim) +
                          ", model has " + std::to_string(config.intermediateDim);
        if (outError) *outError = err;
        return false;
    }
    if (val.config.nLayers != config.nLayers) {
        std::string err = "Layers mismatch: checkpoint has " + std::to_string(val.config.nLayers) +
                          ", model has " + std::to_string(config.nLayers);
        if (outError) *outError = err;
        return false;
    }
    if (val.config.nHeads != config.nHeads) {
        std::string err = "Heads mismatch: checkpoint has " + std::to_string(val.config.nHeads) +
                          ", model has " + std::to_string(config.nHeads);
        if (outError) *outError = err;
        return false;
    }
    if (val.config.useRope != config.useRope) {
        std::string err = "RoPE mode mismatch: checkpoint useRope=" + std::string(val.config.useRope ? "true" : "false") +
                          ", model useRope=" + std::string(config.useRope ? "true" : "false");
        if (outError) *outError = err;
        return false;
    }
    if (val.config.tieEmbeddings != config.tieEmbeddings) {
        std::string err = "Weight-tying mismatch: checkpoint tieEmbeddings=" + std::string(val.config.tieEmbeddings ? "true" : "false") +
                          ", model tieEmbeddings=" + std::string(config.tieEmbeddings ? "true" : "false");
        if (outError) *outError = err;
        return false;
    }
    if (!config.useRope && val.config.maxSeqLen != config.maxSeqLen) {
        std::string err = "maxSeqLen mismatch with learned wPos: checkpoint has " + std::to_string(val.config.maxSeqLen) +
                          ", model has " + std::to_string(config.maxSeqLen);
        if (outError) *outError = err;
        return false;
    }

    config.ropeTheta = val.config.ropeTheta;
    config.tokenizerHash = val.config.tokenizerHash;

    std::ifstream in(filePath, std::ios::binary);
    if (!in.is_open()) {
        if (outError) *outError = "Failed to open file: " + filePath;
        return false;
    }

    if (outStep) *outStep = val.step;
    if (outLoss) *outLoss = val.loss;

    in.seekg(128, std::ios::beg);
    uint32_t loadedParams = 0;

    for (uint32_t t = 0; t < val.tensorCount; ++t) {
        uint32_t nameLen = 0;
        in.read(reinterpret_cast<char*>(&nameLen), sizeof(uint32_t));
        std::string name(nameLen, '\0');
        in.read(&name[0], nameLen);

        uint32_t rank = 0;
        in.read(reinterpret_cast<char*>(&rank), sizeof(uint32_t));
        std::vector<int64_t> dims(rank);
        for (uint32_t r = 0; r < rank; ++r) {
            in.read(reinterpret_cast<char*>(&dims[r]), sizeof(int64_t));
        }

        int64_t numel = 0;
        in.read(reinterpret_cast<char*>(&numel), sizeof(int64_t));

        // Find parameter in model
        bool found = false;
        for (auto* p : allParams) {
            if (p->name == name) {
                if (p->value.numel() != numel || p->value.shape().dims != dims) {
                    if (outError) *outError = "Tensor shape mismatch for " + name;
                    return false;
                }
                in.read(reinterpret_cast<char*>(p->value.dataFloat()), numel * sizeof(float));
                if (in.fail()) {
                    if (outError) *outError = "Failed reading tensor bytes for " + name;
                    return false;
                }
                found = true;
                loadedParams++;
                break;
            }
        }
        if (!found) {
            in.seekg(numel * sizeof(float), std::ios::cur);
        }
    }

    if (loadedParams != allParams.size()) {
        if (outError) *outError = "Incomplete checkpoint: loaded " + std::to_string(loadedParams) +
                                  " of " + std::to_string(allParams.size()) + " required tensors";
        return false;
    }

    if (opt && val.hasOptimizerState && in.good()) {
        opt->loadState(in);
    }

    return true;
}

} // namespace nexa
