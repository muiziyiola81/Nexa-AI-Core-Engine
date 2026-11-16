/**
 * NEXA Native Training Engine - AdamW Optimizer Implementation
 */

#include "../include/nexa/optimizer.hpp"
#include <cmath>
#include <iostream>

namespace nexa {

AdamW::AdamW(const std::vector<GradTensor*>& parameters, const AdamWConfig& cfg)
    : config(cfg), params(parameters) {
    for (auto* p : params) {
        if (!p->requiresGrad) continue;
        ParamState st;
        st.m = Tensor(p->value.shape(), DataType::FLOAT32);
        st.v = Tensor(p->value.shape(), DataType::FLOAT32);
        st.m.fillZero();
        st.v.fillZero();
        states[p->name] = std::move(st);
    }
}

void AdamW::zeroGrad() {
    for (auto* p : params) {
        p->zeroGrad();
    }
}

float AdamW::step() {
    return step(config.lr);
}

float AdamW::step(float currentLr) {
    if (params.empty()) return 0.0f;

    // 1. Compute pre-clipping global gradient norm
    float gradNorm = AutogradOps::computeGlobalGradNorm(params);

    // 2. Gradient clipping
    if (config.maxGradNorm > 0.0f) {
        AutogradOps::clipGradients(params, config.maxGradNorm);
    }

    stepCount++;
    const float beta1 = config.beta1;
    const float beta2 = config.beta2;
    const float eps = config.eps;
    const float lr = currentLr;
    const float wd = config.weightDecay;

    const float biasCorrection1 = 1.0f - std::pow(beta1, static_cast<float>(stepCount));
    const float biasCorrection2 = 1.0f - std::pow(beta2, static_cast<float>(stepCount));

    for (auto* p : params) {
        if (!p->requiresGrad) continue;

        auto it = states.find(p->name);
        if (it == states.end()) {
            ParamState st;
            st.m = Tensor(p->value.shape(), DataType::FLOAT32);
            st.v = Tensor(p->value.shape(), DataType::FLOAT32);
            st.m.fillZero();
            st.v.fillZero();
            states[p->name] = std::move(st);
            it = states.find(p->name);
        }

        ParamState& st = it->second;
        float* w = p->value.dataFloat();
        const float* g = p->grad.dataFloat();
        float* m = st.m.dataFloat();
        float* v = st.v.dataFloat();
        const int64_t n = p->value.numel();
        const bool applyWd = (p->value.rank() > 1) && (p->name.find("embed") == std::string::npos);
        const float paramWd = applyWd ? wd : 0.0f;

        for (int64_t i = 0; i < n; ++i) {
            float gradVal = g[i];

            // Moment updates
            m[i] = beta1 * m[i] + (1.0f - beta1) * gradVal;
            v[i] = beta2 * v[i] + (1.0f - beta2) * (gradVal * gradVal);

            // Bias-corrected estimates
            float mHat = m[i] / biasCorrection1;
            float vHat = v[i] / biasCorrection2;

            // Decoupled weight decay + Adam update
            float stepVal = (mHat / (std::sqrt(vHat) + eps)) + (paramWd * w[i]);
            w[i] -= lr * stepVal;
        }
    }

    return gradNorm;
}

void AdamW::saveState(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&stepCount), sizeof(uint64_t));
    uint32_t numStates = static_cast<uint32_t>(states.size());
    out.write(reinterpret_cast<const char*>(&numStates), sizeof(uint32_t));

    for (const auto& [name, st] : states) {
        uint32_t nameLen = static_cast<uint32_t>(name.size());
        out.write(reinterpret_cast<const char*>(&nameLen), sizeof(uint32_t));
        out.write(name.data(), nameLen);

        int64_t numel = st.m.numel();
        out.write(reinterpret_cast<const char*>(&numel), sizeof(int64_t));
        out.write(reinterpret_cast<const char*>(st.m.dataFloat()), numel * sizeof(float));
        out.write(reinterpret_cast<const char*>(st.v.dataFloat()), numel * sizeof(float));
    }
}

void AdamW::loadState(std::istream& in) {
    in.read(reinterpret_cast<char*>(&stepCount), sizeof(uint64_t));
    uint32_t numStates = 0;
    in.read(reinterpret_cast<char*>(&numStates), sizeof(uint32_t));

    for (uint32_t s = 0; s < numStates; ++s) {
        uint32_t nameLen = 0;
        in.read(reinterpret_cast<char*>(&nameLen), sizeof(uint32_t));
        std::string name(nameLen, '\0');
        in.read(&name[0], nameLen);

        int64_t numel = 0;
        in.read(reinterpret_cast<char*>(&numel), sizeof(int64_t));

        auto it = states.find(name);
        if (it != states.end()) {
            if (it->second.m.numel() == numel) {
                in.read(reinterpret_cast<char*>(it->second.m.dataFloat()), numel * sizeof(float));
                in.read(reinterpret_cast<char*>(it->second.v.dataFloat()), numel * sizeof(float));
            } else {
                // Shape mismatch, skip bytes
                in.seekg(numel * sizeof(float) * 2, std::ios::cur);
            }
        } else {
            // Unknown param, skip bytes
            in.seekg(numel * sizeof(float) * 2, std::ios::cur);
        }
    }
}

} // namespace nexa
