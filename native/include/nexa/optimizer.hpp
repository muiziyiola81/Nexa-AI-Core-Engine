/**
 * NEXA Native Training Engine - AdamW Optimizer Header
 * Implements decoupled AdamW optimization with first & second moment tracking,
 * bias correction, weight decay, gradient clipping, learning rate scheduling (warmup + cosine decay),
 * and state serialization.
 */

#ifndef NEXA_OPTIMIZER_HPP
#define NEXA_OPTIMIZER_HPP

#include "autograd.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <cmath>

namespace nexa {

struct AdamWConfig {
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weightDecay = 0.01f;
    float maxGradNorm = 1.0f;
};

struct LrScheduleConfig {
    float maxLr = 3e-3f;
    float minLr = 1e-4f;
    uint32_t warmupSteps = 10;
    uint32_t totalSteps = 100;

    float getLr(uint32_t currentStep) const {
        if (warmupSteps > 0 && currentStep < warmupSteps) {
            float progress = static_cast<float>(currentStep + 1) / static_cast<float>(warmupSteps);
            return minLr + (maxLr - minLr) * progress;
        }
        if (currentStep >= totalSteps) {
            return minLr;
        }
        uint32_t decaySteps = totalSteps - warmupSteps;
        if (decaySteps == 0) return minLr;

        float progress = static_cast<float>(currentStep - warmupSteps) / static_cast<float>(decaySteps);
        float cosineDecay = 0.5f * (1.0f + std::cos(progress * 3.14159265358979323846f));
        return minLr + (maxLr - minLr) * cosineDecay;
    }
};

struct ParamState {
    Tensor m; // First moment vector (same shape as param)
    Tensor v; // Second moment vector (same shape as param)
};

class AdamW {
public:
    AdamWConfig config;
    uint64_t stepCount = 0;
    std::vector<GradTensor*> params;
    std::unordered_map<std::string, ParamState> states;

    AdamW() = default;
    explicit AdamW(const std::vector<GradTensor*>& parameters, const AdamWConfig& cfg = AdamWConfig());

    void zeroGrad();
    float step();
    float step(float currentLr);

    // Checkpointing: save and load optimizer states (stepCount, m, v)
    void saveState(std::ostream& out) const;
    void loadState(std::istream& in);
};

} // namespace nexa

#endif // NEXA_OPTIMIZER_HPP
