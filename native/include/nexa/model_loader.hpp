/**
 * NEXA Native Runtime - Model Loader Header
 * Inspects .nexa model files on disk, validates headers, and checks memory budgets.
 */

#ifndef NEXA_MODEL_LOADER_HPP
#define NEXA_MODEL_LOADER_HPP

#include "model_format.hpp"
#include <string>
#include <memory>
#include <unordered_map>

namespace nexa {

struct MemoryRequirement {
    uint64_t weightBytes;
    uint64_t kvCacheBytes;
    uint64_t activationBytes;
    uint64_t totalRequiredBytes;
    bool fitsInAvailableRam;
    uint64_t availableRamBytes;
};

class ModelLoader {
public:
    static NexaModelMetadata inspectModel(const std::string& filePath);
    static MemoryRequirement calculateMemoryRequirement(const NexaFileHeader& header, uint64_t availableRam);
    static bool validateArchitectureCompatibility(const NexaFileHeader& header, std::string& errorMsg);

    static std::unordered_map<std::string, Tensor> loadTensors(
        const std::string& filePath,
        const NexaModelMetadata& meta
    );
};

} // namespace nexa

#endif // NEXA_MODEL_LOADER_HPP
