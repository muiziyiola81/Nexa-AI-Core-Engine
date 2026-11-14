/**
 * NEXA Native Runtime - Model Loader Implementation
 */

#include "../include/nexa/model_loader.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace nexa {

NexaModelMetadata ModelLoader::inspectModel(const std::string& filePath) {
    NexaModelMetadata meta;
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        meta.isValid = false;
        return meta;
    }

    // Read fixed header
    file.read(reinterpret_cast<char*>(&meta.header), sizeof(NexaFileHeader));
    if (file.gcount() != sizeof(NexaFileHeader)) {
        meta.isValid = false;
        return meta;
    }

    // Verify magic bytes: 0x4E455841 ('NEXA')
    if (meta.header.magic != NEXA_MAGIC) {
        meta.isValid = false;
        return meta;
    }

    if (meta.header.version != 1 && meta.header.version != 2) {
        meta.isValid = false;
        return meta;
    }

    meta.tensors.reserve(meta.header.nTensors);
    uint64_t totalBytes = 0;

    if (meta.header.version == 1) {
        // Version 1: Descriptor table at start
        for (uint32_t i = 0; i < meta.header.nTensors; ++i) {
            uint32_t nameLen = 0;
            file.read(reinterpret_cast<char*>(&nameLen), sizeof(uint32_t));
            if (nameLen > 256) {
                meta.isValid = false;
                return meta;
            }

            std::string name(nameLen, '\0');
            file.read(&name[0], nameLen);

            uint32_t dtypeVal = 0;
            file.read(reinterpret_cast<char*>(&dtypeVal), sizeof(uint32_t));

            uint32_t nDims = 0;
            file.read(reinterpret_cast<char*>(&nDims), sizeof(uint32_t));

            std::vector<int64_t> dims(nDims);
            for (uint32_t d = 0; d < nDims; ++d) {
                uint64_t dimVal = 0;
                file.read(reinterpret_cast<char*>(&dimVal), sizeof(uint64_t));
                dims[d] = static_cast<int64_t>(dimVal);
            }

            uint64_t offset = 0;
            uint64_t sizeBytes = 0;
            file.read(reinterpret_cast<char*>(&offset), sizeof(uint64_t));
            file.read(reinterpret_cast<char*>(&sizeBytes), sizeof(uint64_t));

            NexaTensorDescriptor desc;
            desc.name = name;
            desc.dtype = static_cast<DataType>(dtypeVal);
            desc.dims = dims;
            desc.offset = offset;
            desc.sizeBytes = sizeBytes;

            totalBytes += sizeBytes;
            meta.tensors.push_back(desc);
        }
    } else {
        // Version 2: Fixed 128-byte header followed by tensors [name, rank, dims, numel, data]
        file.seekg(128, std::ios::beg);
        for (uint32_t i = 0; i < meta.header.nTensors; ++i) {
            if (file.eof() || file.fail()) break;
            uint32_t nameLen = 0;
            file.read(reinterpret_cast<char*>(&nameLen), sizeof(uint32_t));
            if (nameLen > 256) {
                meta.isValid = false;
                return meta;
            }

            std::string name(nameLen, '\0');
            file.read(&name[0], nameLen);

            uint32_t rank = 0;
            file.read(reinterpret_cast<char*>(&rank), sizeof(uint32_t));
            std::vector<int64_t> dims(rank);
            for (uint32_t d = 0; d < rank; ++d) {
                file.read(reinterpret_cast<char*>(&dims[d]), sizeof(int64_t));
            }

            int64_t numel = 0;
            file.read(reinterpret_cast<char*>(&numel), sizeof(int64_t));
            uint64_t sizeBytes = static_cast<uint64_t>(numel) * sizeof(float);
            uint64_t offset = file.tellg();

            NexaTensorDescriptor desc;
            desc.name = name;
            desc.dtype = DataType::FLOAT32;
            desc.dims = dims;
            desc.offset = offset;
            desc.sizeBytes = sizeBytes;

            totalBytes += sizeBytes;
            meta.tensors.push_back(desc);

            file.seekg(sizeBytes, std::ios::cur);
        }
    }

    meta.totalWeightBytes = totalBytes;
    meta.isValid = true;
    return meta;
}

MemoryRequirement ModelLoader::calculateMemoryRequirement(const NexaFileHeader& hp, uint64_t availableRam) {
    MemoryRequirement req;

    // 1. Model weights memory
    // Estimate: vocab * dim + layers * (4 * dim^2 + 3 * dim * interDim + 2 * dim)
    uint64_t headDim = hp.nHeads > 0 ? (hp.dim / hp.nHeads) : 64;
    uint64_t bytesPerFloat = 4; // FP32 default
    if (hp.quantType == 1) bytesPerFloat = 2; // FP16
    if (hp.quantType == 2) bytesPerFloat = 1; // INT8
    if (hp.quantType == 3) bytesPerFloat = 1; // INT4 (half-byte)

    uint64_t embedWeights = static_cast<uint64_t>(hp.vocabSize) * hp.dim * bytesPerFloat;
    uint64_t layerWeights = hp.nLayers * (
        (static_cast<uint64_t>(hp.dim) * hp.dim * 4) +          // Q, K, V, Out
        (static_cast<uint64_t>(hp.dim) * hp.intermediateDim * 3) // Gate, Up, Down
    ) * bytesPerFloat;

    req.weightBytes = embedWeights + layerWeights;

    // 2. KV Cache memory: 2 (K and V) * nLayers * nKvHeads * headDim * maxSeqLen * sizeof(float)
    req.kvCacheBytes = 2ULL * hp.nLayers * hp.nKvHeads * headDim * hp.maxSeqLen * sizeof(float);

    // 3. Activation working buffer (scratchpad): ~4 * maxSeqLen * dim * sizeof(float)
    req.activationBytes = 4ULL * hp.maxSeqLen * hp.dim * sizeof(float);

    req.totalRequiredBytes = req.weightBytes + req.kvCacheBytes + req.activationBytes;
    req.availableRamBytes = availableRam;
    req.fitsInAvailableRam = req.totalRequiredBytes <= availableRam;

    return req;
}

bool ModelLoader::validateArchitectureCompatibility(const NexaFileHeader& hp, std::string& errorMsg) {
    if (hp.dim == 0 || hp.nLayers == 0 || hp.nHeads == 0) {
        errorMsg = "Invalid model hyperparameters: dimension, layers, or heads is zero.";
        return false;
    }
    if (hp.dim % hp.nHeads != 0) {
        errorMsg = "Hidden dimension must be divisible by attention heads count.";
        return false;
    }
    if (hp.nKvHeads > hp.nHeads || hp.nHeads % hp.nKvHeads != 0) {
        errorMsg = "Query heads must be a multiple of KV heads (Grouped-Query Attention invariant).";
        return false;
    }
    return true;
}

} // namespace nexa
