/**
 * NEXA Native Runtime - Model Format Header
 * Binary specification and descriptor definitions for .nexa model weight files.
 */

#ifndef NEXA_MODEL_FORMAT_HPP
#define NEXA_MODEL_FORMAT_HPP

#include "tensor.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace nexa {

constexpr uint32_t NEXA_MAGIC = 0x4E455841; // 'N', 'E', 'X', 'A'
constexpr uint32_t NEXA_VERSION = 1;

#pragma pack(push, 1)
struct NexaFileHeader {
    uint32_t magic;           // Must match NEXA_MAGIC (0x4E455841)
    uint32_t version;         // Format version (currently 1)
    uint32_t archType;        // 1 = Transformer Decoder (Dense Llama-style)
    uint32_t vocabSize;
    uint32_t dim;
    uint32_t intermediateDim;
    uint32_t nLayers;
    uint32_t nHeads;
    uint32_t nKvHeads;
    uint32_t maxSeqLen;
    float normEps;
    float ropeTheta;
    uint32_t quantType;       // 0: FP32, 1: FP16, 2: INT8, 3: INT4
    uint32_t nTensors;        // Total tensors in weights table
};
#pragma pack(pop)

struct NexaTensorDescriptor {
    std::string name;
    DataType dtype;
    std::vector<int64_t> dims;
    uint64_t offset;
    uint64_t sizeBytes;
};

struct NexaModelMetadata {
    NexaFileHeader header;
    std::vector<NexaTensorDescriptor> tensors;
    uint64_t totalWeightBytes = 0;
    uint64_t estimatedRamBytes = 0;
    bool isValid = false;
};

} // namespace nexa

#endif // NEXA_MODEL_FORMAT_HPP
