/**
 * NEXA Native Runtime - Tensor Implementation
 */

#include "../include/nexa/tensor.hpp"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sstream>
#include <algorithm>

namespace nexa {

size_t getDataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT16: return 2;
        case DataType::INT8:    return 1;
        case DataType::INT4:    return 1; // packed
        default: return 4;
    }
}

const char* getDataTypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "FLOAT32";
        case DataType::FLOAT16: return "FLOAT16";
        case DataType::INT8:    return "INT8";
        case DataType::INT4:    return "INT4";
        default: return "UNKNOWN";
    }
}

std::string Shape::toString() const {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        oss << dims[i];
        if (i + 1 < dims.size()) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

AlignedMemory::AlignedMemory(size_t bytes, size_t alignment) : allocatedBytes(bytes) {
    if (bytes == 0) {
        rawPtr = nullptr;
        return;
    }
    // POSIX aligned memory allocation
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        throw std::bad_alloc();
    }
    rawPtr = ptr;
    std::memset(rawPtr, 0, bytes);
}

AlignedMemory::~AlignedMemory() {
    if (rawPtr) {
        std::free(rawPtr);
        rawPtr = nullptr;
    }
}

AlignedMemory::AlignedMemory(AlignedMemory&& other) noexcept
    : rawPtr(other.rawPtr), allocatedBytes(other.allocatedBytes) {
    other.rawPtr = nullptr;
    other.allocatedBytes = 0;
}

AlignedMemory& AlignedMemory::operator=(AlignedMemory&& other) noexcept {
    if (this != &other) {
        if (rawPtr) std::free(rawPtr);
        rawPtr = other.rawPtr;
        allocatedBytes = other.allocatedBytes;
        other.rawPtr = nullptr;
        other.allocatedBytes = 0;
    }
    return *this;
}

Tensor::Tensor(const Shape& shape, DataType dtype)
    : shape_(shape), dtype_(dtype), strides_(shape.computeStrides()) {
    size_t bytes = static_cast<size_t>(shape_.numel()) * getDataTypeSize(dtype_);
    memory_ = std::make_shared<AlignedMemory>(bytes, 64);
}

Tensor Tensor::createView(const Tensor& base, const Shape& newShape, size_t elementOffset) {
    Tensor view;
    view.shape_ = newShape;
    view.dtype_ = base.dtype_;
    view.memory_ = base.memory_;
    view.strides_ = newShape.computeStrides();
    view.offset_ = base.offset_ + elementOffset;
    return view;
}

void Tensor::fillZero() {
    if (memory_ && memory_->data()) {
        std::memset(dataFloat(), 0, static_cast<size_t>(numel()) * sizeof(float));
    }
}

void Tensor::fillConstant(float val) {
    float* ptr = dataFloat();
    int64_t n = numel();
    for (int64_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
}

void Tensor::matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.rank() != 2 || B.rank() != 2 || C.rank() != 2) {
        throw std::invalid_argument("matmul expects 2D tensors");
    }
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t B_K = B.shape()[0];
    int64_t N = B.shape()[1];

    if (K != B_K) {
        throw std::invalid_argument("Inner matrix dimensions do not match for matmul");
    }
    if (C.shape()[0] != M || C.shape()[1] != N) {
        throw std::invalid_argument("Output matrix shape does not match expected (M, N)");
    }

    const float* aData = A.dataFloat();
    const float* bData = B.dataFloat();
    float* cData = C.dataFloat();

    // Cache-blocked Matrix Multiplication for high-throughput CPU cache residency
    constexpr int64_t BLOCK_SIZE = 64;
    C.fillZero();

    for (int64_t ii = 0; ii < M; ii += BLOCK_SIZE) {
        int64_t iMax = std::min(M, ii + BLOCK_SIZE);
        for (int64_t kk = 0; kk < K; kk += BLOCK_SIZE) {
            int64_t kMax = std::min(K, kk + BLOCK_SIZE);
            for (int64_t jj = 0; jj < N; jj += BLOCK_SIZE) {
                int64_t jMax = std::min(N, jj + BLOCK_SIZE);

                for (int64_t i = ii; i < iMax; ++i) {
                    for (int64_t k = kk; k < kMax; ++k) {
                        float aVal = aData[i * K + k];
                        for (int64_t j = jj; j < jMax; ++j) {
                            cData[i * N + j] += aVal * bData[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

void Tensor::vecAdd(const Tensor& A, const Tensor& B, Tensor& out) {
    int64_t n = A.numel();
    if (B.numel() != n || out.numel() != n) {
        throw std::invalid_argument("Dimension mismatch in vecAdd");
    }
    const float* a = A.dataFloat();
    const float* b = B.dataFloat();
    float* o = out.dataFloat();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] + b[i];
    }
}

void Tensor::vecMul(const Tensor& A, const Tensor& B, Tensor& out) {
    int64_t n = A.numel();
    if (B.numel() != n || out.numel() != n) {
        throw std::invalid_argument("Dimension mismatch in vecMul");
    }
    const float* a = A.dataFloat();
    const float* b = B.dataFloat();
    float* o = out.dataFloat();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] * b[i];
    }
}

void Tensor::vecScale(const Tensor& A, float scalar, Tensor& out) {
    int64_t n = A.numel();
    if (out.numel() != n) {
        throw std::invalid_argument("Dimension mismatch in vecScale");
    }
    const float* a = A.dataFloat();
    float* o = out.dataFloat();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] * scalar;
    }
}

float Tensor::dotProduct(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void Tensor::softmax(const Tensor& input, Tensor& output, int axis) {
    if (input.shape() != output.shape()) {
        throw std::invalid_argument("Shape mismatch in softmax");
    }
    const float* in = input.dataFloat();
    float* out = output.dataFloat();
    int64_t total = input.numel();
    int64_t lastDim = input.shape()[input.rank() - 1];
    int64_t batches = total / lastDim;

    for (int64_t b = 0; b < batches; ++b) {
        const float* inRow = in + b * lastDim;
        float* outRow = out + b * lastDim;

        // 1. Numerically stable max extraction
        float maxVal = inRow[0];
        for (int64_t i = 1; i < lastDim; ++i) {
            if (inRow[i] > maxVal) maxVal = inRow[i];
        }

        // 2. Exponential and accumulation
        float sum = 0.0f;
        for (int64_t i = 0; i < lastDim; ++i) {
            float expVal = std::exp(inRow[i] - maxVal);
            outRow[i] = expVal;
            sum += expVal;
        }

        // 3. Normalization
        float invSum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int64_t i = 0; i < lastDim; ++i) {
            outRow[i] *= invSum;
        }
    }
}

} // namespace nexa
