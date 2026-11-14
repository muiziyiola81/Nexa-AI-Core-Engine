/**
 * NEXA Native Runtime - Tensor Computation Header
 * Defines Tensor representation, shapes, data types, aligned CPU memory allocation,
 * and high-performance vector/matrix math primitives.
 */

#ifndef NEXA_TENSOR_HPP
#define NEXA_TENSOR_HPP

#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace nexa {

enum class DataType : uint8_t {
    FLOAT32 = 0,
    FLOAT16 = 1,
    INT8    = 2,
    INT4    = 3
};

size_t getDataTypeSize(DataType dtype);
const char* getDataTypeName(DataType dtype);

class Shape {
public:
    std::vector<int64_t> dims;

    Shape() = default;
    Shape(std::initializer_list<int64_t> list) : dims(list) {}
    explicit Shape(const std::vector<int64_t>& d) : dims(d) {}

    size_t rank() const { return dims.size(); }
    int64_t operator[](size_t idx) const { return dims[idx]; }
    int64_t& operator[](size_t idx) { return dims[idx]; }

    int64_t numel() const {
        if (dims.empty()) return 0;
        int64_t total = 1;
        for (auto d : dims) total *= d;
        return total;
    }

    std::vector<int64_t> computeStrides() const {
        std::vector<int64_t> strides(dims.size());
        if (dims.empty()) return strides;
        strides.back() = 1;
        for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        return strides;
    }

    std::string toString() const;
    bool operator==(const Shape& other) const { return dims == other.dims; }
    bool operator!=(const Shape& other) const { return dims != other.dims; }
};

/**
 * 64-byte Cache-Line Aligned Memory Allocator
 * Ensures SIMD compatibility (AVX2, AVX-512, ARM NEON).
 */
class AlignedMemory {
public:
    void* rawPtr = nullptr;
    size_t allocatedBytes = 0;

    AlignedMemory() = default;
    explicit AlignedMemory(size_t bytes, size_t alignment = 64);
    ~AlignedMemory();

    // Move-only semantics to prevent double-free
    AlignedMemory(const AlignedMemory&) = delete;
    AlignedMemory& operator=(const AlignedMemory&) = delete;
    AlignedMemory(AlignedMemory&& other) noexcept;
    AlignedMemory& operator=(AlignedMemory&& other) noexcept;

    void* data() { return rawPtr; }
    const void* data() const { return rawPtr; }
};

class Tensor {
private:
    Shape shape_;
    DataType dtype_ = DataType::FLOAT32;
    std::shared_ptr<AlignedMemory> memory_;
    std::vector<int64_t> strides_;
    size_t offset_ = 0; // element offset for views/slices

public:
    Tensor() = default;
    Tensor(const Shape& shape, DataType dtype = DataType::FLOAT32);
    static Tensor createView(const Tensor& base, const Shape& newShape, size_t elementOffset = 0);

    const Shape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    int64_t numel() const { return shape_.numel(); }
    size_t rank() const { return shape_.rank(); }
    const std::vector<int64_t>& strides() const { return strides_; }

    float* dataFloat() {
        if (dtype_ != DataType::FLOAT32) throw std::runtime_error("Tensor is not FLOAT32");
        return static_cast<float*>(memory_->data()) + offset_;
    }

    const float* dataFloat() const {
        if (dtype_ != DataType::FLOAT32) throw std::runtime_error("Tensor is not FLOAT32");
        return static_cast<const float*>(memory_->data()) + offset_;
    }

    void* rawData() { return memory_ ? memory_->data() : nullptr; }
    const void* rawData() const { return memory_ ? memory_->data() : nullptr; }

    void fillZero();
    void fillConstant(float val);

    // Initial Core Tensor Math Operations
    static void matmul(const Tensor& A, const Tensor& B, Tensor& C);
    static void vecAdd(const Tensor& A, const Tensor& B, Tensor& out);
    static void vecMul(const Tensor& A, const Tensor& B, Tensor& out);
    static void vecScale(const Tensor& A, float scalar, Tensor& out);
    static float dotProduct(const float* a, const float* b, size_t n);
    static void softmax(const Tensor& input, Tensor& output, int axis = -1);
};

} // namespace nexa

#endif // NEXA_TENSOR_HPP
