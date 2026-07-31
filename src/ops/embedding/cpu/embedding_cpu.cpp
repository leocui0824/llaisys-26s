#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t N, size_t D) {
    for (size_t i = 0; i < N; i++) {
        int64_t row = index[i];
        // Copy weight[row] (D elements) to out[i]
        std::memcpy(out + i * D, weight + row * D, D * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t N, size_t D) {
    const int64_t *idx = reinterpret_cast<const int64_t *>(index);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), idx,
                          reinterpret_cast<const float *>(weight), N, D);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), idx,
                          reinterpret_cast<const llaisys::bf16_t *>(weight), N, D);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), idx,
                          reinterpret_cast<const llaisys::fp16_t *>(weight), N, D);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
