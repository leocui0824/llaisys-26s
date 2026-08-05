#include "embedding_nvidia.cuh"
#include "../../../utils.hpp"
#include <cuda_runtime.h>

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t N, size_t D) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * D) return;

    size_t row = i / D;
    size_t col = i % D;
    int64_t idx = index[row];
    out[i] = weight[idx * D + col];
}

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t N, size_t D) {
    size_t total = N * D;
    size_t block_size = 256;
    size_t grid_size = (total + block_size - 1) / block_size;
    auto *idx = reinterpret_cast<const int64_t *>(index);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<float *>(out), idx,
            reinterpret_cast<const float *>(weight), N, D);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<llaisys::fp16_t *>(out), idx,
            reinterpret_cast<const llaisys::fp16_t *>(weight), N, D);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<llaisys::bf16_t *>(out), idx,
            reinterpret_cast<const llaisys::bf16_t *>(weight), N, D);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::nvidia
