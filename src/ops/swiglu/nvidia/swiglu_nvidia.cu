#include "swiglu_nvidia.cuh"
#include "../../nvidia_util.cuh"
#include <cuda_runtime.h>
#include <cmath>

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float g = d2f(gate[i]);
    float u = d2f(up[i]);
    float silu = g / (1.0f + expf(-g));
    out[i] = f2d<T>(u * silu);
}

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t N) {
    size_t block_size = 256;
    size_t grid_size = (N + block_size - 1) / block_size;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up), N);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(gate),
            reinterpret_cast<const llaisys::fp16_t *>(up), N);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(gate),
            reinterpret_cast<const llaisys::bf16_t *>(up), N);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::nvidia
