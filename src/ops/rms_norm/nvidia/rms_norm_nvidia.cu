#include "rms_norm_nvidia.cuh"
#include "../../nvidia_util.cuh"
#include <cuda_runtime.h>
#include <cmath>

// Each block handles one row
template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                 size_t M, size_t D, float eps) {
    size_t row = blockIdx.x;
    if (row >= M) return;

    __shared__ float s_sq_sum[256];

    float sq_sum = 0.0f;
    for (size_t j = threadIdx.x; j < D; j += blockDim.x) {
        float v = d2f(in[row * D + j]);
        sq_sum += v * v;
    }

    // Block-level sum reduction
    size_t tid = threadIdx.x;
    s_sq_sum[tid] = sq_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sq_sum[tid] += s_sq_sum[tid + s];
        }
        __syncthreads();
    }

    float r_rsqrt = rsqrtf(s_sq_sum[0] / d2f(D) + eps);

    // Apply normalization
    for (size_t j = threadIdx.x; j < D; j += blockDim.x) {
        float in_val = d2f(in[row * D + j]);
        float w_val = d2f(weight[j]);
        out[row * D + j] = f2d<T>(in_val * r_rsqrt * w_val);
    }
}

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t M, size_t D, float eps) {
    size_t block_size = 256;
    size_t grid_size = M;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight), M, D, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight), M, D, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight), M, D, eps);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::nvidia
