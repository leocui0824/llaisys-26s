#include "rope_nvidia.cuh"
#include <cuda_runtime.h>
#include <cmath>

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                             size_t seq_len, size_t n_heads, size_t head_dim,
                             float theta) {
    size_t half = head_dim / 2;
    size_t row_size = n_heads * head_dim;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * n_heads * half;
    if (idx >= total) return;

    // Decompose linear index: idx = pos * (n_heads * half) + head * half + pair_i
    size_t pos = idx / (n_heads * half);
    size_t rem = idx % (n_heads * half);
    size_t head = rem / half;
    size_t i = rem % half;

    float pos_val = static_cast<float>(pos_ids[pos]);
    float freq = pos_val / powf(theta, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
    float cos_val = cosf(freq);
    float sin_val = sinf(freq);

    size_t base = pos * row_size + head * head_dim;
    size_t idx_a = base + i;
    size_t idx_b = base + half + i;

    float a = static_cast<float>(in[idx_a]);
    float b = static_cast<float>(in[idx_b]);

    out[idx_a] = static_cast<T>(a * cos_val - b * sin_val);
    out[idx_b] = static_cast<T>(b * cos_val + a * sin_val);
}

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t type, size_t seq_len, size_t n_heads,
          size_t head_dim, float theta) {
    size_t half = head_dim / 2;
    size_t total = seq_len * n_heads * half;
    size_t block_size = 256;
    size_t grid_size = (total + block_size - 1) / block_size;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            pos_ids, seq_len, n_heads, head_dim, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            pos_ids, seq_len, n_heads, head_dim, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            pos_ids, seq_len, n_heads, head_dim, theta);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::nvidia
