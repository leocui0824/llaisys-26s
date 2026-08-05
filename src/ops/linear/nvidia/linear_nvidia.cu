#include "linear_nvidia.cuh"
#include <cuda_runtime.h>

#define TILE_SIZE 16

template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                               size_t M, size_t N, size_t K) {
    __shared__ T s_in[TILE_SIZE][TILE_SIZE];
    __shared__ T s_weight[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (size_t t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Cooperative load of in tile
        size_t k = t * TILE_SIZE + threadIdx.x;
        if (row < M && k < K) {
            s_in[threadIdx.y][threadIdx.x] = in[row * K + k];
        } else {
            s_in[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        // Cooperative load of weight tile (weight[j][k] at index j*K+k)
        size_t w_row = t * TILE_SIZE + threadIdx.y;
        if (col < N && w_row < K) {
            s_weight[threadIdx.y][threadIdx.x] = weight[col * K + w_row];
        } else {
            s_weight[threadIdx.y][threadIdx.x] = static_cast<T>(0);
        }
        __syncthreads();

        for (size_t kk = 0; kk < TILE_SIZE; kk++) {
            sum += static_cast<float>(s_in[threadIdx.y][kk]) *
                   static_cast<float>(s_weight[kk][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        if (bias != nullptr) {
            sum += static_cast<float>(bias[col]);
        }
        out[row * N + col] = static_cast<T>(sum);
    }
}

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type,
            size_t M, size_t N, size_t K) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_kernel<<<grid, block>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias), M, N, K);
        break;
    case LLAISYS_DTYPE_F16:
        linear_kernel<<<grid, block>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            reinterpret_cast<const llaisys::fp16_t *>(bias), M, N, K);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_kernel<<<grid, block>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            reinterpret_cast<const llaisys::bf16_t *>(bias), M, N, K);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::nvidia
