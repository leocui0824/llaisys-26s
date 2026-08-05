#include "argmax_nvidia.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstring>

// Block-level argmax reduction kernel
// Each block finds the max value+index in its chunk, writes to global partial arrays
template <typename T>
__global__ void argmax_reduce_kernel(const T *vals, size_t N,
                                      float *block_vals, int64_t *block_idxs) {
    __shared__ float s_vals[256];
    __shared__ int64_t s_idxs[256];

    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;

    float val = (gid < N) ? static_cast<float>(vals[gid]) : -FLT_MAX;
    int64_t idx = (gid < N) ? static_cast<int64_t>(gid) : -1;

    s_vals[tid] = val;
    s_idxs[tid] = idx;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = s_vals[tid + s];
            if (s_vals[tid] < other) {
                s_vals[tid] = other;
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = s_vals[0];
        block_idxs[blockIdx.x] = s_idxs[0];
    }
}

// Final reduction: 1 block reduces all block-level results
__global__ void argmax_final_kernel(const float *block_vals, const int64_t *block_idxs,
                                     size_t num_blocks,
                                     float *out_val, int64_t *out_idx) {
    __shared__ float s_vals[256];
    __shared__ int64_t s_idxs[256];

    size_t tid = threadIdx.x;
    float val = (tid < num_blocks) ? block_vals[tid] : -FLT_MAX;
    int64_t idx = (tid < num_blocks) ? block_idxs[tid] : -1;

    s_vals[tid] = val;
    s_idxs[tid] = idx;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = s_vals[tid + s];
            if (s_vals[tid] < other) {
                s_vals[tid] = other;
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_val = s_vals[0];
        *out_idx = s_idxs[0];
    }
}

template <typename T>
static void launch_argmax(const T *d_vals, size_t N,
                          std::byte *d_max_idx, std::byte *d_max_val,
                          cudaStream_t stream = nullptr) {
    size_t block_size = 256;
    size_t grid_size = (N + block_size - 1) / block_size;

    // Allocate temp memory for per-block partials
    float *d_partial_vals;
    int64_t *d_partial_idxs;
    cudaMalloc(&d_partial_vals, grid_size * sizeof(float));
    cudaMalloc(&d_partial_idxs, grid_size * sizeof(int64_t));

    // Step 1: per-block reduction
    argmax_reduce_kernel<<<grid_size, block_size, 0, stream>>>(
        d_vals, N, d_partial_vals, d_partial_idxs);

    // Step 2: final reduction (single block)
    argmax_final_kernel<<<1, 256, 0, stream>>>(
        d_partial_vals, d_partial_idxs, grid_size,
        reinterpret_cast<float *>(d_max_val),
        reinterpret_cast<int64_t *>(d_max_idx));

    cudaFree(d_partial_vals);
    cudaFree(d_partial_idxs);
}

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t N) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_argmax(reinterpret_cast<const float *>(vals), N, max_idx, max_val);
        break;
    case LLAISYS_DTYPE_F16:
        launch_argmax(reinterpret_cast<const llaisys::fp16_t *>(vals), N, max_idx, max_val);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_argmax(reinterpret_cast<const llaisys::bf16_t *>(vals), N, max_idx, max_val);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::nvidia
