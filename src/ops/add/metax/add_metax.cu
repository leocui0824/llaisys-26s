#include "add_metax.cuh"
#include "../../metax_util.cuh"
#include <mc_runtime.h>

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float val = d2f(a[i]) + d2f(b[i]);
    c[i] = f2d<T>(val);
}

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t N) {
    size_t block_size = 256;
    size_t grid_size = (N + block_size - 1) / block_size;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b), N);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<llaisys::fp16_t *>(c),
            reinterpret_cast<const llaisys::fp16_t *>(a),
            reinterpret_cast<const llaisys::fp16_t *>(b), N);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<llaisys::bf16_t *>(c),
            reinterpret_cast<const llaisys::bf16_t *>(a),
            reinterpret_cast<const llaisys::bf16_t *>(b), N);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::nvidia
