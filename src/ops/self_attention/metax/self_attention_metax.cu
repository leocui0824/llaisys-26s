#include "self_attention_metax.cuh"
#include "../../metax_util.cuh"
#include <mc_runtime.h>
#include <cmath>
#include <cfloat>

template <typename T>
__global__ void self_attention_kernel(
    T *out, const T *q, const T *k, const T *v,
    size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {

    __shared__ float s_scores[1024];
    size_t n_rep = nh / nkvh;
    size_t q_stride = nh * hd;
    size_t k_stride = nkvh * hd;
    size_t q_head = blockIdx.y;
    size_t q_pos = blockIdx.x;
    if (q_head >= nh || q_pos >= qlen) return;
    size_t kv_head = q_head / n_rep;
    size_t tid = threadIdx.x;

    // Thread 0 computes all scores into shared memory
    if (tid == 0) {
        float m = -FLT_MAX;
        for (size_t j = 0; j < kvlen; j++) {
            if (static_cast<int64_t>(j) > static_cast<int64_t>(q_pos) +
                static_cast<int64_t>(kvlen - qlen)) {
                s_scores[j] = -FLT_MAX;
            } else {
                float dot = 0.0f;
                for (size_t dm = 0; dm < hd; dm++) {
                    size_t q_idx = q_pos * q_stride + q_head * hd + dm;
                    size_t k_idx = j * k_stride + kv_head * hd + dm;
                    dot += d2f(q[q_idx]) * d2f(k[k_idx]);
                }
                s_scores[j] = dot * scale;
            }
            if (s_scores[j] > m) m = s_scores[j];
        }
        float sum = 0.0f;
        for (size_t j = 0; j < kvlen; j++) {
            if (s_scores[j] != -FLT_MAX) {
                s_scores[j] = expf(s_scores[j] - m);
                sum += s_scores[j];
            } else {
                s_scores[j] = 0.0f;
            }
        }
        for (size_t j = 0; j < kvlen; j++) {
            s_scores[j] /= sum;
        }
    }
    __syncthreads();

    // All threads compute weighted V sum for their output dims
    for (size_t d = tid; d < hd; d += blockDim.x) {
        float val = 0.0f;
        for (size_t j = 0; j < kvlen; j++) {
            size_t v_idx = j * k_stride + kv_head * hd + d;
            val += s_scores[j] * d2f(v[v_idx]);
        }
        size_t out_idx = q_pos * q_stride + q_head * hd + d;
        out[out_idx] = f2d<T>(val);
    }
}

namespace llaisys::ops::nvidia {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type,
                    size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                    size_t hd, float scale) {
    dim3 grid(qlen, nh);
    dim3 block(256);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<<<grid, block>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            qlen, kvlen, nh, nkvh, hd, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<<<grid, block>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            qlen, kvlen, nh, nkvh, hd, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<<<grid, block>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            qlen, kvlen, nh, nkvh, hd, scale);
        break;
    default:
        break;
    }
}
} // namespace llaisys::ops::nvidia
