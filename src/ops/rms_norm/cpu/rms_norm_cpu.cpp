#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t M, size_t D, float eps) {
    for (size_t i = 0; i < M; i++) {
        float sq_sum = 0.0f;
        for (size_t j = 0; j < D; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float val = llaisys::utils::cast<float>(in[i * D + j]);
                sq_sum += val * val;
            } else {
                float val = static_cast<float>(in[i * D + j]);
                sq_sum += val * val;
            }
        }

        float r_rsqrt = 1.0f / std::sqrt(sq_sum / static_cast<float>(D) + eps);

        for (size_t j = 0; j < D; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float in_val = llaisys::utils::cast<float>(in[i * D + j]);
                float w_val = llaisys::utils::cast<float>(weight[j]);
                out[i * D + j] = llaisys::utils::cast<T>(in_val * r_rsqrt * w_val);
            } else {
                float in_val = static_cast<float>(in[i * D + j]);
                float w_val = static_cast<float>(weight[j]);
                out[i * D + j] = static_cast<T>(in_val * r_rsqrt * w_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t M, size_t D, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), M, D, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), M, D, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), M, D, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
