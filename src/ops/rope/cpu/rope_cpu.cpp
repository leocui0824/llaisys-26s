#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;
    size_t row_size = n_heads * head_dim; // elements per position: H * D

    for (size_t pos = 0; pos < seq_len; pos++) {
        float pos_val = static_cast<float>(pos_ids[pos]);

        for (size_t i = 0; i < half_dim; i++) {
            // Compute rotation angle for this dimension pair
            float freq = pos_val / std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
            float cos_val = std::cos(freq);
            float sin_val = std::sin(freq);

            // Apply same rotation to all heads at this position
            for (size_t head = 0; head < n_heads; head++) {
                size_t base = pos * row_size + head * head_dim;
                size_t idx_a = base + i;               // first half element
                size_t idx_b = base + half_dim + i;     // second half element

                // Read a and b, cast to float
                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in[idx_a]);
                    b = llaisys::utils::cast<float>(in[idx_b]);
                } else {
                    a = static_cast<float>(in[idx_a]);
                    b = static_cast<float>(in[idx_b]);
                }

                // 2D rotation: [a'] = [cos -sin] [a]
                //              [b'] = [sin  cos] [b]
                float out_a = a * cos_val - b * sin_val;
                float out_b = b * cos_val + a * sin_val;

                // Write back
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[idx_a] = llaisys::utils::cast<T>(out_a);
                    out[idx_b] = llaisys::utils::cast<T>(out_b);
                } else {
                    out[idx_a] = static_cast<T>(out_a);
                    out[idx_b] = static_cast<T>(out_b);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t type, size_t seq_len, size_t n_heads,
          size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in),
                     pos_ids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in),
                     pos_ids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in),
                     pos_ids, seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
