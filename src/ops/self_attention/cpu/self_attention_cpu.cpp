#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                     size_t hd, float scale) {
    size_t n_rep = nh / nkvh;
    size_t q_stride_hd = nh * hd;      // elements per query position: nh * hd
    size_t k_stride_hd = nkvh * hd;    // elements per key position: nkvh * hd

    // Temporary buffer for attention scores per query position
    std::vector<float> scores(kvlen);

    for (size_t q_head = 0; q_head < nh; q_head++) {
        size_t kv_head = q_head / n_rep;  // GQA: map query head to kv head

        for (size_t i = 0; i < qlen; i++) {
            float max_score = -std::numeric_limits<float>::infinity();

            // Step 1: Compute dot products Q[i][q_head] · K[j][kv_head]
            for (size_t j = 0; j < kvlen; j++) {
                // Causal mask: j > i + (kvlen - qlen) → -inf
                if (static_cast<int64_t>(j) > static_cast<int64_t>(i) + static_cast<int64_t>(kvlen - qlen)) {
                    max_score = 0.0f;  // at least one non-masked entry exists
                    scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float dot = 0.0f;
                for (size_t d = 0; d < hd; d++) {
                    size_t q_idx = i * q_stride_hd + q_head * hd + d;
                    size_t k_idx = j * k_stride_hd + kv_head * hd + d;

                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        dot += llaisys::utils::cast<float>(q[q_idx]) *
                               llaisys::utils::cast<float>(k[k_idx]);
                    } else {
                        dot += static_cast<float>(q[q_idx]) *
                               static_cast<float>(k[k_idx]);
                    }
                }
                scores[j] = dot * scale;
                if (scores[j] > max_score) {
                    max_score = scores[j];
                }
            }

            // Step 2: Softmax (numerically stable)
            float sum = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                if (scores[j] == -std::numeric_limits<float>::infinity()) {
                    scores[j] = 0.0f;
                } else {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum += scores[j];
                }
            }
            for (size_t j = 0; j < kvlen; j++) {
                scores[j] /= sum;
            }

            // Step 3: Weighted sum over V
            for (size_t d = 0; d < hd; d++) {
                float val = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    size_t v_idx = j * k_stride_hd + kv_head * hd + d;

                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        val += scores[j] * llaisys::utils::cast<float>(v[v_idx]);
                    } else {
                        val += scores[j] * static_cast<float>(v[v_idx]);
                    }
                }

                size_t out_idx = i * q_stride_hd + q_head * hd + d;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[out_idx] = llaisys::utils::cast<T>(val);
                } else {
                    out[out_idx] = static_cast<T>(val);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type,
                    size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                    size_t hd, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(out),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(out),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(out),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
