#include "qwen2.hpp"

#include "../../utils.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Meta &meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id), _cached_len(0) {

    core::context().setDevice(_device, _device_id);

    _in_embed = Tensor::create({meta.voc, meta.hs}, meta.dtype, _device, _device_id);
    _out_embed = Tensor::create({meta.voc, meta.hs}, meta.dtype, _device, _device_id);
    _out_norm_w = Tensor::create({meta.hs}, meta.dtype, _device, _device_id);

    _layers.resize(meta.nlayer);
    for (size_t i = 0; i < meta.nlayer; i++) {
        auto &l = _layers[i];
        l.attn_norm_w = Tensor::create({meta.hs}, meta.dtype, _device, _device_id);
        l.attn_q_w = Tensor::create({meta.nh * meta.dh, meta.hs}, meta.dtype, _device, _device_id);
        l.attn_q_b = Tensor::create({meta.nh * meta.dh}, meta.dtype, _device, _device_id);
        l.attn_k_w = Tensor::create({meta.nkvh * meta.dh, meta.hs}, meta.dtype, _device, _device_id);
        l.attn_k_b = Tensor::create({meta.nkvh * meta.dh}, meta.dtype, _device, _device_id);
        l.attn_v_w = Tensor::create({meta.nkvh * meta.dh, meta.hs}, meta.dtype, _device, _device_id);
        l.attn_v_b = Tensor::create({meta.nkvh * meta.dh}, meta.dtype, _device, _device_id);
        l.attn_o_w = Tensor::create({meta.hs, meta.nh * meta.dh}, meta.dtype, _device, _device_id);
        l.mlp_norm_w = Tensor::create({meta.hs}, meta.dtype, _device, _device_id);
        l.mlp_gate_w = Tensor::create({meta.di, meta.hs}, meta.dtype, _device, _device_id);
        l.mlp_up_w = Tensor::create({meta.di, meta.hs}, meta.dtype, _device, _device_id);
        l.mlp_down_w = Tensor::create({meta.hs, meta.di}, meta.dtype, _device, _device_id);
        l.k_cache = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, _device, _device_id);
        l.v_cache = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, _device, _device_id);
    }
}

int64_t Qwen2Model::forward(const int64_t *token_ids, size_t ntoken) {
    core::context().setDevice(_device, _device_id);
    auto &rt = core::context().runtime();

    size_t hs = _meta.hs, nh = _meta.nh, nkvh = _meta.nkvh, dh = _meta.dh;
    size_t di = _meta.di, voc = _meta.voc;

    // 1. Embedding
    auto tok_tensor = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    tok_tensor->load(token_ids);
    auto hidden = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
    ops::embedding(hidden, tok_tensor, _in_embed);

    // 2. Position IDs
    auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    {
        std::vector<int64_t> pos_host(ntoken);
        for (size_t i = 0; i < ntoken; i++)
            pos_host[i] = static_cast<int64_t>(_cached_len + i);
        pos_ids->load(pos_host.data());
    }

    // 3. Skip layers temporarily for debugging
    for (size_t l = 0; l < _meta.nlayer; l++) {
        auto &layer = _layers[l];

        auto attn_normed = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
        ops::rms_norm(attn_normed, hidden, layer.attn_norm_w, _meta.epsilon);

        auto q = Tensor::create({ntoken, nh * dh}, _meta.dtype, _device, _device_id);
        ops::linear(q, attn_normed, layer.attn_q_w, layer.attn_q_b);
        auto k = Tensor::create({ntoken, nkvh * dh}, _meta.dtype, _device, _device_id);
        ops::linear(k, attn_normed, layer.attn_k_w, layer.attn_k_b);
        auto v = Tensor::create({ntoken, nkvh * dh}, _meta.dtype, _device, _device_id);
        ops::linear(v, attn_normed, layer.attn_v_w, layer.attn_v_b);

        auto q_3d = q->view({ntoken, nh, dh});
        auto k_3d = k->view({ntoken, nkvh, dh});
        auto v_3d = v->view({ntoken, nkvh, dh});

        auto q_roped = Tensor::create({ntoken, nh, dh}, _meta.dtype, _device, _device_id);
        auto k_roped = Tensor::create({ntoken, nkvh, dh}, _meta.dtype, _device, _device_id);
        ops::rope(q_roped, q_3d, pos_ids, _meta.theta);
        ops::rope(k_roped, k_3d, pos_ids, _meta.theta);

        // Copy to KV cache
        {
            auto k_slice = layer.k_cache->slice(0, _cached_len, _cached_len + ntoken);
            auto v_slice = layer.v_cache->slice(0, _cached_len, _cached_len + ntoken);
            size_t nbytes = ntoken * nkvh * dh * utils::dsize(_meta.dtype);
            rt.api()->memcpy_sync(k_slice->data(), k_roped->data(), nbytes, LLAISYS_MEMCPY_D2D);
            rt.api()->memcpy_sync(v_slice->data(), v_3d->data(), nbytes, LLAISYS_MEMCPY_D2D);
        }

        size_t total_len = _cached_len + ntoken;
        auto k_all = layer.k_cache->slice(0, 0, total_len);
        auto v_all = layer.v_cache->slice(0, 0, total_len);

        auto attn_out = Tensor::create({ntoken, nh, dh}, _meta.dtype, _device, _device_id);
        float scale = 1.0f / std::sqrt(static_cast<float>(dh));
        ops::self_attention(attn_out, q_roped, k_all, v_all, scale);

        auto attn_2d = attn_out->view({ntoken, nh * dh});
        auto attn_proj = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
        ops::linear(attn_proj, attn_2d, layer.attn_o_w, nullptr);

        auto hidden_attn = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
        ops::add(hidden_attn, hidden, attn_proj);

        // MLP
        auto mlp_normed = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
        ops::rms_norm(mlp_normed, hidden_attn, layer.mlp_norm_w, _meta.epsilon);

        auto gate = Tensor::create({ntoken, di}, _meta.dtype, _device, _device_id);
        auto up = Tensor::create({ntoken, di}, _meta.dtype, _device, _device_id);
        ops::linear(gate, mlp_normed, layer.mlp_gate_w, nullptr);
        ops::linear(up, mlp_normed, layer.mlp_up_w, nullptr);

        auto activated = Tensor::create({ntoken, di}, _meta.dtype, _device, _device_id);
        ops::swiglu(activated, gate, up);

        auto mlp_out = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
        ops::linear(mlp_out, activated, layer.mlp_down_w, nullptr);

        auto new_hidden = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
        ops::add(new_hidden, hidden_attn, mlp_out);
        hidden = new_hidden;
    }

    // 4. Final RMS Norm
    auto final_normed = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
    ops::rms_norm(final_normed, hidden, _out_norm_w, _meta.epsilon);

    // 5. LM Head
    auto logits = Tensor::create({ntoken, voc}, _meta.dtype, _device, _device_id);
    ops::linear(logits, final_normed, _out_embed, nullptr);

    // 6. Argmax on last token
    auto last_logits_2d = logits->slice(0, ntoken - 1, ntoken);
    auto last_logits_1d = last_logits_2d->view({voc});
    auto next_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    auto next_val = Tensor::create({1}, _meta.dtype, _device, _device_id);
    ops::argmax(next_idx, next_val, last_logits_1d);

    // 7. Read result
    int64_t result = 0;
    rt.api()->memcpy_sync(&result, next_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    _cached_len += ntoken;
    return result;
}

} // namespace llaisys::models
