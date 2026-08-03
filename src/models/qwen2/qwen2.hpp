#pragma once

#include "../../core/llaisys_core.hpp"
#include "../../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models {

struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

struct Qwen2Layer {
    // Attention weights
    tensor_t attn_norm_w;
    tensor_t attn_q_w, attn_q_b;
    tensor_t attn_k_w, attn_k_b;
    tensor_t attn_v_w, attn_v_b;
    tensor_t attn_o_w;
    // MLP weights
    tensor_t mlp_norm_w;
    tensor_t mlp_gate_w, mlp_up_w, mlp_down_w;
    // KV cache
    tensor_t k_cache, v_cache;
};

class Qwen2Model {
public:
    Qwen2Model(const Qwen2Meta &meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model() = default;

    // Access weights for loading (const refs for bridge to wrap into LlaisysTensor)
    tensor_t &in_embed() { return _in_embed; }
    tensor_t &out_embed() { return _out_embed; }
    tensor_t &out_norm_w() { return _out_norm_w; }
    const std::vector<Qwen2Layer> &layers() const { return _layers; }

    // Forward pass: takes token_ids and returns the next predicted token
    int64_t forward(const int64_t *token_ids, size_t ntoken);

    const Qwen2Meta &meta() const { return _meta; }

private:
    Qwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    size_t _cached_len;

    tensor_t _in_embed, _out_embed, _out_norm_w;
    std::vector<Qwen2Layer> _layers;
};

} // namespace llaisys::models
