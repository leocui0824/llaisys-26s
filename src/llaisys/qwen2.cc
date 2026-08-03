#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../models/qwen2/qwen2.hpp"

#include <cstring>
#include <vector>

// Internal model struct holding C++ model + C API weights struct
struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model *model;

    // The weights struct returned to Python
    LlaisysQwen2Weights weights;

    // LlaisysTensor wrappers for scalar weights
    LlaisysTensor in_embed_wrap;
    LlaisysTensor out_embed_wrap;
    LlaisysTensor out_norm_wrap;

    // Per-layer LlaisysTensor arrays (owned here, exposed via pointers in weights struct)
    std::vector<LlaisysTensor> attn_norm_w_arr;
    std::vector<LlaisysTensor> attn_q_w_arr;
    std::vector<LlaisysTensor> attn_q_b_arr;
    std::vector<LlaisysTensor> attn_k_w_arr;
    std::vector<LlaisysTensor> attn_k_b_arr;
    std::vector<LlaisysTensor> attn_v_w_arr;
    std::vector<LlaisysTensor> attn_v_b_arr;
    std::vector<LlaisysTensor> attn_o_w_arr;
    std::vector<LlaisysTensor> mlp_norm_w_arr;
    std::vector<LlaisysTensor> mlp_gate_w_arr;
    std::vector<LlaisysTensor> mlp_up_w_arr;
    std::vector<LlaisysTensor> mlp_down_w_arr;

    // Per-layer raw pointers (for the weights struct's pointer arrays)
    std::vector<llaisysTensor_t> attn_norm_w_ptrs;
    std::vector<llaisysTensor_t> attn_q_w_ptrs;
    std::vector<llaisysTensor_t> attn_q_b_ptrs;
    std::vector<llaisysTensor_t> attn_k_w_ptrs;
    std::vector<llaisysTensor_t> attn_k_b_ptrs;
    std::vector<llaisysTensor_t> attn_v_w_ptrs;
    std::vector<llaisysTensor_t> attn_v_b_ptrs;
    std::vector<llaisysTensor_t> attn_o_w_ptrs;
    std::vector<llaisysTensor_t> mlp_norm_w_ptrs;
    std::vector<llaisysTensor_t> mlp_gate_w_ptrs;
    std::vector<llaisysTensor_t> mlp_up_w_ptrs;
    std::vector<llaisysTensor_t> mlp_down_w_ptrs;

    LlaisysQwen2Model(const llaisys::models::Qwen2Meta &meta,
                      llaisysDeviceType_t device, int *device_ids, int ndevice)
        : model(nullptr) {

        int device_id = (ndevice > 0) ? device_ids[0] : 0;
        model = new llaisys::models::Qwen2Model(meta, device, device_id);

        size_t nlayer = meta.nlayer;

        // Resize all arrays
        attn_norm_w_arr.resize(nlayer);
        attn_q_w_arr.resize(nlayer);
        attn_q_b_arr.resize(nlayer);
        attn_k_w_arr.resize(nlayer);
        attn_k_b_arr.resize(nlayer);
        attn_v_w_arr.resize(nlayer);
        attn_v_b_arr.resize(nlayer);
        attn_o_w_arr.resize(nlayer);
        mlp_norm_w_arr.resize(nlayer);
        mlp_gate_w_arr.resize(nlayer);
        mlp_up_w_arr.resize(nlayer);
        mlp_down_w_arr.resize(nlayer);

        attn_norm_w_ptrs.resize(nlayer);
        attn_q_w_ptrs.resize(nlayer);
        attn_q_b_ptrs.resize(nlayer);
        attn_k_w_ptrs.resize(nlayer);
        attn_k_b_ptrs.resize(nlayer);
        attn_v_w_ptrs.resize(nlayer);
        attn_v_b_ptrs.resize(nlayer);
        attn_o_w_ptrs.resize(nlayer);
        mlp_norm_w_ptrs.resize(nlayer);
        mlp_gate_w_ptrs.resize(nlayer);
        mlp_up_w_ptrs.resize(nlayer);
        mlp_down_w_ptrs.resize(nlayer);

        // Scalar weights
        in_embed_wrap.tensor = model->in_embed();
        out_embed_wrap.tensor = model->out_embed();
        out_norm_wrap.tensor = model->out_norm_w();

        // Per-layer weights
        const auto &layers = model->layers();
        for (size_t i = 0; i < nlayer; i++) {
            attn_norm_w_arr[i].tensor = layers[i].attn_norm_w;
            attn_q_w_arr[i].tensor = layers[i].attn_q_w;
            attn_q_b_arr[i].tensor = layers[i].attn_q_b;
            attn_k_w_arr[i].tensor = layers[i].attn_k_w;
            attn_k_b_arr[i].tensor = layers[i].attn_k_b;
            attn_v_w_arr[i].tensor = layers[i].attn_v_w;
            attn_v_b_arr[i].tensor = layers[i].attn_v_b;
            attn_o_w_arr[i].tensor = layers[i].attn_o_w;
            mlp_norm_w_arr[i].tensor = layers[i].mlp_norm_w;
            mlp_gate_w_arr[i].tensor = layers[i].mlp_gate_w;
            mlp_up_w_arr[i].tensor = layers[i].mlp_up_w;
            mlp_down_w_arr[i].tensor = layers[i].mlp_down_w;

            attn_norm_w_ptrs[i] = &attn_norm_w_arr[i];
            attn_q_w_ptrs[i] = &attn_q_w_arr[i];
            attn_q_b_ptrs[i] = &attn_q_b_arr[i];
            attn_k_w_ptrs[i] = &attn_k_w_arr[i];
            attn_k_b_ptrs[i] = &attn_k_b_arr[i];
            attn_v_w_ptrs[i] = &attn_v_w_arr[i];
            attn_v_b_ptrs[i] = &attn_v_b_arr[i];
            attn_o_w_ptrs[i] = &attn_o_w_arr[i];
            mlp_norm_w_ptrs[i] = &mlp_norm_w_arr[i];
            mlp_gate_w_ptrs[i] = &mlp_gate_w_arr[i];
            mlp_up_w_ptrs[i] = &mlp_up_w_arr[i];
            mlp_down_w_ptrs[i] = &mlp_down_w_arr[i];
        }

        // Fill the weights struct
        std::memset(&weights, 0, sizeof(weights));
        weights.in_embed = &in_embed_wrap;
        weights.out_embed = &out_embed_wrap;
        weights.out_norm_w = &out_norm_wrap;
        weights.attn_norm_w = attn_norm_w_ptrs.data();
        weights.attn_q_w = attn_q_w_ptrs.data();
        weights.attn_q_b = attn_q_b_ptrs.data();
        weights.attn_k_w = attn_k_w_ptrs.data();
        weights.attn_k_b = attn_k_b_ptrs.data();
        weights.attn_v_w = attn_v_w_ptrs.data();
        weights.attn_v_b = attn_v_b_ptrs.data();
        weights.attn_o_w = attn_o_w_ptrs.data();
        weights.mlp_norm_w = mlp_norm_w_ptrs.data();
        weights.mlp_gate_w = mlp_gate_w_ptrs.data();
        weights.mlp_up_w = mlp_up_w_ptrs.data();
        weights.mlp_down_w = mlp_down_w_ptrs.data();
    }

    ~LlaisysQwen2Model() {
        delete model;
    }
};

// ============================================================
// C API
// ============================================================

extern "C" {

LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                            llaisysDeviceType_t device,
                                            int *device_ids,
                                            int ndevice) {
    llaisys::models::Qwen2Meta cpp_meta;
    cpp_meta.dtype = meta->dtype;
    cpp_meta.nlayer = meta->nlayer;
    cpp_meta.hs = meta->hs;
    cpp_meta.nh = meta->nh;
    cpp_meta.nkvh = meta->nkvh;
    cpp_meta.dh = meta->dh;
    cpp_meta.di = meta->di;
    cpp_meta.maxseq = meta->maxseq;
    cpp_meta.voc = meta->voc;
    cpp_meta.epsilon = meta->epsilon;
    cpp_meta.theta = meta->theta;
    cpp_meta.end_token = meta->end_token;

    return new LlaisysQwen2Model(cpp_meta, device, device_ids, ndevice);
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *m) {
    delete m;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *m) {
    return &m->weights;
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *m, int64_t *token_ids, size_t ntoken) {
    return m->model->forward(token_ids, ntoken);
}

} // extern "C"
