import json
import re
from ctypes import POINTER, c_int, c_int64
from pathlib import Path
from typing import Sequence

import safetensors

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights


class Qwen2:
    def __init__(self, model_path: str, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # Read model config
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract hyperparameters
        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        nkvh = config["num_key_value_heads"]
        dh = hs // nh
        meta = LlaisysQwen2Meta(
            dtype=DataType.BF16,
            nlayer=config["num_hidden_layers"],
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=dh,
            di=config["intermediate_size"],
            maxseq=config["max_position_embeddings"],
            voc=config["vocab_size"],
            epsilon=config.get("rms_norm_eps", 1e-6),
            theta=config.get("rope_theta", 10000.0),
            end_token=config.get("eos_token_id", 151643),
        )
        self._meta = meta
        self._end_token = meta.end_token

        # Create C++ model
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            POINTER(LlaisysQwen2Meta)(meta),
            device.value,
            device_ids,
            1,
        )

        # Get weights struct
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._weights = weights_ptr.contents

        # Load weights from safetensors files
        self._load_weights(model_path)

    def _load_weights(self, model_path: Path):
        # Compile regex for layer matching
        layer_re = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

        # Map component names to weight struct fields
        component_map = {
            "input_layernorm.weight": "attn_norm_w",
            "self_attn.q_proj.weight": "attn_q_w",
            "self_attn.q_proj.bias": "attn_q_b",
            "self_attn.k_proj.weight": "attn_k_w",
            "self_attn.k_proj.bias": "attn_k_b",
            "self_attn.v_proj.weight": "attn_v_w",
            "self_attn.v_proj.bias": "attn_v_b",
            "self_attn.o_proj.weight": "attn_o_w",
            "post_attention_layernorm.weight": "mlp_norm_w",
            "mlp.gate_proj.weight": "mlp_gate_w",
            "mlp.up_proj.weight": "mlp_up_w",
            "mlp.down_proj.weight": "mlp_down_w",
        }

        # Global weight keys
        global_map = {
            "model.embed_tokens.weight": "in_embed",
            "lm_head.weight": "out_embed",
            "model.norm.weight": "out_norm_w",
        }

        safetensor_files = sorted(model_path.glob("*.safetensors"))
        for file in safetensor_files:
            with safetensors.safe_open(str(file), framework="pt") as sf:
                for key in sf.keys():
                    tensor = sf.get_tensor(key)
                    raw_ptr = tensor.data_ptr()

                    # Try global weights first
                    if key in global_map:
                        field = global_map[key]
                        handle = getattr(self._weights, field)
                        LIB_LLAISYS.tensorLoad(handle, raw_ptr)
                        continue

                    # Try per-layer weights
                    m = layer_re.match(key)
                    if m:
                        layer_idx = int(m.group(1))
                        component = m.group(2)
                        if component in component_map:
                            field = component_map[component]
                            arr = getattr(self._weights, field)
                            handle = arr[layer_idx]
                            LIB_LLAISYS.tensorLoad(handle, raw_ptr)
                            continue

                    # Unknown key - skip (e.g. __metadata__)
                    print(f"[Qwen2] Skipping unknown weight key: {key}")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        all_tokens = list(inputs)

        # Prefill: process all input tokens, get first generated token
        input_array = (c_int64 * len(all_tokens))(*all_tokens)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model, input_array, len(all_tokens)
        )
        all_tokens.append(next_token)

        # Autoregressive generation
        for _ in range(max_new_tokens - 1):
            if next_token == self._end_token:
                break
            token_array = (c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, token_array, 1
            )
            all_tokens.append(next_token)

        return all_tokens

    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
