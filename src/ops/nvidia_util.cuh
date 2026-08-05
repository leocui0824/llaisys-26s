#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../utils.hpp"

// Convert any value to float (works for int, size_t, fp16, bf16, float, etc.)
template<typename U>
__device__ __forceinline__ float d2f(U v) { return static_cast<float>(v); }

template<>
__device__ __forceinline__ float d2f(llaisys::fp16_t v) {
    return __half2float(*reinterpret_cast<const __half*>(&v));
}
template<>
__device__ __forceinline__ float d2f(llaisys::bf16_t v) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&v));
}

// Convert float to target type T (explicit specialization)
template<typename T>
__device__ __forceinline__ T f2d(float v) { return static_cast<T>(v); }

template<>
__device__ __forceinline__ float f2d<float>(float v) { return v; }
template<>
__device__ __forceinline__ llaisys::fp16_t f2d<llaisys::fp16_t>(float v) {
    __half h = __float2half(v);
    return *reinterpret_cast<const llaisys::fp16_t*>(&h);
}
template<>
__device__ __forceinline__ llaisys::bf16_t f2d<llaisys::bf16_t>(float v) {
    __nv_bfloat16 b = __float2bfloat16(v);
    return *reinterpret_cast<const llaisys::bf16_t*>(&b);
}
