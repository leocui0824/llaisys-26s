#pragma once

#include <cuda_fp16.h>
#include "../utils.hpp"

template<typename U>
__device__ __forceinline__ float d2f(U v) { return static_cast<float>(v); }

template<>
__device__ __forceinline__ float d2f(llaisys::fp16_t v) {
    return __half2float(*reinterpret_cast<const __half*>(&v));
}
template<>
__device__ __forceinline__ float d2f(llaisys::bf16_t v) {
    uint16_t raw;
    __builtin_memcpy(&raw, &v, 2);
    uint32_t bits = static_cast<uint32_t>(raw) << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

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
    uint32_t bits;
    __builtin_memcpy(&bits, &v, 4);
    uint16_t raw = static_cast<uint16_t>(bits >> 16);
    llaisys::bf16_t result;
    __builtin_memcpy(&result, &raw, 2);
    return result;
}
