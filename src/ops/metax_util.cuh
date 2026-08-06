#pragma once

#include <mc_runtime.h>
#include "../utils.hpp"

// Device-side type conversion for Metax MACA platform
// Uses manual bit manipulation for portability (no CUDA intrinsics)

template<typename U>
__device__ __forceinline__ float d2f(U v) { return static_cast<float>(v); }

template<>
__device__ __forceinline__ float d2f(llaisys::fp16_t v) {
    // IEEE 754 half → float via bit manipulation
    uint16_t raw;
    __builtin_memcpy(&raw, &v, 2);
    uint16_t sign = (raw >> 15) & 1;
    uint16_t exp  = (raw >> 10) & 0x1f;
    uint16_t mant = raw & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        // Zero or subnormal
        if (mant == 0) {
            bits = sign << 31;
        } else {
            // Subnormal: normalize
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        // Infinity or NaN
        bits = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        // Normal
        bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
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

// Convert float to target type
template<typename T>
__device__ __forceinline__ T f2d(float v) { return static_cast<T>(v); }

template<>
__device__ __forceinline__ float f2d<float>(float v) { return v; }

template<>
__device__ __forceinline__ llaisys::fp16_t f2d<llaisys::fp16_t>(float v) {
    // float → IEEE 754 half via bit manipulation (truncation)
    uint32_t bits;
    __builtin_memcpy(&bits, &v, 4);
    uint16_t sign = (bits >> 31) & 1;
    uint32_t exp  = (bits >> 23) & 0xff;
    uint32_t mant = bits & 0x7fffff;
    uint16_t raw;
    if (exp == 0) {
        raw = sign << 15;
    } else if (exp < 113) {
        // Subnormal
        mant |= 0x800000;
        int shift = 113 - static_cast<int>(exp);
        mant >>= shift;
        raw = (sign << 15) | static_cast<uint16_t>(mant & 0x3ff);
    } else if (exp > 142) {
        // Overflow → infinity
        raw = (sign << 15) | (0x1f << 10);
    } else {
        raw = (sign << 15) | (static_cast<uint16_t>(exp - 112) << 10) | static_cast<uint16_t>((mant >> 13) & 0x3ff);
    }
    llaisys::fp16_t result;
    __builtin_memcpy(&result, &raw, 2);
    return result;
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
