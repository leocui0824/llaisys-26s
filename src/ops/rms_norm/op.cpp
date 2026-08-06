#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/rms_norm_metax.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);

    // Validate input: 2D [M, D]
    ASSERT(in->ndim() == 2, "rms_norm: in must be 2D tensor");
    ASSERT(in->isContiguous(), "rms_norm: in must be contiguous");

    size_t M = in->shape()[0];
    size_t D = in->shape()[1];

    // Validate weight: 1D [D]
    ASSERT(weight->ndim() == 1, "rms_norm: weight must be 1D tensor");
    ASSERT(weight->shape()[0] == D, "rms_norm: weight.shape[0] must match in.shape[1]");
    ASSERT(weight->isContiguous(), "rms_norm: weight must be contiguous");

    // Validate out: 2D [M, D], same shape as in
    ASSERT(out->ndim() == 2, "rms_norm: out must be 2D tensor");
    ASSERT(out->shape()[0] == M && out->shape()[1] == D,
           "rms_norm: out shape must match in shape");
    ASSERT(out->isContiguous(), "rms_norm: out must be contiguous");

    // Validate dtype consistency
    ASSERT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
           "rms_norm: out, in, and weight must have same dtype");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), M, D, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), M, D, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        size_t M_d = in->shape()[0];
        size_t D_d = in->shape()[1];
        return nvidia::rms_norm(out->data(), in->data(), weight->data(),
                                out->dtype(), M_d, D_d, eps);
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
