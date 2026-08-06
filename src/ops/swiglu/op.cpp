#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/swiglu_metax.cuh"
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);

    // Validate shapes: gate, up, out must all have same shape
    ASSERT(gate->ndim() == up->ndim(),
           "swiglu: gate and up must have same number of dimensions");
    for (size_t d = 0; d < gate->ndim(); d++) {
        ASSERT(gate->shape()[d] == up->shape()[d],
               "swiglu: gate and up must have same shape");
    }
    ASSERT(out->ndim() == gate->ndim(),
           "swiglu: out must have same number of dimensions as gate");
    for (size_t d = 0; d < out->ndim(); d++) {
        ASSERT(out->shape()[d] == gate->shape()[d],
               "swiglu: out must have same shape as gate");
    }

    // Validate contiguity and dtypes
    ASSERT(gate->isContiguous(), "swiglu: gate must be contiguous");
    ASSERT(up->isContiguous(), "swiglu: up must be contiguous");
    ASSERT(out->isContiguous(), "swiglu: out must be contiguous");
    ASSERT(out->dtype() == gate->dtype() && out->dtype() == up->dtype(),
           "swiglu: out, gate, and up must have same dtype");

    size_t numel = gate->numel();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                           out->dtype(), numel);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                           out->dtype(), numel);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
