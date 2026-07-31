#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);

    // Validate input: 2D [M, K]
    ASSERT(in->ndim() == 2, "linear: in must be 2D tensor");
    ASSERT(in->isContiguous(), "linear: in must be contiguous");

    // Validate weight: 2D [N, K]
    ASSERT(weight->ndim() == 2, "linear: weight must be 2D tensor");
    ASSERT(weight->isContiguous(), "linear: weight must be contiguous");

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];
    ASSERT(weight->shape()[1] == K, "linear: weight.shape[1] must match in.shape[1]");

    // Validate out: 2D [M, N]
    ASSERT(out->ndim() == 2, "linear: out must be 2D tensor");
    ASSERT(out->shape()[0] == M && out->shape()[1] == N,
           "linear: out shape must be [in.shape[0], weight.shape[0]]");
    ASSERT(out->isContiguous(), "linear: out must be contiguous");

    // Validate dtype consistency
    ASSERT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
           "linear: out, in, and weight must have same dtype");

    // Validate bias if present
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        ASSERT(bias->ndim() == 1, "linear: bias must be 1D tensor");
        ASSERT(bias->shape()[0] == N, "linear: bias.shape[0] must match weight.shape[0]");
        ASSERT(bias->isContiguous(), "linear: bias must be contiguous");
        ASSERT(bias->dtype() == out->dtype(), "linear: bias dtype must match out dtype");
    }

    // Get bias data pointer (nullptr if no bias)
    const void *bias_data = bias ? static_cast<const void *>(bias->data()) : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(),
                           static_cast<const std::byte *>(bias_data),
                           out->dtype(), M, N, K);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                           static_cast<const std::byte *>(bias_data),
                           out->dtype(), M, N, K);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
