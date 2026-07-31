#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DEVICE(out, pos_ids);

    ASSERT(in->ndim() == 3, "rope: in must be 3D tensor");
    ASSERT(in->isContiguous(), "rope: in must be contiguous");

    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    ASSERT(head_dim % 2 == 0, "rope: head_dim must be even");

    ASSERT(pos_ids->ndim() == 1, "rope: pos_ids must be 1D tensor");
    ASSERT(pos_ids->shape()[0] == seq_len, "rope: pos_ids.shape[0] must match in.shape[0]");
    ASSERT(pos_ids->isContiguous(), "rope: pos_ids must be contiguous");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be i64 dtype");

    ASSERT(out->ndim() == 3, "rope: out must be 3D tensor");
    ASSERT(out->shape()[0] == seq_len && out->shape()[1] == n_heads && out->shape()[2] == head_dim,
           "rope: out shape must match in shape");
    ASSERT(out->isContiguous(), "rope: out must be contiguous");
    ASSERT(out->dtype() == in->dtype(), "rope: out and in must have same dtype");

    // Get pos_ids data pointer
    auto *pid = reinterpret_cast<const int64_t *>(pos_ids->data());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pid, out->dtype(),
                         seq_len, n_heads, head_dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pid, out->dtype(),
                         seq_len, n_heads, head_dim, theta);
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
