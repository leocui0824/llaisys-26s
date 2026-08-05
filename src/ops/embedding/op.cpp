#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);

    // Validate index: 1D contiguous i64 tensor, 1 row per lookup
    ASSERT(index->ndim() == 1, "embedding: index must be 1D tensor");
    ASSERT(index->isContiguous(), "embedding: index must be contiguous");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be i64 dtype");

    // Validate weight: 2D contiguous table [vocab_size, embed_dim]
    ASSERT(weight->ndim() == 2, "embedding: weight must be 2D tensor");
    ASSERT(weight->isContiguous(), "embedding: weight must be contiguous");

    // Validate out: shape = [N, D]
    size_t N = index->shape()[0];
    size_t D = weight->shape()[1];
    ASSERT(out->ndim() == 2, "embedding: out must be 2D tensor");
    ASSERT(out->shape()[0] == N && out->shape()[1] == D,
           "embedding: out shape must be [index.shape[0], weight.shape[1]]");
    ASSERT(out->isContiguous(), "embedding: out must be contiguous");
    ASSERT(out->dtype() == weight->dtype(), "embedding: out and weight must have same dtype");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              weight->dtype(), N, D);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              weight->dtype(), N, D);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), weight->dtype(),
                                 index->shape()[0], weight->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
