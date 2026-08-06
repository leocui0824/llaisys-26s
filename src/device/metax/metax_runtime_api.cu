#include "../runtime_api.hpp"

#include <mc_runtime.h>
#include <cstdlib>

namespace llaisys::device::metax {

namespace runtime_api {

int getDeviceCount() {
    int count = 0;
    mcGetDeviceCount(&count);
    return count;
}

void setDevice(int id) {
    mcSetDevice(id);
}

void deviceSynchronize() {
    mcDeviceSynchronize();
}

llaisysStream_t createStream() {
    mcStream_t stream = nullptr;
    mcStreamCreate(&stream);
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    mcStreamDestroy(reinterpret_cast<mcStream_t>(stream));
}

void streamSynchronize(llaisysStream_t stream) {
    mcStreamSynchronize(reinterpret_cast<mcStream_t>(stream));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    mcMalloc(&ptr, size);
    return ptr;
}

void freeDevice(void *ptr) {
    mcFree(ptr);
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    mcMallocHost(&ptr, size);
    return ptr;
}

void freeHost(void *ptr) {
    mcFreeHost(ptr);
}

static mcMemcpyKind mapMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2D:
        return mcMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return mcMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
    default:
        return mcMemcpyDeviceToDevice;
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    mcMemcpy(dst, src, size, mapMemcpyKind(kind));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind,
                 llaisysStream_t stream) {
    mcMemcpyAsync(dst, src, size, mapMemcpyKind(kind),
                  reinterpret_cast<mcStream_t>(stream));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::metax
