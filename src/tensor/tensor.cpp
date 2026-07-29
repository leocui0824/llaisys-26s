#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    size_t ndim_ = ndim();
    if (ndim_ == 0) {
        return true;
    }
    if (_meta.strides[ndim_ - 1] != 1) {
        return false;
    }
    for (int64_t i = static_cast<int64_t>(ndim_) - 2; i >= 0; i--) {
        if (_meta.strides[i] != _meta.strides[i + 1] * static_cast<ptrdiff_t>(_meta.shape[i + 1])) {
            return false;
        }
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(order.size() == ndim(), "permute: order size must match ndim");
    std::vector<size_t> new_shape(ndim());
    std::vector<ptrdiff_t> new_strides(ndim());
    for (size_t i = 0; i < ndim(); i++) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &new_shape) const {
    // Validate total element count
    size_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    CHECK_ARGUMENT(new_numel == numel(), "view: total element count mismatch");
    size_t new_ndim = new_shape.size();

    // Contiguous case: simply recompute strides
    if (isContiguous()) {
        std::vector<ptrdiff_t> new_strides(new_ndim);
        ptrdiff_t stride = 1;
        for (int64_t i = static_cast<int64_t>(new_ndim) - 1; i >= 0; i--) {
            new_strides[i] = stride;
            stride *= static_cast<ptrdiff_t>(new_shape[i]);
        }
        TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
        return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, _offset));
    }

    // Non-contiguous case: right-to-left merge/split algorithm
    int64_t oi = static_cast<int64_t>(ndim()) - 1;
    int64_t ni = static_cast<int64_t>(new_ndim) - 1;
    size_t old_chunk = _meta.shape[oi];
    size_t new_chunk = new_shape[ni];
    std::vector<ptrdiff_t> new_strides(new_ndim, 0);
    new_strides[ni] = _meta.strides[oi];

    while (oi > 0 || ni > 0) {
        if (old_chunk == new_chunk) {
            oi--; ni--;
            CHECK_ARGUMENT(oi >= 0 && ni >= 0, "view: shape not compatible");
            old_chunk = _meta.shape[oi];
            new_chunk = new_shape[ni];
            new_strides[ni] = _meta.strides[oi];
        } else if (old_chunk < new_chunk) {
            // Merge: absorb one more old dimension
            oi--;
            CHECK_ARGUMENT(oi >= 0, "view: cannot merge more old dimensions");
            CHECK_ARGUMENT(
                _meta.strides[oi] == _meta.strides[oi + 1] * static_cast<ptrdiff_t>(_meta.shape[oi + 1]),
                "view: old dimensions not contiguous in memory");
            old_chunk *= _meta.shape[oi];
            // new_strides[ni] stays as the rightmost stride in this merged group
        } else {
            // Split: absorb one more new dimension
            CHECK_ARGUMENT(old_chunk % new_chunk == 0, "view: cannot split dimension evenly");
            old_chunk /= new_chunk;
            ni--;
            CHECK_ARGUMENT(ni >= 0, "view: not enough new dimensions");
            new_chunk = new_shape[ni];
            new_strides[ni] = new_strides[ni + 1] * static_cast<ptrdiff_t>(new_shape[ni + 1]);
        }
    }
    CHECK_ARGUMENT(old_chunk == new_chunk, "view: shape not compatible after dimension matching");
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < ndim(), "slice dim out of range");
    CHECK_ARGUMENT(start < end, "slice start must be less than end");
    CHECK_ARGUMENT(end <= _meta.shape[dim], "slice end out of range");

    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;

    TensorMeta new_meta{_meta.dtype, new_shape, _meta.strides};

    size_t new_offset = _offset + start * _meta.strides[dim] * elementSize();

    return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, new_offset));
}

void Tensor::load(const void *src_) {
    // Switch to the tensor's device, then copy data from host to this tensor
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->memcpy_sync(
        this->data(),                            // dst: tensor's device memory
        const_cast<void *>(src_),                // src: host memory pointer
        this->numel() * this->elementSize(),     // total bytes to copy
        LLAISYS_MEMCPY_H2D);                     // direction: Host → Device
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
