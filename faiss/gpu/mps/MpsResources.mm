// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <Metal/Metal.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/mps/MpsStream.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <limits>

namespace faiss {
namespace gpu {

namespace {
constexpr int kNumStreams = 2;

void waitOnStream(cudaStream_t stream) {
    if (!stream) {
        return;
    }
    auto* mpsStream = reinterpret_cast<MpsStream*>(stream);
    id<MTLCommandBuffer> buffer = [mpsStream->queue commandBuffer];
    [buffer commit];
    [buffer waitUntilCompleted];
}

void* alignedAlloc(size_t alignment, size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
}
} // namespace

std::string allocTypeToString(AllocType t) {
    switch (t) {
        case AllocType::Other:
            return "Other";
        case AllocType::FlatData:
            return "FlatData";
        case AllocType::IVFLists:
            return "IVFLists";
        case AllocType::Quantizer:
            return "Quantizer";
        case AllocType::QuantizerPrecomputedCodes:
            return "QuantizerPrecomputedCodes";
        case AllocType::TemporaryMemoryBuffer:
            return "TemporaryMemoryBuffer";
        case AllocType::TemporaryMemoryOverflow:
            return "TemporaryMemoryOverflow";
        default:
            return "Unknown";
    }
}

std::string memorySpaceToString(MemorySpace s) {
    switch (s) {
        case MemorySpace::Temporary:
            return "Temporary";
        case MemorySpace::Device:
            return "Device";
        case MemorySpace::Unified:
            return "Unified";
        default:
            return "Unknown";
    }
}

std::string AllocInfo::toString() const {
    std::stringstream ss;
    ss << "type " << allocTypeToString(type) << " dev " << device << " space "
       << memorySpaceToString(space) << " stream " << (void*)stream;
    return ss.str();
}

std::string AllocRequest::toString() const {
    std::stringstream ss;
    ss << AllocInfo::toString() << " size " << size << " bytes";
    return ss.str();
}

AllocInfo makeDevAlloc(AllocType at, cudaStream_t st) {
    return AllocInfo(at, getCurrentDevice(), MemorySpace::Device, st);
}

AllocInfo makeTempAlloc(AllocType at, cudaStream_t st) {
    return AllocInfo(at, getCurrentDevice(), MemorySpace::Temporary, st);
}

AllocInfo makeSpaceAlloc(AllocType at, MemorySpace sp, cudaStream_t st) {
    return AllocInfo(at, getCurrentDevice(), sp, st);
}

//
// GpuMemoryReservation
//

GpuMemoryReservation::GpuMemoryReservation()
        : res(nullptr), device(0), stream(nullptr), data(nullptr), size(0) {}

GpuMemoryReservation::GpuMemoryReservation(
        GpuResources* r,
        int dev,
        cudaStream_t str,
        void* p,
        size_t sz)
        : res(r), device(dev), stream(str), data(p), size(sz) {}

GpuMemoryReservation::GpuMemoryReservation(GpuMemoryReservation&& m) noexcept {
    res = m.res;
    m.res = nullptr;
    device = m.device;
    m.device = 0;
    stream = m.stream;
    m.stream = nullptr;
    data = m.data;
    m.data = nullptr;
    size = m.size;
    m.size = 0;
}

GpuMemoryReservation& GpuMemoryReservation::operator=(
        GpuMemoryReservation&& m) {
    FAISS_ASSERT(
            !(res && res == m.res && device == m.device && data == m.data));
    release();
    res = m.res;
    m.res = nullptr;
    device = m.device;
    m.device = 0;
    stream = m.stream;
    m.stream = nullptr;
    data = m.data;
    m.data = nullptr;
    size = m.size;
    m.size = 0;
    return *this;
}

void GpuMemoryReservation::release() {
    if (res) {
        res->deallocMemory(device, data);
        res = nullptr;
        device = 0;
        stream = nullptr;
        data = nullptr;
        size = 0;
    }
}

GpuMemoryReservation::~GpuMemoryReservation() {
    if (res) {
        res->deallocMemory(device, data);
    }
}

//
// GpuResources (MPS implementation)
//

GpuResources::~GpuResources() = default;

bool GpuResources::supportsBFloat16CurrentDevice() {
    return supportsBFloat16(getCurrentDevice());
}

cublasHandle_t GpuResources::getBlasHandleCurrentDevice() {
    return getBlasHandle(getCurrentDevice());
}

cudaStream_t GpuResources::getDefaultStreamCurrentDevice() {
    return getDefaultStream(getCurrentDevice());
}

std::vector<cudaStream_t> GpuResources::getAlternateStreamsCurrentDevice() {
    return getAlternateStreams(getCurrentDevice());
}

cudaStream_t GpuResources::getAsyncCopyStreamCurrentDevice() {
    return getAsyncCopyStream(getCurrentDevice());
}

void GpuResources::syncDefaultStream(int device) {
    waitOnStream(getDefaultStream(device));
}

void GpuResources::syncDefaultStreamCurrentDevice() {
    syncDefaultStream(getCurrentDevice());
}

GpuMemoryReservation GpuResources::allocMemoryHandle(const AllocRequest& req) {
    return GpuMemoryReservation(
            this, req.device, req.stream, allocMemory(req), req.size);
}

size_t GpuResources::getTempMemoryAvailableCurrentDevice() const {
    return getTempMemoryAvailable(getCurrentDevice());
}

GpuResourcesProvider::~GpuResourcesProvider() = default;

GpuResourcesProviderFromInstance::GpuResourcesProviderFromInstance(
        std::shared_ptr<GpuResources> p)
        : res_(std::move(p)) {}

GpuResourcesProviderFromInstance::~GpuResourcesProviderFromInstance() = default;

std::shared_ptr<GpuResources> GpuResourcesProviderFromInstance::getResources() {
    return res_;
}

//
// StandardGpuResourcesImpl (MPS)
//

StandardGpuResourcesImpl::StandardGpuResourcesImpl()
        : pinnedMemAlloc_(nullptr),
          pinnedMemAllocSize_(0),
          tempMemSize_(0),
          pinnedMemSize_(0),
          allocLogging_(false) {}

StandardGpuResourcesImpl::~StandardGpuResourcesImpl() {
    for (auto& entry : allocs_) {
        FAISS_ASSERT_MSG(
                entry.second.empty(),
                "MPS resources destroyed with allocations outstanding");
    }

    for (auto& entry : defaultStreams_) {
        auto* stream = reinterpret_cast<MpsStream*>(entry.second);
        delete stream;
    }
    for (auto& entry : alternateStreams_) {
        for (auto stream : entry.second) {
            delete reinterpret_cast<MpsStream*>(stream);
        }
    }
    for (auto& entry : asyncCopyStreams_) {
        delete reinterpret_cast<MpsStream*>(entry.second);
    }

    if (pinnedMemAlloc_) {
        std::free(pinnedMemAlloc_);
    }
}

bool StandardGpuResourcesImpl::supportsBFloat16(int device) {
    (void)device;
    return true;
}

void StandardGpuResourcesImpl::noTempMemory() {
    tempMemSize_ = 0;
}

void StandardGpuResourcesImpl::setTempMemory(size_t size) {
    tempMemSize_ = size;
}

void StandardGpuResourcesImpl::setPinnedMemory(size_t size) {
    pinnedMemSize_ = size;
}

void StandardGpuResourcesImpl::setDefaultStream(int device, cudaStream_t stream) {
    userDefaultStreams_[device] = stream;
}

void StandardGpuResourcesImpl::revertDefaultStream(int device) {
    userDefaultStreams_.erase(device);
}

cudaStream_t StandardGpuResourcesImpl::getDefaultStream(int device) {
    auto it = userDefaultStreams_.find(device);
    if (it != userDefaultStreams_.end()) {
        return it->second;
    }
    initializeForDevice(device);
    return defaultStreams_.at(device);
}

void StandardGpuResourcesImpl::setDefaultNullStreamAllDevices() {
    userDefaultStreams_.clear();
}

void StandardGpuResourcesImpl::setLogMemoryAllocations(bool enable) {
    allocLogging_ = enable;
}

void StandardGpuResourcesImpl::initializeForDevice(int device) {
    if (defaultStreams_.count(device) != 0) {
        return;
    }

    auto devices = MTLCopyAllDevices();
    FAISS_ASSERT(device >= 0 && device < devices.count);
    id<MTLDevice> dev = devices[device];

    auto* defaultStream = new MpsStream{[dev newCommandQueue]};
    defaultStreams_[device] = reinterpret_cast<cudaStream_t>(defaultStream);

    auto* asyncStream = new MpsStream{[dev newCommandQueue]};
    asyncCopyStreams_[device] = reinterpret_cast<cudaStream_t>(asyncStream);

    std::vector<cudaStream_t> deviceStreams;
    for (int i = 0; i < kNumStreams; ++i) {
        auto* stream = new MpsStream{[dev newCommandQueue]};
        deviceStreams.push_back(reinterpret_cast<cudaStream_t>(stream));
    }
    alternateStreams_[device] = std::move(deviceStreams);

    if (pinnedMemAlloc_ == nullptr && pinnedMemSize_ > 0) {
        pinnedMemAlloc_ = alignedAlloc(64, pinnedMemSize_);
        pinnedMemAllocSize_ = pinnedMemSize_;
    }
}

cublasHandle_t StandardGpuResourcesImpl::getBlasHandle(int device) {
    (void)device;
    return nullptr;
}

std::vector<cudaStream_t> StandardGpuResourcesImpl::getAlternateStreams(
        int device) {
    initializeForDevice(device);
    return alternateStreams_.at(device);
}

void* StandardGpuResourcesImpl::allocMemory(const AllocRequest& req) {
    FAISS_ASSERT(req.size > 0);
    auto* ptr = alignedAlloc(16, req.size);
    FAISS_THROW_IF_NOT_MSG(ptr, "MPS allocMemory failed");
    allocs_[req.device][ptr] = req;
    return ptr;
}

void StandardGpuResourcesImpl::deallocMemory(int device, void* in) {
    if (!in) {
        return;
    }
    auto it = allocs_.find(device);
    if (it != allocs_.end()) {
        it->second.erase(in);
    }
    std::free(in);
}

size_t StandardGpuResourcesImpl::getTempMemoryAvailable(int device) const {
    (void)device;
    return tempMemSize_;
}

std::map<int, std::map<std::string, std::pair<int, size_t>>>
StandardGpuResourcesImpl::getMemoryInfo() const {
    std::map<int, std::map<std::string, std::pair<int, size_t>>> info;
    for (auto& entry : allocs_) {
        size_t total = 0;
        int count = 0;
        for (auto& alloc : entry.second) {
            total += alloc.second.size;
            ++count;
        }
        info[entry.first]["all"] = std::make_pair(count, total);
    }
    return info;
}

std::pair<void*, size_t> StandardGpuResourcesImpl::getPinnedMemory() {
    if (!pinnedMemAlloc_ || pinnedMemAllocSize_ == 0) {
        return std::make_pair(nullptr, 0);
    }
    return std::make_pair(pinnedMemAlloc_, pinnedMemAllocSize_);
}

cudaStream_t StandardGpuResourcesImpl::getAsyncCopyStream(int device) {
    initializeForDevice(device);
    return asyncCopyStreams_.at(device);
}

bool StandardGpuResourcesImpl::isInitialized(int device) const {
    return defaultStreams_.count(device) != 0;
}

size_t StandardGpuResourcesImpl::getDefaultTempMemForGPU(
        int device,
        size_t requested) {
    (void)device;
    return requested;
}

//
// StandardGpuResources (MPS)
//

StandardGpuResources::StandardGpuResources()
        : res_(new StandardGpuResourcesImpl) {}

StandardGpuResources::~StandardGpuResources() = default;

std::shared_ptr<GpuResources> StandardGpuResources::getResources() {
    return res_;
}

bool StandardGpuResources::supportsBFloat16(int device) {
    return res_->supportsBFloat16(device);
}

bool StandardGpuResources::supportsBFloat16CurrentDevice() {
    return res_->supportsBFloat16(getCurrentDevice());
}

void StandardGpuResources::noTempMemory() {
    res_->noTempMemory();
}

void StandardGpuResources::setTempMemory(size_t size) {
    res_->setTempMemory(size);
}

void StandardGpuResources::setPinnedMemory(size_t size) {
    res_->setPinnedMemory(size);
}

void StandardGpuResources::setDefaultStream(int device, cudaStream_t stream) {
    res_->setDefaultStream(device, stream);
}

void StandardGpuResources::revertDefaultStream(int device) {
    res_->revertDefaultStream(device);
}

cudaStream_t StandardGpuResources::getDefaultStream(int device) {
    return res_->getDefaultStream(device);
}

size_t StandardGpuResources::getTempMemoryAvailable(int device) const {
    return res_->getTempMemoryAvailable(device);
}

void StandardGpuResources::syncDefaultStreamCurrentDevice() {
    res_->syncDefaultStreamCurrentDevice();
}

void StandardGpuResources::setDefaultNullStreamAllDevices() {
    res_->setDefaultNullStreamAllDevices();
}

void StandardGpuResources::setLogMemoryAllocations(bool enable) {
    res_->setLogMemoryAllocations(enable);
}

std::map<int, std::map<std::string, std::pair<int, size_t>>>
StandardGpuResources::getMemoryInfo() const {
    return res_->getMemoryInfo();
}

} // namespace gpu
} // namespace faiss
