/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <Metal/Metal.h>
#include <faiss/gpu/mps/MpsStream.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <mutex>
#include <unordered_map>

namespace faiss {
namespace gpu {

namespace {
thread_local int gCurrentDevice = 0;

cudaDeviceProp makeDeviceProp(id<MTLDevice> device) {
    cudaDeviceProp prop{};
    prop.major = 1;
    prop.minor = 0;
    prop.warpSize = 32;
    prop.maxThreadsPerBlock = 1024;
    prop.maxGridSize[0] = 2147483647;
    prop.maxGridSize[1] = 65535;
    prop.maxGridSize[2] = 65535;
    prop.sharedMemPerBlock = 0;
    prop.totalGlobalMem = static_cast<size_t>(device.recommendedMaxWorkingSetSize);
    return prop;
}

std::vector<id<MTLDevice>> getMpsDevices() {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    std::vector<id<MTLDevice>> result;
    result.reserve(devices.count);
    for (id<MTLDevice> dev in devices) {
        result.push_back(dev);
    }
    return result;
}

} // namespace

void streamWaitMps(
        const std::vector<cudaStream_t>& waiting,
        const std::vector<cudaStream_t>& waitOn) {
    (void)waiting;
    for (auto stream : waitOn) {
        if (!stream) {
            continue;
        }
        auto* mpsStream = reinterpret_cast<MpsStream*>(stream);
        id<MTLCommandBuffer> buffer = [mpsStream->queue commandBuffer];
        [buffer commit];
        [buffer waitUntilCompleted];
    }
}

int getCurrentDevice() {
    return gCurrentDevice;
}

void setCurrentDevice(int device) {
    FAISS_ASSERT(device >= 0);
    gCurrentDevice = device;
}

int getNumDevices() {
    auto devices = getMpsDevices();
    return static_cast<int>(devices.size());
}

void profilerStart() {
    // No-op for MPS.
}

void profilerStop() {
    // No-op for MPS.
}

void synchronizeAllDevices() {
    // MPS operations are synchronized per command buffer. This is a no-op.
}

const cudaDeviceProp& getDeviceProperties(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, cudaDeviceProp> properties;

    std::lock_guard<std::mutex> guard(mutex);
    auto it = properties.find(device);
    if (it == properties.end()) {
        auto devices = getMpsDevices();
        FAISS_ASSERT(device >= 0 && device < static_cast<int>(devices.size()));
        properties.emplace(device, makeDeviceProp(devices[device]));
        it = properties.find(device);
    }
    return it->second;
}

const cudaDeviceProp& getCurrentDeviceProperties() {
    return getDeviceProperties(getCurrentDevice());
}

int getMaxThreads(int device) {
    return getDeviceProperties(device).maxThreadsPerBlock;
}

int getMaxThreadsCurrentDevice() {
    return getMaxThreads(getCurrentDevice());
}

dim3 getMaxGrid(int device) {
    auto& prop = getDeviceProperties(device);
    return dim3(prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

dim3 getMaxGridCurrentDevice() {
    return getMaxGrid(getCurrentDevice());
}

size_t getMaxSharedMemPerBlock(int device) {
    return getDeviceProperties(device).sharedMemPerBlock;
}

size_t getMaxSharedMemPerBlockCurrentDevice() {
    return getMaxSharedMemPerBlock(getCurrentDevice());
}

int getDeviceForAddress(const void* p) {
    (void)p;
    return -1;
}

bool getFullUnifiedMemSupport(int device) {
    (void)device;
    return true;
}

bool getFullUnifiedMemSupportCurrentDevice() {
    return getFullUnifiedMemSupport(getCurrentDevice());
}

bool getTensorCoreSupport(int device) {
    (void)device;
    return false;
}

bool getTensorCoreSupportCurrentDevice() {
    return getTensorCoreSupport(getCurrentDevice());
}

int getWarpSize(int device) {
    return getDeviceProperties(device).warpSize;
}

int getWarpSizeCurrentDevice() {
    return getWarpSize(getCurrentDevice());
}

size_t getFreeMemory(int device) {
    auto devices = getMpsDevices();
    FAISS_ASSERT(device >= 0 && device < static_cast<int>(devices.size()));
    id<MTLDevice> dev = devices[device];
    if ([dev respondsToSelector:@selector(currentAllocatedSize)]) {
        size_t allocated = static_cast<size_t>(dev.currentAllocatedSize);
        size_t total = static_cast<size_t>(dev.recommendedMaxWorkingSetSize);
        return total > allocated ? total - allocated : 0;
    }
    return static_cast<size_t>(dev.recommendedMaxWorkingSetSize);
}

size_t getFreeMemoryCurrentDevice() {
    return getFreeMemory(getCurrentDevice());
}

DeviceScope::DeviceScope(int device) {
    if (device >= 0 && device != getCurrentDevice()) {
        prevDevice_ = getCurrentDevice();
        setCurrentDevice(device);
        return;
    }
    prevDevice_ = -1;
}

DeviceScope::~DeviceScope() {
    if (prevDevice_ != -1) {
        setCurrentDevice(prevDevice_);
    }
}

CublasHandleScope::CublasHandleScope() {
    blasHandle_ = nullptr;
}

CublasHandleScope::~CublasHandleScope() {}

CudaEvent::CudaEvent(cudaStream_t stream, bool timer)
        : event_(nullptr) {
    (void)stream;
    (void)timer;
}

CudaEvent::CudaEvent(CudaEvent&& event) noexcept
        : event_(event.event_) {
    event.event_ = nullptr;
}

CudaEvent::~CudaEvent() {}

CudaEvent& CudaEvent::operator=(CudaEvent&& event) noexcept {
    event_ = event.event_;
    event.event_ = nullptr;
    return *this;
}

void CudaEvent::streamWaitOnEvent(cudaStream_t stream) {
    (void)stream;
}

void CudaEvent::cpuWaitOnEvent() {}

} // namespace gpu
} // namespace faiss
