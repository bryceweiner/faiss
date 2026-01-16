/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(FAISS_ENABLE_MPS)
#include <cstdint>

// Minimal CUDA type shims for public headers when building the MPS backend.
// These are opaque and only used to preserve the existing public API surface.
struct CUstream_st;
struct CUevent_st;
typedef CUstream_st* cudaStream_t;
typedef struct faissGpuBlasHandleOpaque* cublasHandle_t;
typedef CUevent_st* cudaEvent_t;
typedef int cudaError_t;

struct cudaDeviceProp {
    int major;
    int minor;
    int warpSize;
    int maxThreadsPerBlock;
    int maxGridSize[3];
    size_t sharedMemPerBlock;
    size_t totalGlobalMem;
};

struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
            : x(vx), y(vy), z(vz) {}
};

static constexpr cudaError_t cudaSuccess = 0;

inline const char* cudaGetErrorString(cudaError_t) {
    return "FAISS MPS backend";
}
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
