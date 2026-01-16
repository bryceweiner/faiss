/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MPSGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphArithmeticOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMatrixMultiplicationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMemoryOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphReductionOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorData.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorShapeOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphTopKOps.h>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <cstring>
#include <limits>
#include <vector>

namespace faiss {
namespace gpu {

bool should_use_cuvs(GpuDistanceParams args) {
    (void)args;
    return false;
}

namespace {
const float* asFloatPointer(const void* data, DistanceDataType type) {
    FAISS_THROW_IF_NOT_MSG(
            type == DistanceDataType::F32,
            "MPS backend only supports float32 distance data");
    return static_cast<const float*>(data);
}

void writeIndices(
        IndicesDataType type,
        idx_t n,
        const int64_t* src,
        void* dst) {
    if (type == IndicesDataType::I64) {
        std::memcpy(dst, src, sizeof(int64_t) * n);
        return;
    }
    if (type == IndicesDataType::I32) {
        auto* out = static_cast<int32_t*>(dst);
        for (idx_t i = 0; i < n; ++i) {
            out[i] = static_cast<int32_t>(src[i]);
        }
        return;
    }
    FAISS_THROW_MSG("Unsupported indices type");
}

id<MTLDevice> getMpsDeviceForIndex(int device) {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    FAISS_THROW_IF_NOT_FMT(
            device >= 0 && device < devices.count,
            "Invalid MPS device %d",
            device);
    return devices[device];
}

MPSGraphTensorData* makeTensorData(
        id<MTLDevice> device,
        const float* data,
        size_t rows,
        size_t cols) {
    size_t byteSize = rows * cols * sizeof(float);
    id<MTLBuffer> buffer = [device newBufferWithBytes:data
                                               length:byteSize
                                              options:MTLResourceStorageModeShared];
    NSArray<NSNumber*>* shape = @[ @(rows), @(cols) ];
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                  shape:shape
                                               dataType:MPSDataTypeFloat32];
}

void runGraphTopK(
        faiss::MetricType metric,
        const float* queries,
        const float* vectors,
        size_t nq,
        size_t nb,
        int dims,
        int k,
        float* outDistances,
        int64_t* outIndices) {
    @autoreleasepool {
        int device = getCurrentDevice();
        id<MTLDevice> mpsDevice = getMpsDeviceForIndex(device);
        id<MTLCommandQueue> queue = [mpsDevice newCommandQueue];

        MPSGraph* graph = [[MPSGraph alloc] init];
        NSArray<NSNumber*>* qShape = @[ @(nq), @(dims) ];
        NSArray<NSNumber*>* vShape = @[ @(nb), @(dims) ];
        MPSGraphTensor* q = [graph placeholderWithShape:qShape
                                              dataType:MPSDataTypeFloat32
                                                  name:@"queries"];
        MPSGraphTensor* v = [graph placeholderWithShape:vShape
                                              dataType:MPSDataTypeFloat32
                                                  name:@"vectors"];
        MPSGraphTensor* vT =
                [graph transposeTensor:v dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* prod =
                [graph matrixMultiplicationWithPrimaryTensor:q
                                             secondaryTensor:vT
                                                        name:nil];

        MPSGraphTensor* dist = prod;
        if (metric == METRIC_L2) {
            MPSGraphTensor* qsq =
                    [graph squareWithTensor:q name:nil];
            MPSGraphTensor* vsq =
                    [graph squareWithTensor:v name:nil];
            MPSGraphTensor* qnorm =
                    [graph reductionSumWithTensor:qsq axis:1 name:nil];
            MPSGraphTensor* vnorm =
                    [graph reductionSumWithTensor:vsq axis:1 name:nil];
            MPSGraphTensor* qnorm2 =
                    [graph reshapeTensor:qnorm withShape:@[@(nq), @1] name:nil];
            MPSGraphTensor* vnorm2 =
                    [graph reshapeTensor:vnorm withShape:@[@1, @(nb)] name:nil];
            MPSGraphTensor* two =
                    [graph constantWithScalar:-2.0f shape:@[@1] dataType:MPSDataTypeFloat32];
            MPSGraphTensor* twoProd =
                    [graph multiplicationWithPrimaryTensor:prod
                                           secondaryTensor:two
                                                      name:nil];
            MPSGraphTensor* sum =
                    [graph additionWithPrimaryTensor:qnorm2
                                     secondaryTensor:vnorm2
                                                name:nil];
            dist = [graph additionWithPrimaryTensor:sum
                                     secondaryTensor:twoProd
                                                name:nil];
        }

        MPSGraphTensor* valuesTensor = dist;
        MPSGraphTensor* indicesTensor = nil;

        if (k > 0) {
            NSArray<MPSGraphTensor*>* topk = nil;
            if (metric == METRIC_L2) {
                topk = [graph bottomKWithSourceTensor:dist axis:1 k:k name:nil];
            } else {
                topk = [graph topKWithSourceTensor:dist axis:1 k:k name:nil];
            }
            valuesTensor = topk[0];
            indicesTensor = topk[1];
        }

        MPSGraphTensorData* qData = makeTensorData(mpsDevice, queries, nq, dims);
        MPSGraphTensorData* vData = makeTensorData(mpsDevice, vectors, nb, dims);
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
                @{ q : qData, v : vData };
        NSArray<MPSGraphTensor*>* targets =
                indicesTensor ? @[ valuesTensor, indicesTensor ] : @[ valuesTensor ];
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
                [graph runWithMTLCommandQueue:queue
                                       feeds:feeds
                               targetTensors:targets
                            targetOperations:nil];

        MPSGraphTensorData* valuesData = results[valuesTensor];
        MPSNDArray* valuesArray = [valuesData mpsndarray];
        [valuesArray readBytes:outDistances strideBytes:nil];
        if (k > 0 && indicesTensor) {
            MPSGraphTensorData* indicesData = results[indicesTensor];
            MPSNDArray* indicesArray = [indicesData mpsndarray];
            std::vector<int32_t> tmp(nq * k);
            [indicesArray readBytes:tmp.data() strideBytes:nil];
            for (size_t i = 0; i < nq * k; ++i) {
                outIndices[i] = static_cast<int64_t>(tmp[i]);
            }
        }
    }
}
} // namespace

void bfKnn(GpuResourcesProvider* resources, const GpuDistanceParams& args) {
    (void)resources;
    const float* vectors = asFloatPointer(args.vectors, args.vectorType);
    const float* queries = asFloatPointer(args.queries, args.queryType);

    FAISS_THROW_IF_NOT_MSG(args.vectorsRowMajor, "MPS bfKnn expects row-major vectors");
    FAISS_THROW_IF_NOT_MSG(args.queriesRowMajor, "MPS bfKnn expects row-major queries");

    if (args.k == -1) {
        FAISS_THROW_IF_NOT_MSG(
                args.outDistances != nullptr,
                "outDistances required for all-pairs distance");
        runGraphTopK(
                args.metric,
                queries,
                vectors,
                args.numQueries,
                args.numVectors,
                args.dims,
                -1,
                args.outDistances,
                nullptr);
        return;
    }

    FAISS_THROW_IF_NOT_MSG(args.k > 0, "k must be > 0 or -1");
    int k = std::min<int>(args.k, args.numVectors);

    std::vector<float> tmpDistances(args.numQueries * k);
    std::vector<int64_t> indices(args.numQueries * k);
    if (args.metric != METRIC_L2 && args.metric != METRIC_INNER_PRODUCT) {
        FAISS_THROW_MSG("Unsupported metric for MPS bfKnn");
    }
    runGraphTopK(
            args.metric,
            queries,
            vectors,
            args.numQueries,
            args.numVectors,
            args.dims,
            k,
            tmpDistances.data(),
            indices.data());

    float fill = args.metric == METRIC_INNER_PRODUCT
            ? -std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::infinity();
    for (idx_t qi = 0; qi < args.numQueries; ++qi) {
        std::memcpy(
                args.outDistances + qi * args.k,
                tmpDistances.data() + qi * k,
                sizeof(float) * k);
        for (int i = k; i < args.k; ++i) {
            args.outDistances[qi * args.k + i] = fill;
        }
        if (args.outIndices) {
            writeIndices(
                    args.outIndicesType,
                    k,
                    indices.data() + qi * k,
                    static_cast<char*>(args.outIndices) +
                            qi * args.k *
                                    (args.outIndicesType ==
                                                     IndicesDataType::I64
                                             ? sizeof(int64_t)
                                             : sizeof(int32_t)));
            if (k < args.k) {
                if (args.outIndicesType == IndicesDataType::I64) {
                    auto* out = static_cast<int64_t*>(args.outIndices) +
                            qi * args.k;
                    for (int i = k; i < args.k; ++i) {
                        out[i] = -1;
                    }
                } else {
                    auto* out = static_cast<int32_t*>(args.outIndices) +
                            qi * args.k;
                    for (int i = k; i < args.k; ++i) {
                        out[i] = -1;
                    }
                }
            }
        }
    }
}

void bfKnn_tiling(
        GpuResourcesProvider* resources,
        const GpuDistanceParams& args,
        size_t vectorsMemoryLimit,
        size_t queriesMemoryLimit) {
    (void)vectorsMemoryLimit;
    (void)queriesMemoryLimit;
    bfKnn(resources, args);
}

void bruteForceKnn(
        GpuResourcesProvider* resources,
        faiss::MetricType metric,
        const float* vectors,
        bool vectorsRowMajor,
        idx_t numVectors,
        const float* queries,
        bool queriesRowMajor,
        idx_t numQueries,
        int dims,
        int k,
        float* outDistances,
        idx_t* outIndices) {
    GpuDistanceParams params;
    params.metric = metric;
    params.k = k;
    params.dims = dims;
    params.vectors = vectors;
    params.vectorType = DistanceDataType::F32;
    params.vectorsRowMajor = vectorsRowMajor;
    params.numVectors = numVectors;
    params.queries = queries;
    params.queryType = DistanceDataType::F32;
    params.queriesRowMajor = queriesRowMajor;
    params.numQueries = numQueries;
    params.outDistances = outDistances;
    params.outIndices = outIndices;
    params.outIndicesType = IndicesDataType::I64;
    bfKnn(resources, params);
}

} // namespace gpu
} // namespace faiss
