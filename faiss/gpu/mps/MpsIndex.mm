/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>

namespace faiss {
namespace gpu {

bool should_use_cuvs(GpuIndexConfig config_) {
    (void)config_;
    return false;
}

GpuIndex::GpuIndex(
        std::shared_ptr<GpuResources> resources,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        GpuIndexConfig config)
        : Index(dims, metric), resources_(std::move(resources)), config_(config) {
    FAISS_THROW_IF_NOT_MSG(dims > 0, "Invalid number of dimensions");
    FAISS_THROW_IF_NOT_FMT(
            config_.device < getNumDevices(),
            "Invalid GPU device %d",
            config_.device);
    metric_arg = metricArg;
    FAISS_ASSERT(resources_);
    resources_->initializeForDevice(config_.device);
    minPagedSize_ = 0;
}

int GpuIndex::getDevice() const {
    return config_.device;
}

std::shared_ptr<GpuResources> GpuIndex::getResources() {
    return resources_;
}

void GpuIndex::setMinPagingSize(size_t size) {
    minPagedSize_ = size;
}

size_t GpuIndex::getMinPagingSize() const {
    return minPagedSize_;
}

void GpuIndex::copyFrom(const faiss::Index* index) {
    d = index->d;
    metric_type = index->metric_type;
    metric_arg = index->metric_arg;
    ntotal = index->ntotal;
    is_trained = index->is_trained;
}

void GpuIndex::copyTo(faiss::Index* index) const {
    index->d = d;
    index->metric_type = metric_type;
    index->metric_arg = metric_arg;
    index->ntotal = ntotal;
    index->is_trained = is_trained;
}

void GpuIndex::add_ex(idx_t n, const void* x, NumericType numeric_type) {
    add_with_ids_ex(n, x, numeric_type, nullptr);
}

void GpuIndex::add(idx_t n, const float* x) {
    add_ex(n, x, NumericType::Float32);
}

void GpuIndex::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* ids) {
    add_with_ids_ex(n, x, NumericType::Float32, ids);
}

void GpuIndex::add_with_ids_ex(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        const idx_t* ids) {
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index not trained");
    std::vector<idx_t> generatedIds;
    if (!ids && addImplRequiresIDs_()) {
        generatedIds.resize(n);
        for (idx_t i = 0; i < n; ++i) {
            generatedIds[i] = ntotal + i;
        }
        ids = generatedIds.data();
    }
    addImpl_ex_(n, x, numeric_type, ids);
    ntotal += n;
}

void GpuIndex::assign(idx_t n, const float* x, idx_t* labels, idx_t k) const {
    std::vector<float> distances(n * k);
    search(n, x, k, distances.data(), labels);
}

void GpuIndex::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    search_ex(n, x, NumericType::Float32, k, distances, labels, params);
}

void GpuIndex::search_ex(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (n == 0) {
        return;
    }
    searchImpl_ex_(n, x, numeric_type, k, distances, labels, params);
}

void GpuIndex::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* recons,
        const SearchParameters* params) const {
    search(n, x, k, distances, labels, params);
    reconstruct_batch(n * k, labels, recons);
}

void GpuIndex::compute_residual(
        const float* x,
        float* residual,
        idx_t key) const {
    FAISS_THROW_IF_NOT_MSG(false, "GpuIndex::compute_residual not implemented for MPS");
}

void GpuIndex::compute_residual_n(
        idx_t n,
        const float* xs,
        float* residuals,
        const idx_t* keys) const {
    FAISS_THROW_IF_NOT_MSG(false, "GpuIndex::compute_residual_n not implemented for MPS");
}

GpuIndex* tryCastGpuIndex(faiss::Index* index) {
    return dynamic_cast<GpuIndex*>(index);
}

bool isGpuIndex(faiss::Index* index) {
    return tryCastGpuIndex(index) != nullptr;
}

bool isGpuIndexImplemented(faiss::Index* index) {
    return dynamic_cast<faiss::IndexFlat*>(index) != nullptr ||
            dynamic_cast<faiss::IndexBinaryFlat*>(index) != nullptr ||
            dynamic_cast<faiss::IndexIVFFlat*>(index) != nullptr ||
            dynamic_cast<faiss::IndexIVFPQ*>(index) != nullptr;
}

} // namespace gpu
} // namespace faiss
