/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <cstring>

namespace faiss {
namespace gpu {

class FlatIndex {
   public:
    FlatIndex(int dims, faiss::MetricType metric, bool useFloat16)
            : dims_(dims), metric_(metric), useFloat16_(useFloat16) {}

    void add(idx_t n, const float* x) {
        data_.insert(data_.end(), x, x + n * dims_);
    }

    void reset() {
        data_.clear();
    }

    size_t getNumVecs() const {
        return data_.size() / dims_;
    }

    void search(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels) const {
        faiss::gpu::GpuDistanceParams args;
        args.metric = metric_;
        args.k = k;
        args.dims = dims_;
        args.vectors = data_.data();
        args.vectorType = faiss::gpu::DistanceDataType::F32;
        args.vectorsRowMajor = true;
        args.numVectors = getNumVecs();
        args.queries = x;
        args.queryType = faiss::gpu::DistanceDataType::F32;
        args.queriesRowMajor = true;
        args.numQueries = n;
        args.outDistances = distances;
        args.outIndices = labels;
        args.outIndicesType = faiss::gpu::IndicesDataType::I64;
        args.device = faiss::gpu::getCurrentDevice();
        faiss::gpu::bfKnn(nullptr, args);
    }

    void reconstruct(idx_t key, float* out) const {
        std::memcpy(out, data_.data() + key * dims_, sizeof(float) * dims_);
    }

    void reconstruct_n(idx_t i0, idx_t num, float* out) const {
        std::memcpy(out, data_.data() + i0 * dims_, sizeof(float) * num * dims_);
    }

    void reconstruct_batch(idx_t n, const idx_t* keys, float* out) const {
        for (idx_t i = 0; i < n; ++i) {
            reconstruct(keys[i], out + i * dims_);
        }
    }

    void copyFrom(const faiss::IndexFlat* index) {
        data_.assign(index->get_xb(), index->get_xb() + index->ntotal * dims_);
    }

    void copyTo(faiss::IndexFlat* index) const {
        index->reset();
        index->add(getNumVecs(), data_.data());
    }

   private:
    int dims_;
    faiss::MetricType metric_;
    std::vector<float> data_;
    bool useFloat16_;
};

GpuIndexFlat::GpuIndexFlat(
        GpuResourcesProvider* provider,
        const faiss::IndexFlat* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider->getResources(), index, config) {}

GpuIndexFlat::GpuIndexFlat(
        std::shared_ptr<GpuResources> resources,
        const faiss::IndexFlat* index,
        GpuIndexFlatConfig config)
        : GpuIndex(resources, index->d, index->metric_type, index->metric_arg, config),
          flatConfig_(config) {
    resetIndex_(index->d);
    copyFrom(index);
}

GpuIndexFlat::GpuIndexFlat(
        GpuResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider->getResources(), dims, metric, config) {}

GpuIndexFlat::GpuIndexFlat(
        std::shared_ptr<GpuResources> resources,
        int dims,
        faiss::MetricType metric,
        GpuIndexFlatConfig config)
        : GpuIndex(resources, dims, metric, 0.0f, config), flatConfig_(config) {
    resetIndex_(dims);
    is_trained = true;
}

GpuIndexFlat::~GpuIndexFlat() = default;

void GpuIndexFlat::copyFrom(const faiss::IndexFlat* index) {
    GpuIndex::copyFrom(index);
    data_->copyFrom(index);
}

void GpuIndexFlat::copyTo(faiss::IndexFlat* index) const {
    GpuIndex::copyTo(index);
    data_->copyTo(index);
}

size_t GpuIndexFlat::getNumVecs() const {
    return data_->getNumVecs();
}

void GpuIndexFlat::reset() {
    data_->reset();
    ntotal = 0;
    is_trained = true;
}

void GpuIndexFlat::train(idx_t n, const float* x) {
    (void)n;
    (void)x;
    is_trained = true;
}

void GpuIndexFlat::add(idx_t n, const float* x) {
    GpuIndex::add(n, x);
}

void GpuIndexFlat::reconstruct(idx_t key, float* out) const {
    data_->reconstruct(key, out);
}

void GpuIndexFlat::reconstruct_n(idx_t i0, idx_t num, float* out) const {
    data_->reconstruct_n(i0, num, out);
}

void GpuIndexFlat::reconstruct_batch(idx_t n, const idx_t* keys, float* out)
        const {
    data_->reconstruct_batch(n, keys, out);
}

void GpuIndexFlat::compute_residual(const float* x, float* residual, idx_t key)
        const {
    std::vector<float> tmp(d);
    reconstruct(key, tmp.data());
    for (int i = 0; i < d; ++i) {
        residual[i] = x[i] - tmp[i];
    }
}

void GpuIndexFlat::compute_residual_n(
        idx_t n,
        const float* xs,
        float* residuals,
        const idx_t* keys) const {
    for (idx_t i = 0; i < n; ++i) {
        compute_residual(xs + i * d, residuals + i * d, keys[i]);
    }
}

void GpuIndexFlat::resetIndex_(int dims) {
    data_ = std::make_unique<FlatIndex>(dims, metric_type, flatConfig_.useFloat16);
}

bool GpuIndexFlat::addImplRequiresIDs_() const {
    return false;
}

void GpuIndexFlat::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    (void)ids;
    data_->add(n, x);
}

void GpuIndexFlat::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    (void)params;
    DeviceScope scope(getDevice());
    data_->search(n, x, k, distances, labels);
}

GpuIndexFlatL2::GpuIndexFlatL2(
        GpuResourcesProvider* provider,
        faiss::IndexFlatL2* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, index, config) {}

GpuIndexFlatL2::GpuIndexFlatL2(
        std::shared_ptr<GpuResources> resources,
        faiss::IndexFlatL2* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, index, config) {}

GpuIndexFlatL2::GpuIndexFlatL2(
        GpuResourcesProvider* provider,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, dims, faiss::METRIC_L2, config) {}

GpuIndexFlatL2::GpuIndexFlatL2(
        std::shared_ptr<GpuResources> resources,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, dims, faiss::METRIC_L2, config) {}

void GpuIndexFlatL2::copyFrom(faiss::IndexFlat* index) {
    GpuIndexFlat::copyFrom(index);
}

void GpuIndexFlatL2::copyTo(faiss::IndexFlat* index) {
    GpuIndexFlat::copyTo(index);
}

GpuIndexFlatIP::GpuIndexFlatIP(
        GpuResourcesProvider* provider,
        faiss::IndexFlatIP* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, index, config) {}

GpuIndexFlatIP::GpuIndexFlatIP(
        std::shared_ptr<GpuResources> resources,
        faiss::IndexFlatIP* index,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, index, config) {}

GpuIndexFlatIP::GpuIndexFlatIP(
        GpuResourcesProvider* provider,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(provider, dims, faiss::METRIC_INNER_PRODUCT, config) {}

GpuIndexFlatIP::GpuIndexFlatIP(
        std::shared_ptr<GpuResources> resources,
        int dims,
        GpuIndexFlatConfig config)
        : GpuIndexFlat(resources, dims, faiss::METRIC_INNER_PRODUCT, config) {}

void GpuIndexFlatIP::copyFrom(faiss::IndexFlat* index) {
    GpuIndexFlat::copyFrom(index);
}

void GpuIndexFlatIP::copyTo(faiss::IndexFlat* index) {
    GpuIndexFlat::copyTo(index);
}

} // namespace gpu
} // namespace faiss
