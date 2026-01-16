/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexBinaryFlat.h>
#include <faiss/gpu/GpuIndexBinaryFlat.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace gpu {

class BinaryFlatIndex {
   public:
    explicit BinaryFlatIndex(int dims)
            : cpuIndex_(std::make_unique<faiss::IndexBinaryFlat>(dims)) {}

    void add(idx_t n, const uint8_t* x) {
        cpuIndex_->add(n, x);
    }

    void reset() {
        cpuIndex_->reset();
    }

    void search(
            idx_t n,
            const uint8_t* x,
            int k,
            int32_t* distances,
            idx_t* labels) const {
        cpuIndex_->search(n, x, k, distances, labels);
    }

    void reconstruct(idx_t key, uint8_t* recons) const {
        cpuIndex_->reconstruct(key, recons);
    }

    void copyFrom(const faiss::IndexBinaryFlat* index) {
        *cpuIndex_ = *index;
    }

    void copyTo(faiss::IndexBinaryFlat* index) const {
        *index = *cpuIndex_;
    }

   private:
    std::unique_ptr<faiss::IndexBinaryFlat> cpuIndex_;
};

GpuIndexBinaryFlat::GpuIndexBinaryFlat(
        GpuResourcesProvider* resources,
        const faiss::IndexBinaryFlat* index,
        GpuIndexBinaryFlatConfig config)
        : IndexBinary(index->d),
          resources_(resources->getResources()),
          binaryFlatConfig_(config) {
    FAISS_ASSERT(resources_);
    data_ = std::make_unique<BinaryFlatIndex>(index->d);
    copyFrom(index);
}

GpuIndexBinaryFlat::GpuIndexBinaryFlat(
        GpuResourcesProvider* resources,
        int dims,
        GpuIndexBinaryFlatConfig config)
        : IndexBinary(dims),
          resources_(resources->getResources()),
          binaryFlatConfig_(config) {
    FAISS_ASSERT(resources_);
    data_ = std::make_unique<BinaryFlatIndex>(dims);
}

GpuIndexBinaryFlat::~GpuIndexBinaryFlat() = default;

int GpuIndexBinaryFlat::getDevice() const {
    return binaryFlatConfig_.device;
}

std::shared_ptr<GpuResources> GpuIndexBinaryFlat::getResources() {
    return resources_;
}

void GpuIndexBinaryFlat::copyFrom(const faiss::IndexBinaryFlat* index) {
    d = index->d;
    ntotal = index->ntotal;
    data_->copyFrom(index);
}

void GpuIndexBinaryFlat::copyTo(faiss::IndexBinaryFlat* index) const {
    index->d = d;
    index->ntotal = ntotal;
    data_->copyTo(index);
}

void GpuIndexBinaryFlat::add(faiss::idx_t n, const uint8_t* x) {
    data_->add(n, x);
    ntotal += n;
}

void GpuIndexBinaryFlat::reset() {
    data_->reset();
    ntotal = 0;
}

void GpuIndexBinaryFlat::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        faiss::idx_t* labels,
        const faiss::SearchParameters* params) const {
    (void)params;
    searchNonPaged_(n, x, static_cast<int>(k), distances, labels);
}

void GpuIndexBinaryFlat::reconstruct(faiss::idx_t key, uint8_t* recons) const {
    data_->reconstruct(key, recons);
}

void GpuIndexBinaryFlat::searchFromCpuPaged_(
        idx_t n,
        const uint8_t* x,
        int k,
        int32_t* outDistancesData,
        idx_t* outIndicesData) const {
    data_->search(n, x, k, outDistancesData, outIndicesData);
}

void GpuIndexBinaryFlat::searchNonPaged_(
        idx_t n,
        const uint8_t* x,
        int k,
        int32_t* outDistancesData,
        idx_t* outIndicesData) const {
    data_->search(n, x, k, outDistancesData, outIndicesData);
}

} // namespace gpu
} // namespace faiss
