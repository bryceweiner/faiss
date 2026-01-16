/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/clone_index.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/InvertedLists.h>
#include <cstring>

namespace faiss {
namespace gpu {

namespace {
Index* clone_to_cpu(Index* index) {
    if (!index) {
        return nullptr;
    }
    if (!isGpuIndex(index)) {
        return index;
    }
    return faiss::clone_index(index);
}
} // namespace

GpuIndexIVF::GpuIndexIVF(
        GpuResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        idx_t nlistIn,
        GpuIndexIVFConfig config)
        : GpuIndex(provider->getResources(), dims, metric, metricArg, config),
          IndexIVFInterface(nullptr, nlistIn),
          ivfConfig_(config) {
    init_();
}

GpuIndexIVF::GpuIndexIVF(
        GpuResourcesProvider* provider,
        Index* coarseQuantizer,
        int dims,
        faiss::MetricType metric,
        float metricArg,
        idx_t nlistIn,
        GpuIndexIVFConfig config)
        : GpuIndex(provider->getResources(), dims, metric, metricArg, config),
          IndexIVFInterface(coarseQuantizer, nlistIn),
          ivfConfig_(config) {
    init_();
}

GpuIndexIVF::~GpuIndexIVF() = default;

void GpuIndexIVF::init_() {
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be > 0");
    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        cp.spherical = true;
    }
    cp.niter = 10;
    cp.verbose = verbose;

    if (!quantizer) {
        if (metric_type == faiss::METRIC_L2) {
            quantizer = new faiss::IndexFlatL2(d);
        } else if (metric_type == faiss::METRIC_INNER_PRODUCT) {
            quantizer = new faiss::IndexFlatIP(d);
        } else {
            FAISS_THROW_FMT("unsupported metric type %d", (int)metric_type);
        }
        own_fields = true;
        is_trained = false;
    } else {
        Index* cpu_quantizer = clone_to_cpu(quantizer);
        if (cpu_quantizer != quantizer) {
            if (own_fields && quantizer) {
                delete quantizer;
            }
            quantizer = cpu_quantizer;
            own_fields = true;
        }
        is_trained = quantizer->is_trained && quantizer->ntotal == nlist;
    }

    verifyIVFSettings_();
}

void GpuIndexIVF::verifyIVFSettings_() const {
    FAISS_THROW_IF_NOT(quantizer);
    FAISS_THROW_IF_NOT(d == quantizer->d);
    if (is_trained) {
        FAISS_THROW_IF_NOT(quantizer->is_trained);
        FAISS_THROW_IF_NOT_FMT(
                quantizer->ntotal == nlist,
                "IVF nlist count (%lld) does not match trained coarse quantizer size (%lld)",
                static_cast<long long>(nlist),
                static_cast<long long>(quantizer->ntotal));
    } else {
        FAISS_THROW_IF_NOT(ntotal == 0);
    }
}

void GpuIndexIVF::setCpuIndex_(std::unique_ptr<IndexIVF> index) {
    cpuIndex_ = std::move(index);
    if (cpuIndex_) {
        quantizer = cpuIndex_->quantizer;
        own_fields = false;
        nlist = cpuIndex_->nlist;
        nprobe = cpuIndex_->nprobe;
        ntotal = cpuIndex_->ntotal;
        is_trained = cpuIndex_->is_trained;
    }
}

IndexIVF* GpuIndexIVF::getCpuIndex_() const {
    return cpuIndex_.get();
}

void GpuIndexIVF::copyFrom(const faiss::IndexIVF* index) {
    GpuIndex::copyFrom(index);
    nlist = index->nlist;
    nprobe = index->nprobe;
    std::unique_ptr<IndexIVF> cloned(
            dynamic_cast<IndexIVF*>(faiss::clone_index(index)));
    FAISS_THROW_IF_NOT_MSG(cloned, "Failed to clone IndexIVF");
    setCpuIndex_(std::move(cloned));
}

void GpuIndexIVF::copyTo(faiss::IndexIVF* index) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "CPU IVF index is not initialized");
    FAISS_THROW_IF_NOT_MSG(index, "Target IndexIVF is null");

    index->d = cpuIndex_->d;
    index->metric_type = cpuIndex_->metric_type;
    index->metric_arg = cpuIndex_->metric_arg;
    index->nlist = cpuIndex_->nlist;
    index->nprobe = cpuIndex_->nprobe;
    index->is_trained = cpuIndex_->is_trained;
    index->cp = cpuIndex_->cp;

    if (index->quantizer && index->own_fields) {
        delete index->quantizer;
    }
    index->quantizer = faiss::clone_index(cpuIndex_->quantizer);
    index->own_fields = true;

    if (index->invlists && index->own_invlists) {
        delete index->invlists;
    }
    auto* inv = new faiss::ArrayInvertedLists(
            cpuIndex_->nlist, cpuIndex_->invlists->code_size);
    cpuIndex_->invlists->copy_subset_to(
            *inv,
            faiss::InvertedLists::SUBSET_TYPE_INVLIST,
            0,
            cpuIndex_->nlist);
    index->invlists = inv;
    index->own_invlists = true;
    index->ntotal = inv->compute_ntotal();
}

int GpuIndexIVF::getCurrentNProbe_(const SearchParameters* params) const {
    if (params) {
        auto ivf_params = dynamic_cast<const SearchParametersIVF*>(params);
        if (ivf_params && ivf_params->nprobe > 0) {
            return ivf_params->nprobe;
        }
    }
    return nprobe;
}

bool GpuIndexIVF::addImplRequiresIDs_() const {
    return true;
}

void GpuIndexIVF::trainQuantizer_(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "CPU IVF index is not initialized");
    cpuIndex_->train(n, x);
    is_trained = cpuIndex_->is_trained;
}

void GpuIndexIVF::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "CPU IVF index is not initialized");
    cpuIndex_->add_with_ids(n, x, ids);
    ntotal = cpuIndex_->ntotal;
}

void GpuIndexIVF::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "CPU IVF index is not initialized");
    cpuIndex_->nprobe = getCurrentNProbe_(params);
    cpuIndex_->search(n, x, k, distances, labels, params);
}

idx_t GpuIndexIVF::getNumLists() const {
    return cpuIndex_ ? cpuIndex_->nlist : nlist;
}

idx_t GpuIndexIVF::getListLength(idx_t listId) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "CPU IVF index is not initialized");
    return cpuIndex_->invlists->list_size(listId);
}

std::vector<uint8_t> GpuIndexIVF::getListVectorData(
        idx_t listId,
        bool gpuFormat) const {
    (void)gpuFormat;
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "CPU IVF index is not initialized");
    size_t listSize = cpuIndex_->invlists->list_size(listId);
    size_t codeSize = cpuIndex_->invlists->code_size;
    std::vector<uint8_t> data(listSize * codeSize);
    const uint8_t* codes = cpuIndex_->invlists->get_codes(listId);
    std::memcpy(data.data(), codes, data.size());
    cpuIndex_->invlists->release_codes(listId, codes);
    return data;
}

std::vector<idx_t> GpuIndexIVF::getListIndices(idx_t listId) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "CPU IVF index is not initialized");
    size_t listSize = cpuIndex_->invlists->list_size(listId);
    std::vector<idx_t> ids(listSize);
    const idx_t* src = cpuIndex_->invlists->get_ids(listId);
    std::memcpy(ids.data(), src, listSize * sizeof(idx_t));
    cpuIndex_->invlists->release_ids(listId, src);
    return ids;
}

void GpuIndexIVF::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const SearchParametersIVF* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "CPU IVF index is not initialized");
    cpuIndex_->search_preassigned(
            n,
            x,
            k,
            assign,
            centroid_dis,
            distances,
            labels,
            store_pairs,
            params,
            stats);
}

void GpuIndexIVF::range_search_preassigned(
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        RangeSearchResult* result,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT_MSG(cpuIndex_, "CPU IVF index is not initialized");
    cpuIndex_->range_search_preassigned(
            nx,
            x,
            radius,
            keys,
            coarse_dis,
            result,
            store_pairs,
            params,
            stats);
}

} // namespace gpu
} // namespace faiss
