/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/clone_index.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace faiss {
namespace gpu {

namespace {
std::unique_ptr<faiss::Index> clone_quantizer(Index* quantizer) {
    if (!quantizer) {
        return nullptr;
    }
    if (isGpuIndex(quantizer)) {
        FAISS_THROW_MSG("MPS IVF requires a CPU coarse quantizer");
    }
    return std::unique_ptr<faiss::Index>(faiss::clone_index(quantizer));
}

void fill_empty_results(
        int k,
        float* distances,
        idx_t* labels,
        faiss::MetricType metric) {
    float fill = metric == faiss::METRIC_INNER_PRODUCT
            ? -std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::infinity();
    for (int i = 0; i < k; ++i) {
        distances[i] = fill;
        labels[i] = -1;
    }
}
} // namespace

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFFlat* index,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(
                  provider,
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  index->nlist,
                  config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    copyFrom(index);
}

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        int dims,
        idx_t nlist,
        faiss::MetricType metric,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(provider, dims, metric, 0.0f, nlist, config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    setIndex_(getResources().get(), dims, nlist, metric, metric_arg, false, nullptr,
              ivfFlatConfig_.interleavedLayout, ivfConfig_.indicesOptions, config_.memorySpace);
}

GpuIndexIVFFlat::GpuIndexIVFFlat(
        GpuResourcesProvider* provider,
        Index* coarseQuantizer,
        int dims,
        idx_t nlist,
        faiss::MetricType metric,
        GpuIndexIVFFlatConfig config)
        : GpuIndexIVF(provider, dims, metric, 0.0f, nlist, config),
          ivfFlatConfig_(config),
          reserveMemoryVecs_(0) {
    if (own_fields && quantizer) {
        delete quantizer;
        quantizer = nullptr;
        own_fields = false;
    }
    auto cpu_quantizer = clone_quantizer(coarseQuantizer);
    FAISS_THROW_IF_NOT_MSG(cpu_quantizer, "Missing coarse quantizer");
    quantizer = cpu_quantizer.get();
    auto index_ptr = std::make_unique<faiss::IndexIVFFlat>(
            quantizer, dims, nlist, metric);
    index_ptr->own_fields = true;
    cpu_quantizer.release();
    setCpuIndex_(std::move(index_ptr));
}

GpuIndexIVFFlat::~GpuIndexIVFFlat() = default;

void GpuIndexIVFFlat::reserveMemory(size_t numVecs) {
    reserveMemoryVecs_ = numVecs;
}

void GpuIndexIVFFlat::copyFrom(const faiss::IndexIVFFlat* index) {
    GpuIndexIVF::copyFrom(index);
    std::unique_ptr<IndexIVF> cloned(
            dynamic_cast<IndexIVF*>(faiss::clone_index(index)));
    FAISS_THROW_IF_NOT_MSG(cloned, "Failed to clone IndexIVFFlat");
    setCpuIndex_(std::move(cloned));
}

void GpuIndexIVFFlat::copyTo(faiss::IndexIVFFlat* index) const {
    GpuIndexIVF::copyTo(index);
}

size_t GpuIndexIVFFlat::reclaimMemory() {
    return 0;
}

void GpuIndexIVFFlat::reset() {
    if (getCpuIndex_()) {
        getCpuIndex_()->reset();
        ntotal = 0;
        is_trained = getCpuIndex_()->is_trained;
    }
}

void GpuIndexIVFFlat::updateQuantizer() {
    if (getCpuIndex_()) {
        quantizer = getCpuIndex_()->quantizer;
    }
}

void GpuIndexIVFFlat::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(getCpuIndex_(), "CPU IVF index is not initialized");
    getCpuIndex_()->train(n, x);
    is_trained = getCpuIndex_()->is_trained;
}

void GpuIndexIVFFlat::reconstruct_n(idx_t i0, idx_t n, float* out) const {
    FAISS_THROW_IF_NOT_MSG(getCpuIndex_(), "CPU IVF index is not initialized");
    getCpuIndex_()->reconstruct_n(i0, n, out);
}

void GpuIndexIVFFlat::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    DeviceScope scope(getDevice());
    auto* cpu_index = dynamic_cast<faiss::IndexIVFFlat*>(getCpuIndex_());
    FAISS_THROW_IF_NOT_MSG(cpu_index, "CPU IVFFlat index is not initialized");
    const int nprobe = getCurrentNProbe_(params);
    std::vector<float> coarse_dis(n * nprobe);
    std::vector<idx_t> coarse_ids(n * nprobe);
    cpu_index->quantizer->search(
            n, x, nprobe, coarse_dis.data(), coarse_ids.data());

    for (idx_t qi = 0; qi < n; ++qi) {
        std::vector<std::pair<float, idx_t>> candidates;
        candidates.reserve(k * nprobe);

        for (int pi = 0; pi < nprobe; ++pi) {
            idx_t list_id = coarse_ids[qi * nprobe + pi];
            if (list_id < 0) {
                continue;
            }
            size_t list_size = cpu_index->invlists->list_size(list_id);
            if (list_size == 0) {
                continue;
            }
            const uint8_t* codes = cpu_index->invlists->get_codes(list_id);
            const idx_t* list_ids = cpu_index->invlists->get_ids(list_id);
            const float* list_vectors =
                    reinterpret_cast<const float*>(codes);
            int list_k = std::min<int>(k, list_size);
            std::vector<float> list_dist(list_k);
            std::vector<int64_t> list_idx(list_k);

            faiss::gpu::GpuDistanceParams args;
            args.metric = metric_type;
            args.k = list_k;
            args.dims = d;
            args.vectors = list_vectors;
            args.vectorType = faiss::gpu::DistanceDataType::F32;
            args.vectorsRowMajor = true;
            args.numVectors = list_size;
            args.queries = x + qi * d;
            args.queryType = faiss::gpu::DistanceDataType::F32;
            args.queriesRowMajor = true;
            args.numQueries = 1;
            args.outDistances = list_dist.data();
            args.outIndices = list_idx.data();
            args.outIndicesType = faiss::gpu::IndicesDataType::I64;
            args.device = getDevice();
            faiss::gpu::bfKnn(nullptr, args);

            for (int i = 0; i < list_k; ++i) {
                idx_t local = static_cast<idx_t>(list_idx[i]);
                candidates.emplace_back(list_dist[i], list_ids[local]);
            }
        }

        if (candidates.empty()) {
            fill_empty_results(
                    k, distances + qi * k, labels + qi * k, metric_type);
            continue;
        }

        auto cmp = [&](const std::pair<float, idx_t>& a,
                       const std::pair<float, idx_t>& b) {
            if (metric_type == faiss::METRIC_INNER_PRODUCT) {
                return a.first > b.first;
            }
            return a.first < b.first;
        };
        if (static_cast<int>(candidates.size()) > k) {
            std::nth_element(
                    candidates.begin(),
                    candidates.begin() + k,
                    candidates.end(),
                    cmp);
            candidates.resize(k);
        }
        std::sort(candidates.begin(), candidates.end(), cmp);
        for (int i = 0; i < k; ++i) {
            if (i < static_cast<int>(candidates.size())) {
                distances[qi * k + i] = candidates[i].first;
                labels[qi * k + i] = candidates[i].second;
            } else {
                fill_empty_results(
                        1, distances + qi * k + i, labels + qi * k + i, metric_type);
            }
        }
    }
}

void GpuIndexIVFFlat::setIndex_(
        GpuResources* resources,
        int dim,
        int nlist,
        faiss::MetricType metric,
        float metricArg,
        bool useResidual,
        faiss::ScalarQuantizer* scalarQ,
        bool interleavedLayout,
        IndicesOptions indicesOptions,
        MemorySpace space) {
    (void)resources;
    (void)metricArg;
    (void)useResidual;
    (void)scalarQ;
    (void)interleavedLayout;
    (void)indicesOptions;
    (void)space;

    if (!quantizer) {
        if (metric == faiss::METRIC_L2) {
            quantizer = new faiss::IndexFlatL2(dim);
        } else if (metric == faiss::METRIC_INNER_PRODUCT) {
            quantizer = new faiss::IndexFlatIP(dim);
        } else {
            FAISS_THROW_FMT("unsupported metric type %d", (int)metric);
        }
        own_fields = true;
    }

    auto index_ptr = std::make_unique<faiss::IndexIVFFlat>(
            quantizer, dim, nlist, metric);
    index_ptr->own_fields = true;
    setCpuIndex_(std::move(index_ptr));
}

} // namespace gpu
} // namespace faiss
