/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/clone_index.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
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
        FAISS_THROW_MSG("MPS IVFPQ requires a CPU coarse quantizer");
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

GpuIndexIVFPQ::GpuIndexIVFPQ(
        GpuResourcesProvider* provider,
        const faiss::IndexIVFPQ* index,
        GpuIndexIVFPQConfig config)
        : GpuIndexIVF(
                  provider,
                  index->d,
                  index->metric_type,
                  index->metric_arg,
                  index->nlist,
                  config),
          ivfpqConfig_(config),
          usePrecomputedTables_(index->use_precomputed_table != 0),
          subQuantizers_(index->pq.M),
          bitsPerCode_(index->pq.nbits),
          reserveMemoryVecs_(0) {
    copyFrom(index);
}

GpuIndexIVFPQ::GpuIndexIVFPQ(
        GpuResourcesProvider* provider,
        int dims,
        idx_t nlist,
        idx_t subQuantizers,
        idx_t bitsPerCode,
        faiss::MetricType metric,
        GpuIndexIVFPQConfig config)
        : GpuIndexIVF(provider, dims, metric, 0.0f, nlist, config),
          ivfpqConfig_(config),
          usePrecomputedTables_(config.usePrecomputedTables),
          subQuantizers_(static_cast<int>(subQuantizers)),
          bitsPerCode_(static_cast<int>(bitsPerCode)),
          reserveMemoryVecs_(0) {
    setIndex_(
            getResources().get(),
            dims,
            nlist,
            metric,
            metric_arg,
            subQuantizers_,
            bitsPerCode_,
            config.useFloat16LookupTables,
            config.useMMCodeDistance,
            config.interleavedLayout,
            nullptr,
            config.indicesOptions,
            config_.memorySpace);
}

GpuIndexIVFPQ::GpuIndexIVFPQ(
        GpuResourcesProvider* provider,
        Index* coarseQuantizer,
        int dims,
        idx_t nlist,
        idx_t subQuantizers,
        idx_t bitsPerCode,
        faiss::MetricType metric,
        GpuIndexIVFPQConfig config)
        : GpuIndexIVF(provider, dims, metric, 0.0f, nlist, config),
          ivfpqConfig_(config),
          usePrecomputedTables_(config.usePrecomputedTables),
          subQuantizers_(static_cast<int>(subQuantizers)),
          bitsPerCode_(static_cast<int>(bitsPerCode)),
          reserveMemoryVecs_(0) {
    if (own_fields && quantizer) {
        delete quantizer;
        quantizer = nullptr;
        own_fields = false;
    }
    auto cpu_quantizer = clone_quantizer(coarseQuantizer);
    FAISS_THROW_IF_NOT_MSG(cpu_quantizer, "Missing coarse quantizer");
    quantizer = cpu_quantizer.get();
    auto index_ptr = std::make_unique<faiss::IndexIVFPQ>(
            quantizer,
            dims,
            nlist,
            subQuantizers_,
            bitsPerCode_,
            metric);
    index_ptr->own_fields = true;
    cpu_quantizer.release();
    setCpuIndex_(std::move(index_ptr));
    pq = dynamic_cast<faiss::IndexIVFPQ*>(getCpuIndex_())->pq;
}

GpuIndexIVFPQ::~GpuIndexIVFPQ() = default;

void GpuIndexIVFPQ::copyFrom(const faiss::IndexIVFPQ* index) {
    GpuIndexIVF::copyFrom(index);
    std::unique_ptr<IndexIVF> cloned(
            dynamic_cast<IndexIVF*>(faiss::clone_index(index)));
    FAISS_THROW_IF_NOT_MSG(cloned, "Failed to clone IndexIVFPQ");
    setCpuIndex_(std::move(cloned));
    auto* cpu_index = dynamic_cast<faiss::IndexIVFPQ*>(getCpuIndex_());
    FAISS_THROW_IF_NOT_MSG(cpu_index, "CPU IVFPQ index type mismatch");
    pq = cpu_index->pq;
    subQuantizers_ = cpu_index->pq.M;
    bitsPerCode_ = cpu_index->pq.nbits;
    usePrecomputedTables_ = cpu_index->use_precomputed_table != 0;
}

void GpuIndexIVFPQ::copyTo(faiss::IndexIVFPQ* index) const {
    GpuIndexIVF::copyTo(index);
    auto* cpu_index = dynamic_cast<faiss::IndexIVFPQ*>(getCpuIndex_());
    FAISS_THROW_IF_NOT_MSG(cpu_index, "CPU IVFPQ index type mismatch");
    index->pq = cpu_index->pq;
    index->use_precomputed_table = cpu_index->use_precomputed_table;
    index->scan_table_threshold = cpu_index->scan_table_threshold;
    index->polysemous_ht = cpu_index->polysemous_ht;
}

void GpuIndexIVFPQ::reserveMemory(size_t numVecs) {
    reserveMemoryVecs_ = numVecs;
}

void GpuIndexIVFPQ::setPrecomputedCodes(bool enable) {
    usePrecomputedTables_ = enable;
    auto* cpu_index = dynamic_cast<faiss::IndexIVFPQ*>(getCpuIndex_());
    if (cpu_index) {
        cpu_index->use_precomputed_table = enable ? 1 : 0;
    }
}

bool GpuIndexIVFPQ::getPrecomputedCodes() const {
    return usePrecomputedTables_;
}

int GpuIndexIVFPQ::getNumSubQuantizers() const {
    return subQuantizers_;
}

int GpuIndexIVFPQ::getBitsPerCode() const {
    return bitsPerCode_;
}

int GpuIndexIVFPQ::getCentroidsPerSubQuantizer() const {
    return 1 << bitsPerCode_;
}

size_t GpuIndexIVFPQ::reclaimMemory() {
    return 0;
}

void GpuIndexIVFPQ::reset() {
    if (getCpuIndex_()) {
        getCpuIndex_()->reset();
        ntotal = 0;
        is_trained = getCpuIndex_()->is_trained;
    }
}

void GpuIndexIVFPQ::updateQuantizer() {
    if (getCpuIndex_()) {
        quantizer = getCpuIndex_()->quantizer;
    }
}

void GpuIndexIVFPQ::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(getCpuIndex_(), "CPU IVF index is not initialized");
    getCpuIndex_()->train(n, x);
    is_trained = getCpuIndex_()->is_trained;
    auto* cpu_index = dynamic_cast<faiss::IndexIVFPQ*>(getCpuIndex_());
    if (cpu_index) {
        pq = cpu_index->pq;
    }
}

void GpuIndexIVFPQ::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    DeviceScope scope(getDevice());
    auto* cpu_index = dynamic_cast<faiss::IndexIVFPQ*>(getCpuIndex_());
    FAISS_THROW_IF_NOT_MSG(cpu_index, "CPU IVFPQ index is not initialized");
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

            std::vector<float> decoded(list_size * d);
            std::vector<idx_t> list_keys(list_size, list_id);
            cpu_index->decode_multiple(
                    list_size, list_keys.data(), codes, decoded.data());

            int list_k = std::min<int>(k, list_size);
            std::vector<float> list_dist(list_k);
            std::vector<int64_t> list_idx(list_k);

            faiss::gpu::GpuDistanceParams args;
            args.metric = metric_type;
            args.k = list_k;
            args.dims = d;
            args.vectors = decoded.data();
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

void GpuIndexIVFPQ::setIndex_(
        GpuResources* resources,
        int dim,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        int numSubQuantizers,
        int bitsPerSubQuantizer,
        bool useFloat16LookupTables,
        bool useMMCodeDistance,
        bool interleavedLayout,
        float* pqCentroidData,
        IndicesOptions indicesOptions,
        MemorySpace space) {
    (void)resources;
    (void)metricArg;
    (void)useFloat16LookupTables;
    (void)useMMCodeDistance;
    (void)interleavedLayout;
    (void)pqCentroidData;
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

    auto index_ptr = std::make_unique<faiss::IndexIVFPQ>(
            quantizer, dim, nlist, numSubQuantizers, bitsPerSubQuantizer, metric);
    index_ptr->own_fields = true;
    setCpuIndex_(std::move(index_ptr));
    auto* cpu_index = dynamic_cast<faiss::IndexIVFPQ*>(getCpuIndex_());
    if (cpu_index) {
        pq = cpu_index->pq;
    }
}

void GpuIndexIVFPQ::verifyPQSettings_() const {}

void GpuIndexIVFPQ::trainResidualQuantizer_(idx_t n, const float* x) {
    train(n, x);
}

} // namespace gpu
} // namespace faiss
