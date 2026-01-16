// @lint-ignore-every LICENSELINT
#include <gtest/gtest.h>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/distances.h>

TEST(MpsKnn, BruteForceL2) {
    constexpr int d = 4;
    constexpr int nb = 5;
    constexpr int nq = 3;
    constexpr int k = 2;
    float xb[nb * d] = {
            0.0f, 0.1f, 0.2f, 0.3f,
            1.0f, 1.1f, 1.2f, 1.3f,
            2.0f, 2.1f, 2.2f, 2.3f,
            3.0f, 3.1f, 3.2f, 3.3f,
            4.0f, 4.1f, 4.2f, 4.3f,
    };
    float xq[nq * d] = {
            0.05f, 0.1f, 0.2f, 0.35f,
            3.1f, 3.0f, 3.2f, 3.2f,
            2.4f, 2.5f, 2.6f, 2.7f,
    };

    float cpu_dist[nq * k];
    int64_t cpu_ids[nq * k];
    faiss::knn_L2sqr(xq, xb, d, nq, nb, k, cpu_dist, cpu_ids);

    float gpu_dist[nq * k];
    int64_t gpu_ids[nq * k];

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuDistanceParams params;
    params.metric = faiss::METRIC_L2;
    params.k = k;
    params.dims = d;
    params.vectors = xb;
    params.vectorType = faiss::gpu::DistanceDataType::F32;
    params.vectorsRowMajor = true;
    params.numVectors = nb;
    params.queries = xq;
    params.queryType = faiss::gpu::DistanceDataType::F32;
    params.queriesRowMajor = true;
    params.numQueries = nq;
    params.outDistances = gpu_dist;
    params.outIndices = gpu_ids;
    params.outIndicesType = faiss::gpu::IndicesDataType::I64;

    faiss::gpu::bfKnn(&res, params);

    for (int i = 0; i < nq * k; ++i) {
        ASSERT_EQ(cpu_ids[i], gpu_ids[i]);
        ASSERT_NEAR(cpu_dist[i], gpu_dist[i], 1e-5f);
    }
}
