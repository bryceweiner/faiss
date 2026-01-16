// @lint-ignore-every LICENSELINT
#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

TEST(MpsIndexFlat, BasicSearchL2) {
    constexpr int d = 4;
    constexpr int nb = 5;
    constexpr int nq = 2;
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
    };

    faiss::IndexFlat cpu_index(d, faiss::METRIC_L2);
    cpu_index.add(nb, xb);

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlat gpu_index(&res, d, faiss::METRIC_L2);
    gpu_index.add(nb, xb);

    constexpr int k = 2;
    float cpu_dist[nq * k];
    faiss::idx_t cpu_ids[nq * k];
    float gpu_dist[nq * k];
    faiss::idx_t gpu_ids[nq * k];

    cpu_index.search(nq, xq, k, cpu_dist, cpu_ids);
    gpu_index.search(nq, xq, k, gpu_dist, gpu_ids);

    for (int i = 0; i < nq * k; ++i) {
        ASSERT_EQ(cpu_ids[i], gpu_ids[i]);
        ASSERT_NEAR(cpu_dist[i], gpu_dist[i], 1e-5f);
    }
}
