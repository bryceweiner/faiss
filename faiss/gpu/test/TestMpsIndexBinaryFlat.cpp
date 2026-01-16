// @lint-ignore-every LICENSELINT
#include <gtest/gtest.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/gpu/GpuIndexBinaryFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

TEST(MpsIndexBinaryFlat, BasicSearch) {
    constexpr int d = 64;
    constexpr int nb = 4;
    constexpr int nq = 2;
    uint8_t xb[nb * d / 8] = {
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
            0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
    };
    uint8_t xq[nq * d / 8] = {
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    };

    faiss::IndexBinaryFlat cpu_index(d);
    cpu_index.add(nb, xb);

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexBinaryFlat gpu_index(&res, d);
    gpu_index.add(nb, xb);

    constexpr int k = 2;
    int32_t cpu_dist[nq * k];
    faiss::idx_t cpu_ids[nq * k];
    int32_t gpu_dist[nq * k];
    faiss::idx_t gpu_ids[nq * k];

    cpu_index.search(nq, xq, k, cpu_dist, cpu_ids);
    gpu_index.search(nq, xq, k, gpu_dist, gpu_ids);

    for (int i = 0; i < nq * k; ++i) {
        ASSERT_EQ(cpu_ids[i], gpu_ids[i]);
        ASSERT_EQ(cpu_dist[i], gpu_dist[i]);
    }
}
