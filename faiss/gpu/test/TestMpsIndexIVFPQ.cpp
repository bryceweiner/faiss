// @lint-ignore-every LICENSELINT
#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>

TEST(MpsIndexIVFPQ, BasicSearch) {
    constexpr int d = 8;
    constexpr int nb = 128;
    constexpr int nq = 6;
    constexpr int nlist = 8;
    constexpr int m = 2;
    constexpr int nbits = 4;
    constexpr int k = 3;

    std::vector<float> xb(nb * d);
    for (int i = 0; i < nb * d; ++i) {
        xb[i] = static_cast<float>(i) * 0.005f;
    }
    std::vector<float> xq(nq * d);
    for (int i = 0; i < nq * d; ++i) {
        xq[i] = static_cast<float>(i) * 0.007f;
    }

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFPQ cpu_index(&quantizer, d, nlist, m, nbits, faiss::METRIC_L2);
    cpu_index.train(nb, xb.data());
    cpu_index.add(nb, xb.data());
    cpu_index.nprobe = 4;

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexIVFPQ gpu_index(&res, d, nlist, m, nbits, faiss::METRIC_L2);
    gpu_index.train(nb, xb.data());
    gpu_index.add(nb, xb.data());
    gpu_index.nprobe = 4;

    std::vector<float> cpu_dist(nq * k);
    std::vector<faiss::idx_t> cpu_ids(nq * k);
    std::vector<float> gpu_dist(nq * k);
    std::vector<faiss::idx_t> gpu_ids(nq * k);

    cpu_index.search(nq, xq.data(), k, cpu_dist.data(), cpu_ids.data());
    gpu_index.search(nq, xq.data(), k, gpu_dist.data(), gpu_ids.data());

    for (int i = 0; i < nq * k; ++i) {
        ASSERT_EQ(cpu_ids[i], gpu_ids[i]);
        ASSERT_NEAR(cpu_dist[i], gpu_dist[i], 1e-5f);
    }
}
