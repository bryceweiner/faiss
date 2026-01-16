# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

target_sources(faiss_c PRIVATE
  DeviceUtils_c.cpp
  GpuAutoTune_c.cpp
  GpuClonerOptions_c.cpp
  GpuIndex_c.cpp
  GpuResources_c.cpp
  StandardGpuResources_c.cpp
)

file(GLOB FAISS_C_API_GPU_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
faiss_install_headers("${FAISS_C_API_GPU_HEADERS}" c_api/gpu)

if (FAISS_ENABLE_MPS)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
  find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  target_link_libraries(
    faiss_c
    PUBLIC
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${MPSGRAPH_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK})
  target_link_libraries(
    faiss_c_avx2
    PUBLIC
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${MPSGRAPH_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK})
  target_link_libraries(
    faiss_c_avx512
    PUBLIC
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${MPSGRAPH_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK})
  target_link_libraries(
    faiss_c_avx512_spr
    PUBLIC
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${MPSGRAPH_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK})
  target_link_libraries(
    faiss_c_sve
    PUBLIC
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${MPSGRAPH_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK})
elseif (FAISS_ENABLE_ROCM)
  target_link_libraries(faiss_c PUBLIC hip::host roc::hipblas)
  target_link_libraries(faiss_c_avx2 PUBLIC hip::host roc::hipblas)
  target_link_libraries(faiss_c_avx512 PUBLIC hip::host roc::hipblas)
  target_link_libraries(faiss_c_avx512_spr PUBLIC hip::host roc::hipblas)
  target_link_libraries(faiss_c_sve PUBLIC hip::host roc::hipblas)
else()
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(faiss_c PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
  target_link_libraries(faiss_c_avx2 PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
  target_link_libraries(faiss_c_avx512 PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
  target_link_libraries(faiss_c_avx512_spr PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
  target_link_libraries(faiss_c_sve PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
endif()

add_executable(example_gpu_c EXCLUDE_FROM_ALL example_gpu_c.c)
target_link_libraries(example_gpu_c PRIVATE faiss_c)
