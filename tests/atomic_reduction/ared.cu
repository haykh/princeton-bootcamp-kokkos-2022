#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define DEVICE "CUDA"

__global__ void
atomicReduce(const float* Particles, int* Bins, std::size_t nprtls, std::size_t nbins) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < nprtls) {
    std::size_t bin = (std::size_t)((double)(Particles[p]) * (double)(nbins));
    atomicAdd(&Bins[bin], 1);
  }
}

auto main() -> int {
  unsigned long duration;
  auto          NPrtls         = (std::size_t)(1e8);
  auto          NBins          = (std::size_t)(1e6);
  auto          prtls_mem_size = (std::size_t)(NPrtls * sizeof(float));
  auto          bins_mem_size  = (std::size_t)(NBins * sizeof(int));

  auto h_Particles = new float[NPrtls];
  auto h_Bins      = new int[NBins];

  for (std::size_t p = 0; p < NPrtls; ++p) {
    h_Particles[p] = rand() / (float)RAND_MAX;
  }
  for (std::size_t b = 0; b < NBins; ++b) {
    h_Bins[b] = 0;
  }

  float* d_Particles = nullptr;
  int*   d_Bins      = nullptr;

  cudaMalloc((void**)&d_Particles, prtls_mem_size);
  cudaMalloc((void**)&d_Bins, bins_mem_size);

  cudaMemcpy(d_Particles, h_Particles, prtls_mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Bins, h_Bins, bins_mem_size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid   = (NPrtls + threadsPerBlock - 1) / threadsPerBlock;

  auto start = std::chrono::high_resolution_clock::now();
  atomicReduce<<<blocksPerGrid, threadsPerBlock>>>(d_Particles, d_Bins, NPrtls, NBins);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  cudaMemcpy(h_Bins, d_Bins, bins_mem_size, cudaMemcpyDeviceToHost);

  std::cout << DEVICE << " time: " << (double)(duration) / 1000.0 << " ms" << std::endl;

  cudaFree(d_Particles);
  cudaFree(d_Bins);

  delete[] h_Particles;
  delete[] h_Bins;

  return 0;
}