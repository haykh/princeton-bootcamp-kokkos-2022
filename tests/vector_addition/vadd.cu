#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define DEVICE "CUDA"

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

auto main() -> int {
  unsigned long duration;
  auto   N        = (std::size_t)(1e8);
  auto   mem_size = (std::size_t)(N * sizeof(float));

  auto h_A = new float[N];
  auto h_B = new float[N];
  auto h_C = new float[N];

  for (std::size_t i = 0; i < N; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;

  cudaMalloc((void**)&d_A, mem_size);
  cudaMalloc((void**)&d_B, mem_size);
  cudaMalloc((void**)&d_C, mem_size);

  cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 512;
  int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

  auto start = std::chrono::high_resolution_clock::now();
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  cudaMemcpy(h_C, d_C, mem_size, cudaMemcpyDeviceToHost);

  for (std::size_t i {0}; i < N; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      std::cerr << DEVICE << " verification failed at element " << i << std::endl;
    }
  }

  std::cout << DEVICE << " time: " << (double)(duration) / 1000.0 << " ms" << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}
