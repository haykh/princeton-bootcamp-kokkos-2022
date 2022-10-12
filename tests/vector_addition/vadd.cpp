#include <iostream>
#include <cmath>
#include <chrono>

#ifdef _OPENACC
#  define DEVICE "OpenACC (GPU)"
#elif _OPENMP
#  define DEVICE "OpenMP (CPU)"
#else
#  define DEVICE "Serial (CPU)"
#endif

#include <omp.h>

int main() {
  double      duration;
  std::size_t N = (std::size_t)(1e8);

#ifdef _OPENACC
  float* restrict A;
  float* restrict B;
  float* restrict C;

  A = (float*)malloc(N * sizeof(float));
  B = (float*)malloc(N * sizeof(float));
  C = (float*)malloc(N * sizeof(float));
#else
  auto A = new float[N];
  auto B = new float[N];
  auto C = new float[N];
#endif

  for (std::size_t i {0}; i < N; ++i) {
    A[i] = rand() / (float)RAND_MAX;
    B[i] = rand() / (float)RAND_MAX;
  }

// clang-format off
#ifdef _OPENACC
  #pragma acc data copyin(A[0:N], B[0:N]), copyout(C[0:N])
#endif
  {
    auto start = std::chrono::high_resolution_clock::now();
#ifdef _OPENACC
    #pragma acc parallel loop
#elif _OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t i = 0; i < N; ++i) {
      C[i] = A[i] + B[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  // clang-format on

  for (std::size_t i {0}; i < N; ++i) {
    if (fabs(A[i] + B[i] - C[i]) > 1e-5) {
      std::cerr << DEVICE << " verification failed at element " << i << std::endl;
    }
  }

  std::cout << DEVICE << " time: " << (double)(duration) / 1000.0 << " ms" << std::endl;

  free(A);
  free(B);
  free(C);

  return 0;
}