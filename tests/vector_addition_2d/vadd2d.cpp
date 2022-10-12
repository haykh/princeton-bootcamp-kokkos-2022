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
  double                duration;
  constexpr std::size_t Nx = (std::size_t)(1e4);
  constexpr std::size_t Ny = (std::size_t)(1e4);

#ifdef _OPENACC
  float** restrict A;
  float** restrict B;
  float** restrict C;

  A = (float**)malloc(Nx * sizeof(float*));
  B = (float**)malloc(Nx * sizeof(float*));
  C = (float**)malloc(Nx * sizeof(float*));

  for (std::size_t i = 0; i < Nx; ++i) {
    A[i] = (float*)malloc(Ny * sizeof(float));
    B[i] = (float*)malloc(Ny * sizeof(float));
    C[i] = (float*)malloc(Ny * sizeof(float));
  }

#else
  auto A = new float[Nx][Ny];
  auto B = new float[Nx][Ny];
  auto C = new float[Nx][Ny];
#endif

  for (std::size_t i {0}; i < Nx; ++i) {
    for (std::size_t j {0}; j < Ny; ++j) {
      A[i][j] = rand() / (float)RAND_MAX;
      B[i][j] = rand() / (float)RAND_MAX;
    }
  }

// clang-format off
#ifdef _OPENACC
  #pragma acc data copyin(A[0:Nx][0:Ny], B[0:Nx][0:Ny]), copyout(C[0:Nx][0:Ny])
#endif
  {
    auto start = std::chrono::high_resolution_clock::now();
  #ifdef _OPENACC
    #pragma acc parallel loop collapse(2)
  #elif _OPENMP
    #pragma omp parallel for collapse(2)
  #endif
    for (std::size_t i = 0; i < Nx; ++i) {
      for (std::size_t j = 0; j < Ny; ++j) {
        C[i][j] = A[i][j] + B[i][j];
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  // clang-format on

  for (std::size_t i {0}; i < Nx; ++i) {
    for (std::size_t j {0}; j < Ny; ++j) {
      if (fabs(C[i][j] - (A[i][j] + B[i][j])) > 1e-5) {
        std::cerr << DEVICE << " verification failed at element " << i << std::endl;
      }
    }
  }

  std::cout << DEVICE << " time: " << (double)(duration) / 1000.0 << " ms" << std::endl;

  free(A);
  free(B);
  free(C);

  return 0;
}