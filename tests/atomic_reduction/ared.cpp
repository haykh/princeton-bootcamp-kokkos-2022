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
  std::size_t NPrtls = (std::size_t)(1e8);
  std::size_t NBins  = (std::size_t)(1e6);

#ifdef _OPENACC
  float* restrict Particles;
  unsigned int* restrict Bins;
  unsigned int* restrict Bins_test;

  Particles = (float*)malloc(NPrtls * sizeof(float));
  Bins      = (unsigned int*)malloc(NBins * sizeof(unsigned int));
  Bins_test = (unsigned int*)malloc(NBins * sizeof(unsigned int));
#else
  auto Particles = new float[NPrtls];
  auto Bins      = new unsigned int[NBins];
  auto Bins_test = new unsigned int[NBins];
#endif

  for (std::size_t p {0}; p < NPrtls; ++p) {
    Particles[p] = rand() / (float)RAND_MAX;
  }

  for (std::size_t b {0}; b < NBins; ++b) {
    Bins[b]      = 0;
    Bins_test[b] = 0;
  }

// clang-format off
#ifdef _OPENACC
  #pragma acc data copyin(Particles[0:NPrtls], Bins[0:NBins]), copyout(Bins[0:NBins])
#endif
  {
    auto start = std::chrono::high_resolution_clock::now();
#ifdef _OPENACC
    #pragma acc parallel loop
#elif _OPENMP
    #pragma omp parallel for
#endif
    for (std::size_t p = 0; p < NPrtls; ++p) {
      std::size_t idx = (std::size_t)((double)(Particles[p]) * (double)(NBins));
#ifdef _OPENACC
      #pragma acc atomic update
#elif _OPENMP
      #pragma omp atomic
#endif
      Bins[idx] += 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  // clang-format on

  for (std::size_t p = 0; p < NPrtls; ++p) {
    std::size_t idx = (std::size_t)((double)(Particles[p]) * (double)(NBins));
    Bins_test[idx] += 1;
  }
  for (std::size_t b = 0; b < NBins; ++b) {
    if (Bins[b] != Bins_test[b]) {
      std::cerr << DEVICE << " verification failed at element " << b << std::endl;
    }
  }

  std::cout << DEVICE << " time: " << (double)(duration) / 1000.0 << " ms" << std::endl;

  free(Particles);
  free(Bins);

  return 0;
}