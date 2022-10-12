#include <Kokkos_Core.hpp>

#include <iostream>
#include <chrono>

#define DEVICE "Kokkos (GPU)"

#define Lambda KOKKOS_LAMBDA
using array_t = Kokkos::View<float*>;
using index_t = const std::size_t;

auto main() -> int {
  unsigned long duration;
  auto          N = (std::size_t)(1e8);
  Kokkos::initialize();
  {
    array_t A("A", N);
    array_t B("B", N);
    array_t C("C", N);

    auto A_h = Kokkos::create_mirror_view(A);
    auto B_h = Kokkos::create_mirror_view(B);
    auto C_h = Kokkos::create_mirror_view(C);

    for (std::size_t i {0}; i < N; ++i) {
      A_h(i) = rand() / (float)RAND_MAX;
      B_h(i) = rand() / (float)RAND_MAX;
    }

    Kokkos::deep_copy(A, A_h);
    Kokkos::deep_copy(B, B_h);

    auto start = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for(
      N, Lambda(index_t i) { C(i) = A(i) + B(i); });
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    Kokkos::deep_copy(C_h, C);
    for (std::size_t i {0}; i < N; ++i) {
      if (fabs(A_h(i) + B_h(i) - C_h(i)) > 1e-5) {
        std::cerr << DEVICE << " verification failed at element " << i << std::endl;
      }
    }

    std::cout << DEVICE << " time: " << (double)(duration) / 1000.0 << " ms" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}