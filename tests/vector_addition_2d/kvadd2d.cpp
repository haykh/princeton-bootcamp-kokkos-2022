#include <Kokkos_Core.hpp>

#include <iostream>
#include <chrono>

#define DEVICE "Kokkos (GPU)"

#define Lambda KOKKOS_LAMBDA
using array_t = Kokkos::View<float**>;
using index_t = const std::size_t;

auto main() -> int {
  unsigned long duration;
  auto          Nx = (std::size_t)(1e4);
  auto          Ny = (std::size_t)(1e4);
  Kokkos::initialize();
  {
    array_t A("A", Nx, Ny);
    array_t B("B", Nx, Ny);
    array_t C("C", Nx, Ny);

    auto A_h = Kokkos::create_mirror_view(A);
    auto B_h = Kokkos::create_mirror_view(B);
    auto C_h = Kokkos::create_mirror_view(C);

    for (std::size_t i {0}; i < Nx; ++i) {
      for (std::size_t j {0}; j < Ny; ++j) {
        A_h(i, j) = rand() / (float)RAND_MAX;
        B_h(i, j) = rand() / (float)RAND_MAX;
      }
    }

    Kokkos::deep_copy(A, A_h);
    Kokkos::deep_copy(B, B_h);

    auto start = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
      Lambda(index_t i, index_t j) { C(i, j) = A(i, j) + B(i, j); });
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    Kokkos::deep_copy(C_h, C);
    for (std::size_t i {0}; i < Nx; ++i) {
      for (std::size_t j {0}; j < Ny; ++j) {
        if (fabs(A_h(i, j) + B_h(i, j) - C_h(i, j)) > 1e-5) {
          std::cerr << DEVICE << " verification failed at element " << i << std::endl;
        }
      }
    }

    std::cout << DEVICE << " time: " << (double)(duration) / 1000.0 << " ms" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}