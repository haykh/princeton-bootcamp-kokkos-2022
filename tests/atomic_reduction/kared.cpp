#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>
#include <chrono>

#define DEVICE "Kokkos (GPU)"

#define Lambda KOKKOS_LAMBDA
using array_t = Kokkos::View<float*>;
using index_t = const std::size_t;

auto main() -> int {
  unsigned long duration;
  auto          NPrtls = (std::size_t)(1e8);
  auto          NBins  = (std::size_t)(1e6);
  Kokkos::initialize();
  {
    array_t Particles("Particles", NPrtls);
    array_t Bins("Bins", NBins);

    auto Particles_h = Kokkos::create_mirror_view(Particles);
    auto Bins_h      = Kokkos::create_mirror_view(Bins);

    for (std::size_t p {0}; p < NPrtls; ++p) {
      Particles_h(p) = rand() / (float)RAND_MAX;
    }

    for (std::size_t b {0}; b < NBins; ++b) {
      Bins_h(b) = 0;
    }

    Kokkos::deep_copy(Particles, Particles_h);
    Kokkos::deep_copy(Bins, Bins_h);

    auto Bins_scatter = Kokkos::Experimental::create_scatter_view(Bins);

    auto start = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for(
      "histogram", NPrtls, Lambda(index_t p) {
        auto bins_access = Bins_scatter.access();
        auto idx         = (std::size_t)((double)(Particles(p)) * (double)(NBins));
        bins_access(idx) += 1;
      });
    Kokkos::Experimental::contribute(Bins, Bins_scatter);
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << DEVICE << " time: " << (double)(duration) / 1000.0 << " ms" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}