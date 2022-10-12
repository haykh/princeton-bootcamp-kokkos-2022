cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME "vector add comparison for 2d")
set(SHORT_NAME "vadd2d")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(SOURCE "${SHORT_NAME}.cpp")
set(CUDA_SOURCE "${SHORT_NAME}.cu")
set(KOKKOS_SOURCE "k${SHORT_NAME}.cpp")

set(SERIAL_TARGET "${SHORT_NAME}.serial")
add_executable(${SERIAL_TARGET} ${SOURCE})
target_compile_options(${SERIAL_TARGET} PRIVATE -O3)

find_package(OpenMP REQUIRED)
set(OMP_TARGET "${SHORT_NAME}.omp")
add_executable(${OMP_TARGET})
target_sources(${OMP_TARGET} PRIVATE ${SOURCE})
target_compile_options(${OMP_TARGET} PRIVATE -fopenmp -O3)
target_link_libraries(${OMP_TARGET} PRIVATE OpenMP::OpenMP_CXX)

find_package(OpenACC REQUIRED)
set(OACC_TARGET "${SHORT_NAME}.oacc")
add_executable(${OACC_TARGET})
target_sources(${OACC_TARGET} PRIVATE ${SOURCE})
target_compile_options(${OACC_TARGET} PRIVATE -acc -ta=tesla:cc70)
target_link_libraries(${OACC_TARGET} PRIVATE OpenACC::OpenACC_CXX)

set(Kokkos_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../extern/kokkos-install/lib64/cmake/Kokkos/")
find_package(Kokkos REQUIRED)
set(KOKKOS_TARGET "${SHORT_NAME}.kokkos")
add_executable(${KOKKOS_TARGET})
target_sources(${KOKKOS_TARGET} PRIVATE ${KOKKOS_SOURCE})
target_link_libraries(${KOKKOS_TARGET} PRIVATE Kokkos::kokkos)