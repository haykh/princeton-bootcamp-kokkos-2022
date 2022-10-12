# About Kokkos @ Princeton Bootcamp

Slides for the talk are available in pdf and fig formats.

## Resources

* [Kokkos github](https://github.com/kokkos/kokkos)
* [Kokkos wiki](https://kokkos.github.io/kokkos-core-wiki/)
* [Kokkos lecture videos](https://www.youtube.com/watch?v=rUIcWtFU5qM)
* [Kokkos tutorials](https://github.com/kokkos/kokkos-tutorials)
* [Kokkos team slack](https://kokkosteam.slack.com/)

## Compiling and running the tests

> _Disclaimer_: some of the tests contain lines of code that are very far from following the best practices. They are there to test the compilation process and the runtime, not to teach you how to write good code. 

Tests can be compiled using both CMake and GNU Makefile. To build with the latter simply proceed to the test directory and run 

```shell
# for GNU Makefile
make all -j
make compare
```

This will run the appropriate test problem and compare the runtimes.

The CMake build, while being the recommended one, requires an installation of `kokkos` (only for the purposes of this tutorial). This can be done by proceeding to the `extern/kokkos` directory and running the following:

```shell
# to install kokkos for CMake build
cmake -B build -D CMAKE_INSTALL_PREFIX=../kokkos-install -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_CUDA_LAMBDA=ON -D CMAKE_CXX_STANDARD=17
cd build
make -j
make install
```

This will install `kokkos` in a temporary directory, which can then be found by the CMake. After that proceed to the test directory and run:

```shell
# for CMake
cmake -B build
cd build
make -j
cd ..
bash ../compare.sh
```

> In-tree builds are [fully supported](https://kokkos.github.io/kokkos-core-wiki/building.html) by `kokkos` itself, but I found out they conflict with pure `CUDA` builds performed in these tests, which is why I had to simplify the build process for the purposes of this tutorial. 
