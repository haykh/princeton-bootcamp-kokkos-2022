./build/ared.serial > build/serial.out &&\
./build/ared.omp > build/omp.out &&\
./build/ared.oacc > build/oacc.out &&\
./build/ared.cuda > build/cuda.out &&\
./build/ared.kokkos > build/kokkos.out &&\
echo  &&\
echo "Bins[Particles[i]] += 1 on 1e8 particles and 1e6 bins:" &&\
cat build/serial.out &&\
cat build/omp.out &&\
cat build/oacc.out &&\
cat build/cuda.out &&\
cat build/kokkos.out