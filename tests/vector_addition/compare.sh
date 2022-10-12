./build/vadd.serial > build/serial.out &&\
./build/vadd.omp > build/omp.out &&\
./build/vadd.oacc > build/oacc.out &&\
./build/vadd.cuda > build/cuda.out &&\
./build/vadd.kokkos > build/kokkos.out &&\
echo  &&\
echo "C[i] = A[i] + B[i] on 1e8 elements:" &&\
cat build/serial.out &&\
cat build/omp.out &&\
cat build/oacc.out &&\
cat build/cuda.out &&\
cat build/kokkos.out