./build/vadd2d.serial > build/serial.out &&\
./build/vadd2d.omp > build/omp.out &&\
./build/vadd2d.oacc > build/oacc.out &&\
./build/vadd2d.kokkos > build/kokkos.out &&\
echo  &&\
echo "C[i,j] = A[i,j] + B[i,j] on 1e4 x 1e4 elements:" &&\
cat build/serial.out &&\
cat build/omp.out &&\
cat build/oacc.out &&\
cat build/kokkos.out