BUILD_DIR=build
KOKKOS_PATH=../../extern/kokkos

TEST_NAME=vadd

SERIAL_COMPILER=g++ -Ofast
OMP_COMPILER=g++ -fopenmp -Ofast -march=native
OACC_COMPILER=nvc++ -acc -ta=tesla:cc70
CUDA_COMPILER=nvcc
KOKKOS_COMPILER=$(KOKKOS_PATH)/bin/nvcc_wrapper
KOKKOS_DEVICES="Cuda"
KOKKOS_ARCH="Volta70"
KOKKOS_CUDA_OPTIONS="enable_lambda"

SOURCE=${TEST_NAME}.cpp
CUDA_SOURCE=${TEST_NAME}.cu
KOKKOS_SOURCE=k${TEST_NAME}.cpp

SERIAL_TARGET=$(BUILD_DIR)/${TEST_NAME}.serial
OMP_TARGET=$(BUILD_DIR)/${TEST_NAME}.omp
OACC_TARGET=$(BUILD_DIR)/${TEST_NAME}.oacc
CUDA_TARGET=$(BUILD_DIR)/${TEST_NAME}.cuda
KOKKOS_TARGET=$(BUILD_DIR)/${TEST_NAME}.kokkos

TARGETS=${SERIAL_TARGET} ${OMP_TARGET} ${OACC_TARGET} ${CUDA_TARGET} ${KOKKOS_TARGET}

default: all

serial: ${SERIAL_TARGET}

omp: ${OMP_TARGET}

oacc: ${OACC_TARGET}

cuda: ${CUDA_TARGET}

kokkos: ${KOKKOS_TARGET}

${SERIAL_TARGET}: ${SOURCE}
	@mkdir -p ${BUILD_DIR}
	${SERIAL_COMPILER} $^ -o $@

${OMP_TARGET}: ${SOURCE}
	@mkdir -p ${BUILD_DIR}
	${OMP_COMPILER} $^ -o $@

${OACC_TARGET}: ${SOURCE}
	@mkdir -p ${BUILD_DIR}
	${OACC_COMPILER} $^ -o $@

${CUDA_TARGET}: ${CUDA_SOURCE}
	@mkdir -p ${BUILD_DIR}
	${CUDA_COMPILER} $^ -o $@

CXX=${KOKKOS_COMPILER}
include $(KOKKOS_PATH)/Makefile.kokkos
${KOKKOS_TARGET}: ${KOKKOS_SOURCE} ${KOKKOS_LINK_DEPENDS}
	@mkdir -p ${BUILD_DIR}
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(KOKKOS_LDFLAGS) $^ $(KOKKOS_LIBS) $(LIB) -o $@

all : ${TARGETS}

compare : all
	@bash ./compare.sh

clean:
	rm -rf ${BUILD_DIR}
	rm -f Kokkos* *.a *.o
