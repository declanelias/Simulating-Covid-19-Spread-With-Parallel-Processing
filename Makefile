CXX=g++
PGCC=pgc++
MPICC=mpicc
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGET=modlemc
CC_ACC_M= -fast -ta=tesla:cc75,managed -Minfo=accel -lcurand -Mcuda


all: $(TARGET)  modlemc_mpi  GPUmodle  fillRand.o

$(TARGET): modlemc.cpp
	$(CXX) -o $(TARGET) modlemc.cpp -I$(INCLUDE) -l$(LIB)


modlemc_mpi: modlemc_mpi.cpp
	mpic++ modlemc_mpi.cpp -o modlemc_mpi -I$(INCLUDE) -l$(LIB)

fillRand.o: fillRand.cu
	nvcc -c fillRand.cu -o fillRand.o

GPUmodle: modlemc_openacc.cpp fillRand.cu
	$(PGCC) ${CC_ACC_M} -o GPUmodle modlemc_openacc.cpp
######### clean
clean:
	rm -f $(TARGET) GPUmodle fillRand.o