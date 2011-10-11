NVCC		:= /usr/local/cuda/bin/nvcc
#cdLD_LIBRARY_PATH	:= /usr/local/cuda/lib64
CUDA_INCLUDE_PATH	:= /home/josh/lib
CUDAFORTRAN_FLAGS := -L$(LD_LIBRARY_PATH) -lcuda -I$(CUDA_INCLUDE_PATH)
PGPLOT_FLAGS := -L/usr/local/pgplot -lcpgplot -lpgplot -lX11 -lgcc -lm
PGPLOT_DIR = /usr/local/pgplot/
NVCCFLAGS	:=  -m64 -O3 -pg -g -G -use_fast_math -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -ccbin /opt/intel/Compiler/11.1/073/bin/intel64/icc -I$(CUDA_INCLUDE_PATH) 
CC = icc
CPP = icpc

all: operator_costs scan_test structure_test
	
operator_costs: operator_costs.cu
	$(NVCC) operator_costs.cu  -L/home/josh/lib -lcutil_x86_64 -L/usr/local/cuda/lib64 -lcudart $(NVCCFLAGS)
	
scan_test: scan_test.cu
	$(NVCC) scan_test.cu  -L/home/josh/lib -lcutil_x86_64 -L/usr/local/cuda/lib64 -lcudart $(NVCCFLAGS)

structure_test: structure_test.cu
	$(NVCC) structure_test.cu -o structure_test -L/home/josh/lib -lcutil_x86_64 -L/usr/local/cuda/lib64 -lcudart $(NVCCFLAGS)

