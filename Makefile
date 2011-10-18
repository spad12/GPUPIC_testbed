NVCC		:= /usr/local/cuda/bin/nvcc
#cdLD_LIBRARY_PATH	:= /usr/local/cuda/lib64
CUDA_INCLUDE_PATH	:= /home/josh/CUDA/Nubeam/include,/home/josh/lib,/home/josh/lib/cudpp
CUDAFORTRAN_FLAGS := -L$(LD_LIBRARY_PATH) -lcuda -I$(CUDA_INCLUDE_PATH)
PGPLOT_FLAGS := -L/usr/local/pgplot -lcpgplot -lpgplot -lX11 -lgcc -lm
PGPLOT_DIR = /usr/local/pgplot/
NVCCFLAGS	:=  -m64 -O3 --use_fast_math -Xptxas -O3 -Xptxas -maxrregcount=40 -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -ccbin /opt/intel/Compiler/11.1/073/bin/intel64/icc -I$(CUDA_INCLUDE_PATH) 
CC = icc
CPP = icpc

all: cuda
	
cuda: $(RADIXSORTOBJ) gpumove.cu
	$(NVCC) gpumove.cu $(RADIXSORTOBJ) -L/home/josh/lib -lcutil_x86_64 $(NVCCFLAGS)
	

