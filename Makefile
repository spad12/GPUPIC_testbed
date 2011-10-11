NVCC		:= /usr/local/cuda/bin/nvcc
#cdLD_LIBRARY_PATH	:= /usr/local/cuda/lib64
CUDA_INCLUDE_PATH	:= /home/josh/CUDA/Nubeam/include,/home/josh/lib,/home/josh/lib/cudpp
CUDAFORTRAN_FLAGS := -L$(LD_LIBRARY_PATH) -lcuda -I$(CUDA_INCLUDE_PATH)
PGPLOT_FLAGS := -L/usr/local/pgplot -lcpgplot -lpgplot -lX11 -lgcc -lm
PGPLOT_DIR = /usr/local/pgplot/
NVCCFLAGS	:= -m64 -O3 -run -gencode arch=compute_20,code=sm_20 --ptxas-options=-v -I$(CUDA_INCLUDE_PATH) 
RADIXSORTOBJDIR = radixSort/obj/x86_64/release/
RADIXSORTOBJ = $(RADIXSORTOBJDIR)radixsort.cpp.o $(RADIXSORTOBJDIR)radixsort.cu.o

all: cuda
	
cuda: $(RADIXSORTOBJ) gpumove.cu
	$(NVCC) gpumove.cu $(RADIXSORTOBJ) -L/home/josh/lib/cudpp -lcudpp_x86_64 -L/home/josh/lib -lcutil_x86_64 $(NVCCFLAGS)
	

