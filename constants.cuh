#define __CUDACC__
#define __cplusplus


#include "cudamatrix_types.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include "cuPrintf.cu"
#include <ctime>
#include <cstring>
#include "cuda.h"
#include "cutil.h"
#include "cuda_runtime.h"
#include "common_functions.h"
#include "sm_20_intrinsics.h"
#include "host_defines.h"
#include <iostream>
#include "math.h"

#  define CUDA_SAFE_KERNEL(call) {                                         \
	call;																					\
	cudaDeviceSynchronize();														\
	cudaError err = cudaGetLastError();										\
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
                exit(EXIT_FAILURE);                                                  \
    } }



const int threadsPerBlock = 256;

template<typename T>
__device__
T reduce(T* shared_array,T* temp)
{
	int i = blockDim.x/2;
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+threadIdx.x;

	T result;

	__syncthreads();

	while (i != 0)
	{
		if (idx < i)
		{
			shared_array[idx] += shared_array[idx+i];
			__threadfence_block();
		}
		i/=2;
		__syncthreads();
	}

		result = shared_array[0];
	__syncthreads();
	return result;
}
