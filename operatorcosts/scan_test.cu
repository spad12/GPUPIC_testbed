#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <ctime>
#include <cstring>
#include "cuda.h"
#include "common_functions.h"
#include "sm_20_intrinsics.h"
#include "host_defines.h"
#include <iostream>
#include <curand_kernel.h>
#include "curand.h"
#include "cuda_texture_types.h"
#include "texture_fetch_functions.h"
#include "builtin_types.h"
#include "cutil.h"
#include "device_functions.h"

#define BLOCK_SIZE 512

#  define CUDA_SAFE_KERNEL(call) {                                         \
	call;																					\
	cudaDeviceSynchronize();														\
	cudaError err = cudaGetLastError();										\
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
                exit(EXIT_FAILURE);                                                  \
    } }

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2*LOG_NUM_BANKS))

template<typename T>
__device__
void prescan0(T* temp,int n)
{

	unsigned int idx = threadIdx.x;
	unsigned int offset = 1;


	for(int d = n>>1; d>0;d>>=1)
	{
		__syncthreads();

		if(idx < d)
		{
			int ai = offset*(2*idx+1)-1;
			int bi = offset*(2*idx+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}
		offset*=2;


	}

	if(idx==0){temp[n-1+CONFLICT_FREE_OFFSET(n-1)] = 0;}

	for(int d=1;d<n;d*=2)
	{
		offset >>= 1;
		__syncthreads();

		if(idx<d)
		{
			int ai = offset*(2*idx+1)-1;
			int bi = offset*(2*idx+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			unsigned int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

}


template<typename T,int blocksize>
__device__
void prescanfast(T* sdatafast,int n)
{

	volatile T* vsdatafast = sdatafast;

	__shared__ int soffset;

	unsigned int idx = threadIdx.x;
	unsigned int offset = 1;
	unsigned int ai;
	unsigned int bi;
	T t;

	// Because we know how big the block is, we can just unroll these loops

	__syncthreads();
	if(blocksize>=512)
	{
		if(idx<256)
		{
			ai = offset*(2*idx+1)-1;
			bi = offset*(2*idx+2)-1;

			sdatafast[bi] += sdatafast[ai];
		}
		offset*=2;
		__syncthreads();
	}

	if(blocksize>=256)
	{
		if(idx<128)
		{
			ai = offset*(2*idx+1)-1;
			bi = offset*(2*idx+2)-1;

			sdatafast[bi] += sdatafast[ai];
		}
		offset*=2;
		__syncthreads();
	}

	if(blocksize>=128)
	{
		if(idx<64)
		{
			ai = offset*(2*idx+1)-1;
			bi = offset*(2*idx+2)-1;

			sdatafast[bi] += sdatafast[ai];
		}
		offset*=2;
		__syncthreads();
	}

	if(idx < 32)
	{
		if(blocksize>=64)
		{
			if(idx < 32)
			{
				ai = offset*(2*idx+1)-1;
				bi = offset*(2*idx+2)-1;

				vsdatafast[bi] += vsdatafast[ai];
				offset*=2;
			}
		}
		if(blocksize>=32)
		{
			if(idx < 16)
			{
				ai = offset*(2*idx+1)-1;
				bi = offset*(2*idx+2)-1;

				vsdatafast[bi] += vsdatafast[ai];
				offset*=2;
			}
		}
		if(blocksize>=16)
		{
			if(idx < 8)
			{
				ai = offset*(2*idx+1)-1;
				bi = offset*(2*idx+2)-1;

				vsdatafast[bi] += vsdatafast[ai];
				offset*=2;
			}
		}
		if(blocksize>=8)
		{
			if(idx < 4)
			{
				ai = offset*(2*idx+1)-1;
				bi = offset*(2*idx+2)-1;

				vsdatafast[bi] += vsdatafast[ai];
				offset*=2;
			}
		}
		if(blocksize>=4)
		{
			if(idx < 2)
			{
				ai = offset*(2*idx+1)-1;
				bi = offset*(2*idx+2)-1;

				vsdatafast[bi] += vsdatafast[ai];
				offset*=2;
			}
		}
		if(blocksize>=2)
		{
			if(idx < 1)
			{
				ai = offset*(2*idx+1)-1;
				bi = offset*(2*idx+2)-1;

				vsdatafast[bi] += vsdatafast[ai];
				offset*=2;
			}
		}
	}



	// While d is less than 32, everything executes in the same warp

	if(idx==0)
	{
		sdatafast[2*blocksize-1] = 0;
		soffset = offset;
	}

	__syncthreads();

	offset = soffset;


	for(int d=1;d<(2*blocksize-1);d*=2)
	{
		offset >>= 1;
		__syncthreads();

		if(idx<d)
		{
			ai = offset*(2*idx+1)-1;
			bi = offset*(2*idx+2)-1;

			t = sdatafast[ai];
			sdatafast[ai] = sdatafast[bi];
			sdatafast[bi] += t;
		}
	}

	__syncthreads();

}



__global__
void prescantest_kernel(unsigned int* g_idata,unsigned int* g_odata,int itest,int n)
{
	__shared__ unsigned int prescansdata[2*(BLOCK_SIZE+4)];

	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+threadIdx.x;

	int nout;
	unsigned int blocksize = BLOCK_SIZE;

	const int nprecalc = 8;

	uint low[nprecalc];
	uint high[nprecalc];

	nout = 2*BLOCK_SIZE;

	// Load data from global memory
	for(int i=0;i<nprecalc;i++)
	{
		if((2*nprecalc*gidx+i)<n)
		{
			low[i] = g_idata[2*nprecalc*gidx+i];
		}
		else
		{
			low[i] = 0;
		}

		if((2*nprecalc*gidx+nprecalc+i)<n)
		{
			high[i] = g_idata[2*nprecalc*gidx+i+nprecalc];
		}
		else
		{
			high[i] = 0;
		}
	}

	// Calculate the sum for 2 sets of 4 elements
	for(int i=1;i<nprecalc;i++)
	{
		low[i] += low[i-1];
		high[i] += high[i-1];
	}

	int ai = 2*idx;
	int bi = 2*idx+1;
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

/*
	if(2*(blockIdx.x+1)*blockDim.x > n)
	{
		nout = n-2*blockDim.x*blockIdx.x;
	}
	else
	{
		nout = 2*blockDim.x;
	}
*/

	prescansdata[ai+bankOffsetA] = low[nprecalc-1];

	prescansdata[bi+bankOffsetB] = high[nprecalc-1];

	__syncthreads();

	switch(itest)
	{
	case 0:
		prescan0(prescansdata,nout);
		break;
	case 1:
		prescanfast<unsigned int,512>(prescansdata,nout);
		break;
	default:
		break;
	}

	__syncthreads();

	//Apply result to local series
	ai = 2*idx;
	bi = 2*idx+1;
	for(int i=0;i<nprecalc;i++)
	{
		low[i] += prescansdata[ai+bankOffsetA];
		high[i] += prescansdata[bi+bankOffsetB];
	}

	for(int i=0;i<nprecalc;i++)
	{
		if((2*nprecalc*gidx+i)<n)
		{
			g_odata[2*nprecalc*gidx+i] = low[i];
		}

		if((2*nprecalc*gidx+nprecalc+i)<n)
		{
			g_odata[2*nprecalc*gidx+i+nprecalc] = high[i];
		}
	}





}

__global__
void check_results_kernel(uint* g_results0,uint* g_results1,int n)
{
	uint idx = threadIdx.x;
	uint gidx = blockDim.x*blockIdx.x+idx;

	uint result0;
	uint result1;

	if(gidx < n)
	{
		result0 = g_results0[gidx];
		result1 = g_results1[gidx];

		if(result0!=result1)
		{
			printf("%i != %i for thread %i \n",result0,result1,gidx);
		}
	}
}


int main(void)
{
	int cudaGridSize = 1;
	int cudaBlockSize = BLOCK_SIZE;
	int data_size = 16*cudaBlockSize*cudaGridSize;
	int niterations = 100;

	dim3 cudaGridSize3(1,1,1);
	dim3 cudaBlockSize3(1,1,1);
	cudaGridSize3.x = cudaGridSize;
	cudaBlockSize3.x = cudaBlockSize;

	unsigned int timer = 0;
	unsigned int timer2 = 0;
	cutCreateTimer(&timer);
	cutCreateTimer(&timer2);

	float control_time;
	float sqrttest_time;

	uint* data_h = (uint*)malloc(data_size*sizeof(uint));

	uint* data_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&data_d,2*sizeof(uint)*data_size));

	uint* results_d0;
	uint* results_d1;
	CUDA_SAFE_CALL(cudaMalloc((void**)&results_d0,2*sizeof(uint)*data_size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&results_d1,2*sizeof(uint)*data_size));

	// Fill data_h with random numbers;

	for(int i=0;i<data_size;i++)
	{
		data_h[i] = 1;
	}

	CUDA_SAFE_CALL(cudaMemcpy(data_d,data_h,data_size*sizeof(uint),cudaMemcpyHostToDevice));

	// Run the control test
	cutStartTimer(timer);
	for(int i=0;i<niterations;i++)
	{
		CUDA_SAFE_KERNEL((prescantest_kernel<<<cudaGridSize,cudaBlockSize>>>(
										data_d,results_d0,0,data_size)));
	}
	cutStopTimer(timer);
	control_time = cutGetTimerValue(timer);
	printf( "\nEmpty test took: %f (ms)\n\n", control_time);

	// Run the sqrt test
	cutStartTimer(timer2);
	for(int i=0;i<niterations;i++)
	{
//		CUDA_SAFE_KERNEL((prescantest_kernel<<<cudaGridSize,cudaBlockSize>>>(
//										data_d,results_d1,1,data_size)));
	}
	cutStopTimer(timer2);
	sqrttest_time = cutGetTimerValue(timer2);
	printf( "\nSqrttest test took: %f (ms)\n\n", sqrttest_time);

	for(int i=1;i<data_size;i++)
	{
		data_h[i] += data_h[i-1];
	}
	CUDA_SAFE_CALL(cudaMemcpy(results_d1,data_h,data_size*sizeof(uint),cudaMemcpyHostToDevice));

	CUDA_SAFE_KERNEL((check_results_kernel<<<2*cudaGridSize,cudaBlockSize>>>(
									results_d0,results_d1,data_size)));

	cudaFree(data_d);
	cudaFree(results_d0);
	cudaFree(results_d1);
	free(data_h);

	return 0;

}






















