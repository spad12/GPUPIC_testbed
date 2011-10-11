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
//#include "host_defines.h"
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

typedef float (*FunctionPtr)(float);


class XPchunk
{
public:
	float x,y,z,vx,vy,vz;
};

class XParray
{
public:
	float* x;
	float* y;
	float* z;
	float* vx;
	float* vy;
	float* vz;
};

static __inline__ __device__
float sqrttest_define(float data)
{
	return sqrt(data);
}

static __inline__ __device__
float emptytest_define(float data)
{
	return data;
}

__device__ FunctionPtr sqrttest = &sqrttest_define;
__device__ FunctionPtr emptytest = &emptytest_define;

__device__ FunctionPtr operations[2] = {&emptytest_define,&sqrttest_define};



__global__
void operator_costs_kernel(float* data, float* result,int op, int npoints)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;

	__shared__ float sdata[BLOCK_SIZE];
	__shared__ float sresults[BLOCK_SIZE];

	sdata[idx] = data[gidx];
	__syncthreads();

	for(int i=0;i<BLOCK_SIZE;i++)
	{
		sresults[idx] += operations[op](sdata[i]);
	}

	__syncthreads();

	result[gidx] = sresults[idx];

}

__global__
void blockstruct_test_kernel(XPchunk* g_particles,float* results,int nparticles)
{
	uint idx = threadIdx.x;
	uint gidx = idx+blockIdx.x*blockDim.x;

//	__shared__ XPchunk particles[BLOCK_SIZE];

	if(gidx < nparticles)
	{
		results[gidx] = g_particles[gidx].x+g_particles[gidx].y+g_particles[gidx].z+
				g_particles[gidx].vx+g_particles[gidx].vy+g_particles[gidx].vz;
	}


}

__global__
void arraystruct_test_kernel(XParray g_particles,float* results,int nparticles)
{
	uint idx = threadIdx.x;
	uint gidx = idx+blockIdx.x*blockDim.x;

//	__shared__ XPchunk particles[BLOCK_SIZE];

	if(gidx < nparticles)
	{
		results[gidx] = g_particles.x[gidx]+g_particles.y[gidx]+g_particles.z[gidx]+
				g_particles.vx[gidx]+g_particles.vy[gidx]+g_particles.vz[gidx];
	}


}




int main(void)
{
	int cudaGridSize = 512;
	int cudaBlockSize = BLOCK_SIZE;
	int data_size = cudaBlockSize*cudaGridSize;
	int niterations = 100;
	int npartmax = pow(2,18);

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


	float* data_h = (float*)malloc(data_size*sizeof(float));

	float* data_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&data_d,sizeof(float)*data_size));

	float* results_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&results_d,sizeof(float)*data_size));

	// Fill data_h with random numbers;

	for(int i=0;i<data_size;i++)
	{
		data_h[i] = sqrt((rand()%100000)/1000.0);
	}

	CUDA_SAFE_CALL(cudaMemcpy(data_d,data_h,data_size*sizeof(float),cudaMemcpyHostToDevice));

	// Run the control test
	cutStartTimer(timer);
	for(int i=0;i<niterations;i++)
	{
		CUDA_SAFE_KERNEL((operator_costs_kernel<<<cudaGridSize,cudaBlockSize>>>(
										data_d,results_d,0,data_size)));
	}
	cutStopTimer(timer);
	control_time = cutGetTimerValue(timer);
	printf( "\nEmpty test took: %f (ms)\n\n", control_time);

	// Run the sqrt test
	cutStartTimer(timer2);
	for(int i=0;i<niterations;i++)
	{
		CUDA_SAFE_KERNEL((operator_costs_kernel<<<cudaGridSize,cudaBlockSize>>>(
										data_d,results_d,1,data_size)));
	}
	cutStopTimer(timer2);
	sqrttest_time = cutGetTimerValue(timer2);
	printf( "\nSqrttest test took: %f (ms)\n\n", sqrttest_time);

	cudaFree(data_d);
	cudaFree(results_d);
	free(data_h);

	return 0;

}

















































