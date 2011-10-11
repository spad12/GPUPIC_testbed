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


__global__
void blockstruct_test_kernel(XPchunk* g_particles,float* results,int nparticles)
{
	uint idx = threadIdx.x;
	uint gidx = idx+blockIdx.x*blockDim.x;

	__shared__ XPchunk particles[BLOCK_SIZE];


	//float x,y,z,vx,vy,vz;

	if(gidx < nparticles)
	{
		particles[idx].x = g_particles[gidx].x;
		particles[idx].y = g_particles[gidx].y;
		particles[idx].z = g_particles[gidx].z;
		particles[idx].vx = g_particles[gidx].vx;
		particles[idx].vy = g_particles[gidx].vy;
		particles[idx].vz = g_particles[gidx].vz;

		particles[idx].x += particles[idx].vx;
		particles[idx].y += particles[idx].vy;
		particles[idx].z += particles[idx].vz;

		g_particles[gidx].x = particles[idx].x;
		g_particles[gidx].y = particles[idx].y;
		g_particles[gidx].z = particles[idx].z;
/*
		x = g_particles[gidx].x;
		y = g_particles[gidx].y;
		z = g_particles[gidx].z;

		vx = g_particles[gidx].vx;
		vy = g_particles[gidx].vy;
		vz = g_particles[gidx].vz;



		x += vx;
		y += vy;
		z += vz;

		g_particles[gidx].x = x;
		g_particles[gidx].y = y;
		g_particles[gidx].z = z;
*/
	}



}

__global__
void arraystruct_test_kernel(XParray g_particles,float* results,int nparticles)
{
	uint idx = threadIdx.x;
	uint gidx = idx+blockIdx.x*blockDim.x;

//	__shared__ XPchunk particles[BLOCK_SIZE];

	float x,y,z,vx,vy,vz;

	if(gidx < nparticles)
	{


		x = g_particles.x[gidx];
		y = g_particles.y[gidx];
		z = g_particles.z[gidx];

		vx = g_particles.vx[gidx];
		vy = g_particles.vy[gidx];
		vz = g_particles.vz[gidx];

		x += vx;
		y += vy;
		z += vz;

		g_particles.x[gidx] = x;
		g_particles.y[gidx] = y;
		g_particles.z[gidx] = z;

	}


}




int main(void)
{

	CUDA_SAFE_CALL(cudaSetDevice(1));

	int cudaGridSize = 512;
	int cudaBlockSize = BLOCK_SIZE;
	int data_size = cudaBlockSize*cudaGridSize;
	int niterations = 100;
	int npartmax = cudaBlockSize*cudaGridSize;

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

	XPchunk cparticles_h[npartmax];
	XPchunk* cparticles_d;
	XParray aparticles_h;
	XParray aparticles_d;

	aparticles_h.x = (float*)malloc(npartmax*sizeof(float));
	aparticles_h.y = (float*)malloc(npartmax*sizeof(float));
	aparticles_h.z = (float*)malloc(npartmax*sizeof(float));
	aparticles_h.vx = (float*)malloc(npartmax*sizeof(float));
	aparticles_h.vy = (float*)malloc(npartmax*sizeof(float));
	aparticles_h.vz = (float*)malloc(npartmax*sizeof(float));

	CUDA_SAFE_CALL(cudaMalloc((void**)&(aparticles_d.x),sizeof(float)*npartmax));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(aparticles_d.y),sizeof(float)*npartmax));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(aparticles_d.z),sizeof(float)*npartmax));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(aparticles_d.vx),sizeof(float)*npartmax));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(aparticles_d.vy),sizeof(float)*npartmax));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(aparticles_d.vz),sizeof(float)*npartmax));

	CUDA_SAFE_CALL(cudaMalloc((void**)&cparticles_d,sizeof(XPchunk)*npartmax));



	float* results_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&results_d,sizeof(float)*npartmax));

	// Fill data_h with random numbers;

	for(int i=0;i<npartmax;i++)
	{
		aparticles_h.x[i] = sqrt((rand()%100000)/10000.0);
		aparticles_h.y[i] = sqrt((rand()%100000)/10000.0);
		aparticles_h.z[i] = sqrt((rand()%100000)/10000.0);
		aparticles_h.vx[i] = sqrt((rand()%100000)/10000.0);
		aparticles_h.vy[i] = sqrt((rand()%100000)/10000.0);
		aparticles_h.vz[i] = sqrt((rand()%100000)/10000.0);

		cparticles_h[i].x = aparticles_h.x[i];
		cparticles_h[i].y = aparticles_h.y[i];
		cparticles_h[i].z = aparticles_h.z[i];
		cparticles_h[i].vx = aparticles_h.vx[i];
		cparticles_h[i].vy = aparticles_h.vy[i];
		cparticles_h[i].vz = aparticles_h.vz[i];
	}

	CUDA_SAFE_CALL(cudaMemcpy(cparticles_d,cparticles_h,npartmax*sizeof(XPchunk),cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpy(aparticles_d.x,aparticles_h.x,npartmax*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(aparticles_d.y,aparticles_h.y,npartmax*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(aparticles_d.z,aparticles_h.z,npartmax*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(aparticles_d.vx,aparticles_h.vx,npartmax*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(aparticles_d.vy,aparticles_h.vy,npartmax*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(aparticles_d.vz,aparticles_h.vz,npartmax*sizeof(float),cudaMemcpyHostToDevice));





	// Run the control test
	cutStartTimer(timer);
	for(int i=0;i<niterations;i++)
	{
		CUDA_SAFE_KERNEL((blockstruct_test_kernel<<<cudaGridSize,cudaBlockSize>>>(
										cparticles_d,results_d,npartmax)));
	}
	cutStopTimer(timer);
	control_time = cutGetTimerValue(timer);
	printf( "\nChunk test took: %f (ms)\n\n", control_time);

	// Run the sqrt test
	cutStartTimer(timer2);
	for(int i=0;i<niterations;i++)
	{
		CUDA_SAFE_KERNEL((arraystruct_test_kernel<<<cudaGridSize,cudaBlockSize>>>(
										aparticles_d,results_d,npartmax)));
	}
	cutStopTimer(timer2);
	sqrttest_time = cutGetTimerValue(timer2);
	printf( "\nArray test took: %f (ms)\n\n", sqrttest_time);

	cudaFree(cparticles_d);
	cudaFree(aparticles_d.x);
	cudaFree(aparticles_d.y);
	cudaFree(aparticles_d.z);
	cudaFree(aparticles_d.vx);
	cudaFree(aparticles_d.vy);
	cudaFree(aparticles_d.vz);
	cudaFree(results_d);
	free(aparticles_h.x);
	free(aparticles_h.y);
	free(aparticles_h.z);
	free(aparticles_h.vx);
	free(aparticles_h.vy);
	free(aparticles_h.vz);

	return 0;

}

















































