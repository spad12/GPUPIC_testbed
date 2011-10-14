//#define __CUDACC__
//#define __cplusplus

#include <thrust/sort.h>

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
#include "cuda_texture_types.h"
#include "texture_fetch_functions.h"
#include "builtin_types.h"
#include "cutil.h"
#include "device_functions.h"







#  define CUDA_SAFE_KERNEL(call) {                                         \
	call;																					\
	cudaDeviceSynchronize();														\
	cudaError err = cudaGetLastError();										\
    if ( cudaSuccess != err) {                                               \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
                exit(EXIT_FAILURE);                                                  \
    } }


#define DEFRAG_BLOCK_SIZE 512
#define BLOCK_SIZE 512


class XPlist
{
public:
	float* px;
	float* py;
	float* pz;

	float* vx;
	float* vy;
	float* vz;

	uint* binindex;

	__host__
	void Free();
};

__host__
void XPlist::Free()
{
	cudaFree(px);
	cudaFree(py);
	cudaFree(pz);

	cudaFree(vx);
	cudaFree(vy);
	cudaFree(vz);

}


class Particlebin
{
public:
	uint* ifirstp;
	uint* ilastp;
	uint* binidx;
	uint* nptcls_out;
	uint* nptcls_in;
	uint* nptcls_stayed;
	int* nptcls_max;

	__device__
	bool compare_binids(uint binindex_in);
};

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

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

			T t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

}


__global__
void count_moved_particles(XPlist particles,Particlebin bins,uint* emptySlots,uint* compactedIDs,uint* scan_results)
{
	uint idx = threadIdx.x;
	uint gidx = blockIdx.x*blockDim.x+idx;
	uint binid = blockIdx.x;
	uint pidx;
	uint block_start = 0;
	uint ilastp = bins.ilastp[binid] - bins.ifirstp[binid];
	uint nptcls_bin = bins.nptcls_max[binid];

	uint didileave;
	uint new_id;

	__shared__ uint scan_offset;

	__shared__ uint scan_data[BLOCK_SIZE];

	// Shift the particle list pointer so that it starts where this bin starts
	particles.binindex += bins.ifirstp[binid];
	emptySlots += bins.ifirstp[binid];
	compactedIDs += bins.ifirstp[binid];
	scan_results += bins.ifirstp[binid];

	if(idx ==0) scan_offset = 0;
	__syncthreads();

	while(block_start < nptcls_bin)
	{
		pidx = block_start+idx;

		if(pidx <= ilastp){ didileave = (particles.binindex[pidx] != binid);}
		else{ didileave = 1;}

		scan_data[idx] = didileave+scan_offset;

		// Scan the data
		prescan0(scan_data,BLOCK_SIZE);

		// because the scan is exclusive, we need to shift everything to the left by 1

		scan_data[idx] += didileave;

		if(pidx < nptcls_bin)
		{
			scan_results[pidx] = scan_data[idx];

			if(didileave)
			{
				new_id = scan_data[idx] - 1;
				emptySlots[new_id] = pidx;
			}
			else
			{
				new_id = pidx - scan_data[idx];
				compactedIDs[new_id] = pidx;
			}
		}

		__syncthreads();

		if(idx == 0) scan_offset = scan_data[BLOCK_SIZE-1];

		block_start += blockDim.x;

		if(pidx == ilastp)
		{
			bins.nptcls_out[binid] = scan_data[idx];
			bins.nptcls_stayed[binid] = (ilastp+1)-scan_data[idx];
		}
		__syncthreads();
	}







}


__host__
void reorder_particles(XPlist particles,Particlebin bins,int nptcls,int gridsize)
{

	int* emptySlots;
	int* compactedIDs;

	// First we need to figure out how many particles are going to be moving

	// Run a scan on each bin to calculate the empty slot id's, the number of particles that are leaving each bin

	// Now we need to allocate an XPlist that will store all of our particle data to be moved.

	//




}






































