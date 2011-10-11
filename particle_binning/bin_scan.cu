#include <thrust/sort.h>

#include "cudamatrix_types.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
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


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5


#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

#define PRESCAN_BLOCK_SIZE 512

int scan_level=0;

__global__
void binning_prescan(cudaMatrixui blockoffsets, cudaMatrixui gridoffsets,int nptcls_bin)
{
	uint idx = threadIdx.x;
	uint gidx = blockIdx.x*blockDim.x+idx;
	uint binid = blockIdx.y;

	int offset = 1;


	const uint n = 2*PRESCAN_BLOCK_SIZE;

	__shared__ uint sdata[2*n];

	uint low[4];
	uint high[4];

	// Load data from global memory
	for(int i=0;i<4;i++)
	{
		if((8*gidx+i)<nptcls_bin)
		{
			low[i] = blockoffsets(8*gidx+i,binid);
		}
		else
		{
			low[i] = 0;
		}

		if((8*gidx+4+i)<nptcls_bin)
		{
			high[i] = blockoffsets(8*gidx+4+i,binid);
		}
		else
		{
			high[i] = 0;
		}
	}

	// Calculate the sum for 2 sets of 4 elements
	for(int i=1;i<4;i++)
	{
		low[i] += low[i-1];
		high[i] += high[i-1];
	}


	int ai = 2*idx;
	int bi = 2*idx+1;
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);


	// Load local results into shared memory
	sdata[ai+bankOffsetA] = low[3];

	sdata[bi+bankOffsetB] = high[3];


	for(int d=n>>1;d>0;d>>=1) // build sum in place up the tree
	{
		__syncthreads();

		if(idx < d)
		{
			ai = offset*(2*idx+1)-1;
			bi = offset*(2*idx+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			sdata[bi] += sdata[ai];
		}

		offset *= 2;

	}

	if(idx == 0){sdata[n-1 + CONFLICT_FREE_OFFSET(n-1)] = 0;}

	// Traverse down the tree and build scan
	for(int d = 1; d<n;d*=2)
	{
		offset >>= 1;
		__syncthreads();

		if(idx < d)
		{
			ai = offset*(2*idx+1)-1;
			bi = offset*(2*idx+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			uint t = sdata[ai];
			sdata[ai] = sdata[bi];
			sdata[bi] += t;
		}
	}

	__syncthreads();

	//Apply result to local series
	ai = 2*idx;
	bi = 2*idx+1;
	for(int i=0;i<4;i++)
	{
		low[i] += sdata[ai+bankOffsetA];
		high[i] += sdata[bi+bankOffsetB];
	}

	// Write results back to global memory
	for(int i=0;i<4;i++)
	{
		if((8*gidx+i)<nptcls_bin)
		{
			blockoffsets(8*gidx+i,binid) = low[i];
		}

		if((8*gidx+4+i)<nptcls_bin)
		{
			blockoffsets(8*gidx+4+i,binid) = high[i];
		}
	}

	// Store the sum of the block for further processing
	if(idx == (blockDim.x-1)){gridoffsets(blockIdx.x,binid) = high[3];}






}

__global__
void bin_scan_back_add(cudaMatrixui blockoffsets,cudaMatrixui gridoffsets,int nptcls_bin)
{
	uint idx = threadIdx.x;
	uint gidx = blockIdx.x*blockDim.x+idx;
	uint bidx = blockIdx.x;
	uint binid = blockIdx.y;
	uint block_start = 8*blockIdx.x*blockDim.x;

	uint thid;

	uint offset;

	if((bidx >  0)){ offset = gridoffsets(bidx-1,binid);}
	if(bidx == 0){offset = 0;}
	__syncthreads();

	for(int i=0;i<8;i++)
	{
		thid = block_start+idx+i*blockDim.x;
		if(thid<nptcls_bin) blockoffsets(thid,binid) += offset;

	}


}




__host__
void bin_scan(cudaMatrixui &blockoffsets,int nptcls_bin,int nbins)
{
	// This function takes an input array blockoffsets, and calculates the cumulative sum of all the elements.
	// This function can process up to 8*PRESCAN_BLOCK_SIZE elements per block

	// This function was tested and verified on 9/30/2011

	int nptcls_block = nptcls_bin/8;
	dim3 cudaBlockSize(1,1,1);
	dim3 cudaGridSize(1,1,1);


	printf("scan_level = %i\n",scan_level);
	scan_level++;

	cudaBlockSize.x = PRESCAN_BLOCK_SIZE;
	cudaGridSize.x = (nptcls_bin+8*PRESCAN_BLOCK_SIZE-1)/(8*PRESCAN_BLOCK_SIZE);
	cudaGridSize.y = nbins;

	// Allocate space to store the blockwise results
	cudaMatrixui gridoffsets(cudaGridSize.x,nbins);

	// Scan
	CUDA_SAFE_KERNEL((binning_prescan<<<cudaGridSize,cudaBlockSize>>>(
									blockoffsets,gridoffsets,nptcls_bin)));

	// If multiple blocks are used then we have to calculate the cumulative sum of the results of each block
	if(cudaGridSize.x > 1)
	{
		// Recurse
		bin_scan(gridoffsets,cudaGridSize.x,nbins);

		// Add the block totals to each of the elements in the block
		CUDA_SAFE_KERNEL((bin_scan_back_add<<<cudaGridSize,cudaBlockSize>>>(
										blockoffsets,gridoffsets,nptcls_bin)));

	}

	// Free memory
	gridoffsets.cudaMatrixFree();






}






































