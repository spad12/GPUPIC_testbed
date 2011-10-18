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
	uint ifirstp;
	uint ilastp;
	uint binidx;
	uint bin_level;

	__device__
	bool compare_binids(uint binindex_in);
};



__device__
void swap_particles(XPlist* shared_list, float3* position,float3* velocity,uint* binindex,int new_id)
{
	float3  temp_position;
	float3 temp_velocity;
	uint temp_binindex;

	// Store the Particle being replaced in registers
	temp_position.x = shared_list->px[new_id];
	temp_position.y = shared_list->py[new_id];
	temp_position.z = shared_list->pz[new_id];

	temp_velocity.x = shared_list->vx[new_id];
	temp_velocity.y = shared_list->vy[new_id];
	temp_velocity.z = shared_list->vz[new_id];

	temp_binindex = shared_list->binindex[new_id];

	// Push my particle to shared memory
	shared_list->px[new_id] = position->x;
	shared_list->py[new_id] = position->y;
	shared_list->pz[new_id] = position->z;

	shared_list->vx[new_id] = velocity->x;
	shared_list->vy[new_id] = velocity->y;
	shared_list->vz[new_id] = velocity->z;

	shared_list->binindex[new_id] = *binindex;

	// Replace my particle with the one that I just pushed to
	*position = temp_position;
	*velocity = temp_velocity;
	*binindex = temp_binindex;

}

__device__
bool compare_binids(uint binindex,uint bin_level)
{

	return ((binindex & ((0x0001) << (bin_level))) > 0);
}

__device__
bool Particlebin::compare_binids(uint binindex_in)
{
	return (binidx != (binindex_in >> (bin_level)));
}

__device__
void defrag_upsweep(XPlist g_particles,XPlist s_particles,Particlebin &parentbin,int* exchange_que,uint bin_level)
{
	uint idx = threadIdx.x;

	uint regidx;
	uint sharedidx;
	uint reg_que;

	int shared_que;

	int swap_idx = 0;
	int nptcls_moved = 0;
	int blockSize = blockDim.x;

	int iloop = 0;

	float3 l_position;
	float3 l_velocity;
	uint l_binindex;

	regidx = parentbin.ifirstp+idx;
	sharedidx = parentbin.ifirstp+idx;
	shared_que = parentbin.ifirstp;
	reg_que = parentbin.ifirstp; // Index that the first thread in the block is looking at

	while(reg_que <= parentbin.ilastp)
	{
		if((*exchange_que > 0)||(iloop == 0))
		{
			if(sharedidx <= parentbin.ilastp)
			{
				// Read in particle data to shared memory
				s_particles.px[idx] = g_particles.px[sharedidx];
				s_particles.py[idx] = g_particles.py[sharedidx];
				s_particles.pz[idx] = g_particles.pz[sharedidx];

				s_particles.vx[idx] = g_particles.vx[sharedidx];
				s_particles.vy[idx] = g_particles.vy[sharedidx];
				s_particles.vz[idx] = g_particles.vz[sharedidx];

				s_particles.binindex[idx] = g_particles.binindex[sharedidx];

			}
			__syncthreads();
		}

		if(idx == 0) *exchange_que = 0;
		__syncthreads();

		if(regidx <= parentbin.ilastp)
		{
			// Read in particle data to local registers
			l_position.x = g_particles.px[regidx];
			l_position.y = g_particles.py[regidx];
			l_position.z = g_particles.pz[regidx];

			l_velocity.x = g_particles.vx[regidx];
			l_velocity.y = g_particles.vy[regidx];
			l_velocity.z = g_particles.vz[regidx];

			l_binindex = g_particles.binindex[regidx];
		}



		if(regidx <= parentbin.ilastp)
		{
			if(parentbin.compare_binids(l_binindex))
			{
				swap_idx = atomicAdd(exchange_que,1);


				while((parentbin.compare_binids(s_particles.binindex[swap_idx]))&&(swap_idx < DEFRAG_BLOCK_SIZE))
				{
					swap_idx = atomicAdd(exchange_que,1);
				}

				if(swap_idx < DEFRAG_BLOCK_SIZE)
				{
				//	printf("particle %i belongs to bin %i not bin %i\n swapping with %i in bin %i \n",regidx,l_binindex,blockIdx.x,swap_idx,s_particles.binindex[swap_idx]);
					swap_particles(&s_particles,&l_position,&l_velocity,&l_binindex,swap_idx);
				}


			}
		}
		__syncthreads();

		nptcls_moved += min(*exchange_que,DEFRAG_BLOCK_SIZE);

		if(*exchange_que > 0)
		{
			// Write everything back to global memory, but only if stuff has changed
			if(sharedidx <= parentbin.ilastp)
			{
				// Write the shared memory buffer back to global memory
				g_particles.px[sharedidx] = s_particles.px[idx];
				g_particles.py[sharedidx] = s_particles.py[idx];
				g_particles.pz[sharedidx] = s_particles.pz[idx];

				g_particles.vx[sharedidx] = s_particles.vx[idx];
				g_particles.vy[sharedidx] = s_particles.vy[idx];
				g_particles.vz[sharedidx] = s_particles.vz[idx];

				g_particles.binindex[sharedidx] = s_particles.binindex[idx];

			}
			__syncthreads();
			if((regidx <= parentbin.ilastp)&&(regidx > parentbin.ifirstp+nptcls_moved))
			{
				// Write particle in local memory back to global memory
				g_particles.px[regidx] = l_position.x;
				g_particles.py[regidx] = l_position.y;
				g_particles.pz[regidx] = l_position.z;

				g_particles.vx[regidx] = l_velocity.x;
				g_particles.vy[regidx] = l_velocity.y;
				g_particles.vz[regidx] = l_velocity.z;

				g_particles.binindex[regidx] = l_binindex;

			}

			__syncthreads();
		}
		// update counters
		sharedidx += min(*exchange_que,DEFRAG_BLOCK_SIZE);
		shared_que += min(*exchange_que,DEFRAG_BLOCK_SIZE);
		regidx += blockSize;
		reg_que += blockSize;
		iloop++;

		__syncthreads();

	}

	// Update the particle bin
	if(idx == 0)
	{
		parentbin.ifirstp += nptcls_moved;
	}

}

__device__
void defrag_downsweep(XPlist g_particles,XPlist s_particles,Particlebin &parentbin,int* exchange_que,uint bin_level)
{
	uint idx = threadIdx.x;

	int blockSize = blockDim.x;

	int regidx;
	int sharedidx;
	int reg_que;
	int shared_que;

	int swap_idx = 0;
	int nptcls_moved = 0;
	int ifirstp = parentbin.ifirstp;

	int iloop = 0;

	float3 l_position;
	float3 l_velocity;
	uint l_binindex;


	regidx = parentbin.ilastp-idx;
	sharedidx = parentbin.ilastp-idx;
	shared_que = parentbin.ilastp;
	reg_que = parentbin.ilastp; // Index that the first thread in the block is looking at

	while(reg_que >= ifirstp)
	{
		//regidx = reg_que - invidx;
		//sharedidx = shared_que - invidx;

		if((*exchange_que > 0)||(iloop == 0))
		{
			if(sharedidx >= ifirstp)
			{
				// Read in particle data to shared memory
				s_particles.px[idx] = g_particles.px[sharedidx];
				s_particles.py[idx] = g_particles.py[sharedidx];
				s_particles.pz[idx] = g_particles.pz[sharedidx];

				s_particles.vx[idx] = g_particles.vx[sharedidx];
				s_particles.vy[idx] = g_particles.vy[sharedidx];
				s_particles.vz[idx] = g_particles.vz[sharedidx];

				s_particles.binindex[idx] = g_particles.binindex[sharedidx];

			}
			__syncthreads();
		}

		if(idx == 0) *exchange_que = 0;
		__syncthreads();

		if((regidx >= ifirstp)&&(regidx <= parentbin.ilastp))
		{
			// Read in particle data to local registers
			l_position.x = g_particles.px[regidx];
			l_position.y = g_particles.py[regidx];
			l_position.z = g_particles.pz[regidx];

			l_velocity.x = g_particles.vx[regidx];
			l_velocity.y = g_particles.vy[regidx];
			l_velocity.z = g_particles.vz[regidx];

			l_binindex = g_particles.binindex[regidx];
		}
		__syncthreads();



		if(regidx >= ifirstp)
		{
			if(parentbin.compare_binids(l_binindex))
			{

				swap_idx = atomicAdd(exchange_que,1);

				while((parentbin.compare_binids(s_particles.binindex[swap_idx]))&&(swap_idx < DEFRAG_BLOCK_SIZE))
				{
					swap_idx = atomicAdd(exchange_que,1);
				}

				if(swap_idx < DEFRAG_BLOCK_SIZE)
				{
					//printf("particle %i belongs to bin %i not bin %i\n swapping with %i in bin %i \n",regidx,l_binindex,blockIdx.x,(swap_idx),s_particles.binindex[swap_idx]);
					swap_particles(&s_particles,&l_position,&l_velocity,&l_binindex,swap_idx);
				}


			}
		}
		__syncthreads();
		nptcls_moved += min(*exchange_que,(DEFRAG_BLOCK_SIZE));

		if(*exchange_que > 0)
		{
			// Write everything back to global memory
			if(sharedidx >= ifirstp)
			{
				// Write the shared memory buffer back to global memory
				g_particles.px[sharedidx] = s_particles.px[idx];
				g_particles.py[sharedidx] = s_particles.py[idx];
				g_particles.pz[sharedidx] = s_particles.pz[idx];

				g_particles.vx[sharedidx] = s_particles.vx[idx];
				g_particles.vy[sharedidx] = s_particles.vy[idx];
				g_particles.vz[sharedidx] = s_particles.vz[idx];

				g_particles.binindex[sharedidx] = s_particles.binindex[idx];

			}
			__syncthreads();

			if((regidx >= ifirstp)&&(regidx < (parentbin.ilastp-nptcls_moved)))
			{
				// Write particle in local memory back to global memory
				g_particles.px[regidx] = l_position.x;
				g_particles.py[regidx] = l_position.y;
				g_particles.pz[regidx] = l_position.z;

				g_particles.vx[regidx] = l_velocity.x;
				g_particles.vy[regidx] = l_velocity.y;
				g_particles.vz[regidx] = l_velocity.z;

				g_particles.binindex[regidx] = l_binindex;

			}

			__syncthreads();
		}

		// update counters
		sharedidx -= min(*exchange_que,(DEFRAG_BLOCK_SIZE));
		shared_que -= min(*exchange_que,(DEFRAG_BLOCK_SIZE));
		regidx -= blockSize;
		reg_que -= blockSize;
		iloop++;

		__syncthreads();
/*
		loop_count++;
		if(loop_count > 1000)
		{
		//	if(idx == 0) printf("Error, breaking on maximum loop count, reg_que = %i\n",reg_que);
			break;
		}
*/
	}

	// Update the particle bin
	if(idx == 0)
	{
		parentbin.ilastp -= nptcls_moved;
	}

}

__global__
void __launch_bounds__(DEFRAG_BLOCK_SIZE,2)
defrag_particle_list(XPlist g_particles,Particlebin* bin_tree,uint bin_level,int* nptcls_max)
{
	uint idx = threadIdx.x;
	uint binid = blockIdx.x;

	__shared__ float spx[DEFRAG_BLOCK_SIZE];
	__shared__ float spy[DEFRAG_BLOCK_SIZE];
	__shared__ float spz[DEFRAG_BLOCK_SIZE];

	__shared__ float svx[DEFRAG_BLOCK_SIZE];
	__shared__ float svy[DEFRAG_BLOCK_SIZE];
	__shared__ float svz[DEFRAG_BLOCK_SIZE];

	__shared__ uint sbinindex[DEFRAG_BLOCK_SIZE+1];
	__shared__ int exchange_que;
	Particlebin parentbin;
	XPlist s_particles;


	s_particles.px = spx;
	s_particles.py = spy;
	s_particles.pz = spz;

	s_particles.vx = svx;
	s_particles.vy = svy;
	s_particles.vz = svz;

	s_particles.binindex = sbinindex;
	parentbin = bin_tree[binid];

	if(idx == 0)
	{
		exchange_que = 0;

		//printf("ilastp in bin %i = %i, with ifirstp = %i\n",parentbin.binidx,parentbin.ilastp,parentbin.ifirstp);
	}
	__syncthreads();

	switch((parentbin.binidx & 0x0001))
	{
	case 1:
		// Odd bin index means that we do an upsweep
		defrag_upsweep(g_particles,s_particles,parentbin,&exchange_que,bin_level);
		break;
	case 0:
		// Even bin index means that we do a downsweep
		defrag_downsweep(g_particles,s_particles,parentbin,&exchange_que,bin_level);
		break;
	default:
		break;
	}


	if(idx == 0)
	{
		bin_tree[binid] = parentbin;
	}

/*
	// Update the subbins
	if(idx == 0)
	{
		parentbin = bin_tree[binid];
		subbins[0].ifirstp = parentbin.ifirstp;
		subbins[0].ilastp = parentbin.ifirstp+nptcls_moved-1;
		subbins[1].ifirstp = subbins[0].ilastp+1;
		subbins[1].ilastp = parentbin.ilastp;

		subbins[0].binidx = 2*binid;
		subbins[1].binidx = 2*binid+1;

		printf("partentbin stuff = %i, %i, binindex = %i \n",parentbin.ifirstp,parentbin.ilastp,parentbin.binidx);
		printf("subbin0 stuff = %i, %i, binindex = %i \n",subbins[0].ifirstp,subbins[0].ilastp,subbins[0].binidx);
		printf("subbin1 stuff = %i, %i, binindex = %i \n",subbins[1].ifirstp,subbins[1].ilastp,subbins[1].binidx);

		bin_tree[gridDim.x+2*binid] = subbins[0];
		bin_tree[gridDim.x+2*binid+1] = subbins[1];

		int nptcls0 = subbins[0].ilastp-subbins[0].ifirstp+1;
		int nptcls1 = subbins[1].ilastp-subbins[1].ifirstp+1;

		nptcls1 = max(nptcls0,nptcls1);
		atomicMax(nptcls_max,nptcls1);

	}
*/

}

__global__
void resize_bins(XPlist g_particles,Particlebin* bin_tree,uint bin_level,int* nptcls_max)
{
	uint idx = threadIdx.x;
	uint binid = 2*blockIdx.x;
	int blockSize = blockDim.x;

	int left_block_start;
	int right_block_start;

	__shared__ float spx[DEFRAG_BLOCK_SIZE];
	__shared__ float spy[DEFRAG_BLOCK_SIZE];
	__shared__ float spz[DEFRAG_BLOCK_SIZE];

	__shared__ float svx[DEFRAG_BLOCK_SIZE];
	__shared__ float svy[DEFRAG_BLOCK_SIZE];
	__shared__ float svz[DEFRAG_BLOCK_SIZE];

	__shared__ uint sbinindex[DEFRAG_BLOCK_SIZE];
	__shared__ Particlebin leftbin;
	__shared__ Particlebin rightbin;
	__shared__ int exit_flag;
	XPlist s_particles;

	float3 l_position;
	float3 l_velocity;
	uint l_binindex;

	uint binindex_left;
	uint binindex_right;

	int regidx;
	int sharedidx;

	int loop_counter;

	if(idx == 0)
	{
		leftbin = bin_tree[binid];
		rightbin = bin_tree[binid+1];

		//printf("nptcls in bin %i = %i, with ifirstp = %i\n",leftbin.binidx,leftbin.ilastp,leftbin.ifirstp);
		//printf("nptcls in bin %i = %i, with ifirstp = %i\n",rightbin.binidx,rightbin.ilastp,rightbin.ifirstp);
		exit_flag = 0;
	}

	s_particles.px = spx;
	s_particles.py = spy;
	s_particles.pz = spz;

	s_particles.vx = svx;
	s_particles.vy = svy;
	s_particles.vz = svz;

	s_particles.binindex = sbinindex;
	__syncthreads();

	left_block_start = leftbin.ilastp;
	right_block_start = rightbin.ifirstp;

	while(left_block_start < right_block_start)
	{
		sharedidx =	left_block_start + idx;
		regidx = right_block_start - idx;


		if(sharedidx < regidx)
		{
			// Read in particle data to shared memory
			s_particles.px[idx] = g_particles.px[sharedidx];
			s_particles.py[idx] = g_particles.py[sharedidx];
			s_particles.pz[idx] = g_particles.pz[sharedidx];

			s_particles.vx[idx] = g_particles.vx[sharedidx];
			s_particles.vy[idx] = g_particles.vy[sharedidx];
			s_particles.vz[idx] = g_particles.vz[sharedidx];

			s_particles.binindex[idx] = g_particles.binindex[sharedidx];

			// Read in particle data to local registers
			l_position.x = g_particles.px[regidx];
			l_position.y = g_particles.py[regidx];
			l_position.z = g_particles.pz[regidx];

			l_velocity.x = g_particles.vx[regidx];
			l_velocity.y = g_particles.vy[regidx];
			l_velocity.z = g_particles.vz[regidx];

			l_binindex = g_particles.binindex[regidx];

			binindex_left = s_particles.binindex[idx] >> bin_level;
			binindex_right = l_binindex >> bin_level;
		}
		__syncthreads();

		// Have the last thread check to see if we will finish moving all the particles that need moving on this step
		if(idx == (DEFRAG_BLOCK_SIZE-1))
		{
			if(binindex_left == binindex_right) exit_flag = 1;
		}

		// If your index doesn't overlap, then swap your particles
		if((sharedidx < regidx)&&(binindex_left > binindex_right))
		{
			swap_particles(&s_particles,&l_position,&l_velocity,&l_binindex,idx);
		}

		__syncthreads();

		if(sharedidx < regidx)
		{
			// Write particle in local memory back to global memory
			g_particles.px[regidx] = l_position.x;
			g_particles.py[regidx] = l_position.y;
			g_particles.pz[regidx] = l_position.z;

			g_particles.vx[regidx] = l_velocity.x;
			g_particles.vy[regidx] = l_velocity.y;
			g_particles.vz[regidx] = l_velocity.z;

			g_particles.binindex[regidx] = l_binindex;

			// Write the shared memory buffer back to global memory
			g_particles.px[sharedidx] = s_particles.px[idx];
			g_particles.py[sharedidx] = s_particles.py[idx];
			g_particles.pz[sharedidx] = s_particles.pz[idx];

			g_particles.vx[sharedidx] = s_particles.vx[idx];
			g_particles.vy[sharedidx] = s_particles.vy[idx];
			g_particles.vz[sharedidx] = s_particles.vz[idx];

			g_particles.binindex[sharedidx] = s_particles.binindex[idx];
		}

		__syncthreads();

		left_block_start += blockSize;
		right_block_start -= blockSize;
		loop_counter++;

	//	if(loop_counter > 100) break;

		__syncthreads();

		if(exit_flag)
		{
			// This means that the bin of the last particle on the left and the first particle on the right are the same
			break;
		}
	}
	__syncthreads();
	if(idx == 0) exit_flag = 0;
	__syncthreads();

	for(int shared_que=leftbin.ilastp;shared_que<rightbin.ifirstp;shared_que+=blockSize)
	{
		sharedidx = shared_que+idx;

		if(sharedidx <= rightbin.ifirstp)
		{
			s_particles.px[idx] = g_particles.px[sharedidx];
			s_particles.py[idx] = g_particles.py[sharedidx];
			s_particles.pz[idx] = g_particles.pz[sharedidx];

			s_particles.vx[idx] = g_particles.vx[sharedidx];
			s_particles.vy[idx] = g_particles.vy[sharedidx];
			s_particles.vz[idx] = g_particles.vz[sharedidx];

			s_particles.binindex[idx] = g_particles.binindex[sharedidx];
		}

		__syncthreads();

		// Now we need to figure out where the left bin ends and the right one begins
		if((idx > 0)&&(sharedidx <= rightbin.ifirstp))
		{
			binindex_left = s_particles.binindex[idx-1] >> (bin_level);
			binindex_right = s_particles.binindex[idx] >> (bin_level);

			if(binindex_left != binindex_right)
			{
				// Assuming everything before this went well, then the last particle in the left bin is sharedidx-1,
				// and the first particle in the right bin is sharedidx
				exit_flag = 1;

				leftbin.ilastp = sharedidx-1;
				rightbin.ifirstp = sharedidx;
				bin_tree[binid] = leftbin;
				bin_tree[binid+1] = rightbin;
				if(bin_level > 0)
				{
					bin_tree[2*gridDim.x+2*binid].ifirstp = leftbin.ifirstp;
					bin_tree[2*gridDim.x+2*binid+1].ilastp = leftbin.ilastp;
					bin_tree[2*gridDim.x+2*(binid+1)].ifirstp = rightbin.ifirstp;
					bin_tree[2*gridDim.x+2*(binid+1)+1].ilastp = rightbin.ilastp;
				}

				int nptcls0 = leftbin.ilastp-leftbin.ifirstp+1;
				int nptcls1 = rightbin.ilastp-rightbin.ifirstp+1;

				nptcls1 = max(nptcls1,nptcls0);

				atomicMax(nptcls_max,nptcls1);
			}
		}
		__syncthreads();

		if(exit_flag)
		{
			break;
		}
	}





}

__host__
void rebin_particles(Particlebin* bin_tree,XPlist particles,int* nptcls_max_out,int nptcls,int nbins_max)
{

	uint bin_level = 0;
	uint nptcls_max = nptcls;
	int nbins_current = 2;
	dim3 cudaGridSize(1,1,1);
	dim3 cudaBlockSize(1,1,1);
	int nlevels = log(nbins_max)/log(2);

	bin_level = nlevels-1;


	Particlebin* current_bins = bin_tree;


	int* nptcls_max_next;
	CUDA_SAFE_CALL(cudaMalloc((void**)&nptcls_max_next,sizeof(int)));
	int* nptcls_max_next_h = (int*)malloc(sizeof(int));

	// Main tree traversing loop
	for(int i=0;i<(nlevels);i++)
	{
		CUDA_SAFE_CALL(cudaMemset(nptcls_max_next,0,sizeof(int)));

		if(nptcls_max > nptcls)
		{
			printf("error number of particles is growing\n");
			return;
		}
		// Setup the kernel launch parameters
		cudaBlockSize.x = DEFRAG_BLOCK_SIZE;
		cudaGridSize.x = nbins_current;

		// Defragment the particle bins
		CUDA_SAFE_KERNEL((defrag_particle_list<<<cudaGridSize,cudaBlockSize>>>
									 (particles,current_bins,bin_level,nptcls_max_next)));

		cudaGridSize.x /= 2;

		CUDA_SAFE_KERNEL((resize_bins<<<cudaGridSize,cudaBlockSize>>>
									 (particles,current_bins,bin_level,nptcls_max_next)));


		// Advance counters
		bin_level--;
		current_bins = current_bins+nbins_current;
		nbins_current *= 2;
		CUDA_SAFE_CALL(cudaMemcpy(nptcls_max_next_h,nptcls_max_next,sizeof(int),cudaMemcpyDeviceToHost));

		nptcls_max = *nptcls_max_next_h;

		//printf("binlevel = %i, nptcls_max = %i \n", bin_level,nptcls_max);

	}

	*nptcls_max_out = nptcls_max;

	CUDA_SAFE_CALL(cudaFree(nptcls_max_next));

}

__global__
void find_bin_boundaries(XPlist particles,uint* ifirstp,uint* ilastp,int nptcls)
{
	uint idx = threadIdx.x;
	uint gidx = blockIdx.x*blockDim.x+idx;

	uint binindex;
	uint binindex_left;
	uint binindex_right;

	if(gidx < nptcls)
	{
		binindex = particles.binindex[gidx];
		binindex_left = particles.binindex[max(gidx-1,0)];
		binindex_right = particles.binindex[min(gidx+1,nptcls-1)];

		if(binindex_left != binindex)
		{
			ifirstp[binindex] = gidx;
		}

		if(binindex_right != binindex)
		{
			ilastp[binindex] = gidx;
		}

		if(gidx == 0)
		{
			ifirstp[binindex] = 0;
		}
		if(gidx == nptcls-1)
		{
			ilastp[binindex] = gidx;
		}
	}
}

__global__
void populate_bin_data(Particlebin* bins,uint* ifirstp,uint* ilastp,int gridsize)
{
	uint idx = threadIdx.x;
	uint gidx = blockIdx.x*blockDim.x+idx;

	Particlebin mybin;

	if(gidx < gridsize)
	{
		mybin.ifirstp = ifirstp[gidx];
		mybin.ilastp = ilastp[gidx];
		mybin.binidx = gidx;
	}

	__syncthreads();

	if(gidx <gridsize)
	{
		bins[gidx] = mybin;
	}
}

__host__
Particlebin* generate_bins(XPlist particles,int gridsize,int nptcls)
{
	dim3 cudaGridSize(1,1,1);
	dim3 cudaBlockSize(BLOCK_SIZE,1,1);

	uint* ifirstp_d;
	uint* ilastp_d;
	Particlebin* bins_out;

	CUDA_SAFE_CALL(cudaMalloc((void**)&bins_out,gridsize*sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ifirstp_d,gridsize*sizeof(uint)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&ilastp_d,gridsize*sizeof(uint)));

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	cudaGridSize.x = (nptcls+cudaBlockSize.x-1)/cudaBlockSize.x;
	CUDA_SAFE_KERNEL((find_bin_boundaries<<<cudaGridSize,cudaBlockSize>>>
									(particles,ifirstp_d,ilastp_d,nptcls)));
	cudaGridSize.x = (gridsize+cudaBlockSize.x-1)/cudaBlockSize.x;
	CUDA_SAFE_KERNEL((populate_bin_data<<<cudaGridSize,cudaBlockSize>>>
									(bins_out,ifirstp_d,ilastp_d,gridsize)));

	cudaFree(ifirstp_d);
	cudaFree(ilastp_d);

	return bins_out;


}

__global__
void check_sort(XPlist particles,Particlebin* bins,int nptcls_max)
{
	uint idx = threadIdx.x;
	uint gidx = blockDim.x*blockIdx.x+idx;
	uint binid = blockIdx.y;
	uint pidx;
	uint pbinindex;
	int pdata;

	__shared__ Particlebin sbin;

	if(idx == 0)
	{
		sbin = bins[binid];
		//if(gidx == 0)printf("nptcls in bin %i = %i, with ifirstp = %i\n",sbin.binidx,sbin.ilastp-sbin.ifirstp+1,sbin.ifirstp);
	}

	__syncthreads();

	pidx = gidx+sbin.ifirstp;

	if((pidx < sbin.ilastp))
	{
		pbinindex = particles.binindex[pidx];
		if(pbinindex != (sbin.binidx))
		{
			printf("Error, particle %i(%i), with binindex %i is in bin %i\n",pidx,gidx,pbinindex,sbin.binidx);
		}

		pdata = rint(particles.px[pidx]-0.1);

		if(pdata != (pbinindex))
		{
			printf("Error, particle %i(%i), with binindex %i is in bin %i\n",pidx,gidx,pdata,pbinindex);
		}
	}
}

__host__
Particlebin* setup_particle_bins(int nptcls,int gridsize)
{
	int bin_tree_size = 0;
	int nlevels = log(gridsize)/log(2);
	int nbins;


	Particlebin* bins;

	for(int i=0;i<(nlevels+1);i++)
	{
		bin_tree_size += (1<<i);
	}

	CUDA_SAFE_CALL(cudaMalloc((void**)&bins,bin_tree_size*sizeof(Particlebin)));

	Particlebin* bins_h = (Particlebin*)malloc(bin_tree_size*sizeof(Particlebin));

	Particlebin* current_bins = bins_h;
	Particlebin* subbins;

	current_bins[0].ifirstp = 0;
	current_bins[0].ilastp = nptcls-1;
	current_bins[0].binidx = 0;
	current_bins[0].bin_level = nlevels;


	for(int i=0;i<(nlevels);i++)
	{
		nbins = 1<<(i);

		for(int j=0;j<nbins;j++)
		{
			//printf("bin %i spans %i to %i\n",current_bins[j].binidx,current_bins[j].ifirstp,current_bins[j].ilastp);
			subbins = current_bins+nbins+2*j;
			subbins[0].ifirstp = current_bins[j].ifirstp;
			subbins[0].ilastp = subbins[0].ifirstp+(current_bins[j].ilastp-current_bins[j].ifirstp)/2;
			subbins[1].ifirstp = subbins[0].ilastp+1;
			subbins[1].ilastp = current_bins[j].ilastp;
			subbins[0].binidx = 2*j;
			subbins[1].binidx = 2*j+1;
			subbins[0].bin_level = current_bins[j].bin_level-1;
			subbins[1].bin_level = current_bins[j].bin_level-1;
		}

		current_bins += nbins;

	}



	CUDA_SAFE_CALL(cudaMemcpy(bins,bins_h,bin_tree_size*sizeof(Particlebin),cudaMemcpyHostToDevice));

	free(bins_h);

	return bins+1;

}

__global__
void reorder_particle_list(XPlist particles_in, XPlist particles_out,uint* old_ids,int nptcls)
{
	uint idx = threadIdx.x;
	uint gidx = blockIdx.x*blockDim.x+idx;

	float3 temp_velocities;
	float3 temp_positions;

	uint old_id;

	if(gidx < nptcls)
	{
		old_id = old_ids[gidx];
		temp_positions.x = particles_in.px[old_id];
		temp_positions.y = particles_in.py[old_id];
		temp_positions.z = particles_in.pz[old_id];

		temp_velocities.x = particles_in.vx[old_id];
		temp_velocities.y = particles_in.vy[old_id];
		temp_velocities.z = particles_in.vz[old_id];
	}

	__syncthreads();

	if(gidx < nptcls)
	{
		particles_out.px[gidx] = temp_positions.x;
		particles_out.py[gidx] = temp_positions.y;
		particles_out.pz[gidx] = temp_positions.z;

		particles_out.vx[gidx] = temp_velocities.x;
		particles_out.vy[gidx] = temp_velocities.y;
		particles_out.vz[gidx] = temp_velocities.z;
	}
}

__host__
void rough_test(int nptcls,int gridsize)
{
	int max_nptclsperbin;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	Particlebin* bins;

	XPlist particles_h;
	XPlist* particles_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_d,sizeof(XPlist)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles_h.px),nptcls*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles_h.py),nptcls*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles_h.pz),nptcls*sizeof(float)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles_h.vx),nptcls*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles_h.vy),nptcls*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles_h.vz),nptcls*sizeof(float)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles_h.binindex),nptcls*sizeof(uint)));

	uint* pbinindex_h = (uint*)malloc(nptcls*sizeof(uint));
	uint* pindex_h = (uint*)malloc(nptcls*sizeof(uint));
	float* particle_data_h = (float*)malloc(nptcls*sizeof(float));
	uint* pindex_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&pindex_d,nptcls*sizeof(uint)));

	printf("Run parameters: Nptcls = %i, GridSize = %i \n",nptcls,gridsize);

	int temp_index;

	for(int i=0;i<nptcls;i++)
	{
		pbinindex_h[i] = i/(nptcls/gridsize);
		pindex_h[i] = i;

		if((rand()%1000) < 10)
		{
			temp_index = pbinindex_h[i] + rand()%17 - 8;
			if(temp_index < 0)
			{
				temp_index = gridsize-1;
			}
			else if(temp_index > gridsize-1)
			{
				temp_index = 0;
			}

			pbinindex_h[i] = temp_index;

		}

		particle_data_h[i] = (float)pbinindex_h[i];
	}

	CUDA_SAFE_CALL(cudaMemcpy(particles_h.binindex,pbinindex_h,nptcls*sizeof(uint),cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpy(particles_h.px,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_h.py,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_h.pz,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_h.vx,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_h.vy,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_h.vz,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));

	// Figure how big the bin tree needs to be
	int bin_tree_size = 0;
	int nlevels = log(gridsize)/log(2);

	for(int i=1;i<(nlevels+1);i++)
	{
		bin_tree_size += (1<<i);
	}

	// Set up the first guess for the particle bins
	bins = setup_particle_bins(nptcls,gridsize);



	// Copy Particle data to the device
	CUDA_SAFE_CALL(cudaMemcpy(particles_d,&particles_h,sizeof(XPlist),cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	// Rebin the particles
	rebin_particles(bins,particles_h,&max_nptclsperbin,nptcls,gridsize);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Binning Sort took %f milliseconds\n", milliseconds);


	// Check the sort
	dim3 cudaGridSize(1,gridsize,1);
	dim3 cudaBlockSize(BLOCK_SIZE,1,1);

	cudaGridSize.x = (max_nptclsperbin+cudaBlockSize.x-1)/cudaBlockSize.x;
	Particlebin* bottom_bins = bins+(bin_tree_size-gridsize);

//	CUDA_SAFE_KERNEL((check_sort<<<cudaGridSize,cudaBlockSize>>>
	//							 (particles_h,bottom_bins,max_nptclsperbin)));

	// Now let's see how the thrust library sort compares:
	CUDA_SAFE_CALL(cudaMemcpy(particles_h.binindex,pbinindex_h,nptcls*sizeof(uint),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(pindex_d,pindex_h,nptcls*sizeof(uint),cudaMemcpyHostToDevice));

	XPlist particles2_h;

	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles2_h.px),nptcls*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles2_h.py),nptcls*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles2_h.pz),nptcls*sizeof(float)));

	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles2_h.vx),nptcls*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles2_h.vy),nptcls*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&(particles2_h.vz),nptcls*sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpy(particles_h.px,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_h.py,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_h.pz,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_h.vx,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_h.vy,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(particles_h.vz,particle_data_h,nptcls*sizeof(float),cudaMemcpyHostToDevice));

	particles2_h.binindex = particles_h.binindex;


	// wrap raw device pointers with a device_ptr
	thrust::device_ptr<uint> d_keys(particles_h.binindex);
	thrust::device_ptr<uint> d_values(pindex_d);

	cudaEventRecord(start);
	// Sort the data
	thrust::sort_by_key(d_keys,d_keys+nptcls,d_values);

	cudaDeviceSynchronize();


	cudaGridSize.x = (nptcls+BLOCK_SIZE-1)/BLOCK_SIZE;
	cudaGridSize.y = 1;
	// Reorder all of the data in the particle list
	CUDA_SAFE_KERNEL((reorder_particle_list<<<cudaGridSize,cudaBlockSize>>>
								 (particles_h,particles2_h,pindex_d,nptcls)));

	bottom_bins = generate_bins(particles2_h,gridsize,nptcls);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);


	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Thrust Sort took %f milliseconds\n", milliseconds);

	cudaGridSize.x = (max_nptclsperbin+cudaBlockSize.x-1)/cudaBlockSize.x;
	cudaGridSize.y = gridsize;
	CUDA_SAFE_KERNEL((check_sort<<<cudaGridSize,cudaBlockSize>>>
								 (particles2_h,bottom_bins,max_nptclsperbin)));

	cudaFree(bins-1);
	cudaFree(bottom_bins);
	cudaFree(particles_d);
	cudaFree(pindex_d);
	cudaFree(particles_h.binindex);
	particles_h.Free();
	particles2_h.Free();
	free(pbinindex_h);
	free(pindex_h);
	free(particle_data_h);









}



int main(void)
{
	cudaSetDevice(1);
	int nptcls;
	int gridsize;

	nptcls = pow(2,24);
	gridsize = 4096;
	rough_test(nptcls,gridsize);


	return 0;
}














































