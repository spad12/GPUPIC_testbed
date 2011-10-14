/*
 * Particle list object for gpu pic code
 *
 * by Joshua Payne
 */

#include "constants.cuh"
#include <thrust/sort.h>
#include <thrust/scan.h>

// Define Allocation functor classes

class XPMallocHost
{
public:
	__host__ void operator() (void** Ptr, int size)
	{
		cudaHostAlloc(Ptr,size,4);
	}
};

class XPMallocDevice
{
public:
	__host__ void operator() (void** Ptr,int size)
	{
		cudaMalloc(Ptr,size);
	}
};


enum XPlistlocation
{
	host = 0,
	device = 1
};



struct XPdata
{
	float px,py,pz,vx,vy,vz;
};

class XPlist
{
public:

	float* px;
	float* py;
	float* pz;
	float* vx;
	float* vy;
	float* vz;

	// Position indices
	int* index; // x = nx, y = ny, z = nz, w = cellindex

	int nptcls;

	__host__ XPlist(int nptcls_in, enum XPlistlocation location_in)
	{
		nptcls = nptcls_in;
		location = location_in;
		if(!location_in)
		{
		//	printf("allocating particle list on the host \n");
			XPlist_allocate(XPMallocHost());
		}
		else
		{
			//printf("allocating particle list on the device \n");
			XPlist_allocate(XPMallocDevice());
		}

	}
	__host__ __device__ XPlist(){;}
	// Allocate particle list in specified memory
	template<class O>
	__host__ void XPlist_allocate(O op);

	__device__ void XPlist_shared(int nptcls_in);

	// Method to generate a random position and velocity distribution (list must reside in host memory)
	__host__ void random_distribution(int3 gridDims,float3 gridSpacing);

	// Method to sort the particle list (list must reside in particle memory)
	__host__ void sort(float3 Pgridspacing, int3 Pgrid_i_dims);
	// Wrapper method for find_index_kernel()
	__host__ void find_cell_index(int* cellindex_temp);
	// Move the particles
	__host__ void move_cached_unsorted(float3 Pgridspacing, int3 Pgrid_i_dims,
																	cudaMatrixf Phi, cudaMatrixi rho,float dt);
	__host__ void move_cached_semisorted(float3 Pgridspacing, int3 Pgrid_i_dims,
																	cudaMatrixf Phi, cudaMatrixi rho,float dt);
	__host__ void move_shared_sorted(float3 Pgridspacing, int3 Pgrid_i_dims, cudaMatrixf Phi,
																	cudaMatrixi rho,float dt);
	__host__ void reorder(void);

	__host__ void inject_new_particles(int* didileave,int nptcls_left);

	__device__ void periodic_boundary(int3 griddims,float3 gridspacing,bool shared);

	__host__ void cpu_move(float* phi,int* rho,float3 gridspacing,int3 griddims,float dt);

	// Destructor
	__host__ void XPlistFree(void)
	{
		if(!location)
		{
			cudaFreeHost(px);
			cudaFreeHost(py);
			cudaFreeHost(pz);
			cudaFreeHost(vx);
			cudaFreeHost(vy);
			cudaFreeHost(vz);
			cudaFreeHost(index);

		}
		else
		{
			cudaFree(px);
			cudaFree(py);
			cudaFree(pz);
			cudaFree(vx);
			cudaFree(vy);
			cudaFree(vz);
			cudaFree(index);

		}
	}

// Members
	/* not yet needed
	float* charge;
	float* mass;
	*/

private:

	enum XPlistlocation location;

};

template<class O>
__host__ void XPlist::XPlist_allocate(O op)
{
	int intsize = sizeof(int)*nptcls;
	int floatsize = sizeof(float)*nptcls;

		op((void**)&px,floatsize);
		op((void**)&py,floatsize);
		op((void**)&pz,floatsize);
		op((void**)&vx,floatsize);
		op((void**)&vy,floatsize);
		op((void**)&vz,floatsize);
		op((void**)&index,intsize);

	return;

}

__device__ void XPlist::XPlist_shared(int nptcls_in)
{
	__shared__ float data_temp[6*threadsPerBlock];
	__shared__ int index_temp[threadsPerBlock];

	px = data_temp;
	py = px+threadsPerBlock;
	pz = py+threadsPerBlock;
	vx = pz+threadsPerBlock;
	vy = vx+threadsPerBlock;
	vz = vy+threadsPerBlock;
	index = index_temp;

	location = device;

	nptcls = nptcls_in;

}

__host__ void XPlistCopy(XPlist dst, XPlist src,int nptcls_in, enum cudaMemcpyKind kind)
{
	dst.nptcls = src.nptcls;
	size_t intsize = sizeof(int)*(nptcls_in);
	size_t floatsize = sizeof(float)*(nptcls_in);

	cudaMemcpy(dst.px,src.px,floatsize,kind);
	cudaMemcpy(dst.py,src.py,floatsize,kind);
	cudaMemcpy(dst.pz,src.pz,floatsize,kind);
	cudaMemcpy(dst.vx,src.vx,floatsize,kind);
	cudaMemcpy(dst.vy,src.vy,floatsize,kind);
	cudaMemcpy(dst.vz,src.vz,floatsize,kind);
	cudaMemcpy(dst.index,src.index,intsize,kind);

}

__host__ float box_muller(float width, float offset)
{
	float result;
	float pi = 3.14159265358979323846264338327950288419716939937510;
	float u1 = ((float)(rand() % 100000)+1.0)/100000.0;
	float u2 = ((float)(rand() % 100000)+1.0)/100000.0;

	result = width/4.0*sqrt(-2*log(u1))*cos(2*pi*u2)+offset;

	//printf(" result = %f \n",result);

	return result;
}

__host__ void XPlist::random_distribution(int3 gridDims,float3 gridSpacing)
{

	if(location)
	{
		printf("Error, XPlist::random_distribution() cannot write to a particle list existing on a device \n");
		return;
	}

	int xindex;
	int yindex;
	int zindex;
	float3 gridlim;
	gridlim.x = gridDims.x*gridSpacing.x;
	gridlim.y = gridDims.y*gridSpacing.y;
	gridlim.z = gridDims.z*gridSpacing.z;


	for(int i = 0;i < nptcls; i++)
	{

		px[i] = box_muller(gridlim.x/4.0,gridlim.x/2.0);
		py[i] = box_muller(gridlim.y/4.0,gridlim.y/2.0);
		pz[i] = box_muller(gridlim.z/4.0,gridlim.z/2.0);

		vx[i] = box_muller(100.0/1000.0,0);
		vy[i] = box_muller(100.0/1000.0,0);
		vz[i] = box_muller(100.0/1000.0,0);

		xindex = floor(px[i]/gridSpacing.x);
		yindex = floor(py[i]/gridSpacing.y);
		zindex = floor(pz[i]/gridSpacing.z);

		index[i] = zindex* gridDims.x * gridDims.y +
				yindex * gridDims.x + xindex;
		//printf("particle %i, at (%f,%f,%f) index (%i,%i,%i) cellindex %i \n",i,px[i],py[i],pz[i],nx[i],ny[i],nz[i],cellindex[i]);
	}
}

__global__ void find_cell_index_kernel(XPlist particles,
						int* cellindex_out)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;

	if(gidx < particles.nptcls)
	{
		cellindex_out[gidx] = particles.index[gidx];
		 //printf("particle %i, at (%f,%f,%f) index (%i,%i,%i) cellindex %i \n",gidx,x[gidx],y[gidx],z[gidx],nx[gidx],ny[gidx],nz[gidx],cellindex[gidx]);

	}
	return;

}

__host__ void XPlist::find_cell_index(int* cellindex_temp)
{
	if(!location)
	{
		printf("Error, XPlist::find_index() can only be called for a particle list residing on the device. \n");
		return;
	}

		int cudaGridSize = (nptcls+threadsPerBlock-1)/threadsPerBlock;
		find_cell_index_kernel<<<cudaGridSize,threadsPerBlock>>>(*this,cellindex_temp);
}



__global__ void write_xpindex_array(unsigned int* index_array,int nptcls)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;

	if(gidx < nptcls)
	{
		index_array[gidx] = gidx;
	}
}

__global__ void sort_remaining(XPlist particles, XPlist particles_temp, unsigned int* index_array)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;

	XPdata data;
	int index;

	if(gidx < particles.nptcls)
	{

		int ogidx = index_array[gidx]; // What particle is this thread relocating

		data.px = particles.px[ogidx];
		data.py = particles.py[ogidx];
		data.pz = particles.pz[ogidx];

		data.vx = particles.vx[ogidx];
		data.vy = particles.vy[ogidx];
		data.vz = particles.vz[ogidx];

		index = particles.index[ogidx];
	}
	__syncthreads();
	if(gidx < particles.nptcls)
	{
		particles_temp.px[gidx] = data.px;
		particles_temp.py[gidx] = data.py;
		particles_temp.pz[gidx] = data.pz;

		particles_temp.vx[gidx] = data.vx;
		particles_temp.vy[gidx] = data.vy;
		particles_temp.vz[gidx] = data.vz;

		particles_temp.index[gidx] = index;



		//__syncthreads();
/*
		printf("particle %i, at (%f,%f,%f) index (%i,%i,%i) cellindex %i \n",ogidx,particles.data[gidx].px,
				particles.data[gidx].py,particles.data[gidx].pz,particles.index[gidx].x,particles.index[gidx].y,
				particles.index[gidx].z,particles.index[gidx].w);
*/
	}
}

__host__ void XPlist::sort(float3 Pgridspacing, int3 Pgrid_i_dims)
{
	/* TODO
	 * 1) figure out the cell index of each particle
	 * 2) write a particle index array for values array
	 * 3) create a radix object
	 * 4) use radix.sort(d_keys,d_values) keys = cell index, values = particle index
	 * 5) launch a kernel to move the particles to their new array index based on the sorted key/value pairs
	 *
	 */

	cudaError status;

	if(!location)
	{
		printf("Error, XPlist::sort() can only be called for a particle list residing on the device. \n");
		return;
	}

	// Setup temporary particle index array
	unsigned int* XP_index_array;
	cudaMalloc((void**)&XP_index_array,nptcls*sizeof(unsigned int));

	int* cellindex_temp;
	cudaMalloc((void**)&cellindex_temp,nptcls*sizeof(int));

	unsigned int* d_keys = (unsigned int*)cellindex_temp;
	unsigned int* d_values = XP_index_array;

	int cudaGridSize = (nptcls+threadsPerBlock-1)/threadsPerBlock;

	XPlist temp_list(nptcls,device);

	cudaThreadSynchronize();

	// Populate the cellindex array so that it can be used to sort the particles
	find_cell_index(cellindex_temp);
	cudaThreadSynchronize();

	// Write a particle index array for values array
	write_xpindex_array<<<(nptcls+threadsPerBlock-1)/threadsPerBlock,threadsPerBlock>>>(XP_index_array,nptcls);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "Write_xpindex_array %s\n", cudaGetErrorString(status));}


/*
	// Create the RadixSort object
	nvRadixSort::RadixSort radixsort(nptcls);

	// Sort the key / value pairs
	radixsort.sort(d_keys,d_values,nptcls,32);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "radix sort %s\n", cudaGetErrorString(status));}
*/

	// wrap raw device pointers with a device_ptr
	thrust::device_ptr<uint> thrust_keys(d_keys);
	thrust::device_ptr<uint> thrust_values(d_values);

	// Sort the data
	thrust::sort_by_key(thrust_keys,thrust_keys+nptcls,thrust_values);


	// sort the rest of the particle data
	sort_remaining<<<cudaGridSize,threadsPerBlock>>>(*this,temp_list,XP_index_array);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "sort_remaining %s\n", cudaGetErrorString(status));}

	XPlistCopy(*this,temp_list,nptcls,cudaMemcpyDeviceToDevice);
	cudaThreadSynchronize();

	cudaFree(XP_index_array);
	cudaFree(cellindex_temp);
	temp_list.XPlistFree();
	cudaThreadSynchronize();
	return;

}
__global__ void count_particles(XPlist particles,int2* cellInfo, int nptcls, int ncells)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;
	int cellindex;

	__shared__ int XPcellindex[threadsPerBlock+101];

	if(gidx < nptcls-1)
	{
		XPcellindex[idx] = particles.index[gidx];
		if((idx==threadsPerBlock-1)||(gidx == nptcls-2))
		{
			XPcellindex[idx+1] = particles.index[gidx+1];
		}
	}
	__syncthreads();
	if(gidx < nptcls-1)
	{
		if(XPcellindex[idx] != XPcellindex[idx+1])
		{
			cellindex = XPcellindex[idx+1];

			// Set index of first particle in next cell
			for(int i=XPcellindex[idx]+1;i<=cellindex;i++)
			{
				cellInfo[i].x = gidx+1;
			}
			//printf("(GPU) %i particles before cell %i \n",gidx+1,XPcellIndex[gidx]+1);
		}
	}
	else if(gidx == nptcls-1)
	{
		cellindex = XPcellindex[idx]+1;
		cellInfo[0].x = 0;
		cellInfo[cellindex].x = gidx+1;
		cellInfo[ncells].x = nptcls;
	}
}

__global__ void fix_cellinfo(int2* cellinfo,int ncells)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;

	if(gidx < ncells)
	{
		int tempi0;
		int tempi1;
		int j=1;

		if(cellinfo[gidx].x != 0)
		{
			while((cellinfo[gidx+j].x == 0)&&((gidx+j) < ncells))
			{
				cellinfo[gidx+j].x = cellinfo[gidx].x;
				j++;
				__threadfence();
			}
		}

	}

}

__global__ void count_blocks(int2* cellinfo,int* nblocks_percell,int* redoxtemp,int* blocksize,int ncells)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;

	__shared__ int tempCellInfo[threadsPerBlock+1];

	float temp;
	int tempi;

	tempCellInfo[idx] = 0;
	if(idx == 0)
		redoxtemp[blockIdx.x] = 0;
	__syncthreads();
	if(gidx < ncells)
	{
		if(gidx ==0)
			blocksize[0] = 0;


		temp = ((float)(cellinfo[gidx+1].x-cellinfo[gidx].x))/((float)threadsPerBlock);

		tempCellInfo[idx] = ceil(temp);

		cellinfo[gidx].y = tempCellInfo[idx];
		nblocks_percell[gidx] = tempCellInfo[idx];
	}
		__syncthreads();
		//if(gidx < ncells)
		//	printf("(GPU) %i blocks in cell %i with %i - %i = %i particles \n", cellinfo[gidx].y, gidx,cellinfo[gidx+1].x,cellinfo[gidx].x,(int)(temp*threadsPerBlock));



		tempi= reduce(tempCellInfo,redoxtemp);

		if(idx == 0 )
		{
			redoxtemp[blockIdx.x] = tempi;
			__threadfence_system();
			atomicAdd(&blocksize[0],tempi);
			__threadfence_system();
		}
		//printf("(GPU) Total number of blocks = %i \n", redoxtemp[blockIdx.x]);


}

__global__
void populate_blockinfo(int3* blockinfo, int2* cellinfo,int* ifirstblock, int ncells)
{
	int idx = threadIdx.x;
	int gidx = blockIdx.x*blockDim.x+idx;


	int k = 0;
	int firstParticle;

	if(gidx < ncells-1)
	{

		firstParticle = cellinfo[gidx].x;

		for(int j = 0;j<cellinfo[gidx].y;j++)
		{
			k = ifirstblock[gidx]+j;
			blockinfo[k].x = gidx;
			blockinfo[k].y = firstParticle;

			if(j < cellinfo[gidx].y-1)
			{
				blockinfo[k].z = threadsPerBlock;
				firstParticle += threadsPerBlock;
			}
			else
			{
				blockinfo[k].z = cellinfo[gidx+1].x-firstParticle;
			}
		}
	}

}
/*
__global__ void populate_blockinfo(int3* blockinfo_in,int2* cellinfo, int nblocks)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = idx+blockIdx.x*blockDim.x;

	__shared__ bool test[threadsPerBlock];

	int i;
	int mycell;
	int comparecell;

	__shared__ int3 blockinfo[threadsPerBlock+1];
	__shared__ int cellblockID[threadsPerBlock];

	cellblockID[idx] = 0;

	__syncthreads();

	if(gidx < nblocks)
	{
		blockinfo[idx+1] = blockinfo_in[gidx];

		if((idx == 0)&&(gidx != 0))
		{
			blockinfo[idx] = blockinfo_in[gidx-1];
		}
		// figure out what block I am in my cell
		i = 1;
		if(gidx == 0)
			test[idx] = 0;
		else
			test[idx] =1;

		while(test[idx])
		{
			mycell = blockinfo[idx+1].x;
			if(i > idx)
			{ // End of this block, We have to go to the global blockinfo
				comparecell = blockinfo_in[gidx-i].x;
			}
			else
			{
				comparecell = blockinfo[idx-i].x;
			}

			if(i > idx)
			{
				if((!test[idx-i])&&(comparecell == mycell))
				{
					cellblockID[idx] += cellblockID[idx-i]+1;
					test[idx] = 0;
					break;
				}
			}

			cellblockID[idx] += 1;
			test[idx] = (comparecell == mycell);

			i++;
		}
		__syncthreads();

		blockinfo[idx+1].y = cellblockID[idx]*cellinfo[mycell].x; // first particle for this block

		if(cellblockID[idx]+1 < cellinfo[mycell].y)
		{
			blockinfo[idx+1].z = threadsPerBlock;
		}
		else
		{
			blockinfo[idx+1].z = (cellinfo[mycell+1].x-cellinfo[mycell].x)-cellblockID[idx]*threadsPerBlock;
		}
		blockinfo_in[gidx] = blockinfo[idx+1];

	}

}*/

__global__ void condense_list(XPlist particles, XPlist particles_moved,
												XPlist particles_out,
												int3* block_info,int* blockoffsets,
												int* celloffsets,int* offsets, int* didimove,
												int2* cellinfo, int* cellcounter)
{
	/*
	 * This kernel calculates the offsets for each particle for the sorting step
	 */

	unsigned int idx = threadIdx.x;
	unsigned int bidx = blockIdx.x;
	unsigned int gidx = block_info[blockIdx.x].y+idx;

	unsigned int midx; // variable for moved particles new index
	unsigned int cellindex;

	__shared__ int offsets_temp[threadsPerBlock];

	if(idx == 0)
	{
		blockoffsets[bidx] = 0;
		if((bidx != 0)&&(block_info[bidx].x != block_info[bidx-1].x))
		{
			blockoffsets[bidx-1] = celloffsets[block_info[bidx-1].x];
		}
	}
	offsets_temp[idx] = 0;
	if(idx < block_info[blockIdx.x].z)
	{
		offsets_temp[idx] = offsets[gidx];

		__syncthreads();

		for(unsigned int i=1;i < floor(threadsPerBlock/2.0); i <<= 1 )
		{
			if(idx > i)
			{
				offsets_temp[idx] += offsets_temp[idx-i];
			}
			__syncthreads();
		}

		__syncthreads();

		offsets[gidx] = offsets_temp[idx];

		if(idx == 0)
		{
			blockoffsets[bidx] += offsets_temp[threadsPerBlock-1];

			__syncthreads();

			for(unsigned int i = 1; i < floor(gridDim.x/2.0);i <<= 1)
			{
				if(bidx > i)
				{
					blockoffsets[bidx] += blockoffsets[bidx-i];
				}
				__syncthreads();
			}
		}

		__syncthreads();

		if(bidx > 0)
			offsets[gidx] += blockoffsets[bidx-1];

		__syncthreads();

		// Write the particle data to the output list

		if(!didimove[gidx])
		{
			// If the Particle didn't move just use its offset to determine where it will go
			particles_out.px[gidx-offsets[gidx]] = particles.px[gidx];
			particles_out.index[gidx-offsets[gidx]] = particles.index[gidx];
		}
		else
		{
			// The particle moved into a new cell. We are going to use atomic functions so that it can be safely inserted into the list
			cellindex = particles.index[gidx];

			// Safely read what the current available index is and increment it by 1
			midx = atomicAdd(cellcounter+cellindex,1)+cellinfo[cellindex+1].x;






		}






	}

}

__device__ void XPlist::periodic_boundary(int3 griddims,float3 gridspacing,bool shared)
{
	unsigned int gidx;

	if(shared)
	{
		gidx = threadIdx.x;
	}
	else
	{
		gidx = threadIdx.x+blockIdx.x*blockDim.x;
	}

	float3 gridlims;
	gridlims.x = gridspacing.x*griddims.x;
	gridlims.y = gridspacing.y*griddims.y;
	gridlims.z = gridspacing.z*griddims.z;

	if(px[gidx] < 0)
	{
		px[gidx] += gridlims.x;

	}
	if(py[gidx] < 0)
	{
		py[gidx] += gridlims.y;
	}
	if(pz[gidx] < 0)
	{
		pz[gidx] += gridlims.z;
	}
	if(px[gidx] >gridlims.x)
	{
		px[gidx] -= gridlims.x;
	}
	if(py[gidx] >gridlims.y)
	{
		py[gidx] -= gridlims.y;
	}
	if(pz[gidx] >gridlims.z)
	{
		pz[gidx] -= gridlims.z;
	}

	return;
}

__global__ void XPlist_move_kernel_unsorted_cached(XPlist particles,
															float3 Pgridspacing, int3 Pgrid_i_dims, cudaMatrixf Phi,
															cudaMatrixi rho,
															float dt)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;

	float myphi=0;
	float phi_local;
	int nptcls_moved_temp;
	int3 temps;

	int nx;
	int ny;
	int nz;

	int temprho;
	int temprho2;

	__shared__ float temp_array[threadsPerBlock];


	/*
		if(idx == 0)
		{
			nptcls_left[blockIdx.x] = 0;
		}

		__shared__ int reduce_array[threadsPerBlock+1];

		reduce_array[idx] = 0;
		__threadfence_block();
		*/

	if(gidx < particles.nptcls)
	{
		nx = floor(particles.px[gidx]/Pgridspacing.x);
		ny = floor(particles.py[gidx]/Pgridspacing.y);
		nz = floor(particles.pz[gidx]/Pgridspacing.z);


		for(int k=0;k<2;k++)
		{
			for(int j=0;j<2;j++)
			{
				for(int i=0;i<2;i++)
				{
					if((nx >= 0)&&(ny >= 0)&&(nz >= 0)&&
						(nx < Pgrid_i_dims.x)&&(ny < Pgrid_i_dims.y)&&(nz <Pgrid_i_dims.z))
					{
						myphi += Phi(nx+i,ny+j,nz+k)/8.0;
					}
					else
					{
						myphi= 1;
					//	printf("nx = %i, ny = %i, nz = %i \n",nx,ny,nz);
					}
				}
			}
		}
		temp_array[idx] = myphi;
		/*


			// Sum up the new density
			for(int k=0;k<2;k++)
			{
				for(int j=0;j<2;j++)
				{
					for(int i=0;i<2;i++)
					{
						//phi_local = Phi(particles.index[gidx].x+i,particles.index[gidx].y+j,particles.index[gidx].z+k)+1;
						temprho = rint(1.0+myphi);
						temprho2 =atomicAdd(&rho(nx+i,ny+j,nz+k), temprho);
						//printf("temprho = %i at (%i,%i,%i) \n",temprho2+temprho,i+temps.x,j+temps.y,k+temps.z);
					}
				}

			}
			//



		//__syncthreads();
		particles.data[gidx].px += particles.data[gidx].vx*dt*myphi;
		particles.data[gidx].py += particles.data[gidx].vy*dt*myphi;
		particles.data[gidx].pz += particles.data[gidx].vz*dt*myphi;

		// Apply periodic boundary condition

		particles.periodic_boundary(Pgrid_i_dims,Pgridspacing,0);

		particles.index[gidx].x = floor(particles.data[gidx].px/Pgridspacing.x);
		particles.index[gidx].y = floor(particles.data[gidx].py/Pgridspacing.y);
		particles.index[gidx].z = floor(particles.data[gidx].pz/Pgridspacing.z);

		particles.index[gidx].w = particles.index[gidx].z * Pgrid_i_dims.x * Pgrid_i_dims.y +
				particles.index[gidx].y * Pgrid_i_dims.x + particles.index[gidx].x;

*/
	}

		// Total up the number of particles that have left the grid
/*

		nptcls_moved_temp = reduce(reduce_array,nptcls_left);
		__threadfence();
		//if(gidx < particles.nptcls)
		//	printf(" didileave = %i for particle %i \n",didileave[gidx],gidx);

	__threadfence();

	//if(gidx == 0)
		//printf("%i left the grid \n",nptcls_left[0]);

	__threadfence();
	*/


	//__syncthreads();


}

__global__ void XPlist_move_kernel_semisorted_cached(XPlist particles,
															float3 Pgridspacing, int3 Pgrid_i_dims, cudaMatrixf Phi,
															cudaMatrixi rho,
															float dt, int* nptcls_left, int* didileave)
{
	unsigned int idx = threadIdx.x;
	unsigned int gidx = blockIdx.x*blockDim.x+idx;

	float myphi=0;
	float phi_local;
	int nptcls_moved_temp;
	int3 temps;

	int nx;
	int ny;
	int nz;

	int temprho;
	int temprho2;

	__shared__ float temp_array[threadsPerBlock];

	/*
		if(idx == 0)
		{
			nptcls_left[blockIdx.x] = 0;
		}

		__shared__ int reduce_array[threadsPerBlock+1];

		reduce_array[idx] = 0;
		__threadfence_block();
		*/

	if(gidx < particles.nptcls)
	{
		nx = floor(particles.px[gidx]/Pgridspacing.x);
		ny = floor(particles.py[gidx]/Pgridspacing.y);
		nz = floor(particles.pz[gidx]/Pgridspacing.z);

		for(int k=0;k<2;k++)
		{
			for(int j=0;j<2;j++)
			{
				for(int i=0;i<2;i++)
				{
					if((nx >= 0)&&(ny >= 0)&&(nz >= 0)&&
						(nx < Pgrid_i_dims.x)&&(ny < Pgrid_i_dims.y)&&(nz <Pgrid_i_dims.z))
					{
						myphi += Phi(nx+i,ny+j,nz+k)/8.0;
					}
					else
					{
						myphi= 1;
					//	printf("nx = %i, ny = %i, nz = %i \n",nx,ny,nz);
					}
				}
			}
		}

		temp_array[idx] = myphi;


/*
			// Sum up the new density
			for(int k=0;k<2;k++)
			{
				for(int j=0;j<2;j++)
				{
					for(int i=0;i<2;i++)
					{
						//phi_local = Phi(particles.index[gidx].x+i,particles.index[gidx].y+j,particles.index[gidx].z+k)+1;
						temprho = rint(1.0+myphi);
						temprho2 =atomicAdd(&rho(nx+i,ny+j,nz+k), temprho);
						//printf("temprho = %i at (%i,%i,%i) \n",temprho2+temprho,i+temps.x,j+temps.y,k+temps.z);
					}
				}

			}
			//


		//__syncthreads();
		particles.data[gidx].px += particles.data[gidx].vx*dt*myphi;
		particles.data[gidx].py += particles.data[gidx].vy*dt*myphi;
		particles.data[gidx].pz += particles.data[gidx].vz*dt*myphi;

		// Apply periodic boundary condition

		particles.periodic_boundary(Pgrid_i_dims,Pgridspacing,0);

		particles.index[gidx].x = floor(particles.data[gidx].px/Pgridspacing.x);
		particles.index[gidx].y = floor(particles.data[gidx].py/Pgridspacing.y);
		particles.index[gidx].z = floor(particles.data[gidx].pz/Pgridspacing.z);

		particles.index[gidx].w = particles.index[gidx].z * Pgrid_i_dims.x * Pgrid_i_dims.y +
				particles.index[gidx].y * Pgrid_i_dims.x + particles.index[gidx].x;
				*/

	}

		// Total up the number of particles that have left the grid
/*

		nptcls_moved_temp = reduce(reduce_array,nptcls_left);
		__threadfence();
		//if(gidx < particles.nptcls)
		//	printf(" didileave = %i for particle %i \n",didileave[gidx],gidx);

	__threadfence();

	//if(gidx == 0)
		//printf("%i left the grid \n",nptcls_left[0]);

	__threadfence();
	*/


	//__syncthreads();


}

__global__ void XPlist_move_kernel_sorted_shared(XPlist particles,
															float3 Pgridspacing, int3 Pgrid_i_dims, cudaMatrixf Phi,
															cudaMatrixi rho,
															float dt, int* nptcls_left, int* didileave,
															int3* blockinfo)
{
	unsigned int idx = threadIdx.x;
	unsigned int bidx = blockIdx.x;
	unsigned int gidx = blockinfo[blockIdx.x].y+idx;
	int cellindex_old = blockinfo[bidx].x;
	float myphi;

	int temprho;
	int temprho2;
	int3 index;

	__shared__ int3 location;
	__shared__ float Phi_local[8];

	__shared__ int reduce_array[threadsPerBlock];


	reduce_array[idx] = 0;

	if(idx < blockinfo[bidx].z)
	{
		reduce_array[idx] = 1;
	}
	__syncthreads();

	temprho = reduce(reduce_array,didileave);

	__syncthreads();

	if((idx == 0)&&(gidx<particles.nptcls))
	{
		nptcls_left[bidx] = 0;
		location.x = floor(particles.px[gidx]/Pgridspacing.x);
		location.y = floor(particles.py[gidx]/Pgridspacing.y);
		location.z = floor(particles.pz[gidx]/Pgridspacing.z);
		__threadfence_block();
	}
__syncthreads();

	switch(idx)
	{
	case 0:
		Phi_local[0] = Phi(location.x,location.y,location.z);
		atomicAdd(&rho(location.x,location.y,location.z),temprho);
		__threadfence_block();
		break;
	case 1:
		Phi_local[1] = Phi(location.x+1,location.y,location.z);
		atomicAdd(&rho(location.x+1,location.y,location.z),temprho);
		__threadfence_block();
		break;
	case 2:
		Phi_local[2] = Phi(location.x,location.y+1,location.z);
		atomicAdd(&rho(location.x,location.y+1,location.z),temprho);
		__threadfence_block();
		break;
	case 3:
		Phi_local[3] = Phi(location.x+1,location.y+1,location.z);
		atomicAdd(&rho(location.x+1,location.y+1,location.z),temprho);
		__threadfence_block();
		break;
	case 4:
		Phi_local[4] = Phi(location.x,location.y,location.z+1);
		atomicAdd(&rho(location.x,location.y,location.z+1),temprho);
		__threadfence_block();
		break;
	case 5:
		Phi_local[5] = Phi(location.x+1,location.y,location.z+1);
		atomicAdd(&rho(location.x+1,location.y,location.z+1),temprho);
		__threadfence_block();
		break;
	case 6:
		Phi_local[6] = Phi(location.x,location.y+1,location.z+1);
		atomicAdd(&rho(location.x,location.y+1,location.z+1),temprho);
		__threadfence_block();
		break;
	case 7:
		Phi_local[7] = Phi(location.x+1,location.y+1,location.z+1);
		atomicAdd(&rho(location.x+1,location.y+1,location.z+1),temprho);
		__threadfence_block();
		break;
	default:
		break;
	}

__syncthreads();

	if(idx < blockinfo[bidx].z)
	{
		//printf("%i threads in this block \n",blockinfo[bidx].z);


		for(int k=0;k<8;k++)
		{
			myphi += Phi_local[k]/8.0;
		}


		//__syncthreads();
		particles.px[gidx] += particles.vx[gidx]*dt*myphi;
		particles.py[gidx] += particles.vy[gidx]*dt*myphi;
		particles.pz[gidx] += particles.vz[gidx]*dt*myphi;

		// Apply periodic boundary condition

		particles.periodic_boundary(Pgrid_i_dims,Pgridspacing,0);

		index.x = floor(particles.px[gidx]/Pgridspacing.x);
		index.y = floor(particles.py[gidx]/Pgridspacing.y);
		index.z = floor(particles.pz[gidx]/Pgridspacing.z);

		particles.index[gidx] = index.z * Pgrid_i_dims.x * Pgrid_i_dims.y +
				index.y * Pgrid_i_dims.x + index.x;

	}


}

/*

__global__ void XPlist_move_kernel_cached(XPlist particles, float3 Pgridspacing, int3 Pgrid_i_dims,
														 int3* block_info,float dt, int* nptcls_moved, int* offsets,int* didimove)
{

	 // To be launched with each block moving only particles located within a single cell.

	unsigned int idx = threadIdx.x;
	unsigned int gidx = block_info[blockIdx.x].y+idx;
	int cellindex_old;
	int nptcls_moved_temp;
	int3 temps;

	__shared__ int reduce_array[threadsPerBlock];

	if(idx < block_info[blockIdx.x].z)
	{

		cellindex_old = particles.cellindex[gidx];

		particles.px[gidx] += particles.vx[gidx]*dt;
		particles.py[gidx] += particles.vy[gidx]*dt;
		particles.pz[gidx] += particles.vz[gidx]*dt;

		temps.x = floor(particles.px[gidx]/Pgridspacing.x);
		temps.y = floor(particles.py[gidx]/Pgridspacing.y);
		temps.z = floor(particles.pz[gidx]/Pgridspacing.z);

		if((particles.nx[gidx] != temps.x)||(particles.ny[gidx] != temps.y)||(particles.nz[gidx] != temps.z))
		{
			particles.nx[gidx] = temps.x;
			particles.ny[gidx] = temps.y;
			particles.nz[gidx] = temps.z;

			particles.cellindex[gidx] = temps.z * Pgrid_i_dims.x * Pgrid_i_dims.y +
					temps.y * Pgrid_i_dims.x + temps.x;

			nptcls_moved[gidx] = 1;
			didimove[gidx] = 1;
			offsets[gidx+1] = 1; // offsets must be nptcls+1
		}
		else
		{
			nptcls_moved[gidx] = 0;
			didimove[gidx] = 0;
			offsets[gidx] = 0;
		}

		__syncthreads();

		// Total up the total number of particles that have changed cells

		nptcls_moved_temp = reduce(reduce_array,nptcls_moved);

	}
	return;
}
*/

__global__ void inject_new_particles_kernel(XPlist particles,XPlist new_particles,
																		int* didileave,
																		int* offsets,int* blockoffsets,
																		int nptcls_left)
{
	unsigned int idx = threadIdx.x;
	unsigned int bidx = blockIdx.x;
	unsigned int gidx = bidx*blockDim.x+idx;

	unsigned int midx; // variable for moved particles new index

	__shared__ int offsets_temp[threadsPerBlock];


	if(idx == 0)
	{
		blockoffsets[bidx] = 0;
	}

	offsets_temp[idx] = 0;
	offsets[gidx] = 0;
	__threadfence();

	__syncthreads();

	if(gidx < particles.nptcls)
	{
		offsets_temp[idx] = didileave[gidx];
		__threadfence_block();
	}


	for(int i=1;i <= (threadsPerBlock/2.0); i <<= 1 )
	{
		if(idx >= i)
		{
			offsets_temp[idx] += offsets_temp[idx-i];
			__threadfence_block();
		}
		__syncthreads();
	}

	offsets[gidx] = offsets_temp[idx];

	__threadfence();
/*
	if(gidx == 0)
	{
		for(int i=0;i<particles.nptcls;i++)
		{
			//printf(" nptcls left should = %i @ didleave = %i, bidx = %i \n",offsets[i],didileave[i],((i+threadsPerBlock)/threadsPerBlock)-1);
		}
	}
*/

	if(idx == (threadsPerBlock-1))
	{
		blockoffsets[bidx] = offsets[gidx];
		//printf(" blockoffsets = %i @ block %i \n",blockoffsets[bidx],bidx);
		__threadfence_system();
	}
		__syncthreads();

		for(int i = 1; i <= rint(__powf(2.0,floor(__log2f(gridDim.x))-1));i <<= 1)
		{
			if(bidx >= i )
			{
				if(idx == (threadsPerBlock-1))
				{
					atomicAdd(&blockoffsets[bidx],blockoffsets[bidx-i]);
				//	printf(" %i + %i = %i @ block: %i - %i  \n",midx,blockoffsets[bidx-i],midx+blockoffsets[bidx-i],bidx,bidx-i);
					__threadfence_system();
				}
			}
			__syncthreads();
		}

	if(bidx > 0)
	{
		offsets[gidx] += blockoffsets[bidx-1];
		__threadfence_system();
	}


	__syncthreads();

	if(gidx < particles.nptcls)
	{

		if((didileave[gidx])&&(offsets[gidx]-1 < nptcls_left)&&(offsets[gidx] > 1))
		{
			//printf(" particle %i pulling particle %i \n",gidx,offsets[gidx]);

			particles.px[gidx] = new_particles.px[offsets[gidx]-1];

			particles.index[gidx] = new_particles.index[offsets[gidx]-1];
			__threadfence_system();

		}
	}

	__syncthreads();

	if(gidx == gridDim.x*(threadsPerBlock-1))
	{
		printf("nptcls left should = %i \n",blockoffsets[gridDim.x-1]);
		__threadfence_system();
	}

	return;

}

__host__ void XPlist::inject_new_particles(int* didileave,int nptcls_left)
{

	int cudaGridSize = (nptcls+threadsPerBlock-1)/threadsPerBlock;
	cudaError status;
	int* blockoffsets;
	int* offsets;
	cudaMalloc((void**)&offsets,cudaGridSize*threadsPerBlock*sizeof(int));
	cudaMalloc((void**)&blockoffsets,cudaGridSize*sizeof(int));

	XPlist new_particles_h(nptcls_left,host);
	XPlist new_particles_d(nptcls_left,device);

	int3 gridDims;
	float3 gridSpacing;
	new_particles_h.random_distribution(gridDims,gridSpacing);

	cudaThreadSynchronize();

	XPlistCopy(new_particles_d,new_particles_h,nptcls_left,cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, " inject new particles copy %s\n", cudaGetErrorString(status));}

	inject_new_particles_kernel<<<cudaGridSize,threadsPerBlock>>>(*this,new_particles_d,didileave,offsets,blockoffsets,nptcls_left);

	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, " inject new particles kernel %s\n", cudaGetErrorString(status));}

	new_particles_h.XPlistFree();
	new_particles_d.XPlistFree();

	cudaFree(blockoffsets);
	cudaFree(offsets);


}

__host__ void XPlist::move_cached_unsorted(float3 Pgridspacing, int3 Pgrid_i_dims, cudaMatrixf Phi,
																			cudaMatrixi rho,float dt)
{
	int cudaGridSize = (nptcls+threadsPerBlock-1)/threadsPerBlock;
	cudaError status;

	int* nptcls_left_d;
	int* nptcls_left_h = (int*)malloc(sizeof(int));
	int* didileave;

	cudaThreadSynchronize();

	cudaMalloc((void**)&nptcls_left_d,(cudaGridSize)*sizeof(int));
	cudaMalloc((void**)&didileave,nptcls*sizeof(int));

	cudaMemset(nptcls_left_d,0,(cudaGridSize)*sizeof(int));

	cudaThreadSynchronize();

	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "malloc before cached, unsorted move kernel %s\n", cudaGetErrorString(status));}

	XPlist_move_kernel_unsorted_cached<<<cudaGridSize,threadsPerBlock>>>(*this,
																Pgridspacing, Pgrid_i_dims, Phi,
																rho,dt);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "cached, unsorted move kernel %s\n", cudaGetErrorString(status));}

	//cudaMemcpy(nptcls_left_h,nptcls_left_d,sizeof(int),cudaMemcpyDeviceToHost);

	//printf("nptcls_left = %i \n",nptcls_left_h[0]);

	//if(nptcls_left_h[0] < nptcls)
		//inject_new_particles(didileave,nptcls_left_h[0]);

	cudaFree(nptcls_left_d);
	cudaFree(didileave);



}

__host__ void XPlist::move_cached_semisorted(float3 Pgridspacing, int3 Pgrid_i_dims, cudaMatrixf Phi,
																			cudaMatrixi rho,float dt)
{
	int cudaGridSize = (nptcls+threadsPerBlock-1)/threadsPerBlock;
	cudaError status;

	int* nptcls_left_d;
	int* nptcls_left_h = (int*)malloc(sizeof(int));
	int* didileave;

	cudaThreadSynchronize();

	cudaMalloc((void**)&nptcls_left_d,(cudaGridSize+1)*sizeof(int));
	cudaMalloc((void**)&didileave,nptcls*sizeof(int));

	cudaMemset(nptcls_left_d,0,(cudaGridSize+1)*sizeof(int));

	cudaThreadSynchronize();

	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "malloc before cached, semisorted move kernel %s\n", cudaGetErrorString(status));}

	XPlist_move_kernel_semisorted_cached<<<cudaGridSize,threadsPerBlock>>>(*this,
																Pgridspacing, Pgrid_i_dims, Phi,
																rho,dt,nptcls_left_d,didileave);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "cached, semisorted move kernel %s\n", cudaGetErrorString(status));}

	//cudaMemcpy(nptcls_left_h,nptcls_left_d,sizeof(int),cudaMemcpyDeviceToHost);

	//printf("nptcls_left = %i \n",nptcls_left_h[0]);

	//if(nptcls_left_h[0] < nptcls)
	//	inject_new_particles(didileave,nptcls_left_h[0]);

	cudaFree(nptcls_left_d);
	cudaFree(didileave);


}

__host__ void XPlist::move_shared_sorted(float3 Pgridspacing, int3 Pgrid_i_dims, cudaMatrixf Phi,
																			cudaMatrixi rho,float dt)
{

	cudaError status;

	dim3 cudaGridSize(1,1,1);
	dim3 cudaBlockSize(1,1,1);
	size_t free2 = 0;
	size_t total = 0;

	int cudaGridSize1 = (nptcls+threadsPerBlock-1)/threadsPerBlock;

	int* nptcls_left_d;
	int* nptcls_left_h = (int*)malloc(sizeof(int));
	int* didileave;
	int gridSize = Pgrid_i_dims.x*Pgrid_i_dims.y*Pgrid_i_dims.z;


	int* NptinCell = (int*)malloc(gridSize*sizeof(float));
	int* nBlocks_h = (int*)malloc(sizeof(int));
	int* nBlocks_d;
//	printf("CudaMalloc1 \n");
	cudaMalloc((void**)&nBlocks_d,sizeof(int));
	cudaThreadSynchronize();

	int2* cellInfo_h = (int2*)malloc((gridSize+1)*sizeof(int2));

	int2* cellInfo_d;
	int* nblocks_per_cell;
//	printf("CudaMalloc2 \n");
	cudaMalloc((void**)&cellInfo_d,(gridSize+1)*sizeof(int2));
	cudaMalloc((void**)&nblocks_per_cell,(gridSize+1)*sizeof(int));
	cudaThreadSynchronize();
	cudaMemset(cellInfo_d,0,(gridSize+1)*sizeof(int2));

	int* redoxtemp_d;
//	printf("CudaMalloc4 \n");
	cudaMalloc((void**)&redoxtemp_d,((gridSize+threadsPerBlock+1)/threadsPerBlock)*sizeof(int));
	cudaThreadSynchronize();
	cudaMemset(redoxtemp_d,0,((gridSize+threadsPerBlock+1)/threadsPerBlock)*sizeof(int));


	int3* blockinfo_d; // grid index, first particle index, number of particles in block
	int3* blockinfo_h;

	// Sort the particle List
	cudaThreadSynchronize();
//	printf("Sorting Particles \n");
	sort(Pgridspacing, Pgrid_i_dims);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	 if(status != cudaSuccess){fprintf(stderr, "sort  particles %s\n", cudaGetErrorString(status));}

	// Figure out how many particles are in each cell
	cudaGridSize.x = (nptcls+threadsPerBlock-2)/threadsPerBlock;

//	printf("Launching Count Particles Kernel \n");
	count_particles<<<cudaGridSize,threadsPerBlock>>>(*this,cellInfo_d,nptcls,gridSize);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	 if(status != cudaSuccess){fprintf(stderr, "count particles %s\n", cudaGetErrorString(status));}

	 // Fix the cellinfo array for errors from cells with 0 particles
	cudaGridSize = (gridSize+threadsPerBlock-1)/threadsPerBlock;

/*
//	printf("Launching Fix Cellinfo Kernel \n");
	fix_cellinfo<<<cudaGridSize, threadsPerBlock>>>(cellInfo_d,gridSize);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	 if(status != cudaSuccess){fprintf(stderr, "Fix Cell Info %s\n", cudaGetErrorString(status));}
*/
	// Figure out how many thread blocks are needed for each cell and the total number of thread blocks

//	printf("Launching Count Blocks Kernel \n");
	CUDA_SAFE_KERNEL((count_blocks<<<cudaGridSize, threadsPerBlock>>>(cellInfo_d,nblocks_per_cell,redoxtemp_d,nBlocks_d,gridSize)));
	cudaThreadSynchronize();
	status = cudaGetLastError();
	 if(status != cudaSuccess){fprintf(stderr, "count blocks %s\n", cudaGetErrorString(status));}
//	cudaMemcpy(cellInfo_h,cellInfo_d,(gridSize+1)*sizeof(int2),cudaMemcpyDeviceToHost);
//	cudaMemcpy(nBlocks_h,nBlocks_d,sizeof(int),cudaMemcpyDeviceToHost);
	status = cudaGetLastError();
	 if(status != cudaSuccess){fprintf(stderr, "cpy block count %s\n", cudaGetErrorString(status));}
	cudaThreadSynchronize();

	// Do a scan to get the first block in each cell
	thrust::device_ptr<int> thrust_data(nblocks_per_cell);
	thrust::exclusive_scan(thrust_data,thrust_data+gridSize+1,thrust_data);


	//printf(" nblocks = %i \n",nBlocks_h[0]);

	if(nBlocks_h[0] > nptcls)
	{
		printf(" error, nBlocks way to big, returning \n");
		return;
	}

	cudaMemGetInfo(&free2,&total);
	//printf("Free Memory = %i mb\nUsed mememory = %i mb\n",(int)(free2)/(1<<20),(int)(total-free2)/(1<<20));


	// Populate blockInfo
	cudaGridSize1 = nBlocks_h[0];
	cudaMalloc((void**)&blockinfo_d,(nBlocks_h[0]+1)*sizeof(int3));
	blockinfo_h = (int3*)malloc(nBlocks_h[0]*sizeof(int3));
//	printf("Finished Move Kernel \n");


	cudaMalloc((void**)&nptcls_left_d,(cudaGridSize1+1)*sizeof(int));
	cudaMalloc((void**)&didileave,nptcls*sizeof(int));
	cudaMemset(nptcls_left_d,0,(cudaGridSize1+1)*sizeof(int));

	cudaGridSize.x = (gridSize+threadsPerBlock-1)/threadsPerBlock;
	cudaGridSize.y = 1;
	cudaBlockSize.x = threadsPerBlock;
	CUDA_SAFE_KERNEL((populate_blockinfo<<<cudaGridSize,cudaBlockSize>>>(blockinfo_d,cellInfo_d,nblocks_per_cell,gridSize)));

//	cudaMemcpy(blockinfo_d,blockinfo_h,nBlocks_h[0]*sizeof(int3),cudaMemcpyHostToDevice);

	 // Move the particles

	cudaThreadSynchronize();
//	printf("Launching Move Kernel \n");

	XPlist_move_kernel_sorted_shared<<<cudaGridSize1,threadsPerBlock>>>(*this,
																Pgridspacing, Pgrid_i_dims, Phi,
																rho,dt,nptcls_left_d,didileave,
																blockinfo_d);

	cudaThreadSynchronize();
//	printf("Finished Move Kernel \n");
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "cached, semisorted move kernel %s\n", cudaGetErrorString(status));}

//	if(nptcls_left_h[0] < nptcls)
	//	inject_new_particles(didileave,nptcls_left_h[0]);

	cudaFree(nptcls_left_d);
	cudaFree(didileave);
	cudaFree(blockinfo_d);
	cudaFree(cellInfo_d);
	cudaFree(redoxtemp_d);
	cudaFree(nBlocks_d);
	free(cellInfo_h);
	free(blockinfo_h);

//	printf("Finished freeing stuff \n");
	//cudaThreadSynchronize();


}
/*
void XPlist::cpu_move(float* phi,int* rho,float3 gridspacing,int3 griddims,float dt)
{
	float3 gridlims;

	float myphi;
	int phiidx;

	gridlims.x = gridspacing.x*griddims.x;
	gridlims.y = gridspacing.y*griddims.y;
	gridlims.z = gridspacing.z*griddims.z;

	for(int i=0;i<nptcls;i++)
	{

		myphi = 0;

		for(int k=0;k<2;k++)
		{
			for(int j=0;j<2;j++)
			{
				for(int l=0;l<2;l++)
				{
					phiidx = (index[i].z+l)*32*32+(index[i].y+j)*32+(index[i].x+k);
					myphi += phi[phiidx]/8.0;

				}
			}

		}

		data[i].px += data[i].vx*dt*myphi;
		data[i].py += data[i].vy*dt*myphi;
		data[i].pz += data[i].vz*dt*myphi;

		// Apply periodic boundary condition

		if(data[i].px < 0)
		{
			data[i].px += gridlims.x;

		}
		if(data[i].py< 0)
		{
			data[i].py += gridlims.y;
		}
		if(data[i].pz < 0)
		{
			data[i].pz += gridlims.z;
		}
		if(data[i].px>gridlims.x)
		{
			data[i].px -= gridlims.x;
		}
		if(data[i].py>gridlims.y)
		{
			data[i].py -= gridlims.y;
		}
		if(data[i].pz>gridlims.z)
		{
			data[i].pz -= gridlims.z;
		}

		index[i].x = floor(data[i].px/gridspacing.x);
		index[i].y = floor(data[i].py/gridspacing.y);
		index[i].z = floor(data[i].pz/gridspacing.z);

		index[i].w = index[i].z * griddims.x * griddims.y +
				index[i].y * griddims.x + index[i].x;

		for(int k=0;k<2;k++)
		{
			for(int j=0;j<2;j++)
			{
				for(int l=0;l<2;l++)
				{
					phiidx = (index[i].z+l)*32*32+(index[i].y+j)*32+(index[i].x+k);
					rho[phiidx] += 1;

				}
			}

		}

	}
	return;

}
*/

/*
__host__ void XPlist::reorder(void)
{
	 This method must accomplish the following:
	 *  1) Figure out how much each particle must be shifted up in a cell list -> also counts number of particles that left a cell
	 *  2) Put the particles that have moved into a new cell into a new particle list
	 *  	- Is it faster to
	 *  		a) just use radixsort() to sort the particles, and then put them back into the main list
	 *  	  or
	 *  		b) use atomics to add each particle back to the main list
	 *


	int* offsets_d;
	int* cellinfo_d;

	cudaMalloc((void**)&offsets_d,nptcls*sizeof(int));

}



*/


















































