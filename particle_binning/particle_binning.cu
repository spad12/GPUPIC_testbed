#define __CUDACC__
#define __cplusplus

#include "bin_scan.cu"


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



class particle
{
public:
	uint pindex;
	uint binindex;
};

class Particlebin
{
public:
	Particlebin* parentbin;
	Particlebin* subbins;
	uint nptcls;
	uint ifirstp;

};

__device__
bool compare_binids(uint binindex,uint bin_level)
{

	return ((binindex & ((0x0001) << (bin_level))) > 0);
}

__global__
void populate_splitting_list(particle* particles_in,Particlebin* bins,
												cudaMatrixui splitting_list,
												uint nptcls_max,uint bin_level)
{
	uint idx = threadIdx.x;
	uint gidx = blockDim.x*blockIdx.x+idx;
	uint binid = blockIdx.y;

	uint block_start = blockDim.x*blockIdx.x;
	uint pid;
	uint splitting_condition;

	__shared__ Particlebin parentbin;

	particle my_particle;

	if(idx == 0) parentbin = bins[binid];
	__syncthreads();

	if(gidx < parentbin.nptcls)
	{
		pid = parentbin.ifirstp + gidx;
		/*
		if(pid < 256)
		{
			my_particle = particles_in[pid];
		}
		else
		{
			printf("requesting read of pid %i by thread %i in bin %i with %i ptcls\n",pid,gidx,binid,parentbin.nptcls);
		}
		*/
		my_particle = particles_in[pid];

		splitting_condition = compare_binids(my_particle.binindex,bin_level);

		splitting_list(gidx,binid) = splitting_condition;
	}
	else if(gidx < nptcls_max)
	{
		splitting_list(gidx,binid) = 0;
	}


}


__global__ void
__launch_bounds__(BLOCK_SIZE,3)
find_new_ids(particle* particles_in,particle* particles_out,Particlebin* bins,
							cudaMatrixui sums,uint nptcls_max, uint bin_level,int* nptcls_max_out)
{
	uint idx = threadIdx.x;
	uint gidx = blockDim.x*blockIdx.x+idx;
	uint binid = blockIdx.y;
	uint block_start = blockDim.x*blockIdx.x;
	uint pid;

	particle my_particle;

	// No reason to run this block if it is more than the number of particles in the bin.
	// Might be able to avoid this with a better block mapping system
	if(block_start > bins[binid].nptcls)
		return;

	uint new_bin;
	__shared__ int nptcls_bin;

	int nptcls_max_out_temp;

	uint new_id;

	__shared__ Particlebin parentbin;
	__shared__ Particlebin subbins[2];

	// We need to figure out how many particles are going to be in each sub bin
	// We can also use this section to setup the subbins for the next tree level
	if(idx == 0)
	{
		parentbin = bins[binid];
		subbins[0].parentbin = &bins[binid];
		subbins[1].parentbin = &bins[binid];
		nptcls_bin = parentbin.nptcls;



		parentbin.subbins = (bins+gridDim.y)+2*binid;


		subbins[1].nptcls = max(0,sums(max(nptcls_bin-1,0),binid));
		subbins[0].nptcls = max(0,(nptcls_bin - subbins[1].nptcls));


		subbins[0].ifirstp = parentbin.ifirstp;
		subbins[1].ifirstp = subbins[0].ifirstp+subbins[0].nptcls;

		if(gidx == 0)
		{
		//printf("nptcls in bin %i = %i, with ifirstp = %i\n",2*binid,subbins[0].nptcls,subbins[0].ifirstp);
		//printf("nptcls in bin %i = %i, with ifirstp = %i\n",2*binid+1,subbins[1].nptcls,subbins[1].ifirstp);
		}

		if(gidx == 0)
		{
		parentbin.subbins[0] = subbins[0];
		parentbin.subbins[1] = subbins[1];
		nptcls_max_out_temp = max(subbins[0].nptcls,subbins[1].nptcls);
		atomicMax(nptcls_max_out,nptcls_max_out_temp);
		}
	}

	__syncthreads();

	if(gidx < nptcls_bin)
	{
		pid = gidx+parentbin.ifirstp;
		my_particle = particles_in[pid];
		new_bin = compare_binids(my_particle.binindex,bin_level);

		if(new_bin == 0)
		{
			new_id = gidx-sums(gidx,binid);
		}
		else
		{
			new_id = sums(gidx,binid)-1;
		}



		new_id += subbins[new_bin].ifirstp;
		//printf("particle %i, %i is being moved to index %i\n",pid,2*binid+new_bin,new_id);

		particles_out[new_id] = my_particle;
	}

}


void rebin_particles(Particlebin* bin_tree,particle* &particles,int* nptcls_max_out,int nptcls,int nbins_max)
{

	uint bin_level = 0;
	uint nptcls_max = nptcls;
	int nbins_current = 1;
	dim3 cudaGridSize(1,1,1);
	dim3 cudaBlockSize(1,1,1);
	int nlevels = log(nbins_max)/log(2);

	bin_level = nlevels-1;


	Particlebin* current_bins = bin_tree;

	cudaMatrixui splitting_list;


	particle* particles_temp;
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_temp,nptcls*sizeof(particle)));
	CUDA_SAFE_CALL(cudaMemcpy(particles_temp,particles,nptcls*sizeof(particle),cudaMemcpyDeviceToDevice));

	int* nptcls_max_next;
	CUDA_SAFE_CALL(cudaMalloc((void**)&nptcls_max_next,sizeof(int)));
	int* nptcls_max_next_h = (int*)malloc(sizeof(int));

	particle* particles_in = particles;
	particle* particles_out = particles_temp;
	particle* particles_swap;

	// Main tree traversing loop
	for(int i=0;i<(nlevels);i++)
	{
		CUDA_SAFE_CALL(cudaMemset(nptcls_max_next,0,sizeof(int)));
		printf("binlevel = %i, nptcls_max = %i \n", bin_level,nptcls_max);
		if(nptcls_max > nptcls)
		{
			printf("error number of particles is growing\n");
			return;
		}
		// Setup the kernel launch parameters
		cudaBlockSize.x = BLOCK_SIZE;
		cudaGridSize.x = (nptcls_max+cudaBlockSize.x-1)/cudaBlockSize.x;
		cudaGridSize.y = nbins_current;


		// Allocate space for the splitting list
		splitting_list.cudaMatrix_allocate(nptcls_max,nbins_current,1);

		// Populate the splitting list
		CUDA_SAFE_KERNEL((populate_splitting_list<<<cudaGridSize,cudaBlockSize>>>
										(particles_in,current_bins,splitting_list,nptcls_max,bin_level)));

		// Now we take the splitting list and find the cumulative sum for each bin
		bin_scan(splitting_list,nptcls_max,nbins_current);

		// Use the cumulative sum to calculate the new indices and move the particles
		CUDA_SAFE_KERNEL((find_new_ids<<<cudaGridSize,cudaBlockSize>>>
									 (particles_in,particles_out,current_bins,splitting_list,nptcls_max,bin_level,nptcls_max_next)));

		// Swap particles_in and particles_out
		particles_swap = particles_in;
		particles_in = particles_out;
		particles_out = particles_swap;

		// Advance counters
		bin_level--;
		current_bins = current_bins+nbins_current;
		nbins_current *= 2;
		CUDA_SAFE_CALL(cudaMemcpy(nptcls_max_next_h,nptcls_max_next,sizeof(int),cudaMemcpyDeviceToHost));

		nptcls_max = *nptcls_max_next_h;
		splitting_list.cudaMatrixFree();

	}

	particles = particles_in;
	*nptcls_max_out = nptcls_max;

	CUDA_SAFE_CALL(cudaFree(nptcls_max_next));
	CUDA_SAFE_CALL(cudaFree(particles_out));

}


__global__
void check_sort(particle* particles,Particlebin* bins,int nptcls_max)
{
	uint idx = threadIdx.x;
	uint gidx = blockDim.x*blockIdx.x+idx;
	uint binid = blockIdx.y;
	uint pidx;

	__shared__ Particlebin sbin;
	particle my_particle;

	if(idx == 0)
	{
		sbin = bins[binid];
	//	printf("nptcls in bin %i = %i, with ifirstp = %i\n",binid,sbin.nptcls,sbin.ifirstp);
	}

	__syncthreads();

	pidx = gidx+sbin.ifirstp;

	if(gidx < sbin.nptcls)
	{
		my_particle = particles[pidx];
		if(my_particle.binindex != (binid))
		{
			printf("Error, particle %i(%i), with binindex %i is in bin %i\n",pidx,gidx,my_particle.binindex,binid);
		}
	}
}

__host__
void rough_test(int nptcls)
{
	int gridsize = 4096;
	int max_nptclsperbin;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	particle* particles_h = (particle*)malloc(nptcls*sizeof(particle));

	particle* particles_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_d,nptcls*sizeof(particle)));

	Particlebin* bins;



	for(int i=0;i<nptcls;i++)
	{
		particles_h[i].pindex = i;
		particles_h[i].binindex = (rand()%gridsize);
	}

	// Figure how big the bin tree needs to be
	int bin_tree_size = 0;
	int nlevels = log(gridsize)/log(2);

	for(int i=0;i<(nlevels+1);i++)
	{
		bin_tree_size += (1<<i);
	}
	CUDA_SAFE_CALL(cudaMalloc((void**)&bins,bin_tree_size*sizeof(Particlebin)));

	// Set up the first particle bin

	Particlebin parent;
	parent.ifirstp = 0;
	parent.nptcls = nptcls;

	CUDA_SAFE_CALL(cudaMemcpy(bins,&parent,sizeof(Particlebin),cudaMemcpyHostToDevice));


	// Copy Particle data to the device
	CUDA_SAFE_CALL(cudaMemcpy(particles_d,particles_h,nptcls*sizeof(particle),cudaMemcpyHostToDevice));

	cudaEventRecord(start);
	// Rebin the particles
	rebin_particles(bins,particles_d,&max_nptclsperbin,nptcls,gridsize);
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

	CUDA_SAFE_KERNEL((check_sort<<<cudaGridSize,cudaBlockSize>>>
								 (particles_d,bottom_bins,max_nptclsperbin)));



}

__host__
void semi_sorted_test(int nptcls)
{
	int gridsize = 4096;
	int max_nptclsperbin;
	int temp_index;

	particle* particles_h = (particle*)malloc(nptcls*sizeof(particle));

	particle* particles_d;
	CUDA_SAFE_CALL(cudaMalloc((void**)&particles_d,nptcls*sizeof(particle)));

	Particlebin* bins;


	for(int i=0;i<nptcls;i++)
	{
		particles_h[i].pindex = i;
		particles_h[i].binindex = i/(nptcls/gridsize);

		if((rand()%1000) < 500)
		{
			temp_index = particles_h[i].binindex + rand()%5 - 2;
			if(temp_index < 0)
			{
				temp_index = gridsize-1;
			}
			else if(temp_index > gridsize-1)
			{
				temp_index = 0;
			}

			particles_h[i].binindex = temp_index;

		}
	}

	// Figure how big the bin tree needs to be
	int bin_tree_size = 0;
	int nlevels = log(gridsize)/log(2);

	for(int i=0;i<(nlevels+1);i++)
	{
		bin_tree_size += (1<<i);
	}
	CUDA_SAFE_CALL(cudaMalloc((void**)&bins,bin_tree_size*sizeof(Particlebin)));

	// Set up the first particle bin

	Particlebin parent;
	parent.ifirstp = 0;
	parent.nptcls = nptcls;

	CUDA_SAFE_CALL(cudaMemcpy(bins,&parent,sizeof(Particlebin),cudaMemcpyHostToDevice));


	// Copy Particle data to the device
	CUDA_SAFE_CALL(cudaMemcpy(particles_d,particles_h,nptcls*sizeof(particle),cudaMemcpyHostToDevice));

	// Rebin the particles
	rebin_particles(bins,particles_d,&max_nptclsperbin,nptcls,gridsize);


	// Check the sort
	dim3 cudaGridSize(1,gridsize,1);
	dim3 cudaBlockSize(BLOCK_SIZE,1,1);

	cudaGridSize.x = (max_nptclsperbin+cudaBlockSize.x-1)/cudaBlockSize.x;
	Particlebin* bottom_bins = bins+(bin_tree_size-gridsize);

	CUDA_SAFE_KERNEL((check_sort<<<cudaGridSize,cudaBlockSize>>>
								 (particles_d,bottom_bins,max_nptclsperbin)));



}

__global__
void check_scan_results_kernel(cudaMatrixui g_results0,uint* g_results1,int n)
{
	uint idx = threadIdx.x;
	uint gidx = blockDim.x*blockIdx.x+idx;

	uint result0;
	uint result1;

	if(gidx < n)
	{
		result0 = g_results0(gidx);
		result1 = g_results1[gidx];

		if(result0!=result1)
		{
			printf("%i != %i for thread %i \n",result0,result1,gidx);
		}
	}
}


__host__
void scan_test(int nptcls)
{

	cudaMatrixui sums(nptcls,1,1);
	uint* data_h = (uint*)malloc(nptcls*sizeof(uint));

	uint* results_d1;
	CUDA_SAFE_CALL(cudaMalloc((void**)&results_d1,2*sizeof(uint)*nptcls));

	for(int i=0;i<nptcls;i++)
	{
		data_h[i] = 1;
	}

	sums.cudaMatrixcpy(data_h,cudaMemcpyHostToDevice);

	bin_scan(sums,nptcls,1);

	for(int i=1;i<nptcls;i++)
	{
		data_h[i] += data_h[i-1];
	}

	CUDA_SAFE_CALL(cudaMemcpy(results_d1,data_h,nptcls*sizeof(uint),cudaMemcpyHostToDevice));

	int cudaGridSize = (nptcls+BLOCK_SIZE-1)/BLOCK_SIZE;
	int cudaBlockSize = BLOCK_SIZE;

	CUDA_SAFE_KERNEL((check_scan_results_kernel<<<cudaGridSize,cudaBlockSize>>>(
									sums,results_d1,nptcls)));

	sums.cudaMatrixFree();
	cudaFree(results_d1);
	free(data_h);


}









int main(void)
{
	cudaSetDevice(1);
	int nptcls = pow(2,24);

	rough_test(nptcls);

	return 0;
}















