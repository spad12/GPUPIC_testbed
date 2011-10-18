
#include "particleclass.cuh"
#include <iostream>

int* random_sorteddist( int* nptcls_out,int nx,int ny, int nz)
{

	int particlespercell[nx*ny*nz];
	int nptcls = 0;
	int* XPcellindex;

	srand(time(NULL));
	for(int i = 0;i<nx*ny*nz;i++)
	{
		particlespercell[i] = 20+(rand() % 40);
		nptcls += particlespercell[i];
	}

	XPcellindex = (int*)malloc(nptcls*sizeof(int));

	int k = 0;
	for(int i = 0;i<nx*ny*nz;i++)
	{
		for(int j=0;j<particlespercell[i];j++)
		{
			XPcellindex[k] = i;

			printf("particle %i, in cell %i \n",k,i);
			k++;
		}
	}

	*nptcls_out =nptcls;

	return XPcellindex;

}



__global__ void set_Phi(cudaMatrixf Phi)
{
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int idz = threadIdx.z;

	Phi(idx,idy,idz) = 0.05;
}

//void gpupic(int nx, int ny, int nz, int nptcls, float3* times)
int main(int argc, char* argv[])
{
	cudaSetDevice(1);

	int nx = 16;
	int ny = 16;
	int nz = 16;
	int gridSize = nx*ny*nz;

	int testx = 100;
	int testy = 100;
	int testz = 100;

	float3 times[1];
	float dt = 0.1;

	srand(13445);

	int nptcls = pow(2,22);

	for(int i=0;i<argc;i++)
	{
		if(std::string(argv[i]) == "-nx")
		{
			nx = atoi(argv[i+1]);
			ny = nx;
			nz = nx;
		}
		else if(std::string(argv[i]) == "-nptcls")
		{
			int power = atoi(argv[i+1]);
			nptcls = pow(2,power);
		}
		else if(std::string(argv[i]) == "-dt")
		{
			dt = atof(argv[i+1]);
		}
	}

	printf("Launching GPUMOVE with nx = %i, nptcls = %i, and dt = %f\n",nx,nptcls,dt);


	dim3 cudaGridSize(1,1,1);
	dim3 cudaBlockSize(1,1,1);
	//cudaBlockSize.x = threadsPerBlock;
	cudaError status;
	size_t free = 0;
	size_t total = 0;

	unsigned int timer = 0;
	unsigned int timer2 = 0;
	cutCreateTimer(&timer);
	cutCreateTimer(&timer2);

	cutCreateTimer(&sort_timer);


	float* Phi_h = (float*)malloc(35*35*35*sizeof(float));
	int* Rho_h = (int*)malloc(35*35*35*sizeof(int));


	// Grid Information
	float3 griddims;
	int3 grid_i_dims;
	float3 gridspacing;

	cudaMatrixf Phi(nx+2,ny+2,nz+2);

	cudaMatrixi rho(nx+2,ny+2,nz+2);

	cudaBlockSize.x = 2;
	cudaBlockSize.y = 2;
	cudaBlockSize.z = nz;

	cudaGridSize.x = nx/2;
	cudaGridSize.y = ny/2;

	set_Phi<<<cudaGridSize,cudaBlockSize>>>(Phi);
	cudaThreadSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){fprintf(stderr, "set phi %s\n", cudaGetErrorString(status));}


	// Particle Information lists

	 XPlist particles_h(nptcls,host);

	//XPlist particles_d2(nptcls,device);

	 XPlist particles_d(nptcls,device);

	 // Setup grid parameters

	 grid_i_dims.x = nx;
	 grid_i_dims.y = ny;
	 grid_i_dims.z = nz;

	 griddims.x = 1.0;
	 griddims.y = 1.0;
	 griddims.z = 1.0;

	 gridspacing.x = griddims.x/((float)nx);
	 gridspacing.y = griddims.y/((float)ny);
	 gridspacing.z = griddims.z/((float)nz);

	 // Setup a random spatial and velocity particle distribution

	 particles_h.random_distribution(grid_i_dims,gridspacing);

	 // Copy particles from the host to the device

	//printf("Copying particle list to GPU");
	 XPlistCopy(particles_d, particles_h,nptcls, cudaMemcpyHostToDevice);
	// XPlistCopy(particles_d2, particles_h,nptcls, cudaMemcpyHostToDevice);
		cudaThreadSynchronize();



	/*
	int* XPcellindex_h;

	printf("Setting up Particle list \n");

	XPcellindex_h = random_sorteddist(&nptcls,nx,ny,nz);


	printf("nptcls = %i \n",nptcls);

	int* XPcellindex_d;
	cudaMalloc((void**)&XPcellindex_d,nptcls*sizeof(int));
	status = cudaGetLastError();
	 if(status != cudaSuccess){fprintf(stderr, "particle allocate %s\n", cudaGetErrorString(status));}

	cudaMemcpy(XPcellindex_d,XPcellindex_h,nptcls*sizeof(int),cudaMemcpyHostToDevice);
	*/




	// sort particles if particle list is not already sorted
	particles_d.sort(gridspacing, grid_i_dims);
	cudaThreadSynchronize();

	//cudaMemcpy(particles_h.cellindex, particles_d.cellindex,nptcls*sizeof(int),cudaMemcpyDeviceToHost);
/*
	for(int i = 0; i < nptcls; i++)
	{
		printf("particle %i, in cell %i \n",i,particles_h.cellindex[i]);
	}
*/

	// See how much memory is allocated / free
	cudaMemGetInfo(&free,&total);
	//printf("Free Memory = %i mb\nUsed mememory = %i mb\n",(int)(free)/(1<<20),(int)(total-free)/(1<<20));


	// Move Particles
/*
	printf(" \n unsorted move \n \n");
	cutStartTimer(timer);
	for(int i = 0; i<100;i++)
	{
		particles_d2.move_cached_unsorted(gridspacing,grid_i_dims,Phi,rho,0.1);
		cudaThreadSynchronize();
	}
	cutStopTimer( timer);
	times[0].x= cutGetTimerValue(timer);
	printf( "\n Unsorted, cached  move took: %f (ms)\n\n", cutGetTimerValue( timer));
	cutResetTimer( timer );

	printf(" \n semisorted move \n \n");
	cutStartTimer(timer);
	for(int i = 0; i<100;i++)
	{
		particles_d.sort(gridspacing, grid_i_dims);
		cudaThreadSynchronize();
		particles_d.move_cached_semisorted(gridspacing,grid_i_dims,Phi,rho,0.1);
		cudaThreadSynchronize();
	}
	cutStopTimer( timer);
	times[0].y= cutGetTimerValue(timer);
	printf( "\nSemi-sorted, cached move took: %f (ms)\n\n", cutGetTimerValue( timer));
	cutResetTimer( timer );
*/

	times[0].x = 0;

	 XPlistCopy(particles_d, particles_h,nptcls, cudaMemcpyHostToDevice);

	 int test_index;
	//printf(" \n sorted shred move \n \n");
	cutStartTimer(timer);
	for(int i = 0; i<100;i++)
	{
		particles_d.move_shared_sorted(gridspacing,grid_i_dims,Phi,rho,dt);
/*
	//	printf(" \n finished one move step \n \n");
		cudaThreadSynchronize();

		cutResetTimer( timer2 );
		cutStartTimer(timer2);
		Phi.cudaMatrixcpy(phi_test_in,cudaMemcpyDeviceToHost);
		for(int i=0;i<testx;i++)
		{
			for(int j=0;j<testy;j++)
			{
				for(int k=0;k<testz;k++)
				{
					test_index = i+testx*(j+testy*k);
					phi_test_out[test_index] = (rand()%10)*(1.0+phi_test_in[test_index])/10.0;
				}
			}
		}
		phi_test_d.cudaMatrixcpy(phi_test_out,cudaMemcpyHostToDevice);
		cutStopTimer(timer2);
		times[0].x += cutGetTimerValue(timer2);

*/

	}
	cutStopTimer( timer);
	times[0].z= cutGetTimerValue(timer);
	printf( "\nSorted, shared move took: %f (ms)\n\n", cutGetTimerValue( timer));
	//printf( "\nCopying potential took: %f (ms)\n\n", times[0].x);
	cutResetTimer( timer );

/*
	printf(" \n CPU Move \n \n");
	cutStartTimer(timer);
	for(int i = 0; i<100;i++)
	{
		particles_h.cpu_move(Phi_h,Rho_h,gridspacing,grid_i_dims,0.1);

	//	printf(" \n finished one move step \n \n");
	}
	cutStopTimer( timer);
	printf( "\nCPU move took: %f (ms)\n\n", cutGetTimerValue( timer));
	cutResetTimer( timer );
*/
	cutDeleteTimer( timer);
	cutDeleteTimer( timer2);

	float fraction_moved = ((float)nptcls_moved_total)/100.0/((float)nptcls);
	printf("Fraction of particles that changed cells  = %f \n",fraction_moved);
	printf( "Total Sort time was %f (ms)\n\n", cutGetTimerValue(sort_timer));

	// Sort Particles

	// Return particles

	//cudaMalloc((void**)&(particles_d.px),nptcls*sizeof(float));

	//XPlistCopy(particles_h, particles_d,nptcls, cudaMemcpyDeviceToHost);

	//cudaMemcpy(particles_h.ny,particles_d.ny,nptcls*sizeof(int),cudaMemcpyDeviceToHost);
/*

	 for(int i = 0;i<nptcls;i++)
	 {
		 printf("particle %i, at (%f,%f,%f) index (%i,%i,%i) cellindex %i \n",i,particles_h.px[i],
				 particles_h.py[i],particles_h.pz[i],particles_h.nx[i],particles_h.ny[i],particles_h.nz[i],particles_h.cellindex[i]);

	 }

*/


	particles_h.XPlistFree();
	particles_d.XPlistFree();
	//particles_d2.XPlistFree();

	Phi.cudaMatrixFree();
	rho.cudaMatrixFree();



}
/*
int main(void)
{

	int nx[1] = {32};
	int nptcl_power[7] = {10,12,14,16,18,20,22};
	int nptcls;
	int k =0;

	float3 times[21];

	for(int i=0;i<;i++)
	{
		for(int j=0;j<7;j++)
		{

			nptcls = pow(2.0,(float)nptcl_power[j]);

			gpupic(nx[i],nx[i],nx[i],nptcls,&times[k]);


			k++;
		}

	}

	for(int i=0;i<14;i++)
	{
		printf(" %f, %f, %f, \n",times[i].x,times[i].y,times[i].z);
	}

	return 0;

}


*/














































