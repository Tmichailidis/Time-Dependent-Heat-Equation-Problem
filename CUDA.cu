#include <stdio.h>

 
__global__ 
void update(float *ad, float *bd, int ny,int nx)
{
   int x, y;

   x = blockIdx.x;
   y = threadIdx.x;
   if(x > 0 && y > 0 && x < nx-1 && y < ny-1) 
   	bd[x*ny+y] = ad[x*ny+y] + (ad[(x+1)*ny+y] + ad[(x-1)*ny+y] - 2 * ad[x*ny+y])/10 + (ad[x*ny+(y+1)] + ad[x*ny+(y-1)] -  2 * ad[x*ny+y])/10;
}

extern "C" float updateGPU(float **arr1, float **arr2, int nx, int ny, int steps)
{ 
	float *ad,*bd,s[nx*ny], milli = 0.0;
	int i, j;
	size_t size = nx*ny*sizeof(float);
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("= mpi_heat2D - CUDA Version =\nGrid size: X = %d, Y = %d, Time steps = %d\n",nx,ny,steps);

    for (i = 0;i<nx;i++)
    	for (j = 0;j<ny;j++){
    		s[i*ny+j] = arr1[i][j];
    	}

	cudaMalloc( (void**)&ad, size); 
	cudaMemcpy( ad, s, size, cudaMemcpyHostToDevice );

    cudaMalloc( (void**)&bd, size);

	dim3 threads_per_block(ny);
	dim3 num_blocks(nx,1);
	
	cudaEventRecord(start);
	for(i = 0; i < steps; i++){
		if(i%2 == 0)
			update<<<num_blocks, threads_per_block>>>(ad, bd, ny,nx);
		else
			update<<<num_blocks, threads_per_block>>>(bd, ad, ny,nx);
	}
	cudaEventRecord(stop);
	if (i%2 == 0)
		cudaMemcpy( s, ad, size, cudaMemcpyDeviceToHost );
	else
		cudaMemcpy( s, bd, size, cudaMemcpyDeviceToHost );

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);

	printf("Time Elapsed is %2.6f seconds\n",milli/1000);	
 
	cudaFree( ad );
	cudaFree( bd );

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	for (i = 0;i<nx;i++)
    	for (j = 0;j<ny;j++)
    		 arr1[i][j] = s[i*ny+j];

	return 1;
}