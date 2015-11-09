/****************************************************************************
 * FILE: mpi_heat2D.c
 * OTHER FILES: draw_heat.c  
 * DESCRIPTIONS:  
 *   HEAT2D Example - Parallelized C Version
 *   This example is based on a simplified two-dimensional heat 
 *   equation domain decomposition.  The initial temperature is computed to be 
 *   high in the middle of the domain and zero at the boundaries.  The 
 *   boundaries are held at zero throughout the simulation.  During the 
 *   time-stepping, an array containing two domains is used; these domains 
 *   alternate between old data and new data.
 *
 *   In this parallelized version, the grid is decomposed by the master
 *   process and then distributed by rows to the worker processes.  At each 
 *   time step, worker processes must exchange border data with neighbors, 
 *   because a grid point's current temperature depends upon it's previous
 *   time step value plus the values of the neighboring grid points.  Upon
 *   completion of all time steps, the worker processes return their results
 *   to the master process.
 *
 *   Two data files are produced: an initial data set and a final data set.
 *   An X graphic of these two states displays after all calculations have
 *   completed.
 * AUTHOR: Blaise Barney - adapted from D. Turner's serial C version. Converted
 *   to MPI: George L. Gusciora (1/95)
 * LAST REVISED: 04/02/05
 ****************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NXPROB      240             /* x dimension of problem grid */
#define NYPROB      240              /* y dimension of problem grid */
#define STEPS       100                /* number of time steps */
#define MAXWORKER   8                  /* maximum number of worker tasks */
#define MINWORKER   3                  /* minimum number of worker tasks */
#define BEGIN       1                  /* message tag */
#define LTAG        2                  /* message tag */
#define RTAG        3                  /* message tag */
#define NONE        0                  /* indicates no neighbor */
#define DONE        4                  /* message tag */
#define MASTER      0                  /* taskid of first process */

struct Parms { 
  float cx;
  float cy;
} parms = {0.1, 0.1};

void print(float *tmp[],int size1, int size2,int taskid)
{
	int i,j,k;
	for(i=1; i<size1-1; i++){
		printf("\n");
		for(j=1; j<size2-1; j++){
			printf("%10.1f ",tmp[i][j]);
		}
	}
   	printf("\n");
}

/**************************************************************************
 *  subroutine update
 ****************************************************************************/
void inner_update(int size, float **arr1, float **arr2)
{
   int x, y;
   for (x = 2; x < size; x++) 
      for (y = 2; y < size; y++) 
        arr2[x][y] = arr1[x][y] + 
        parms.cx * (arr1[x+1][y] + arr1[x-1][y] - 2.0 * arr1[x][y]) +
        parms.cy * (arr1[x][y+1] + arr1[x][y-1] -  2.0 * arr1[x][y]);
}

void outer_update(int size, int taskid, int t_sqrt,float **arr1, float **arr2)
{
	int i;
	if (taskid/t_sqrt != 0) //not first row
	{
		for(i=2;i<size;i++)
			arr2[1][i] = arr1[1][i] + parms.cx * (arr1[2][i] + arr1[0][i] - 2.0 * arr1[1][i]) + parms.cy * (arr1[1][i+1] + arr1[1][i-1] -  2.0 * arr1[1][i]);
		if(taskid%t_sqrt != 0)
			arr2[1][1] = arr1[1][1] + parms.cx * (arr1[2][1] + arr1[0][1] - 2.0 * arr1[1][1]) + parms.cy * (arr1[1][2] + arr1[1][0] -  2.0 * arr1[1][1]);
		if(taskid%t_sqrt != t_sqrt-1)
			arr2[1][size] = arr1[1][size] + parms.cx * (arr1[2][size] + arr1[0][size] - 2.0 * arr1[1][size]) + parms.cy * (arr1[1][size+1] + arr1[1][size-1] -  2.0 * arr1[1][size]);
	}
	if(taskid%t_sqrt != 0) //now first column
		for(i=2;i<size;i++)
			arr2[i][1] = arr1[i][1] + parms.cx * (arr1[i+1][1] + arr1[i-1][1] - 2.0 * arr1[i][1]) + parms.cy * (arr1[i][2] + arr1[i][0] -  2.0 * arr1[i][1]);
	if (taskid/t_sqrt != t_sqrt-1){ //not last row
		for(i=2;i<size;i++)
			arr2[size][i] = arr1[size][i] + parms.cx * (arr1[size+1][i] + arr1[size-1][i] - 2.0 * arr1[size][i]) + parms.cy * (arr1[size][i+1] + arr1[size][i-1] -  2.0 * arr1[size][i]);
		if(taskid%t_sqrt != 0)
			arr2[size][1] = arr1[size][1] + parms.cx * (arr1[size+1][1] + arr1[size-1][1] - 2.0 * arr1[size][1]) + parms.cy * (arr1[size][2] + arr1[size][0] -  2.0 * arr1[size][1]);
		if(taskid%t_sqrt != t_sqrt-1)
			arr2[size][size] = arr1[size][size] + parms.cx * (arr1[size+1][size] + arr1[size-1][size] - 2.0 * arr1[size][size]) + parms.cy * (arr1[size][size+1] + arr1[size][size-1] -  2.0 * arr1[size][size]);
	}
	if(taskid%t_sqrt != t_sqrt-1) //not column
		for(i=2;i<size;i++)
			arr2[i][size] = arr1[i][size] + parms.cx * (arr1[i+1][size] + arr1[i-1][size] - 2.0 * arr1[i][size]) + parms.cy * (arr1[i][size+1] + arr1[i][size-1] -  2.0 * arr1[i][size]); 
}

int main (int argc, char *argv[])
{
	void inidat();
	float  ***array;        /* array for grid */
	int	taskid,                     /* this task's unique id */
		numtasks,                   /* number of tasks */
		averow,rows,offset,extra,   /* for sending rows of data */
		dest, source,               /* to - from for message send-receive */
		left,right,        /* neighbor tasks */
		msgtype,                    /* for message types */
		rc,start,end,               /* misc */
		i,x,y,z,it,size,t_sqrt;              /* loop variables */
	MPI_Status status;
   	MPI_Datatype dt,dt2; 
    MPI_Request req, req2,req3,req4,req5;
    double t1,t2;

/* First, find out my taskid and how many tasks are running */
   	MPI_Init(&argc,&argv);
   	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
   	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

	
   	i = 0;
   	while(i*i < (NXPROB*NYPROB)/numtasks)
   		i++;
   	size = i;
   	i = 0;
   	while(i*i<numtasks)
   		i++;
   	t_sqrt = i;
   	if (taskid == 0){
   		printf("Size %d and tasks_sqrt is %d\n",size,t_sqrt);
   	}
   	MPI_Type_contiguous(size+2,MPI_FLOAT, &dt); 
	MPI_Type_commit(&dt);
	MPI_Type_vector(size+2,1,size+2,MPI_FLOAT,&dt2);
	MPI_Type_commit(&dt2); 
	array = malloc(2*sizeof(float**));
	for (i = 0;i<2;i++){
		array[i] = malloc((2+size)*sizeof(float*));
		array[i][0] = malloc(((2+size)*(2+size))*sizeof(float));
		for (x = 1;x<2+size;x++){
			array[i][x] = &(array[i][0][x*(2+size)]);
		}
	}
	for (z=0; z<2; z++){
		for (x=0; x<2+size; x++){
			for (y=0; y<2+size; y++){
				array[z][x][y] = 0.0;
			}
		}
	}
	z = 0;
	inidat(NXPROB,NYPROB,array[z],size*(taskid/t_sqrt),size*(taskid%t_sqrt),size);
	// for (i=0;i<numtasks;i++)
	// {
	// 	if (i == taskid)
	// 		print(array[z],size+2,size+2,0);
	// 	sleep(1);
	// }
	if (taskid == 0)
   	{
   		printf("Grid size: X= %d  Y= %d  Time steps= %d\n",NXPROB,NYPROB,STEPS);
   		t1 = MPI_Wtime();
   	}
	for (i = 1; i <= STEPS; i++)
	{
		if (taskid/t_sqrt != 0) //not first row
		{
			MPI_Isend(array[z][1],1,dt,taskid-t_sqrt,100, MPI_COMM_WORLD, &req);
			MPI_Irecv(array[z][0],1,dt,taskid-t_sqrt,100, MPI_COMM_WORLD, &req2);
		}
		if (taskid/t_sqrt != t_sqrt-1) //not last row
		{
			MPI_Isend(array[z][size],1,dt,taskid+t_sqrt,100, MPI_COMM_WORLD, &req);
			MPI_Irecv(array[z][size+1],1,dt,taskid+t_sqrt,100, MPI_COMM_WORLD, &req3);
		}
		if(taskid%t_sqrt != 0) //not last column
		{
			MPI_Isend(&array[z][0][1],1,dt2,taskid-1,100, MPI_COMM_WORLD, &req);
			MPI_Irecv(&array[z][0][0],1,dt2,taskid-1,100, MPI_COMM_WORLD, &req4);
		}
		if(taskid%t_sqrt != t_sqrt-1) //not last column
		{
			MPI_Isend(&array[z][0][size],1,dt2,taskid+1,100, MPI_COMM_WORLD, &req);
			MPI_Irecv(&array[z][0][size+1],1,dt2,taskid+1,100, MPI_COMM_WORLD, &req5);
		}
		inner_update(size,array[z],array[1-z]);
		if (taskid/t_sqrt != 0) //not first row
			MPI_Wait(&req2,&status);
		if (taskid/t_sqrt != t_sqrt-1) //not last row
			MPI_Wait(&req3,&status);
		if(taskid%t_sqrt != 0) //not first column
			MPI_Wait(&req4,&status);
		if(taskid%t_sqrt != t_sqrt-1) //not last column
			MPI_Wait(&req5,&status);
		outer_update(size,taskid,t_sqrt,array[z],array[1-z]);
		z = 1-z;
	}
	// for (i=0;i<numtasks;i++)
	// {
	// 	if (i == taskid)
	// 	{
	// 		print(array[z],size+2,size+2,0);
	// 	}
	// 	sleep(1);
	// }
	
	if (taskid == 0){
		t2 = MPI_Wtime();
		printf("MPI_Wtime measured: %1.2f\n", t2-t1);
	} 
	for (i = 0;i<2;i++){
		free(array[i][0]);
		free(array[i]);
	}
	free(array);
	MPI_Type_free(&dt);
	MPI_Type_free(&dt2);
	MPI_Finalize();
}

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int ny, float **array,int off1,int off2,int size) {
int x, y;
for (x = 0; x < size; x++) 
  	for (y = 0; y < size; y++){
  		// printf("X is %d and Y is %d\n",x,y);
     	array[x+1][y+1] = (float)((x+off1) * (nx - (x+off1) - 1) * (y+off2) * (ny - (y+off2) - 1));
  	}
}
