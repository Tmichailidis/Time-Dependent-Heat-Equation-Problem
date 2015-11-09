/****************************************************************************
 * FILE: mpi_heat2D.c
 * OTHER FILES: draw_heat.c  
 * DESCRIPTIONS:  
 *   HEAT2D Example - Parallelized C Version
 *   This example is based on a simplified two-dimensional heat 
 *   equation domain decomposition.  The initial temperature is computed to be 
 *   high in the middle of the domain and zero at the boundaries.  The 
 *   boundaries are held at zero throughout the simulation.  During the 
 *   time-stepping, an u containing two domains is used; these domains 
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

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string>
#include <iostream>

using namespace std;


#define NXPROB      1000            /* x dimension of problem grid */
#define NYPROB      1000               /* y dimension of problem grid */
#define STEPS       100                /* number of time steps */

extern "C" float updateGPU(float**, float**, int, int, int);

void print(float *tmp[],int size1, int size2)
{
  int i,j;
  for(i=0; i<size1; i++){
    printf("\n");
    for(j=0; j<size2; j++){
      printf("%10.1f ",tmp[i][j]);
    }
  }
    printf("\n");
}

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int ny, float **array) {
int x, y;
for (x = 0; x < nx-1; x++) 
    for (y = 0; y < ny-1; y++){
      array[x][y] = (float)(x * (nx - x - 1) * y * (ny - y - 1));
    }
}

int main (int argc, char *argv[])
{
  float  ***u;   
  int i,ix,iy,iz;   

      u = new float**[2];
      for (i = 0;i<2;i++){
        u[i] = new float*[NXPROB];
        for (ix = 0;ix<(NXPROB);ix++){
          u[i][ix] = new float[NYPROB];
        }
      }
      /* Initialize everything - including the borders - to zero */
      for (iz=0; iz<2; iz++)
        for (ix=0; ix<(NXPROB); ix++)
          for (iy=0; iy<NYPROB; iy++)
            u[iz][ix][iy] = 0.0;
      inidat(NXPROB, NYPROB, u[0]);
      updateGPU(u[0],u[1],NXPROB,NYPROB,STEPS);
      for (i = 0;i<2;i++){
        for (ix = 0;ix<(NXPROB);ix++){
          delete[] u[i][ix];
        }
        delete[] u[i];
      }
      delete[] u;

      return 1;
 }
