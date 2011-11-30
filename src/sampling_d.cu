// sampling_d.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include <Rmath.h>
#include "mstnrUtils.h"
#include "sampling_d.h"

#define BLOCKSIZE 256

void doSamplingD( double *smat, double *D, double *By,   
      double *alpha, 
      double* beta, double *logpostdens, double *newbetaret,
      int *nsamp1, int *N1, int *F1, int *nk1) 
{

   __global__ void samplingOnDevice( double *smat, double *D, double *By,   
      double *alpha, 
      double * beta, double *logpostdens, double *newbetaret,
      int nsamp, int N, int Fm1, double newalpha) ;

  void checkCUDAError(const char *msg) ;

  double *smat_d, *D_d, *By_d, *alpha_d, *beta_d, *logpostdens_d, 
     *newbetaret_d ;  // pointers to device memory
  double newalpha ;
  int i  ;
  int nsamp = nsamp1[0], N = N1[0], F = F1[0], nk = nk1[0] ;
  int Fm1 = F - 1 ;

  size_t sizes = nsamp * Fm1 *sizeof(double);
  size_t sizeD = N * Fm1 *sizeof(double);
  size_t sizeBy = N*sizeof(double);
  size_t sizea = F * sizeof(double);  
  size_t sizeret = nsamp * sizeof(double) ;

  newalpha = (double) (N - nk) / 2.0 ;
  for (i = 0; i < F; i++)
     newalpha += alpha[i] ;

  // allocate array on host
 
  // allocate array on device 
  cudaMalloc((void **) &smat_d, sizes);
  cudaMalloc((void **) &D_d, sizeD);
  cudaMalloc((void **) &By_d, sizeBy);
  cudaMalloc((void **) &alpha_d, sizea);
  cudaMalloc((void **) &beta_d, sizea);
  cudaMalloc((void **) &logpostdens_d, sizeret);
  cudaMalloc((void **) &newbetaret_d, sizeret);


  // copy matrices from host to device
  cudaMemcpy(smat_d, smat, sizes, cudaMemcpyHostToDevice);
  cudaMemcpy(D_d, D, sizeD, cudaMemcpyHostToDevice);
  cudaMemcpy(By_d, By, sizeBy, cudaMemcpyHostToDevice);
  cudaMemcpy(alpha_d, alpha, sizea, cudaMemcpyHostToDevice);
  cudaMemcpy(beta_d, beta, sizea, cudaMemcpyHostToDevice);

  // Check for any CUDA errors
    checkCUDAError("memcpy");

  // initialize accumulators on device 
  cudaMemset( logpostdens_d, 0, sizeret ) ;  
  cudaMemset( newbetaret_d, 0, sizeret ) ;  

  // Check for any CUDA errors
    checkCUDAError("memset");

  // Compute execution configuration

  int threadx = min(nsamp, BLOCKSIZE) ;
  int blockx = nsamp/threadx + (nsamp%threadx ==0?0:1) ;

  // printf(" %d %d \n", threadx, blockx);

  dim3 threadsPerBlock( threadx );   // block dim
  dim3 numBlocks(blockx);             // grid dim

   // do calc on device 

   samplingOnDevice <<< numBlocks, threadsPerBlock >>> ( smat_d, 
         D_d, By_d, alpha_d,  beta_d, logpostdens_d, newbetaret_d,
       nsamp,  N, Fm1, newalpha) ;

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");


  // Retrieve result from device 

  cudaMemcpy(logpostdens, logpostdens_d, sizeret, cudaMemcpyDeviceToHost);
  cudaMemcpy(newbetaret, newbetaret_d, sizeret, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");

  // clean up

  cudaFree(smat_d); cudaFree(D_d); cudaFree(By_d); cudaFree(alpha_d);  
  cudaFree(beta_d); cudaFree(logpostdens_d)  ; cudaFree(newbetaret_d) ;

  //  printf("after cudaFree \n") ; 
}

   __global__ void samplingOnDevice( double *smat, double *D, double *By,   
      double *alpha, 
      double * beta, double *logpostdens, double *newbetaret,
      int nsamp, int N, int Fm1, double newalpha) 

{
    int idxtot, j, k ;

    idxtot = blockIdx.x*blockDim.x + threadIdx.x ;
    double s0, whole = 0.0, logpostdensnumer = 0.0, logpostdensdenom ;
    double neweigennumer, neweigen, newbeta ;


    s0 = 1.0 ;
    for( j = 0; j < Fm1 ; j++)
    {
       s0 -= smat[idxtot * Fm1 + j] ;
    }

  if( idxtot < nsamp ) /* change 08/05/11 */ 
    {
    for( k = 0; k < N; k++)
      {
       neweigennumer = 0.0 ;
       for( j = 0; j < Fm1; j++)
           neweigennumer += smat[idxtot * Fm1 + j] * D[k*Fm1 + j] ;
       neweigen = s0 * neweigennumer / (s0 + neweigennumer) ;
       logpostdensnumer += neweigen > 0.0 ? log(neweigen) / 2.0 : 0.0 ;
       whole += neweigen * By[k] * By[k] ;
      }
   newbeta = whole / 2.0 + s0 * beta[0] ;
   logpostdensnumer += log(s0) * (alpha[0] - 1.0) ;
   for( j = 0; j < Fm1; j++)
     {
     logpostdensnumer += log(smat[idxtot * Fm1 + j]) * (alpha[j+1] - 1.0) ;
     newbeta += smat[idxtot * Fm1 + j] * beta[j+1] ;
     }
   logpostdensdenom = log(newbeta) * newalpha ;
   logpostdens[idxtot] = logpostdensnumer - logpostdensdenom ;
   newbetaret[idxtot] = newbeta ;
    
   }
}

