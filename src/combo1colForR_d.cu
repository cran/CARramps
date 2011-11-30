// combo1colForR.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include <Rmath.h>
#include "mstnrUtils.h"
#include "combo1colForR_d.h"

#define BLOCKSIZE 16

void doCombo1colD( double *a, double *b, double *D, double *tausqy, 
      double* tausqphi, double *By, double *mean, double *sd,
      int *na1, int *nb1, int *nc1, int *F1) 
{

   __global__ void kronVectMult1colOnDevice(double *a, double *b, double *c, 
   double *mean_d, double *sd_d, int na, int nb, int iter) ;

  void checkCUDAError(const char *msg) ;

  double *a_d, *b_d, *c_d, *mean_d, *sd_d ;  // pointer to device memory
  int i, j, k, iter ;
  int na = na1[0],nb = nb1[0], nab = na1[0] * nb1[0], nc = nc1[0], F=F1[0];
  int Fm1 = F - 1 ;

  double *Bphi ;
  double neweigendenom, normmean, normstd ;

  size_t sizea = na * na*sizeof(double);
  size_t sizeb = nb * nb*sizeof(double);
  size_t sizec = nab * sizeof(double); // Changed from mat ver

  // allocate array on host
 
  Bphi = (double *)malloc(sizec);


  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &mean_d, sizec);
  cudaMalloc((void **) &sd_d, sizec);


  // copy eigenvector matrices from host to device
  cudaMemcpy(a_d, a, sizeof(double)*na*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(double)*nb*nb, cudaMemcpyHostToDevice);

  // Check for any CUDA errors
    checkCUDAError("memcpy");

  // initialize accumulators on device 
  cudaMemset( mean_d, 0, sizec ) ;  
  cudaMemset( sd_d, 0, sizec ) ;

  // Check for any CUDA errors
    checkCUDAError("memset");

  // Compute execution configuration
  // Changed from matrix version

  int threadx = min(na, BLOCKSIZE), thready = min(nb, BLOCKSIZE) ;
  int blockx = (na)/threadx + ((na)%threadx ==0?0:1) ;
  int blocky = nb/thready + (nb%thready ==0?0:1) ;

  // printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim

  // printf("nc %d F %d \n", nc, F) ;

// get R's RNG seed 
GetRNGstate();

for(i = 0; i < nc; i++)  // for each row in output
{
// printf("%d \n", i) ;

  for( j=0; j < nab; j++ )   // for each data element
  {
     neweigendenom = tausqy[i] ;
     for( k = 0; k < Fm1; k++)
        neweigendenom += D[j * Fm1 +k] * tausqphi[i * Fm1 + k ] ;
     normmean = tausqy[i] * By[j] / neweigendenom ;
     normstd = 1.0 / sqrt(neweigendenom) ;
     Bphi[j] = rnorm( normmean, normstd ) ;
  }
  
  cudaMemcpy(c_d, Bphi, sizeof(double)*nab , cudaMemcpyHostToDevice);
  // Check for any CUDA errors
    checkCUDAError("memcpy");

  iter = i + 1 ;

  // do calculation on device:


  kronVectMult1colOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, b_d, c_d, 
  mean_d, sd_d, na, nb, iter );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");
}

// done gen rand numbers; send seed state back to R
PutRNGstate(); 

  // Retrieve result from device 

  cudaMemcpy(mean, mean_d, sizeof(double)*nab, cudaMemcpyDeviceToHost);
  cudaMemcpy(sd, sd_d, sizeof(double)*nab, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");
//}

  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);  cudaFree(mean_d); 
  cudaFree(sd_d)  ;

 // printf("after cudaFree \n") ; 
}

__global__ void kronVectMult1colOnDevice(double *a, double *b, double *c, 
double *mean, double *sd, int na, int nb, int iter)
{
  /*  a is na x na;  b is nb x nb;  c is (na*nb) x nc */

  double Csub = 0.0, currdiff, oldmean, oldsd ;  /* element computed by this thread */
  int N = na * nb,  acol,  bcol ;
  int arow = min(blockIdx.x*blockDim.x + threadIdx.x, na-1); /* output row */  
  int brow = min( blockIdx.y*blockDim.y + threadIdx.y, nb-1) ; /* output col */
  int idxtot = arow * nb + brow ;
  oldmean = mean[idxtot] ;
  oldsd = sd[idxtot] ;
  double newmean ;

  if( idxtot < N ) 
    {

      for( int k = 0; k < N; k++)
         {
           acol = k / nb ;
           bcol = k % nb ;
           Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ] 
               * c[k ] ;  
        }
      currdiff = Csub - oldmean ;
      newmean = oldmean + currdiff / (double) iter ;
      mean[idxtot] = newmean ;
      sd[idxtot] = oldsd + currdiff * (Csub - newmean) ;
    }   
}

