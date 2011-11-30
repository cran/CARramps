// kronVectMult1colForR3Q_d.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include "mstnrUtils.h"
#include "kronVectMult1colForR3Q_d.h"

#define BLOCKSIZE 16

void doKronVectMult1col3QD( double *a, double *b, double *b2, double *c, double *retvect, int *na1, 
      int *nb1, int *nb21, int *nc1) 
{

   __global__ void kronVectMult1colOnDevice3Q(double *a, double *b, double * b2, 
   double *c, 
   double *d, int na, int nb, int nb2) ;

  void checkCUDAError(const char *msg) ;

  double *a_d, *b_d, *b2_d, *c_d, *d_d ;  // pointer to device memory
  int i, na = na1[0],nb = nb1[0], nb2 = nb21[0], nab = na1[0] * nb1[0] * nb21[0]
, nc = nc1[0];

  size_t sizea = na * na*sizeof(double);
  size_t sizeb = nb * nb*sizeof(double);
  size_t sizeb2 = nb2 * nb2*sizeof(double);
  size_t sizec = nab * sizeof(double); // Changed from mat ver


  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &b2_d, sizeb2);
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &d_d, sizec);


  // copy data from host to device
  cudaMemcpy(a_d, a, sizeof(double)*na*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(double)*nb*nb, cudaMemcpyHostToDevice);
  cudaMemcpy(b2_d, b2, sizeof(double)*nb2*nb2, cudaMemcpyHostToDevice);

  // Compute execution configuration
  // Changed from matrix version

  int threadx = min(na, BLOCKSIZE), thready = min(nb*nb2, BLOCKSIZE) ;
  int blockx = (na)/threadx + ((na)%threadx ==0?0:1) ;

  int blocky = nb* nb2/thready + ((nb*nb2)%thready ==0?0:1) ;

  // printf(" %d %d %d %d \n", threadx, thready, blockx, blocky);

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim


for(i = 0; i < nc; i++)
{
  cudaMemcpy(c_d, &c[i * nab], sizeof(double)*nab , cudaMemcpyHostToDevice);
  // Check for any CUDA errors
    checkCUDAError("memcpy");
  cudaMemset( d_d, 0, sizec ) ;
  // Check for any CUDA errors
    checkCUDAError("memset");


  // do calculation on device:

  kronVectMult1colOnDevice3Q <<< numBlocks, threadsPerBlock >>> (a_d, b_d, b2_d,
      c_d, d_d, na, nb, nb2 );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");

  // Retrieve result from device and store in c_hfd

  cudaMemcpy(&retvect[i * nab], d_d, sizeof(double)*nab, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");
}

  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(b2_d); cudaFree(c_d);  cudaFree(d_d) ;
}

__global__ void kronVectMult1colOnDevice3Q(double *a, double *b, double *b2, double *c, 
double *d, int na, int nb, int nb2)
{
  /*  a is na x na;  b is nb x nb;  c is (na*nb) x nc */

  double Csub = 0.0 ;  /* element computed by this thread */
  int N = na * nb * nb2,  acol,  bcol, b2col ;
  int arow = min(blockIdx.x*blockDim.x + threadIdx.x, na-1); /* output row */
  int brow = min( (blockIdx.y*blockDim.y + threadIdx.y) / nb2, nb-1) ;
  int b2row = min( (blockIdx.y*blockDim.y + threadIdx.y) % nb2, nb2-1) ;
  int idxtot = arow * nb * nb2 + brow * nb2 + b2row ;

  if( idxtot < N ) 
    {

      for( int k = 0; k < N; k++)
         {
           acol = k /( nb * nb2 ) ;
           bcol = k % (nb * nb2) / nb2 ;
           b2col = k % nb2 ;
           Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ]
               * b2[b2row * nb2 + b2col ] * c[k ] ;
        }

      d[idxtot] = Csub ;
    }   
}

