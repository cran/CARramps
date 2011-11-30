// combo1colForR3Q_d.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include <Rmath.h>
#include "mstnrUtils.h"
#include "combo1colForR3Q_d.h"

#define BLOCKSIZE 16

void doCombo1col3QD( double *a, double *b, double *b2,  double *D, 
      double *tausqy, 
      double* tausqphi, double *By, double *mean, double *sd,
      int *na1, int *nb1, int *nb21, int *nc1, int *F1) 
{

   __global__ void kronVectMult1colOnDevice3Q(double *a, double *b, 
   double *b2, double *c, 
   double *mean_d, double *sd_d, int na, int nb, int nb2, int iter) ;

  void checkCUDAError(const char *msg) ;

  double *a_d, *b_d, *b2_d, *c_d, *mean_d, *sd_d ;  // pointer to device memory
  int i, j, k, iter ;
  int na = na1[0], nb = nb1[0], nb2 = nb21[0], nab = na1[0] * nb1[0] * nb21[0], 
   nc = nc1[0], F=F1[0];
  int Fm1 = F - 1 ;

  double *Bphi ;
  double neweigendenom, normmean, normstd ;

  size_t sizea = na * na*sizeof(double);
  size_t sizeb = nb * nb*sizeof(double);
  size_t sizeb2 = nb2 * nb2*sizeof(double);
  size_t sizec = nab * sizeof(double); // Changed from mat ver

  // allocate array on host
 
  Bphi = (double *)malloc(sizec);


  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &b2_d, sizeb2);
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &mean_d, sizec);
  cudaMalloc((void **) &sd_d, sizec);


  // copy eigenvector matrices from host to device
  cudaMemcpy(a_d, a, sizeof(double)*na*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(double)*nb*nb, cudaMemcpyHostToDevice);
  cudaMemcpy(b2_d, b2, sizeof(double)*nb2*nb2, cudaMemcpyHostToDevice);

  // Check for any CUDA errors
    checkCUDAError("memcpy");

  // initialize accumulators on device 
  cudaMemset( mean_d, 0, sizec ) ;  
  cudaMemset( sd_d, 0, sizec ) ;

  // Check for any CUDA errors
    checkCUDAError("memset");

  // Compute execution configuration
  // Changed from matrix version

  /* int threadx = min(na, BLOCKSIZE), thready = min(nb, BLOCKSIZE), 
         threadz = min(nb2, BLOCKSIZE) ; */
  int threadx = min(na, BLOCKSIZE), thready = min(nb*nb2, BLOCKSIZE) ;
  int blockx = (na)/threadx + ((na)%threadx ==0?0:1) ;
/*   int blocky = nb/thready + (nb%thready ==0?0:1) ;
  int blockz = nb2/threadz + (nb2%threadz ==0?0:1) ; */

  int blocky = nb* nb2/thready + ((nb*nb2)%thready ==0?0:1) ;

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky);

  //dim3 threadsPerBlock( threadx, thready, threadz );   // block dim
  //dim3 numBlocks(blockx, blocky, blockz);             // grid dim

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim
printf("nab %d nc %d F %d \n", nab, nc, F) ;

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
     if(j < 10)
              printf("%7.4f \n", Bphi[j]) ;
  }
  
  cudaMemcpy(c_d, Bphi, sizeof(double)*nab , cudaMemcpyHostToDevice);
  // Check for any CUDA errors
    checkCUDAError("memcpy");

  iter = i + 1 ;

  // do calculation on device:

  // Call kronVectOnDevice kernel 
// printf("do calc on device \n") ;

  kronVectMult1colOnDevice3Q <<< numBlocks, threadsPerBlock >>> (a_d, b_d, 
     b2_d, c_d, mean_d, sd_d, na, nb, nb2, iter );

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


  // check results

/*  for (i=0; i<na * nb * nc; i+=100) 
     {
       assert(c_h[i] == c_hfd[i]);
       printf("%7.4f  \n", d_h[i]) ;
     }   */

  printf("Worked! \n") ;
  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(b2_d); cudaFree(c_d);  cudaFree(mean_d); 
  cudaFree(sd_d)  ;

 printf("after cudaFree \n") ; 
}


__global__ void kronVectMult1colOnDevice3Q(double *a, double *b, double *b2,
   double *c, double *mean, double *sd, int na, int nb, int nb2, int iter)
{
  /*  a is na x na;  b is nb x nb;  c is (na*nb) x nc */

  double Csub = 0.0, currdiff, oldmean, oldsd ;  /* element computed by this thread */
  int N = na * nb * nb2,  acol,  bcol, b2col ;
  int arow = min(blockIdx.x*blockDim.x + threadIdx.x, na-1); /* output row */  
  int brow = min( (blockIdx.y*blockDim.y + threadIdx.y) / nb2, nb-1) ; 
  int b2row = min( (blockIdx.y*blockDim.y + threadIdx.y) % nb2, nb2-1) ; 
  int idxtot = arow * nb * nb2 + brow * nb2 + b2row ;
  double newmean ;

  if( idxtot < N ) 
    {
     oldmean = mean[idxtot] ;
     oldsd = sd[idxtot] ;

      for( int k = 0; k < N; k++)
         {
           acol = k /( nb * nb2 ) ;
           bcol = k % (nb * nb2) / nb2 ;
           b2col = k % nb2 ;
           Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ] 
               * b2[b2row * nb2 + b2col ] * c[k ] ;  
        }
      currdiff = Csub - oldmean ;
      newmean = oldmean + currdiff / (double) iter ;
      mean[idxtot] = newmean ;
      sd[idxtot] = oldsd + currdiff * (Csub - newmean) ;
    }   
}

// combo1colForR3Q_dnew.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include <Rmath.h>
#include <combo1colForR3Q_d.h>

#define BLOCKSIZE 16

void doCombo1col3QD( double *a, double *b, double *b2,  double *D, 
      double *tausqy, 
      double* tausqphi, double *By, double *mean, double *sd,
      int *na1, int *nb1, int *nb21, int *nc1, int *F1) 
{

   __global__ void kronVectMult1colOnDevice(double *a, double *b, 
   double *b2, double *c, 
   double *mean_d, double *sd_d, int na, int nb, int nb2, int iter) ;

  void checkCUDAError(const char *msg) ;

  double *a_d, *b_d, *b2_d, *c_d, *mean_d, *sd_d ;  // pointer to device memory
  int i, j, k, iter ;
  int na = na1[0],nb = nb1[0], nb2 = nb21[0], nab = na1[0] * nb1[0] * nb21[0], 
   nc = nc1[0], F=F1[0];
  int Fm1 = F - 1 ;

  double *Bphi ;
  double neweigendenom, normmean, normstd ;

  size_t sizea = na * na*sizeof(double);
  size_t sizeb = nb * nb*sizeof(double);
  size_t sizeb2 = nb2 * nb2*sizeof(double);
  size_t sizec = nab * sizeof(double); // Changed from mat ver

  // allocate array on host
 
  Bphi = (double *)malloc(sizec);


  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &b2_d, sizeb2);
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &mean_d, sizec);
  cudaMalloc((void **) &sd_d, sizec);


  // copy eigenvector matrices from host to device
  cudaMemcpy(a_d, a, sizeof(double)*na*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(double)*nb*nb, cudaMemcpyHostToDevice);
  cudaMemcpy(b2_d, b, sizeof(double)*nb2*nb2, cudaMemcpyHostToDevice);

  // Check for any CUDA errors
    checkCUDAError("memcpy");

  // initialize accumulators on device 
  cudaMemset( mean_d, 0, sizec ) ;  
  cudaMemset( sd_d, 0, sizec ) ;

  // Check for any CUDA errors
    checkCUDAError("memset");

  // Compute execution configuration
  // Changed from matrix version

  int threadx = min(nab, BLOCKSIZE) ;
  int blockx = (nab)/threadx + ((nab)%threadx ==0?0:1) ;


  printf(" %d %d  \n", threadx,  blockx);


  dim3 threadsPerBlock( threadx );   // block dim
  dim3 numBlocks(blockx);             // grid dim
printf("nab %d nc %d F %d \n", nab, nc, F) ;

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
     if(j < 10)
              printf("%7.4f \n", Bphi[j]) ;
  }
  
  cudaMemcpy(c_d, Bphi, sizeof(double)*nab , cudaMemcpyHostToDevice);
  // Check for any CUDA errors
    checkCUDAError("memcpy");

  iter = i + 1 ;

  // do calculation on device:

  // Call kronVectOnDevice kernel 
// printf("do calc on device \n") ;

  kronVectMult1colOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, b_d, 
     b2_d, c_d, mean_d, sd_d, na, nb, nb2, iter );

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


  // check results

/*  for (i=0; i<na * nb * nc; i+=100) 
     {
       assert(c_h[i] == c_hfd[i]);
       printf("%7.4f  \n", d_h[i]) ;
     }   */

  printf("Worked! \n") ;
  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(b2_d); cudaFree(c_d);  cudaFree(mean_d); 
  cudaFree(sd_d)  ;

 printf("after cudaFree \n") ; 
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__global__ void kronVectMult1colOnDevice(double *a, double *b, double *b2,
   double *c, double *mean, double *sd, int na, int nb, int nb2, int iter)
{
  /*  a is na x na;  b is nb x nb;  c is (na*nb) x nc */

  double Csub = 0.0, currdiff, oldmean, oldsd ;  /* element computed by this thread */
  int N = na * nb * nb2,  acol,  bcol, b2col ;
  int idxtot = blockIdx.x * blockDim.x +threadIdx.x ;
  int  arow = idxtot /( nb * nb2 ) ;
  int    brow = idxtot % (nb * nb2) / nb2 ;
  int    b2row = idxtot % nb2 ;
  double newmean ;

  if( idxtot < N ) 
    {
     oldmean = mean[idxtot] ;
     oldsd = sd[idxtot] ;

      for( int k = 0; k < N; k++)
         {
           acol = k /( nb * nb2 ) ;
           bcol = k % (nb * nb2) / nb2 ;
           b2col = k % nb2 ;
           Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ] 
               * b2[b2row * nb2 + b2col ] * c[k ] ;  
        }
      currdiff = Csub - oldmean ;
      newmean = oldmean + currdiff / (double) iter ;
      mean[idxtot] = newmean ;
      sd[idxtot] = oldsd + currdiff * (Csub - newmean) ;
    }   
}

// combo1colForR.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include <Rmath.h>
#include <combo1colForR.h>

#define BLOCKSIZE 16

void doCombo1col( float *a, float *b, float *D, float *tausqy, 
      float* tausqphi, float *By, float *mean, float *sd,
      int *na1, int *nb1, int *nc1, int *F1) 
{

   __global__ void kronVectMult1colOnDevice(float *a, float *b, float *c, 
   float *mean_d, float *sd_d, int na, int nb, int iter) ;

  void checkCUDAError(const char *msg) ;

  float *a_d, *b_d, *c_d, *mean_d, *sd_d ;  // pointer to device memory
  int i, j, k, iter ;
  int na = na1[0],nb = nb1[0], nab = na1[0] * nb1[0], nc = nc1[0], F=F1[0];

  float *Bphi ;
  float neweigendenom, normmean, normstd ;

  size_t sizea = na * na*sizeof(float);
  size_t sizeb = nb * nb*sizeof(float);
  size_t sizec = nab * sizeof(float); // Changed from mat ver

  // allocate array on host
 
  Bphi = (float *)malloc(sizec);


  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &mean_d, sizec);
  cudaMalloc((void **) &sd_d, sizec);


  // copy eigenvector matrices from host to device
  cudaMemcpy(a_d, a, sizeof(float)*na*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(float)*nb*nb, cudaMemcpyHostToDevice);

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

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim

printf("nc %d F %d \n", nc, F) ;

for(i = 0; i < nc; i++)  // for each row in output
{
// printf("%d \n", i) ;

  for( j=0; j < nab; j++ )   // for each data element
  {
     neweigendenom = tausqy[i] ;
     for( k = 0; k < (F-1); k++)
        neweigendenom += D[j * (F-1) +k] * tausqphi[i * (F-1) +k ] ;
     normmean = tausqy[i] * By[j] / neweigendenom ;
     normstd = 1.0 / sqrt(neweigendenom) ;
     Bphi[j] = rnorm( normmean, normstd ) ;
    // printf("%6.2f \n", Bphi[j]) ;
  }
  
  cudaMemcpy(c_d, Bphi, sizeof(float)*nab , cudaMemcpyHostToDevice);
  // Check for any CUDA errors
    checkCUDAError("memcpy");

  iter = i + 1 ;

  // do calculation on device:

  // Call kronVectOnDevice kernel 
printf("do calc on device \n") ;

  kronVectMult1colOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, b_d, c_d, 
  mean_d, sd_d, na, nb, iter );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");
}

  // Retrieve result from device 

  cudaMemcpy(mean, mean_d, sizeof(float)*nab, cudaMemcpyDeviceToHost);
  cudaMemcpy(sd, sd_d, sizeof(float)*nab, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");
//}


  // check results

/*  for (i=0; i<na * nb * nc; i+=100) 
     {
       assert(c_h[i] == c_hfd[i]);
       printf("%7.4f  \n", d_h[i]) ;
     }   */

  printf("Worked! \n") ;
  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);  cudaFree(mean_d); 
  cudaFree(sd_d)  ;

 printf("after cudaFree \n") ; 
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__global__ void kronVectMult1colOnDevice(float *a, float *b, float *c, 
float *mean, float *sd, int na, int nb, int iter)
{
  /*  a is na x na;  b is nb x nb;  c is (na*nb) x nc */

  float Csub = 0.0, currdiff, oldmean, oldsd ;  /* element computed by this thread */
  int N = na * nb,  acol,  bcol ;
  int arow = min(blockIdx.x*blockDim.x + threadIdx.x, na-1); /* output row */  
  int brow = min( blockIdx.y*blockDim.y + threadIdx.y, nb-1) ; /* output col */
  int idxtot = arow * nb + brow ;
  oldmean = mean[idxtot] ;
  oldsd = sd[idxtot] ;
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
      mean[idxtot] = oldmean + currdiff / (float) iter ;
      sd[idxtot] = oldsd + currdiff * (Csub - oldmean) ;
    }   
}

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

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim

printf("nc %d F %d \n", nc, F) ;

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

  // Call kronVectOnDevice kernel 
// printf("do calc on device \n") ;

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


  // check results

/*  for (i=0; i<na * nb * nc; i+=100) 
     {
       assert(c_h[i] == c_hfd[i]);
       printf("%7.4f  \n", d_h[i]) ;
     }   */

  printf("Worked! \n") ;
  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);  cudaFree(mean_d); 
  cudaFree(sd_d)  ;

 printf("after cudaFree \n") ; 
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

// kronVect.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>


// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

void kronVectOnHost(float *a, float *b, float *c, int na, int nb)
{
  int i, j, ind;
  for (i=0; i < na; i++) 
     for(j = 0; j < nb; j++)
        {
          ind = i * nb + j ;
          c[ind] = a[i] * b[j] ;
        }
}

__global__ void kronVectOnDevice(float *a, float *b, float *c, int na, int nb)
{
  int idxx = min(blockIdx.x*blockDim.x + threadIdx.x, na-1) ;
  int idxy = min( blockIdx.y*blockDim.y + threadIdx.y, nb-1) ;
  int idxtot = idxx * nb + idxy ;
  int N = na * nb ;
  if (idxtot<N) c[idxtot] = a[idxx] * b[idxy];
}

int main(void)
{
  float *a_h, *b_h, *c_h;           // pointers to host memory
  float *a_d, *b_d, *c_d, *c_hfd ;  // pointer to device memory
  int i, na = 40, nb = 32;

  size_t sizea = na*sizeof(float);
  size_t sizeb = nb*sizeof(float);
  size_t sizec = na * nb*sizeof(float);

  // allocate arrays on host
  a_h = (float *)malloc(sizea);
  b_h = (float *)malloc(sizeb);
  c_h = (float *)malloc(sizec);
  c_hfd = (float *)malloc(sizec);

  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &c_d, sizec);

  // initialization of host data
  for (i=0; i<na; i++) a_h[i] = (float)i;
  for (i=0; i<nb; i++) b_h[i] = (float)i;

  // copy data from host to device
  cudaMemcpy(a_d, a_h, sizeof(float)*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, sizeof(float)*nb, cudaMemcpyHostToDevice);

  // do calculation on host
  kronVectOnHost(a_h, b_h, c_h, na, nb);

  // do calculation on device:

  // Part 1 of 2. Compute execution configuration

  int threadx = min(na, 16), thready = min(nb, 16) ;
  int blockx = na/threadx + (na%threadx ==0?0:1) ;
  int blocky = nb/thready + (nb%thready ==0?0:1) ;

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim

  // Part 2 of 2. Call kronVectOnDevice kernel 

  kronVectOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, b_d, c_d, na, nb );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");

  // Retrieve result from device and store in c_hfd

  cudaMemcpy(c_hfd, c_d, sizeof(float)*na*nb, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");

  // check results

  for (i=0; i<na * nb; i++) 
     {
       assert(c_h[i] == c_hfd[i]);
//       printf("%7.4f %7.4f \n", c_h[i], c_hfd[i]) ;
     }

  printf("Worked! \n") ;

  // clean up

  free(a_h); free(b_h); free(c_h); free(c_hfd); 
  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d); 
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

// kronVect.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>


doKronVect( float *a, float *b, float *c, int *na, int *nb) 
{
   __global__ void kronVectOnDevice( float *, float *, float *, 

void kronVect( float *a, float *b, float *c, int *na, int *nb) ;
 
__global__ void kronVectOnDevice(float *a, float *b, float *c, int na, int nb) ;

void kronVect( float *a, float *b, float *c, int *na, int *nb) 
{
 // float *a_h, *b_h, *c_h;           // pointers to host memory
  float *a_d, *b_d, *c_d ;  // pointer to device memory
  int nna = na[0], nnb = nb[0] ;

  size_t sizea = nna*sizeof(float);
  size_t sizeb = nnb*sizeof(float);
  size_t sizec = nna * nnb*sizeof(float);

  // allocate arrays on host
  // a_h = (float *)malloc(sizea);
  // b_h = (float *)malloc(sizeb);
  // c_h = (float *)malloc(sizec);

  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &c_d, sizec);

  // initialization of host data
//  for (i=0; i<na; i++) a_h[i] = (float)i;
//  for (i=0; i<nb; i++) b_h[i] = (float)i;

  // copy data from host to device
  cudaMemcpy(a_d, a, sizeof(float)*nna, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(float)*nnb, cudaMemcpyHostToDevice);

  // do calculation on device:

  // Part 1 of 2. Compute execution configuration

  int threadx = min(nna, 16), thready = min(nnb, 16) ;
  int blockx = nna/threadx + (nna%threadx ==0?0:1) ;
  int blocky = nnb/thready + (nnb%thready ==0?0:1) ;

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim

  // Part 2 of 2. Call kronVectOnDevice kernel 

  kronVectOnDevice <<< numBlocks, threadsPerBlock >>> (a, b, c, nna, nnb );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
 //   checkCUDAError("kernel invocation");

  // Retrieve result from device and store in c_hfd

  cudaMemcpy(c, c_d, sizeof(float)*nna*nnb, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
  //  checkCUDAError("memcpy");


  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d); 
}

//void checkCUDAError(const char *msg)
//{
 //   cudaError_t err = cudaGetLastError();
 //   if( cudaSuccess != err)
 //   {
   //     fprintf(stderr, "Cuda error: %s: %s.\n", msg,
     //                             cudaGetErrorString( err) );
     //   exit(EXIT_FAILURE);
  //  }
//}

__global__ void kronVectOnDevice(float *a, float *b, float *c, int na, int nb)
{
  int idxx = min(blockIdx.x*blockDim.x + threadIdx.x, na-1) ;
  int idxy = min( blockIdx.y*blockDim.y + threadIdx.y, nb-1) ;
  int idxtot = idxx * nb + idxy ;
  int N = na * nb ;
  if (idxtot<N) c[idxtot] = a[idxx] * b[idxy];
}

// kronVect.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include <kronVectForR.h>


void doKronVect( float *a, float *b, float *retvect, int *na, int *nb) 
{
 // float *a_h, *b_h, *c_h;           // pointers to host memory
  float *a_d, *b_d, *c_d ;  // pointer to device memory
  int nna = na[0], nnb = nb[0] ;

  __global__ void kronVectOnDevice(float *a, float *b, float *c, int na, 
           int nb) ;
  void checkCUDAError(const char *msg) ;

  size_t sizea = nna*sizeof(float);
  size_t sizeb = nnb*sizeof(float);
  size_t sizec = nna * nnb*sizeof(float);

  // allocate arrays on host
  // a_h = (float *)malloc(sizea);
  // b_h = (float *)malloc(sizeb);
  // c_h = (float *)malloc(sizec);

  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &c_d, sizec);

  // initialization of host data
//  for (i=0; i<na; i++) a_h[i] = (float)i;
//  for (i=0; i<nb; i++) b_h[i] = (float)i;

  // copy data from host to device
  cudaMemcpy(a_d, a, sizeof(float)*nna, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(float)*nnb, cudaMemcpyHostToDevice);

  // do calculation on device:

  // Part 1 of 2. Compute execution configuration

  int threadx = min(nna, 16), thready = min(nnb, 16) ;
  int blockx = nna/threadx + (nna%threadx ==0?0:1) ;
  int blocky = nnb/thready + (nnb%thready ==0?0:1) ;

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim

  // Part 2 of 2. Call kronVectOnDevice kernel 

  kronVectOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, b_d, c_d, nna, nnb );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");

  // Retrieve result from device and store in c_hfd

  cudaMemcpy(retvect, c_d, sizeof(float)*nna*nnb, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");


  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d); 
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__global__ void kronVectOnDevice(float *a, float *b, float *c, int na, int nb)
{
  int idxx = min(blockIdx.x*blockDim.x + threadIdx.x, na-1) ;
  int idxy = min( blockIdx.y*blockDim.y + threadIdx.y, nb-1) ;
  int idxtot = idxx * nb + idxy ;
  int N = na * nb ;
  if (idxtot<N) c[idxtot] = a[idxx] * b[idxy];
}

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

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky);

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

  // Call kronVectOnDevice kernel 

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


  // check results

/*  for (i=0; i<na * nb * nc; i+=100) 
     {
       assert(c_h[i] == c_hfd[i]);
       printf("%7.4f  \n", d_h[i]) ;
     }   */

  printf("Worked! \n") ;

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

// kronVectMultForR.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include <kronVectMult1colForR.h>

#define BLOCKSIZE 16

void doKronVectMult1col( float *a, float *b, float *c, float *retvect, int *na1, 
      int *nb1, int *nc1) 
{

   __global__ void kronVectMult1colOnDevice(float *a, float *b, float *c, 
   float *d, int na, int nb) ;

  void checkCUDAError(const char *msg) ;

  float *a_d, *b_d, *c_d, *d_d ;  // pointer to device memory
  int i, na = na1[0],nb = nb1[0], nab = na1[0] * nb1[0], nc = nc1[0];

  size_t sizea = na * na*sizeof(float);
  size_t sizeb = nb * nb*sizeof(float);
  size_t sizec = nab * sizeof(float); // Changed from mat ver


  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &d_d, sizec);


  // copy data from host to device
  cudaMemcpy(a_d, a, sizeof(float)*na*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(float)*nb*nb, cudaMemcpyHostToDevice);

  // Compute execution configuration
  // Changed from matrix version

  int threadx = min(na, BLOCKSIZE), thready = min(nb, BLOCKSIZE) ;
  int blockx = (na)/threadx + ((na)%threadx ==0?0:1) ;
  int blocky = nb/thready + (nb%thready ==0?0:1) ;

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim


for(i = 0; i < nc; i++)
{
  cudaMemcpy(c_d, &c[i * nab], sizeof(float)*nab , cudaMemcpyHostToDevice);
  // Check for any CUDA errors
    checkCUDAError("memcpy");
  cudaMemset( d_d, 0, sizec ) ;
  // Check for any CUDA errors
    checkCUDAError("memset");


  // do calculation on device:

  // Call kronVectOnDevice kernel 

  kronVectMult1colOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, b_d, c_d, d_d,
na, nb );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");

  // Retrieve result from device and store in c_hfd

  cudaMemcpy(&retvect[i * nab], d_d, sizeof(float)*nab, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");
}


  // check results

/*  for (i=0; i<na * nb * nc; i+=100) 
     {
       assert(c_h[i] == c_hfd[i]);
       printf("%7.4f  \n", d_h[i]) ;
     }   */

  printf("Worked! \n") ;

  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);  cudaFree(d_d) ;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__global__ void kronVectMult1colOnDevice(float *a, float *b, float *c, 
float *d, int na, int nb)
{
  /*  a is na x na;  b is nb x nb;  c is (na*nb) x nc */

  float Csub = 0.0 ;  /* element computed by this thread */
  int N = na * nb,  acol,  bcol ;
  int arow = min(blockIdx.x*blockDim.x + threadIdx.x, na-1); /* output row */  
  int brow = min( blockIdx.y*blockDim.y + threadIdx.y, nb-1) ; /* output col */
  int idxtot = arow * nb + brow ;
  if( idxtot < N ) 
    {

      for( int k = 0; k < N; k++)
         {
           acol = k / nb ;
           bcol = k % nb ;
           Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ] 
               * c[k ] ;  
        }
      d[idxtot] = Csub ;
    }   
}

// kronVectMultForR.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include "mstnrUtils.h"
#include "kronVectMult1colForR_d.h"

#define BLOCKSIZE 16

void doKronVectMult1colD( double *a, double *b, double *c, double *retvect, int *na1, 
      int *nb1, int *nc1) 
{

   __global__ void kronVectMult1colOnDevice(double *a, double *b, double *c, 
   double *d, int na, int nb) ;

  void checkCUDAError(const char *msg) ;

  double *a_d, *b_d, *c_d, *d_d ;  // pointer to device memory
  int i, na = na1[0],nb = nb1[0], nab = na1[0] * nb1[0], nc = nc1[0];

  size_t sizea = na * na*sizeof(double);
  size_t sizeb = nb * nb*sizeof(double);
  size_t sizec = nab * sizeof(double); // Changed from mat ver


  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &d_d, sizec);


  // copy data from host to device
  cudaMemcpy(a_d, a, sizeof(double)*na*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(double)*nb*nb, cudaMemcpyHostToDevice);

  // Compute execution configuration
  // Changed from matrix version

  int threadx = min(na, BLOCKSIZE), thready = min(nb, BLOCKSIZE) ;
  int blockx = (na)/threadx + ((na)%threadx ==0?0:1) ;
  int blocky = nb/thready + (nb%thready ==0?0:1) ;

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

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

  // Call kronVectOnDevice kernel 

  kronVectMult1colOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, b_d, c_d, d_d,
na, nb );

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


  // check results

/*  for (i=0; i<na * nb * nc; i+=100) 
     {
       assert(c_h[i] == c_hfd[i]);
       printf("%7.4f  \n", d_h[i]) ;
     }   */

  printf("Worked! \n") ;

  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);  cudaFree(d_d) ;
}

__global__ void kronVectMult1colOnDevice(double *a, double *b, double *c, 
double *d, int na, int nb)
{
  /*  a is na x na;  b is nb x nb;  c is (na*nb) x nc */

  double Csub = 0.0 ;  /* element computed by this thread */
  int N = na * nb,  acol,  bcol ;
  int arow = min(blockIdx.x*blockDim.x + threadIdx.x, na-1); /* output row */  
  int brow = min( blockIdx.y*blockDim.y + threadIdx.y, nb-1) ; /* output col */
  int idxtot = arow * nb + brow ;
  if( idxtot < N ) 
    {

      for( int k = 0; k < N; k++)
         {
           acol = k / nb ;
           bcol = k % nb ;
           Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ] 
               * c[k ] ;  
        }
      d[idxtot] = Csub ;
    }   
}

// kronVectMult.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define BLOCKSIZE 16

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

/* void kronVectOnHost(float *a, float *b, float *c, int na, int nb)
{
  int i, j, ind;
  for (i=0; i < na; i++) 
     for(j = 0; j < nb; j++)
        {
          ind = i * nb + j ;
          c[ind] = a[i] * b[j] ;
        }
} */

__global__ void kronVectMultOnDevice(float *a, float *b, float *c, 
float *d, int na, int nb, int nc)
{
  /*  a is na x na;  b is nb x nb;  c is (na*nb) x nc */

  float Csub = 0.0 ;  /* element computed by this thread */
  int N = na * nb, arow, acol, brow, bcol ;
  int idxx = min(blockIdx.x*blockDim.x + threadIdx.x, N-1); /* output row */  
  int idxy = min( blockIdx.y*blockDim.y + threadIdx.y, nc-1) ; /* output col */
  int idxtot = idxx * nc + idxy ;
  if( idxtot < N * nc) 
    {
      arow = idxx / nb ;
      brow = idxx % nb ;

      for( int k = 0; k < N; k++)
         {
           acol = k / nb ;
           bcol = k % nb ;
           Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ] 
               * c[k * nc + idxy ] ;  
        }
      d[idxtot] = Csub ;
    }   
}

int main(void)
{
  float *a_h, *b_h, *c_h, *d_h;           // pointers to host memory
  float *a_d, *b_d, *c_d, *d_d ;  // pointer to device memory
  int i, na = 10, nb = 5, nc = 12;

  size_t sizea = na * na*sizeof(float);
  size_t sizeb = nb * nb*sizeof(float);
  size_t sizec = na * nb * nc*sizeof(float);

  // allocate arrays on host
  a_h = (float *)malloc(sizea);
  b_h = (float *)malloc(sizeb);
  c_h = (float *)malloc(sizec);
  d_h = (float *)malloc(sizec);

  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &d_d, sizec);

  // initialization of host data
  for (i=0; i<na*na; i++) a_h[i] = (float)i;
  for (i=0; i<nb*nb; i++) b_h[i] = (float)i;
  for (i=0; i<nb*na*nc; i++) c_h[i] = (float)i;
  for (i=0; i<nb*na*nc; i++) d_h[i] = (float)0;

  // copy data from host to device
  cudaMemcpy(a_d, a_h, sizeof(float)*na*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, sizeof(float)*nb*nb, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c_h, sizeof(float)*na * nb * nc, cudaMemcpyHostToDevice);
  // Check for any CUDA errors
    checkCUDAError("memcpy");
  cudaMemset( d_d, 0, sizec ) ;
  // Check for any CUDA errors
    checkCUDAError("memset");

  // do calculation on host
  /* kronVectMultOnHost(a_h, b_h, c_h, na, nb); */

  // do calculation on device:

  // Part 1 of 2. Compute execution configuration

  int threadx = min(na*nb, BLOCKSIZE), thready = min(nc, BLOCKSIZE) ;
  int blockx = (na*nb)/threadx + ((na*nb)%threadx ==0?0:1) ;
  int blocky = nc/thready + (nc%thready ==0?0:1) ;

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim

  // Part 2 of 2. Call kronVectOnDevice kernel 

  kronVectMultOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, b_d, c_d, d_d,
na, nb, nc );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");

  // Retrieve result from device and store in c_hfd

  cudaMemcpy(d_h, d_d, sizeof(float)*na*nb*nc, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");

  // check results

  for (i=0; i<na * nb * nc; i+=100) 
     {
//       assert(c_h[i] == c_hfd[i]);
       printf("%7.4f  \n", d_h[i]) ;
     }

  printf("Worked! \n") ;

  // clean up

  free(a_h); free(b_h); free(c_h); free(d_h); 
  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);  cudaFree(d_d) ;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

// kronVectMult.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define BLOCKSIZE 16

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

/* void kronVectOnHost(double *a, double *b, double *c, int na, int nb)
{
  int i, j, ind;
  for (i=0; i < na; i++) 
     for(j = 0; j < nb; j++)
        {
          ind = i * nb + j ;
          c[ind] = a[i] * b[j] ;
        }
} */

__global__ void kronVectMultOnDevice(double *a, double *b, double *c, 
double *d, int na, int nb, int nc)
{
  /*  a is na x na;  b is nb x nb;  c is (na*nb) x nc */

  double Csub = 0.0 ;  /* element computed by this thread */
  int N = na * nb, arow, acol, brow, bcol ;
  int idxx = min(blockIdx.x*blockDim.x + threadIdx.x, N-1); /* output row */  
  int idxy = min( blockIdx.y*blockDim.y + threadIdx.y, nc-1) ; /* output col */
  int idxtot = idxx * nc + idxy ;
  if( idxtot < N * nc) 
    {
      arow = idxx / nb ;
      brow = idxx % nb ;

      for( int k = 0; k < N; k++)
         {
           acol = k / nb ;
           bcol = k % nb ;
           Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ] 
               * c[k * nc + idxy ] ;  
        }
      d[idxtot] = Csub ;
    }   
}

int main(void)
{
  double *a_h, *b_h, *c_h, *d_h;           // pointers to host memory
  double *a_d, *b_d, *c_d, *d_d ;  // pointer to device memory
  int i, na = 100, nb = 50, nc = 120;

  size_t sizea = na * na*sizeof(double);
  size_t sizeb = nb * nb*sizeof(double);
  size_t sizec = na * nb * nc*sizeof(double);

  // allocate arrays on host
  a_h = (double *)malloc(sizea);
  b_h = (double *)malloc(sizeb);
  c_h = (double *)malloc(sizec);
  d_h = (double *)malloc(sizec);

  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &d_d, sizec);

  // initialization of host data
  for (i=0; i<na*na; i++) a_h[i] = (double)i;
  for (i=0; i<nb*nb; i++) b_h[i] = (double)i;
  for (i=0; i<nb*na*nc; i++) c_h[i] = (double)i;
  for (i=0; i<nb*na*nc; i++) d_h[i] = (double)0;

  // copy data from host to device
  cudaMemcpy(a_d, a_h, sizeof(double)*na*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, sizeof(double)*nb*nb, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c_h, sizeof(double)*na * nb * nc, cudaMemcpyHostToDevice);
  // Check for any CUDA errors
    checkCUDAError("memcpy");
  cudaMemset( d_d, 0, sizec ) ;
  // Check for any CUDA errors
    checkCUDAError("memset");

  // do calculation on host
  /* kronVectMultOnHost(a_h, b_h, c_h, na, nb); */

  // do calculation on device:

  // Part 1 of 2. Compute execution configuration

  int threadx = min(na*nb, BLOCKSIZE), thready = min(nc, BLOCKSIZE) ;
  int blockx = (na*nb)/threadx + ((na*nb)%threadx ==0?0:1) ;
  int blocky = nc/thready + (nc%thready ==0?0:1) ;

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim

  // Part 2 of 2. Call kronVectOnDevice kernel 

  kronVectMultOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, b_d, c_d, d_d,
na, nb, nc );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");

  // Retrieve result from device and store in c_hfd

  cudaMemcpy(d_h, d_d, sizeof(double)*na*nb*nc, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");

  // check results

  for (i=0; i<na * nb * nc; i+=100) 
     {
//       assert(c_h[i] == c_hfd[i]);
       printf("%7.4f  \n", d_h[i]) ;
     }

  printf("Worked! \n") ;

  // clean up

  free(a_h); free(b_h); free(c_h); free(d_h); 
  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);  cudaFree(d_d) ;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

// kronVectMultForR.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include <kronVectMultForR.h>

#define BLOCKSIZE 16

void doKronVectMult( float *a, float *b, float *c, float *retvect, int *na1, 
      int *nb1, int *nc1) 
{

   __global__ void kronVectMultOnDevice(float *a, float *b, float *c, 
   float *d, int na, int nb, int nc) ;

  void checkCUDAError(const char *msg) ;

  // float *a_h, *b_h, *c_h, *d_h;    // pointers to host memory
  float *a_d, *b_d, *c_d, *d_d ;  // pointer to device memory
  int na = na1[0],nb = nb1[0], nc=nc1[0] ;

  size_t sizea = na * na*sizeof(float);
  size_t sizeb = nb * nb*sizeof(float);
  size_t sizec = na * nb * nc*sizeof(float);

  // allocate arrays on host
  /* a_h = (float *)malloc(sizea);
  b_h = (float *)malloc(sizeb);
  c_h = (float *)malloc(sizec);
  d_h = (float *)malloc(sizec); */

  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &d_d, sizec);

  // initialization of host data
/*  for (i=0; i<na*na; i++) a_h[i] = (float)i;
  for (i=0; i<nb*nb; i++) b_h[i] = (float)i;
  for (i=0; i<nb*na*nc; i++) c_h[i] = (float)i;
  for (i=0; i<nb*na*nc; i++) d_h[i] = (float)0; */

  // copy data from host to device
  cudaMemcpy(a_d, a, sizeof(float)*na*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(float)*nb*nb, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c, sizeof(float)*na * nb * nc, cudaMemcpyHostToDevice);
  // Check for any CUDA errors
    checkCUDAError("memcpy");
  cudaMemset( d_d, 0, sizec ) ;
  // Check for any CUDA errors
    checkCUDAError("memset");

  // do calculation on host
  /* kronVectMultOnHost(a_h, b_h, c_h, na, nb); */

  // do calculation on device:

  // Part 1 of 2. Compute execution configuration

  int threadx = min(na*nb, BLOCKSIZE), thready = min(nc, BLOCKSIZE) ;
  int blockx = (na*nb)/threadx + ((na*nb)%threadx ==0?0:1) ;
  int blocky = nc/thready + (nc%thready ==0?0:1) ;

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim

  // Part 2 of 2. Call kronVectOnDevice kernel 

  kronVectMultOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, b_d, c_d, d_d,
na, nb, nc );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");

  // Retrieve result from device and store in c_hfd

  cudaMemcpy(retvect, d_d, sizeof(float)*na*nb*nc, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");

  // check results

/*  for (i=0; i<na * nb * nc; i+=100) 
     {
       assert(c_h[i] == c_hfd[i]);
       printf("%7.4f  \n", d_h[i]) ;
     }   */

  printf("Worked! \n") ;

  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);  cudaFree(d_d) ;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__global__ void kronVectMultOnDevice(float *a, float *b, float *c, 
float *d, int na, int nb, int nc)
{
  /*  a is na x na;  b is nb x nb;  c is (na*nb) x nc */

  float Csub = 0.0 ;  /* element computed by this thread */
  int N = na * nb, arow, acol, brow, bcol ;
  int idxx = min(blockIdx.x*blockDim.x + threadIdx.x, N-1); /* output row */  
  int idxy = min( blockIdx.y*blockDim.y + threadIdx.y, nc-1) ; /* output col */
  int idxtot = idxx * nc + idxy ;
  if( idxtot < N * nc) 
    {
      arow = idxx / nb ;
      brow = idxx % nb ;

      for( int k = 0; k < N; k++)
         {
           acol = k / nb ;
           bcol = k % nb ;
           Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ] 
               * c[k * nc + idxy ] ;  
        }
      d[idxtot] = Csub ;
    }   
}

// kronVectMultForR.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include <kronVectMultForR_d.h>

#define BLOCKSIZE 16

void doKronVectMultD( double *a, double *b, double *c, double *retvect, int *na1, 
      int *nb1, int *nc1) 
{

   __global__ void kronVectMultOnDevice(double *a, double *b, double *c, 
   double *d, int na, int nb, int nc) ;

  void checkCUDAError(const char *msg) ;

  // double *a_h, *b_h, *c_h, *d_h;    // pointers to host memory
  double *a_d, *b_d, *c_d, *d_d ;  // pointer to device memory
  int na = na1[0],nb = nb1[0], nc=nc1[0] ;

  size_t sizea = na * na*sizeof(double);
  size_t sizeb = nb * nb*sizeof(double);
  size_t sizec = na * nb * nc*sizeof(double);

  // allocate arrays on host
  /* a_h = (double *)malloc(sizea);
  b_h = (double *)malloc(sizeb);
  c_h = (double *)malloc(sizec);
  d_h = (double *)malloc(sizec); */

  // allocate array on device 
  cudaMalloc((void **) &a_d, sizea);
  cudaMalloc((void **) &b_d, sizeb);
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &d_d, sizec);

  // copy data from host to device
  cudaMemcpy(a_d, a, sizeof(double)*na*na, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, sizeof(double)*nb*nb, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c, sizeof(double)*na * nb * nc, cudaMemcpyHostToDevice);
  // Check for any CUDA errors
    checkCUDAError("memcpy");

printf("here 1 \n") ;
  cudaMemset( d_d, 0, sizec ) ;
  // Check for any CUDA errors
    checkCUDAError("memset");

printf("here 2 \n") ;
  // do calculation on host
  /* kronVectMultOnHost(a_h, b_h, c_h, na, nb); */

  // do calculation on device:

  // Part 1 of 2. Compute execution configuration

  int threadx = min(na*nb, BLOCKSIZE), thready = min(nc, BLOCKSIZE) ;
  int blockx = (na*nb)/threadx + ((na*nb)%threadx ==0?0:1) ;
  int blocky = nc/thready + (nc%thready ==0?0:1) ;

  printf(" %d %d %d %d \n", threadx, thready, blockx, blocky ) ;

  dim3 threadsPerBlock( threadx, thready );   // block dim
  dim3 numBlocks(blockx, blocky);             // grid dim

  // Part 2 of 2. Call kronVectOnDevice kernel 

  kronVectMultOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, b_d, c_d, d_d,
na, nb, nc );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");

  // Retrieve result from device and store in c_hfd

  cudaMemcpy(retvect, d_d, sizeof(double)*na*nb*nc, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");

  // check results

/*  for (i=0; i<na * nb * nc; i+=100) 
     {
       assert(c_h[i] == c_hfd[i]);
       printf("%7.4f  \n", d_h[i]) ;
     }   */

  printf("Worked! \n") ;

  // clean up

  cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);  cudaFree(d_d) ;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

__global__ void kronVectMultOnDevice(double *a, double *b, double *c, 
double *d, int na, int nb, int nc)
{
  /*  a is na x na;  b is nb x nb;  c is (na*nb) x nc */

  double Csub = 0.0 ;  /* element computed by this thread */
  int N = na * nb, arow, acol, brow, bcol ;
  int idxx = min(blockIdx.x*blockDim.x + threadIdx.x, N-1); /* output row */  
  int idxy = min( blockIdx.y*blockDim.y + threadIdx.y, nc-1) ; /* output col */
  int idxtot = idxx * nc + idxy ;
  if( idxtot < N * nc) 
    {
      arow = idxx / nb ;
      brow = idxx % nb ;

      for( int k = 0; k < N; k++)
         {
           acol = k / nb ;
           bcol = k % nb ;
           Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ] 
               * c[k * nc + idxy ] ;  
        }
      d[idxtot] = Csub ;
    }   
}

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>

#include "mstnrUtils.h"


void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

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

  printf(" %d %d \n", threadx, blockx);

  dim3 threadsPerBlock( threadx );   // block dim
  dim3 numBlocks(blockx);             // grid dim

// get R's RNG seed 
// GetRNGstate();

  // do calculation on device:

  // Call kronVectOnDevice kernel 
// printf("do calc on device \n") ;

   samplingOnDevice <<< numBlocks, threadsPerBlock >>> ( smat_d, 
         D_d, By_d, alpha_d,  beta_d, logpostdens_d, newbetaret_d,
       nsamp,  N, Fm1, newalpha) ;

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");

// done gen rand numbers; send seed state back to R
// PutRNGstate(); 

  // Retrieve result from device 

  cudaMemcpy(logpostdens, logpostdens_d, sizeret, cudaMemcpyDeviceToHost);
  cudaMemcpy(newbetaret, newbetaret_d, sizeret, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");
//}

  printf("Worked! \n") ;
  // clean up

  cudaFree(smat_d); cudaFree(D_d); cudaFree(By_d); cudaFree(alpha_d);  
  cudaFree(beta_d); cudaFree(logpostdens_d)  ; cudaFree(newbetaret_d) ;

 printf("after cudaFree \n") ; 
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

  if( idxtot < N ) 
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

