// combo1colForR1Q_d.cu
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <R.h>
#include <Rmath.h>
#include <combo1colForR1Q_d.h>

#define BLOCKSIZE 256 

void doCombo1col1QD( double *a, double *D, double *tausqy, 
      double *tausqphi, double *By, double *mean, double *sd,
      int *na1, int *nc1, int *F1) 
{

   __global__ void kronVectMult1colOnDevice1Q(double *a, double *c, 
   double *mean_d, double *sd_d, int na, int iter) ;

  void checkCUDAError(const char *msg) ;

  double *a_d, *c_d, *mean_d, *sd_d ;  // pointers to device memory
  int i, j, k, iter ;
  int na = na1[0], nc = nc1[0], F=F1[0];
  int Fm1 = F - 1 ;

  double *Bphi ;
  double neweigendenom, normmean, normstd ;

  size_t sizea = na * na*sizeof(double);
  
  size_t sizec = na * sizeof(double); // Changed from mat ver

  // allocate array on host
 
  Bphi = (double *)malloc(sizec);


  // allocate arrays on device 
  cudaMalloc((void **) &a_d, sizea);
  
  cudaMalloc((void **) &c_d, sizec);
  cudaMalloc((void **) &mean_d, sizec);
  cudaMalloc((void **) &sd_d, sizec);


  // copy eigenvector matrices from host to device
  cudaMemcpy(a_d, a, sizeof(double)*na*na, cudaMemcpyHostToDevice);

  // Check for any CUDA errors
    checkCUDAError("memcpy");

  // initialize accumulators on device 
  cudaMemset( mean_d, 0, sizec ) ;  
  cudaMemset( sd_d, 0, sizec ) ;

  // Check for any CUDA errors
    checkCUDAError("memset");

  // Compute execution configuration
  // Changed from matrix version

  int threadx = min(na, BLOCKSIZE) ;
  int blockx = (na)/threadx + ((na)%threadx ==0?0:1) ;
  


//  dim3 threadsPerBlock( threadx );   // block dim
//  dim3 numBlocks(blockx);             // grid dim


// get R's RNG seed 
GetRNGstate();

for(i = 0; i < nc; i++)  // for each row in output
{
// printf("%d \n", i) ;

  for( j=0; j < na; j++ )   // for each data element
  {
     neweigendenom = tausqy[i] ;
     for( k = 0; k < Fm1; k++)
        neweigendenom += D[j * Fm1 +k] * tausqphi[i * Fm1 + k ] ;
     normmean = tausqy[i] * By[j] / neweigendenom ;
     normstd = 1.0 / sqrt(neweigendenom) ;
     Bphi[j] = rnorm( normmean, normstd ) ;
  }
  
  cudaMemcpy(c_d, Bphi, sizeof(double)*na , cudaMemcpyHostToDevice);
  // Check for any CUDA errors
    checkCUDAError("memcpy");

  iter = i + 1 ;

  // do calculation on device:

  // Call kronVectOnDevice kernel 
 // printf("do calc on device \n") ;


  /* kronVectMult1colOnDevice <<< numBlocks, threadsPerBlock >>> (a_d, c_d, 
  mean_d, sd_d, na, iter ); */

  kronVectMult1colOnDevice1Q <<< blockx, threadx >>> (a_d, c_d, 
  mean_d, sd_d, na, iter );

  // block until the device has completed
    cudaThreadSynchronize();

  // check if kernel execution generated an error
  // Check for any CUDA errors
    checkCUDAError("kernel invocation");
}

// done gen rand numbers; send seed state back to R
PutRNGstate(); 

  // Retrieve result from device 

  cudaMemcpy(mean, mean_d, sizeof(double)*na, cudaMemcpyDeviceToHost);
  cudaMemcpy(sd, sd_d, sizeof(double)*na, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
    checkCUDAError("memcpy");



  // clean up

  cudaFree(a_d) ; cudaFree(c_d);  cudaFree(mean_d); 
  cudaFree(sd_d)  ;

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

__global__ void kronVectMult1colOnDevice1Q(double *a, double *c, 
double *mean, double *sd, int na, int iter)
{
  /*  a is na x na;  c is (na) x nc */

  double Csub = 0.0, currdiff, oldmean, oldsd ;  /* element computed by this thread */
  int N = na ,  acol ;
  int arow = min(blockIdx.x*blockDim.x + threadIdx.x, na-1); /* output row */  
  
  int idxtot = arow ;
  oldmean = mean[idxtot] ;
  oldsd = sd[idxtot] ;
  double newmean ;

  if( idxtot < N ) 
    {

      for( int k = 0; k < N; k++)
         {
           acol = k ;
           
           Csub += a[ arow * na + acol ] * c[k ] ;  
        }
      currdiff = Csub - oldmean ;
      newmean = oldmean + currdiff / (double) iter ;
      mean[idxtot] = newmean ;
      sd[idxtot] = oldsd + currdiff * (Csub - newmean) ;
    }   
}

