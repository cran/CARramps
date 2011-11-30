
// sampling_d.h

#ifndef _SAMP_H
#define	_SAMP_H

#ifdef	__cplusplus
extern "C" {
#endif

void doSamplingD( double *smat, double *D, double *By,
      double *alpha,
      double* beta, double *logpostdens, double *newbetaret,
      int *nsamp1, int *N1, int *F1, int *nk1) ;

#ifdef	__cplusplus
}
#endif

#endif	/* _SAMP_H */

