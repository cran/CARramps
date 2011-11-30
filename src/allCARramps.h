
// combo1colForR3Q_d.h




#ifndef _UTILS_H
#define	_UTILS_H

#ifdef	__cplusplus
extern "C" {
#endif

void doCombo1col3QD( double *a, double *b, double *b2,  double *D,
      double *tausqy,
      double* tausqphi, double *By, double *mean, double *sd,
      int *na1, int *nb1, int *nb21, int *nc1, int *F1) ;


#ifdef	__cplusplus
}
#endif

#endif	/* _UTILS_H */


// combo1colForR_d.h




#ifndef _UTILS_H
#define	_UTILS_H

#ifdef	__cplusplus
extern "C" {
#endif


void doCombo1colD(double *a, double *b, double *D, double *tausqy, 
      double* tausqphi, double *By, double *mean, double *sd,
      int *na1, int *nb1, int *nc1, int *F1) ; 

#ifdef	__cplusplus
}
#endif

#endif	/* _UTILS_H */


// combo1colForR.h




#ifndef _UTILS_H
#define	_UTILS_H

#ifdef	__cplusplus
extern "C" {
#endif


void doCombo1col(float *a, float *b, float *D, float *tausqy, 
      float* tausqphi, float *By, float *mean, float *sd,
      int *na1, int *nb1, int *nc1, int *F1) ; 

#ifdef	__cplusplus
}
#endif

#endif	/* _UTILS_H */


// kronVectForR2.h




#ifndef _UTILS_H
#define	_UTILS_H

#ifdef	__cplusplus
extern "C" {
#endif


   void doKronVect( float *a, float *b, float *retvect, int *na, int *nb) ;



#ifdef	__cplusplus
}
#endif

#endif	/* _UTILS_H */


// kronVectMult1colForR_d.h




#ifndef _UTILS_H
#define	_UTILS_H

#ifdef	__cplusplus
extern "C" {
#endif

void doKronVectMult1col3QD( double *a, double *b, double *b2, double *c, double *retvect, int *na1,
      int *nb1, int *nb21, int *nc1) ;


#ifdef	__cplusplus
}
#endif

#endif	/* _UTILS_H */


// kronVectMult1colForR_d.h




#ifndef _UTILS_H
#define	_UTILS_H

#ifdef	__cplusplus
extern "C" {
#endif


void doKronVectMult1colD( double *a, double *b, double *c, double *retvect, int *na1,
      int *nb1, int *nc1);

#ifdef	__cplusplus
}
#endif

#endif	/* _UTILS_H */


// kronVectMult1colForR.h




#ifndef _UTILS_H
#define	_UTILS_H

#ifdef	__cplusplus
extern "C" {
#endif


void doKronVectMult1col( float *a, float *b, float *c, float *retvect, int *na1,
      int *nb1, int *nc1);

#ifdef	__cplusplus
}
#endif

#endif	/* _UTILS_H */


// kronVectForR2.h




#ifndef _UTILS_H
#define	_UTILS_H

#ifdef	__cplusplus
extern "C" {
#endif


void doKronVectMultD( double *a, double *b, double *c, double *retvect, int *na1,
      int *nb1, int *nc1);

#ifdef	__cplusplus
}
#endif

#endif	/* _UTILS_H */


// kronVectForR2.h




#ifndef _UTILS_H
#define	_UTILS_H

#ifdef	__cplusplus
extern "C" {
#endif


void doKronVectMult( float *a, float *b, float *c, float *retvect, int *na1,
      int *nb1, int *nc1);

#ifdef	__cplusplus
}
#endif

#endif	/* _UTILS_H */


// mstnrUtils.h




#ifndef _UTILS_H
#define	_UTILS_H

#ifdef	__cplusplus
extern "C" {
#endif

void checkCUDAError(const char *msg) ;

#ifdef	__cplusplus
}
#endif

#endif	/* _UTILS_H */


// sampling_d.h

#ifndef _UTILS_H
#define	_UTILS_H

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

#endif	/* _UTILS_H */

