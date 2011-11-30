samplingFromCD <-
function( smat, alpha, beta=beta, D, By, k)
{
   N1 <- length(By)
   F1 <- ncol(smat) + 1
   nsamp1 <- nrow(smat)
   logpostdens <- newbetaret <- rep(0,nsamp1 )
   out <- .C("doSamplingD",  smat=as.double(as.vector(t(smat))), 
      D = as.double(as.vector(t(D))), 
      By = as.double(By),
      alpha = as.double(alpha),
      beta = as.double(beta), logpostdens = as.double(logpostdens), 
      newbetaret = as.double(newbetaret),
      nsamp1 = as.integer(nsamp1), N1 = as.integer(N1), 
      F1 = as.integer(F1), nk1 = as.integer(k), PACKAGE="CARramps")
   return( cbind( out$logpostdens, out$newbetaret) )
}

