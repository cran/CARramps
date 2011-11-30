combo1colFromCD <-
function(a,b,D, tausqy, tausqphi, By)
{
   if(!is.numeric(a) | !is.numeric(b)  )
          stop("a and b must be numeric matrices")

   na1 <- nrow(a)
   nb1 <- nrow(b)
   nc1 <- length(tausqy)
   F1 <- ncol(tausqphi) + 1
   nab <- na1 * nb1
   mmean <- rep(0, nab)
   ssd <- rep(0,nab)

   out <- .C("doCombo1colD", a=as.double(as.vector(t(a))), 
              b = as.double(as.vector(t(b))),
              D = as.double(as.vector(t(D))),
              tausqy = as.double(tausqy) ,
              tausqphi = as.double( as.vector( t(tausqphi) )) ,
    By = as.double(By), mean = as.double(mmean), sd = as.double(ssd),
               na1 = as.integer(na1),
               nb1 = as.integer(nb1), nc1 = as.integer(nc1), F1 = 
               as.integer(F1), PACKAGE="CARramps" )
   return(list( phimean = out$mean, phisd = sqrt(out$sd/(nc1-1) )))
}

