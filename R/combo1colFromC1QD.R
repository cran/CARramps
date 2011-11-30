combo1colFromC1QD <-
function(a,D, tausqy, tausqphi,By)
{
   if(!is.numeric(a) )
          stop("a  must be numeric matrix")
   na1 <- nrow(a)
   nc1 <- length(tausqy)
   tausqphi <- matrix( tausqphi, nrow=nc1)
   F1 <- ncol(tausqphi) + 1
   #print(c("F1",F1))
   mmean <- rep(0, na1)
   ssd <- rep(0,na1)


   #print(c(na1, nc1, F1))

   out <- .C("doCombo1col1QD", a=as.double(as.vector(t(a))), 
              D = as.double(as.vector(t(D))),
              tausqy = as.double(tausqy) ,
              tausqphi = as.double( as.vector( t(tausqphi) )) ,
    By = as.double(By), mean = as.double(mmean), sd = as.double(ssd),
               na1 = as.integer(na1),  nc1 = as.integer(nc1), F1 = 
               as.integer(F1), PACKAGE="CARramps" )
   return(list( phimean = out$mean, phisd = sqrt(out$sd/(nc1-1) )))
}

