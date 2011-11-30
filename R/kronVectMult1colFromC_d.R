kronVectMult1colFromC_d <-
function(a,b,c)
{
   if(!is.numeric(a) | !is.numeric(b) | !is.numeric(c) )
          stop("a and b must be numeric matrices")

   na1 <- nrow(a)
   nb1 <- nrow(b)
   nc1 <- ncol(c)
   a<- as.vector(t(a))
   b<- as.vector(t(b))
   c<- as.vector(c)     # change; not transposed before sent
   retvect <- rep(0, na1 * nb1 * nc1)

   out <- .C("doKronVectMult1colD", a=as.double(a), b = as.double(b),
              c = as.double(c),
               retvect=as.double(retvect), na1 = as.integer(na1),
               nb1 = as.integer(nb1), nc1 = as.integer(nc1), PACKAGE="CARramps" )
   return(matrix(out$retvect, nrow=na1*nb1,byrow=FALSE) ) # change byrow
}

