kronVectMult1colFromC3Q_d <-
function(a,b,b2,c)
{
   if(!is.numeric(a) | !is.numeric(b) | !is.numeric(c) )
          stop("a and b must be numeric matrices")

   na1 <- nrow(a)
   nb1 <- nrow(b)
   nb21 <- nrow(b2)
   nc1 <- ncol(c)
   a<- as.vector(t(a))
   b<- as.vector(t(b))
   b2<- as.vector(t(b2))
   c<- as.vector(c)     # change; not transposed before sent
   retvect <- rep(0, na1 * nb1 * nb21* nc1)

   out <- .C("doKronVectMult1col3QD", a=as.double(a), b = as.double(b),
              b2 = as.double(b2),
              c = as.double(c),
               retvect=as.double(retvect), na1 = as.integer(na1),
               nb1 = as.integer(nb1), nb21=as.integer(nb21), 
               nc1 = as.integer(nc1), PACKAGE="CARramps" )
   return(matrix(out$retvect, nrow=na1*nb1*nb21,byrow=FALSE) ) # change byrow
}

