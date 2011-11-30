\name{kronVectMult1colFromC_d}
\alias{kronVectMult1colFromC_d}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to call CUDA program.
}
\description{
Function to call CUDA program to compute product of a
Kronecker product of two matrices times another matrix.
}
\usage{
kronVectMult1colFromC_d(a, b, c)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{a}{ left matrix in kronecker product
%%     ~~Describe \code{a} here~~
}
  \item{b}{ right matrix in kronecker product
%%     ~~Describe \code{b} here~~
}
  \item{c}{matrix to be premultiplied by kronecker product of 
\code{a} and
\code{b}
%%     ~~Describe \code{c} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
matrix; the product kronecker(a,b) %*% c computed on GPU
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
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
   return(matrix(out$retvect, nrow=na1*nb1,byrow=F) ) # change byrow
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
