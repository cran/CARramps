\name{combo1colFromCD}
\alias{combo1colFromCD}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to call CUDA program to calculate posterior means and
standard deviations of random effects in models with 2 structure
matrices.
}
\description{
Function to call CUDA program to calculate posterior means and
standard deviations of random effects in models with 2 structure
matrices.

}
\usage{
combo1colFromCD(a, b, D, tausqy, tausqphi, By)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{a}{ Eigenvector matrix of 1st Q matrix.}
  \item{b}{ Eigenvector matrix of 2nd Q matrix.}
  \item{D}{ Matrix of eigenvalues from the 3 structure matrices.}
 \item{tausqy}{
Vector of samples of measurement error precision.}
  \item{tausqphi}{
Matrix sampled values of spatial precisions.}
  \item{By}{
Vector resulting from premultiplication of data vector y b
transpose of kronecker sum of eigenvector matrices.
}

}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
 \item{phimean }{Vector of means of posterior densities of random effects}
 \item{phisd }{Vector of standard deviations of marginal posterior}
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
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
