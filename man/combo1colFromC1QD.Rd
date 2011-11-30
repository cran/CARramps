\name{combo1colFromC1QD}
\alias{combo1colFromC1QD}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Call to C/CUDA function.
}
\description{
This function calls the C/CUDA program that calculates the posterior
means and standard deviations of the random effects phi.
}
\usage{
combo1colFromC1QD(a, D, tausqy, tausqphi, By)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{a}{ Eigenvector matrix of Q matrix.
%%     ~~Describe \code{a} here~~
}
  \item{D}{ Eigenvalues of Q matrix.
%%     ~~Describe \code{D} here~~
}
  \item{tausqy}{ Accepted samples of measurement error precision parameter.
%%     ~~Describe \code{tausqy} here~~
}
  \item{tausqphi}{ Accepted samples spatial precision parameter.
%%     ~~Describe \code{tausqphi} here~~
}
  \item{By}{  Vector obtained by premultiplying data vector by transpose
of eigenvector matrix of Q matrix.
%%     ~~Describe \code{By} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
 \item{phimean }{Vector of means of posterior densities of random effects}
 \item{phisd }{Vector of standard deviations of marginal posterior
densities of random effects}
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
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
