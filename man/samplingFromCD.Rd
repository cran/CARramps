\name{samplingFromCD}
\alias{samplingFromCD}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to call CUDA for rejection sampling.
}
\description{
Function to call CUDA for rejection sampling.
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
samplingFromCD(smat, alpha, beta = beta, D, By, k)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{smat}{
%%     ~~Describe \code{smat} here~~
}
  \item{alpha}{
%%     ~~Describe \code{alpha} here~~
}
  \item{beta}{
%%     ~~Describe \code{beta} here~~
}
  \item{D}{
%%     ~~Describe \code{D} here~~
}
  \item{By}{
%%     ~~Describe \code{By} here~~
}
  \item{k}{
%%     ~~Describe \code{k} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
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
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
