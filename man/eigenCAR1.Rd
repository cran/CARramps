\name{eigenCAR1}
\alias{eigenCAR1}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to compute eigenvalues and eigenvectors of CAR1 structure matrix.
The structure matrix is of dimension nr by nc.
}

\description{
Function to compute eigenvalues and eigenvectors of CAR1 structure matrix.
}
\usage{
eigenCAR1( nr, nc )
%- maybe also 'usage' for other objects documented here.
}
\arguments{
  \item{nr }{integer; first dimension of structure matrix }
  \item{nc }{integer; second dimension of structure matrix }
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
\item{values }{eigenvalues}
\item{vectors }{eigenvectors}
}
\references{
Kunsch, H.R. (1994),  "Robust priors for smoothing andimage restoriation,"
Annals of the Institute of Statistical Mathematics, 55, no. 1, 1-19.
}
\author{
Kate Cowles
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
## The function is currently defined as
function(nr,nc)
{

eigenrow <- eigenRW1(nr)
eigencol <- eigenRW1(nc)

list( values = kronecker( rep(1,nc), eigenrow$values) +
        kronecker(eigencol$values, rep(1,nr)) ,
      vectors = kronecker( eigencol$vectors, eigenrow$vectors) )

}
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
