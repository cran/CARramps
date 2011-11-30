\name{eigenRW1}
\alias{eigenRW1}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to compute eigenvalues and eigenvectors of RW1 structure matrix.
}
\description{
Function to compute eigenvalues and eigenvectors of RW1 structure matrix.
}
\usage{
eigenRW1(n)
%- maybe also 'usage' for other objects documented here.
}
\arguments{
  \item{n }{dimension of structure matrix}
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
Kate Cowles}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{

## The function is currently defined as
function(n)
{
# compute eigenvals and eigenvects of RW1

w <-( pi/n) * ((n-1):0)
evals <-  2 * (1-cos(w) )

evects <- sin( outer( 1:n, w ) ) - sin( outer( (0:(n-1)), w))
evects[,n] <- rep(1,n)

evects <- apply( evects, 2, function(v) v / sqrt( sum(v^2) ) )

list( values = evals, vectors = evects)

}
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{internal}
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
