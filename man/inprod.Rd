\name{inprod}
\alias{inprod}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to compute inner product of two vectors.
}
\description{
Function to compute inner product of two vectors.
}
\usage{
inprod(v1, v2)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{v1}{numeric vector
%%     ~~Describe \code{v1} here~~
}
  \item{v2}{numeric vector of same length asv1
%%     ~~Describe \code{v2} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
scalar; inner product of v1 and v2
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
function( v1, v2)
{
    sum( v1 * v2)
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
