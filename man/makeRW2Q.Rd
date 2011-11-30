\name{makeRW2Q}
\alias{makeRW2Q}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to construct structure matrix for 1-dimensional random walk 2.
}
\description{
Function to construct structure matrix for 1-dimensional random walk 2.
}
\usage{
makeRW2Q(n)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{n}{ dimension of structure matrix
%%     ~~Describe \code{n} here~~
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
RW2 structure matrix of dimension n by n
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{Kate Cowles
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
 \code{\link{CARramps.fit}}
}
\examples{
%##---- Should be DIRECTLY executable !! ----
%##-- ==>  Define data, use random,
%##--	or do  help(data=index)  for the standard data sets.

# construct structure matrix for RW(2) on a line at 15 equally-spaced points
makeRW2Q(15)
}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
