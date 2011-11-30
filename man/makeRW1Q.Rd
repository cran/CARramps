\name{makeRW1Q}
\alias{makeRW1Q}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to construct structure matrix for 1-dimenstional random walk 1.
}
\description{
Function to construct structure matrix for 1-dimenstional random walk 1.
}
\usage{
makeRW1Q(n)
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
structure matrix of dimension n by n
%%  \item{comp2 }{Description of 'comp2'}
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
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
%##---- Should be DIRECTLY executable !! ----
%##-- ==>  Define data, use random,
%##--	or do  help(data=index)  for the standard data sets.

# Construct the structure matrix for a RW(1) on a line at 15 equally-spaced
# points
makeRW1Q(15)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
