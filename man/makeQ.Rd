\name{makeQ}
\alias{makeQ}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to construct a structure matrix for 2-dimensional CAR(1) model.
}

\description{
Function to construct a structure matrix for 2-dimensional CAR(1) model.
}
\usage{
makeQ(nr, nc)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{nr}{ number of rows in lattice
%%     ~~Describe \code{nr} here~~
}
  \item{nc}{ number of columns in lattice
%%     ~~Describe \code{nc} here~~
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
structure matrix of dimension (nr * nc) by (nr * nc)
%% ...
}
\references{
%% ~put references to the literature/web site here ~
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
%##---- Should be DIRECTLY executable !! ----
%##-- ==>  Define data, use random,
%##--	or do  help(data=index)  for the standard data sets.

# Construct 300 x 300 structure matrix for intrinsic CAR(1) for
# lattice data on a rectangle with 15 rows and 20 columns
makeQ( 15, 20)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per lin
