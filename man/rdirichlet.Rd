\name{rdirichlet}
\alias{rdirichlet}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Generate random samples from dirichlet distribution.
}
\description{
Generate random samples from dirichlet distribution.
}
\usage{
rdirichlet(n, parms)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{n}{ number of vectors to be drawn
%%     ~~Describe \code{n} here~~
}
  \item{parms}{ vector of parameters of dirichlet distribution
%%     ~~Describe \code{parms} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
n draws from the dirichlet density with parameters in \code{parms}
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

## The function is currently defined as
function (n, parms) 
{
# generate n random vectors from dirichlet
# rejection envelope

# corrected 09/18/09
    #l <- length(alpha)
    l <- length(parms)
    #x <- matrix(rgamma(l * n, alpha), ncol = l, byrow = TRUE)
    x <- matrix(rgamma(l * n, parms), ncol = l, byrow = TRUE)
   sm <- x \%*\% rep(1, l)
    return(x/as.vector(sm))
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
