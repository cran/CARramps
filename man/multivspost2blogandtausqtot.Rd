\name{multivspost2blogandtausqtot}
\alias{multivspost2blogandtausqtot}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to evaluate marginalized and reparameterized posterior.
}
\description{
Function to evaluate marginalized and reparameterized posterior.
}
\usage{
multivspost2blogandtausqtot(smat, alpha, beta, D, y, By, k)
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
  \item{y}{
%%     ~~Describe \code{y} here~~
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
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function(smat, alpha, beta, D, y, By, k )

{
# function to evaluate log marginal posterior of s and generate tausqtot

smat <- as.matrix(smat)

n <- length(y)

F <- ncol(D) + 1

logpostvect <- numeric()
tausqtot <- numeric()

sumalpha <- sum(alpha)


# change to use apply 09/29/09 MKC

genst <- function( s )
{
        s0 <- 1-sum(s)

        neweigennumer <- s[1] * D[,1]
        if( F > 2)
        for(j in 2:(F-1))
                neweigennumer <- neweigennumer + s[j] * D[,j]

        neweigendenom <-  neweigennumer + s0

        neweigen <- s0 * neweigennumer / neweigendenom
# corrected to (alpha - 1)  09/18/09
        #logpostdensnumer <- sum( log(c(s0,s)) * alpha  ) + 
        logpostdensnumer <- sum( log(c(s0,s)) * (alpha-1)  ) + 
                        sum( log(neweigen[ neweigen > 0 ]) ) / 2
        whole <- sum( neweigen * By^2)

        newbeta <- whole/2 +  sum( c(s0,s) * beta )
        newalpha <- (sumalpha + (n-k)/2 )
        logpostdensdenom <-  log( newbeta)* newalpha
        #logpostvect[i] <- logpostdensnumer - logpostdensdenom
        #tausqtot[i] <- rgamma(1, newalpha, newbeta)
        c( logpostdensnumer - logpostdensdenom, 
        rgamma(1, newalpha, newbeta))
  }


if( multicoreflag )
    output <- t( parApply(cl, smat, 1, genst ) )
else
    output <- t( apply( smat, 1, genst ) )
output
 
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
