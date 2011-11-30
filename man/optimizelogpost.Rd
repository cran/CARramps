\name{optimizelogpost}
\alias{optimizelogpost}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Optimize log of marginalized and reparameterized joint posterior.
}
\description{
Optimize log of marginalized and reparameterized joint posterior.
}
\usage{
optimizelogpost(alpha, beta, D, y, By, k, func)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
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
  \item{func}{
%%     ~~Describe \code{func} here~~
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
function( alpha,beta,D, y, By,  k, func )
{

# Note: this function needs to be made generic to accept different values
#   of F; right now it works only if F = 3.

F <- ncol( D )  + 1

# F = number of precision parms (meas err precision and for all Q's)

optimouts <- numeric()

# We need to run optimization from different starting values to increase
# chance of finding global max.  

if( F ==5 ) {
        optimout1 <- optim( c( 6/10, 1/10, 1/10, 1/10 ), func, alpha=alpha, beta = beta,
        D = D,   y = y,By = By,  k = k,              
        control = list(fnscale = -1) )$value

        optimout2 <- optim( c( 1/10, 1/10, 1/10, 1/10 ), func, alpha=alpha, beta = beta,
        D = D, y = y, By = By, k = k,
        control = list(fnscale = -1) )$value

        optimout3 <- optim( c( 1/10, 1/10, 1/10, 6/10 ), func, alpha=alpha, beta = beta,
        D = D,  y = y, By = By, k = k,              
        control = list(fnscale = -1) )$value
        optimout4 <- optim( c( 1/10, 1/10, 6/10, 1/10 ), func, alpha=alpha, beta = beta,
        D = D,  y = y, By = By, k = k,              
        control = list(fnscale = -1) )$value
        optimout5 <- optim( c( 1/10, 6/10, 1/10, 1/10 ), func, alpha=alpha, beta = beta,
        D = D,  y = y, By = By, k = k,              
        control = list(fnscale = -1) )$value

        gmax <- max( optimout1, optimout2, optimout3, optimout4, optimout5 )
  }
else if( F ==4 ) {
        optimout1 <- optim( c( 1/16, 3/16, 5/16 ), func, alpha=alpha, beta = beta,
        D = D,   y = y,By = By,  k = k,              
        control = list(fnscale = -1) )$value

        optimout2 <- optim( c( 3/16, 5/16, 7/16 ), func, alpha=alpha, beta = beta,
        D = D, y = y, By = By, k = k,
        control = list(fnscale = -1) )$value

        optimout3 <- optim( c( 5/16, 7/16, 1/16 ), func, alpha=alpha, beta = beta,
        D = D,  y = y, By = By, k = k,              
        control = list(fnscale = -1) )$value
        optimout4 <- optim( c( 7/16, 1/16, 3/16 ), func, alpha=alpha, beta = beta,
        D = D,  y = y, By = By, k = k,              
        control = list(fnscale = -1) )$value

        gmax <- max( optimout1, optimout2, optimout3, optimout4 )
  }
else if( F ==3 ) {
        #optimout1 <- optim( c( 1/9 , 3/9 ), func, alpha=alpha, beta = beta,
        optimout1 <- optim( c( 1/6 , 1/2 ), func, alpha=alpha, beta = beta,
        D = D,   y = y,By = By,  k = k,              
        control = list(fnscale = -1) )$value

        optimout2 <- optim( c( 1/2 , 1/3 ), func, alpha=alpha, beta = beta,
        D = D, y = y, By = By, k = k,
        control = list(fnscale = -1) )$value

        optimout3 <- optim( c( 1/3 , 1/6 ), func, alpha=alpha, beta = beta,
        D = D,  y = y, By = By, k = k,              
        control = list(fnscale = -1) )$value

        gmax <- max( optimout1, optimout2, optimout3 )
  }

else
    if (F==2) {
        max1 <- optimize( func,c(0,0.5),alpha=alpha, beta = beta,
        D = D, y = y, By = By, k = k,       maximum=T)$objective
        max2 <- optimize( func,c(0.5,1.0), alpha=alpha, beta = beta,
        D = D, y = y, By = By, k = k,       maximum=T)$objective

        gmax<- max(max1,max2)

  }

gmax


  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{internal }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
