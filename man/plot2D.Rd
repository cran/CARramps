\name{plot2D}
\alias{plot2D}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to produce image plot based on CARramps.fit output for 2-Q model.
}
\description{
Function to produce image plot based on CARramps.fit output for 2-Q model.
}
\usage{
plot2D(objname, numcols)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{objname}{ object containing output from \code{
CARramps.fit}
%%     ~~Describe \code{objname} here~~
}
  \item{numcols}{number of shades of colors to be used in plots
%%     ~~Describe \code{numcols} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp2 }{Description of 'comp2'}
two-panel plot; raw data in left panel; posterior means of
spatial random effects in right panel

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
\code{\link{CARramps.fit}}, ~~~
}
\examples{
%##---- Should be DIRECTLY executable !! ----
%##-- ==>  Define data, use random,
%##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function(objname, numcols)
{
# objname: name of object containing CARramps.fit output from 2-Q analysis
# numcols:  number of terrain.colors shades to use
l1 <- length(unique(objname$D[,1]))
 l2 <- length(unique(objname$D[,2]))
 par(mfcol = c(1,2) )
 image(1:l1, 1:l2, matrix(objname$y,nrow=l1),col=terrain.colors(numcols),
xlab='', ylab='', main="Raw data" )            
 image(1:l1, 1:l2, matrix(objname$phi$phimean,nrow=l1),col=terrain.colors(numcols),xlab='',ylab='', main="Estimated underlying truth" )   
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%\keyword{ ~kwd1 }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
