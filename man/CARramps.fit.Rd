\name{CARramps.fit}
\alias{CARramps.fit}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fit Bayesian normal conditional autoregressive model
}
\description{
This function fits CAR models to data.  It draws independent samples from 
the posterior distribution of precision parameters and regression coefficients 
for certain polynomial trend surface.  If the \code{randeffs} argument is
set to true, it will also produce the estimated mean and standard
deviation of the marginal posterior density of each random effect.
}
\usage{
CARramps.fit(alpha, beta, Q, y, nsamp, seed, fixed = FALSE, coefs = FALSE, 
randeffs = FALSE, designMat = NULL, mult = 20, filename = "params.txt")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{alpha}{
   Vector of \code{alpha} parameters for gamma prior densities on
precisions.  The first element is for the prior density on the measurement 
error precision, and
the remaining entries are for the the prior density(ies) on the random effects
precision(s), in the same order as the Q matrices in the list argument Q.
}
  \item{beta}{
   Vector of \code{beta} parameters for gamma prior densities on
precisions.  The first element is for the prior density on the measurement 
error precision, and
the remaining entries are for the the prior density(ies) on the random effects
precisions, in the same order as the Q matrices in the list argument Q.
}
  \item{Q}{
List of specifications of structure matrices for the CAR model.  
Each element of the list Q is itself a list with 2 elements: \\
\code{type}:  one of c("CAR1","RW1","Gen") indicating a conditional 
autoregressive 
structure of order 1 on a rectangular grid, a random walk 1 structure on
a line, or a general (any other) structure, and \\
\code{content}: for  "CAR1," a two-vector giving the dimensions of the 
rectangular grid;
for "RW1," a scalar giving the dimension; and for "Gen," a matrix --
the symmetric structure matrix.
}

  \item{y}{
Vector of observed data values.  If there is more than one structure matrix
in the list called Q, then the data must be ordered accordingly.  For example,
suppose the data are measured on a rectangular lattice with r rows and
c columns, and that Q[[1]] represents the within-row neighborhood structure
while Q[[2]] represents the within-column neighborhood structure.  Then
the data must be in row-major order.
}
  \item{nsamp}{
Integer representing the number of desired draws from the joint posterior
density of the model parameters.
}


 \item{seed}{
Seed for the random number generator.  Must be a positive integer. 
}

  \item{fixed}{
Logical value (TRUE or FALSE).  CARramps.fit utilizes rejection sampling to draw from the marginalized joint
posterior density.  This involves generating a batch of candidate values,
of which some are rejected and some are accepted.  If \code{fixed} is TRUE,
then a single batch of candidates of size \code{nsamp} is generated and the
number of accepted candidates returned is likely to be much smaller than
\code{nsamp}.  If \code{fixed} is FALSE (the default), then additional batches
of candidates will be generated iteratively until at least \code{nsamp}
samples have been accepted.  
}
  \item{coefs}{
Logical value.  If \code{coefs} is TRUE, then regression
coefficients for a polynomial trend surface will be estimated.  The 
design matrix must be provided in the argument \code{designmat}. 
}
  \item{randeffs}{
Logical. If true, random effects corresponding to each observation in the
dataset will be calculated and returned.}
  \item{designMat}{
The design matrix for regression.  It must be set to NULL if 
\code{coefs} is FALSE.  The only regression coefficients that CARramps
can estimate are for polynomial trend surfaces.  The
degree of the polynomial trend surface must agree with the structure
matrix or matrices in the list Q.  For a CAR(1) or RandomWalk(1)
structure matrix, the appropriate design matrix is an intercept (column of
ones) only.   RandomWalk(2) structure matrices correspond to linear trend
surfaces, and the appropriate design matrix consists of an intercept and
a column of consecutive integers.  The example code for this function
demonstrates how to construct the design matrix for a two-dimensional
linear trend surface, which corresponds to a Q list consisting of two
RandomWalk(2) structure matrices.
}
  \item{mult}{
The \code{mult} argument and the \code{nsamp} argument together determine
the size of the batches of candidates generated for rejection sampling,
which is \code{mult} times \code{nsamp}.  Leaving \code{mult} at its
default value of 20 is generally safe.  Decreasing \code{mult} for very
small datasets with high acceptance rates, or increasing it for large
datasets with log acceptance rates, may speed computing.
}
  \item{filename}{
The name for the file in which batches of accepted samples of precision
parameters will be saved while the function is running.  Future releases
of the CARramps package will be able to use this output file to resume
sampling after interruption.
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
 \item{params }{Matrix of samples drawn from joint posterior density
of the precision parameters.  The first column is the measurement
error precision, and subsequent columns are the spatial precisions
in the same order as the Q matrices.}
 \item{phi }{List with two components:  \code{phimean} is vector
of means of posterior densities random effects; \code{phisd} is vector of 
standard deviations of marginal posterior densities of random effects.
Both are in the same order as the observations the \code{y} vector.}
 \item{preds }{Always NULL in current release of CARramps.  Future
releases will enable estimation of means and standard deviations from
posterior predictive distributions, which will be output here.}
 \item{regcoefs }{Matrix of samples drawn from joint posterior
density of regression coefficients.  There is one column for each
coefficient.}
 \item{y }{Data vector}
 \item{acptrate }{Rejection sampling acceptance rate}
 \item{n }{Number of observations in \code{y}}
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
%%##---- Should be DIRECTLY executable !! ----

# load data
  data(iowaNW00)

# construct structure matrix
  Q1<- makeRW2Q(33)       # for rows
  Q2<- makeRW2Q(24)       # for columns


# dimensions of Q1, Q2,  in that order
    na<- nrow(Q1)
    nb<- nrow(Q2)

Q <- list( list(type="Gen",content=Q1), list(type="Gen",content=Q2) )

# construct the design matrix with with as many columns as there are
# in null space of kronecker prod of Q's

X2 <- cbind( rep(1,nb), 1:nb)
X1 <- cbind( rep(1,na), 1:na)
X <-  kronecker( X2, X1)

# parameters of gamma prior densities on tausqy, tausqphi[1], tausqphi[2]
alpha2 = beta2 <- c(.1, .1, .1)

# number of samples
nsamp = 500

#random seed
myseed = 314

output <- CARramps.fit(alpha=alpha2,
            beta=beta2, Q=Q, y=iowaNW00,  nsamp=nsamp,
            seed=myseed,
            fixed = FALSE, randeffs=TRUE, coefs=TRUE,designMat=X,
            mult= 50) 

# summarize marginal posterior densities of precision parameters
library(coda)
summary(as.mcmc( output$params ))

# summarize marginal posterior densities regression coefficients
#    intercept, slope within rows (west-to-east linear trend), 
#    slope for columns (north to south linear trend), 
#    interaction between rows  and columns
summary(as.mcmc( output$regcoefs))

# summary statistics for site-specific random effects at first 10 sites
print( cbind( output$phi$phimean, output$phi$phisd)[1:10,] )

# plot the raw data and the posterior means of the site-specific random effects
plot2D( output, numcols=16)

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ spatial }
%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
