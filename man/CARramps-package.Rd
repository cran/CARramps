\name{CARramps-package}
\alias{CARramps-package}
\alias{CARramps}
\docType{package}
\title{
Draws independent samples from joint posterior in Bayesian CAR models
}
\description{
This package fits Bayesian conditional autoregressive models
(also called intrinsic Gaussian Markov random fields or IGMRFs) for
spatial and spatiotemporal data on a lattice.  It uses Nvidia GPUs to perform
rejection sampling to obtain independent samples from the joint posterior
distribution of model parameters, including spatial(temporal) random effects,
precision parameters,
and regression coefficients for some polynomial trend surfaces.

The CARramps package can handle models with up to three structure matrices
to accommodate different patterns and strengths of spatial association in 
different dimensions.  Such models were used by Besag and Higdon (1999) and 
by Reich, Hodges, and Carlin (2007) are discussed in detail in Kunsch (1994)
and He, Hodges, and Carlin (2007).  

Rue and Held (2005), Chapter 3, describe (polynomial) intrinsic GMRFs
of first, second, and higher orders.  If all of the structure matrices
representing association in a model are random walk 1 or 2, then
the CARramps package can estimate 
the coefficients of the polynomial trend surface, the design matrix of
which spans the null space of the Kronecker sum of the structure matrices.
}
\details{
\tabular{ll}{
Package: \tab CARramps\cr
Type: \tab Package\cr
Version: \tab 0.1.2\cr
Date: \tab 2011-11-30\cr
License: \tab GPL>=3\cr
LazyLoad: \tab yes\cr
}
}
\author{
Kate Cowles and Stephen Bonett, with
thanks to Juan Cervantes, Dong Liang, Alex Sawyer, and Michael Seedorff \cr
Maintainer: Kate Cowles <kate-cowles@uiowa.edu>
}
\references{
Besag, J. E. and Higdon, D. M. (1993), "Bayesian inference for 
agricultural Field experiments." Bull. Int. Statist. Inst., 55, no.1, 121-136.

Cowles, M.K. (2011) "Back to the future: New hardware + an old algorithm
equals fast sampling for Bayesian spatial models."  Department of 
Statistics and Actuarial Science, The University of Iowa.

Cowles M.K., Yan, J., Smith, B. (2009),
"Reparameterized and Marginalized Posterior and Predictive Sampling for
Complex Bayesian Geostatistical Models,"
Journal of Computational and Graphical Statistics,  18(2), 262-282.

He, Y., Hodges, J.S., and Carlin, B.P. (2007), "Reconsidering the variance 
parameterization in multiple precision models," Bayesian Analysis, 
2, 529-556.

Kunsch, H.R. (1994),  "Robust priors for smoothing and image restoration," 
Annals of the Institute of Statistical Mathematics, 55, no. 1, 1-19. 

Reich, B.J., Hodges, J.S., and Carlin, B.P. (2007), "Spatial analyses of 
periodontal data using conditionally autoregressive priors having two 
classes of neighbor relations," J. Amer. Statist. Assoc., 102, 
44-55.

Rue, H. and Held, L. (2005). Gaussian Markov Random Fields: Theory and 
Applications, volume 104 of 
Monographs on Statistics and Applied Probability. Chapman & Hall, London.
}
\keyword{spatial} 
\seealso{
%%\code{\link[ramps:ramps-package]{ramps}} ~~
}
\examples{

# load data
  data(iowaNW00)

# construct structure matrices
  Q1<- makeRW2Q(33)       # RW2 within rows (east-west)
  Q2<- makeRW2Q(24)       # RW2 within columns (north-south)

iowaQ <- list( list( type="Gen", content=Q1 ), list( type="Gen", content=Q2))
# dimenstions of Q1, Q2,  in that order
    na<- nrow(Q1)
    nb<- nrow(Q2)


# construct the design matrix with with as many columns as there are
# in null space of kronecker prod of Q's

X2 <- cbind( rep(1,nb), 1:nb)
X1 <- cbind( rep(1,na), 1:na)
X <-  kronecker( X2, X1)

# parameters of gamma prior densities on tausqy, tausqphi[1], tausqphi[2]
alpha2 = beta2 <- c(.1, .1, .1)

# number of samples
nsamp = 1000

#random seed
myseed = 314

output <- CARramps.fit(alpha=alpha2, beta=beta2, 
            Q=iowaQ, y=iowaNW00,  nsamp=nsamp,
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

