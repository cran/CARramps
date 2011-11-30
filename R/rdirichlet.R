rdirichlet <-
function (n, parms) 
{
# generate n random vectors from dirichlet
# rejection envelope

# corrected 09/18/09
    #l <- length(alpha)
    l <- length(parms)
    #x <- matrix(rgamma(l * n, alpha), ncol = l, byrow = TRUE)
    x <- matrix(rgamma(l * n, parms), ncol = l, byrow = TRUE)
   sm <- x %*% rep(1, l)
    return(x/as.vector(sm))
}

