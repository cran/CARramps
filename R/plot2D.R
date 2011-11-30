plot2D <-
function(objname, numcols)
{
# objname: name of object containing CARramps.fit output from 2-Q analysis
# numcols:  number of terrain.colors shades to use

if(!(length(objname$n==2)))
     print("Not a 2-dimensional problem; cannot plot with plot2D.")
else {

    l1 <- objname$n[1]
    l2 <- objname$n[2]
    par(mfcol = c(1,2) )
    image(1:l1, 1:l2, matrix(objname$y,nrow=l1),col=terrain.colors(numcols),
          xlab='', ylab='', main="Raw data" )            
    image(1:l1, 1:l2, matrix(objname$phi$phimean,nrow=l1),
          col=terrain.colors(numcols),xlab='',ylab='', 
          main="Estimated underlying truth" )   
   }
}

