library(data.table)
library(MASS)

n <- 10000
corM <- rbind(c(1.0, 0.6, 0.9), c(0.6, 1.0, 0.5), c(0.9, 0.5, 1.0))
set.seed(123)
SigmaEV <- eigen(corM)
eps <- rnorm(n * ncol(SigmaEV$vectors))
Meps <- matrix(eps, ncol = n, byrow = TRUE)    
Meps <- SigmaEV$vectors %*% diag(sqrt(SigmaEV$values)) %*% Meps
Meps <- t(Meps)
# target correlation matrix
corM

sigmoid <- function(x){
  return(5/(1+exp(-8*x)))
}
sample_axes <- c(-1,0,1)
for(numrast in 1:1000){
  cat(".")
  curr_samp <- sample(sample_axes,size = 2)
  xaxis <- seq(curr_samp[1],curr_samp[2], length.out = 128)
  curr_samp <- sample(sample_axes,size = 2)
  yaxis <- seq(curr_samp[1],curr_samp[2], length.out = 128)
  
  mat <- matrix(data = NA_real_, nrow = 128, ncol = 128)
  for(i in 1:length(xaxis)){
    for(j in 1:length(yaxis)){
      meanval <- sigmoid(xaxis[i])*exp(yaxis[j])
      mat[i,j] <- rnorm(1,mean = meanval,sd = 2)
    }
  }
  mat2 <- mat^2
  
  if(numrast == 1){
    outrast <- rast(mat2)
  }else{
    temp <- rast(mat2)
    add(outrast) <- temp
  }
}


plot(outrast[[4]])
plot(r$lyr.1)
