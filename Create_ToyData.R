library(data.table)
library(MASS)
library(terra)
library(abind)

no_obs = 4000             # Number of observations per column
means = 1:5               # Mean values of each column
no_cols = 5               # Number of columns

sds = rep(1.5,no_cols)                 # SD of each column
sd = diag(sds)         # SD in a diagonal matrix for later operations

observations = matrix(rnorm(no_cols * no_obs), nrow = no_cols) # Rd draws N(0,1)

# cor_matrix = matrix(c(1.0, 0.7,0.6,0.5,0.4,
#                       0.7, 1.0, 0.5,
#                       0.01, 0.5, 1.0), byrow = T, nrow = 3)     # cor matrix [3 x 3]

cor_matrix <- as.matrix(fread("DoWnGAN_Kiri/ToyDatCovarMat.csv",header = F))
cov_matrix = sd %*% cor_matrix %*% sd                          # The covariance matrix

Chol = chol(cov_matrix)                                        # C
sam_eq_mean = t(observations) %*% Chol          # Generating random MVN (0, cov_matrix)

samples = t(sam_eq_mean) + means
sv <- as.vector(t(sam_eq_mean))
test_mat <- matrix(sv[1:128^2], nrow = 128, ncol = 128)
image(test_mat)
test_r <- rast(test_mat)
plot(test_r)

##generate images with divide
x <- seq(0,32,by = 0.001)
y <- dgamma(x, shape = 4,scale = 3)
plot(x,y, type = "l")

library(OpenImageR)
dat <- rnorm(128^2)
dmat <- matrix(dat, nrow = 128, ncol = 128)
image(dmat)
sloc <- as.integer(round(rgamma(1, shape = 4, scale = 3)))
dmat[,(32+sloc):ncol(dmat)] <- dmat[,(32+sloc):ncol(dmat)]*10
image(dmat)
matds <- down_sample_image(dmat,factor = 32, gaussian_blur = F)
image(matds)

library(OpenImageR)
sigmoid <- function(x){
  return(5/(1+exp(-8*x)))
}
sample_axes <- c(-1,0,1)
##generate single test set
curr_samp <- c(-1,1)
xaxis <- seq(curr_samp[1],curr_samp[2], length.out = 128)
curr_samp <- c(0,1)
yaxis <- seq(curr_samp[1],curr_samp[2], length.out = 128)

mat <- matrix(data = NA_real_, nrow = 128, ncol = 128)
for(i in 1:length(xaxis)){
  for(j in 1:length(yaxis)){
    meanval <- sigmoid(xaxis[i])*exp(yaxis[j])
    mat[i,j] <- rnorm(1,mean = meanval,sd = 2)
  }
}
mat2 <- mat^2
matds <- down_sample_image(mat2,factor = 8, gaussian_blur = T)
image(matds)
image(mat2)
fine_val <- mat2
coarse_val <- matds
for(i in 1:249){
  fine_val <- abind(fine_val, mat2,along = 3)
  coarse_val <- abind(coarse_val, matds, along = 3)
}

##fine validation data - should be stochastic
for(x in 1:250){
  mat <- matrix(data = NA_real_, nrow = 128, ncol = 128)
  for(i in 1:length(xaxis)){
    for(j in 1:length(yaxis)){
      meanval <- sigmoid(xaxis[i])*exp(yaxis[j])
      mat[i,j] <- rnorm(1,mean = meanval,sd = 2)
    }
  }
  mat2 <- mat^2
  if(x == 1){
    outrast <- mat2
  }else{
    outrast <- abind(outrast,mat2,along = 3)
  }
}

library(reticulate)
np <- import("numpy")
np$save("fine_val_toydat.npy",outrast)
np$save("coarse_val_toydat.npy",coarse_val)

for(numrast in 1:8000){
  if(numrast %% 100 == 0) cat("iteration",numrast,"\n")
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
  matds <- down_sample_image(mat2,factor = 8, gaussian_blur = T)
  if(numrast == 1){
    outrast <- mat2
    outcoarse <- matds
  }else{
    outrast <- abind(outrast,mat2,along = 3)
    outcoarse <- abind(outcoarse,matds,along = 3)
  }
}

image(outrast[,,42])
image(outcoarse[,,42])

library(reticulate)
np <- import("numpy")
np$save("fine_train.npy",outrast[,,1:5000])
np$save("coarse_train.npy",outcoarse[,,1:5000])

np$save("fine_test.npy",outrast[,,5001:7000])
np$save("coarse_test.npy",outcoarse[,,5001:7000])

np$save("fine_val.npy",outrast[,,7001:8000])
np$save("coarse_val.npy",outcoarse[,,7001:8000])


image(outrast[,,8])
library(OpenImageR)
dsimg <- down_sample_image(mat2,factor = 8, gaussian_blur = T)
image(dsimg)
