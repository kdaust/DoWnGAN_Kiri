library(data.table)
library(MASS)
library(terra)
library(abind)

nsamp = 128

sigma <- matrix(rep(0,128^2),nsamp)
diag(sigma) <- 1
#sigma[1:127,2:128] <- 0.8
for(i in 1:(nsamp-1)){
  sigma[i,i+1] <- 0.8
  sigma[i+1,i] <- 0.8
}
for(i in 1:(nsamp-2)){
  sigma[i,i+2] <- 0.7
  sigma[i+2,i] <- 0.7
}
for(i in 1:(nsamp-3)){
  sigma[i,i+3] <- 0.5
  sigma[i+3,i] <- 0.5
}
for(i in 1:(nsamp-4)){
  sigma[i,i+4] <- 0.3
  sigma[i+4,i] <- 0.3
}
sds = rep(1,nsamp)                 # SD of each column
sd = diag(sds)
cov_matrix = sd %*% sigma %*% sd
library(lqmm)
sig2 <- make.positive.definite(sigma)
# distr1 <- mvrnorm(n = nsamp,mu = rep(1,128),Sigma = sig2)
# distr2 <- mvrnorm(n = nsamp,mu = rep(5,128),Sigma = sig2)
# binmask <- matrix(rbinom(128^2, size = 1, prob = 0.35),128)
# distr1[binmask == 1] <- distr2[binmask == 1]
# image(distr1)
# plot(density(distr1))
#################################
##create test data

library(OpenImageR)
sigmoid <- function(x){
  return(5/(1+exp(-8*x)))
}

sample_axes <- c(-1,0,1)
nsamp = 128
##Bimodal
##generate stochastic test set
curr_samp <- sample(sample_axes,size = 2)
xaxis <- seq(curr_samp[1],curr_samp[2], length.out = 128)
curr_samp <- sample(sample_axes,size = 2)
yaxis <- seq(curr_samp[1],curr_samp[2], length.out = 128)
dat <- as.data.table(cbind(expand.grid(xaxis,yaxis),expand.grid(xax = 1:128,yax =1:128)))
dat[,Val := sigmoid(Var1)*exp(Var2)]
d2 <- dcast(dat, xax ~ yax, value.var = "Val")
d2[,xax := NULL]
meanmat <- as.matrix(d2)
image(meanmat)
meanmat <- meanmat + 5

for(numrast in 1:500){
  if(numrast %% 100 == 0) cat("iteration",numrast,"\n")
  distr1 <- mvrnorm(n = nsamp,mu = rep(1,128),Sigma = sig2)
  #distr2 <- mvrnorm(n = nsamp,mu = rep(5,128),Sigma = sig2)
  #binmask <- matrix(rbinom(128^2, size = 1, prob = 0.35),128)
  #distr1[binmask == 1] <- distr2[binmask == 1]
  #mat2 <- distr1*meanmat
  mat2 <- meanmat + distr1
  mat2 <- mat2^2
  rfine <- rast(mat2)
  matds <- down_sample_image(mat2,factor = 8, gaussian_blur = T)
  rcoarse <- rast(matds)
  if(numrast == 1){
    outrast <- rfine
    outcoarse <- rcoarse
  }else{
    add(outrast) <- rfine
    add(outcoarse) <- rcoarse
  }
}
plot(outrast[[5]])
farr <- as.array(outrast)
carr <- as.array(outcoarse)

library(reticulate)
np <- import("numpy")
np$save("../Data/synthetic/no_small/fine_stochastic.npy",farr)
np$save("../Data/synthetic/no_small/coarse_stochastic.npy",carr)

#dem <- np$load("Data/Synth_DEM/dem_crop.npy")
image(dem)
dem_weight <- 10
##generate single test set
# xaxis <- seq(curr_samp[1],curr_samp[2], length.out = 128)
# yaxis <- seq(curr_samp[1],curr_samp[2], length.out = 128)
library(terra)
for(numrast in 1:8000){
  if(numrast %% 100 == 0) cat("iteration",numrast,"\n")
  curr_samp <- sample(sample_axes,size = 2)
  xaxis <- seq(curr_samp[1],curr_samp[2], length.out = 128)
  curr_samp <- sample(sample_axes,size = 2)
  yaxis <- seq(curr_samp[1],curr_samp[2], length.out = 128)
  dat <- as.data.table(cbind(expand.grid(xaxis,yaxis),expand.grid(xax = 1:128,yax =1:128)))
  dat[,Val := sigmoid(Var1)*exp(Var2)]
  d2 <- dcast(dat, xax ~ yax, value.var = "Val")
  d2[,xax := NULL]
  meanmat <- as.matrix(d2)
  meanmat <- meanmat + 5
  
  distr1 <- mvrnorm(n = nsamp,mu = rep(0,128),Sigma = sig2)
  #distr2 <- mvrnorm(n = nsamp,mu = rep(5,128),Sigma = sig2)
  #binmask <- matrix(rbinom(128^2, size = 1, prob = 0.35),128)
  #distr1[binmask == 1] <- distr2[binmask == 1]
  mat2 <- meanmat + distr1
  mat2 <- mat2^2
  rfine <- rast(mat2)
  matds <- down_sample_image(mat2,factor = 8, gaussian_blur = T)
  rcoarse <- rast(matds)
  if(numrast == 1){
    outrast <- rfine
    outcoarse <- rcoarse
  }else{
    add(outrast) <- rfine
    add(outcoarse) <- rcoarse
  }
}

plot(outrast[[42]])
plot(outrast[[43]])
plot(outrast[[44]])
plot(outcoarse[[43]])

farr <- as.array(outrast)
carr <- as.array(outcoarse)

library(reticulate)
np <- import("numpy")
np$save("../Data/synthetic/no_small/fine_train.npy",farr[,,1:5000])
np$save("../Data/synthetic/no_small/coarse_train.npy",carr[,,1:5000])

np$save("../Data/synthetic/no_small/fine_test.npy",farr[,,5001:7000])
np$save("../Data/synthetic/no_small/coarse_test.npy",carr[,,5001:7000])

np$save("../Data/synthetic/no_small/fine_val_reg.npy",farr[,,7001:8000])
np$save("../Data/synthetic/no_small/coarse_val_reg.npy",carr[,,7001:8000])

png("Rank_Hists.png",width = 8, height = 4, units = "in", res = 600)
par(mfrow = c(1,3))
regular_max <- np$load("DoWnGAN_Kiri/Rank_Hist_Data_maxpool_Regular.npy")
hist(regular_max, main = "PFS")
sm_max <- np$load("DoWnGAN_Kiri/Rank_Hist_Data_maxpool.npy")
hist(sm_max, main = "Stochastic Mean")
vl_max <- np$load("DoWnGAN_Kiri/Rank_Hist_Data_minpool_varloss.npy")
hist(vl_max, main = "PFS + VarLoss")
dev.off()

library(gstat)
sigmoid <- function(x){
  return(5/(1+exp(-8*x)))
}
xy <- expand.grid(1:128,1:128)
names(xy) <- c('x',"y")
xy$xvar <- sigmoid(xy$x)
xy$yvar <- sigmoid(xy$y)

g.dummy <- gstat(formula=z~1+x+y, locations=~x+y, dummy=T, beta=c(1,0.005,0.005), 
                 model=vgm(psill=0.025, range=5, model='Gau'), nmax=20)
yy <- predict(g.dummy, newdata=xy, nsim=4)
gridded(yy) = ~x+y
spplot(yy)
############################################
no_obs = 128             # Number of observations per column
means = rep(1,128)               # Mean values of each column
no_cols = 128               # Number of columns

sds = rep(1.5,no_cols)                 # SD of each column
sd = diag(sds)         # SD in a diagonal matrix for later operations

observations = matrix(rnorm(no_cols * no_obs), nrow = no_cols) # Rd draws N(0,1)

# cor_matrix = matrix(c(1.0, 0.7,0.6,0.5,0.4,
#                       0.7, 1.0, 0.5,
#                       0.01, 0.5, 1.0), byrow = T, nrow = 3)     # cor matrix [3 x 3]

cov_matrix = sd %*% sigma %*% sd                          # The covariance matrix

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
library(terra)

for(i in 1:6000){
  if((i %% 50) == 0) cat("done",i,"\n")
  dat <- rnorm(128^2)
  dmat <- matrix(dat, nrow = 128, ncol = 128)
  sloc <- as.integer(round(rgamma(1, shape = 4, scale = 3)))
  dmat[,(32+sloc):ncol(dmat)] <- (dmat[,(32+sloc):ncol(dmat)])^2
  matds <- down_sample_image(dmat,factor = 32, gaussian_blur = F)
  if(i == 1){
    outrast <- rast(dmat)
    outds <- rast(matds)
  }else{
    add(outrast) <- rast(dmat)
    add(outds) <- rast(matds)
  }
}
finedat <- as.array(outrast)
coarsedat <- as.array(outds)

library(reticulate)
np <- import("numpy")
dat <- np$load("DoWnGAN_Kiri/Rank_Hist_Data.npy")

np$save("Data/ToyDataSet/VerticalSep/fine_train.npy",finedat[,,1:4000])
np$save("Data/ToyDataSet/VerticalSep/coarse_train.npy",coarsedat[,,1:4000])

np$save("Data/ToyDataSet/VerticalSep/fine_test.npy",finedat[,,4001:5000])
np$save("Data/ToyDataSet/VerticalSep/coarse_test.npy",coarsedat[,,4001:5000])

np$save("Data/ToyDataSet/VerticalSep/fine_val.npy",finedat[,,5001:6000])
np$save("Data/ToyDataSet/VerticalSep/coarse_val.npy",coarsedat[,,5001:6000])

test <- colMeans(dmat)

dat <- rnorm(128^2)
dmat <- matrix(dat, nrow = 128, ncol = 128)
image(dmat)
sloc <- as.integer(round(rgamma(1, shape = 4, scale = 3)))
dmat[,(32+sloc):ncol(dmat)] <- dmat[,(32+sloc):ncol(dmat)]*10
image(dmat)
matds <- down_sample_image(dmat,factor = 32, gaussian_blur = F)
image(matds)



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




image(outrast[,,8])
library(OpenImageR)
dsimg <- down_sample_image(mat2,factor = 8, gaussian_blur = T)
image(dsimg)
