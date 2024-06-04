library(terra)
library(reticulate)
dem <- rast("../../FFEC/Common_Files/WNA_DEM_SRT_30m_cropped.tif")
loc2 <- rast("../Data/loc2_all/fine_train_tq.nc")
plot(loc2)
writeCDF(loc2, "../Data/hr_invariant/hr_topo_30m.nc")
plot(loc2$temp_1990)

var <- rast("../Data/hr_invariant/var_hr.nc")
plot(var)

lake <- rast("../Data/hr_invariant/lkmsk_hr.nc")
plot(lake)

lu <- rast("../Data/hr_invariant/luidx_hr.nc")
plot(lu)


temp <- crop(dem, loc2)
plot(temp)
hr_topo <- resample(lu, loc2)
hr_topo2 <- (hr_topo - (mean(values(hr_topo))))/sd(values(hr_topo))
plot(hr_topo2)
rvals <- as.array(hr_topo2)

writeCDF(hr_topo2, "../Data/wind_loc2/hr_topo_loc2.nc")
trast <- rast("../Data/tile_data/tile_t9/hr_topo_all.nc")
plot(trast)
vals <- py$tspd
values(trast) <- vals
plot(trast)
writeRaster(trast, "GAN_WndSpd.tif")


hr_g3 <- hr_topo2
hr_g2 <- hr_topo2
hr_g1 <- hr_topo2
writeCDF(hr_topo, "../Data/hr_topo_medres.nc")
writeCDF(hr_g2, "../Data/ds_wind_full/hr_topo_g2.nc")
writeCDF(hr_g3, "../Data/ds_wind_full/hr_topo_g3.nc")



library(ggplot2)
library(data.table)
library(reticulate)
library(ggsci)
library(latex2exp)

##ralsd temperature
dat <- data.table(Mean = py$ral, StDev = py$sdral)
dat[,FreqBand := 1:nrow(dat)]

ggplot(dat, aes(x = FreqBand, y = Mean)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = Mean - StDev, ymax = Mean + StDev), alpha = 0.2, linetype = 0)+
  geom_hline(yintercept = 1) +
  theme_bw()+
  ylab("Standardised Amplitude") +
  xlab("Frequency Band") +
  ggtitle("Precip")

ggsave("RAPS_Precip.png", height = 5, width = 8, dpi = 400)


gen <- py$hist_gen
wrf <- py$hist_wrf
xval <- py$hist_wrf_brks
xval <- xval[-1]
diff <- wrf-gen

dat <- data.table(Gen = gen, Real = wrf, Precip = xval)

dat2 <- melt(dat, id.vars = "Precip")

ggplot(dat2, aes(x = Precip, y = value, color = variable)) +
  geom_line() +
  scale_y_log10() +
  geom_vline(xintercept = py$quant99) +
  ylab("Count (log)")

ggsave("Precip_Freq_All_Vars.png", height = 5, width = 7, dpi = 300)

##Rank Hist
temp <- py$results
rh_dat <- data.table(fmae = hist(temp$F_MAE, plot = F, breaks = 100)$counts,
                     smae = hist(temp$S_MAE, plot = F, breaks = 100)$counts,
                     scrps = hist(temp$S_CRPS, plot = F, breaks = 100)$counts)
rh_dat[,`:=`(cdf_fmae = cumsum(fmae),
             cdf_smae = cumsum(smae),
             cdf_scrps = cumsum(scrps))]
rh_dat[, cdf_unif := cumsum(rep(sum(scrps)/length(scrps),length(scrps)))]
fwrite(rh_dat, "RankHistData.csv")
rh_dat <- fread("RankHistData.csv")
pdat <- rh_dat[,.(cdf_fmae,cdf_smae,cdf_scrps,cdf_unif)]
pdat[,rank := seq_along(cdf_unif)]
pdat <- rbind(matrix(rep(0,5),nrow = 1),pdat, use.names = FALSE)
setnames(pdat, c(names(rh_dat[,.(cdf_fmae,cdf_smae,cdf_scrps,cdf_unif)]),"rank"))
pdat <- melt(pdat, id.vars = "rank", value.name = "CDF", variable.name = "Model")
pdat[,CDF := CDF/max(CDF)]
pdat[,rank := rank/max(rank)]


ggplot(pdat[!Model %in%  c("cdf_unif"),], aes(x = rank, y = CDF, col = Model)) +
  geom_line(linewidth = 1) +
  #geom_line(data = pdat[Model == "cdf_noisecov",], aes(x = rank, y = CDF), linetype = "dotted", col = "black", linewidth = 1) +
  geom_line(data = pdat[Model == "cdf_unif",], aes(x = rank, y = CDF, col = "Uniform"), linetype = "dashed", col = "black", linewidth = 0.5)+
  theme_bw() +
  #scale_color_manual(values = c('#dfe0af', '#4696c6', '#00429d') ,labels = lapply(c("$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  xlab("Normalised Rank")

ggsave("RankHist_CDF_Precip.png", height = 5, width = 8, dpi = 400)


##test bivariate vs univariate
ami_data <- read.table("http://static.lib.virginia.edu/statlab/materials/data/ami_data.DAT")
names(ami_data) <- c("TOT","AMI","GEN","AMT","PR","DIAP","QRS")
mlm1 <- lm(cbind(TOT, AMI) ~ GEN + AMT + PR + DIAP + QRS, data = ami_data)
summary(mlm1)

dat <- data.table(NoiseCov = py$ks_ncov, LowNoise = py$ks_l1, MediumNoise = py$ks_l2, FullNoise = py$ks_ninj)
dat2 <- melt(dat)

##synthetic rank hist
temp <- py$results
rh_dat <- data.table(var1 = hist(temp$f_mae_nc, plot = F, breaks = 100)$counts,
                     var2 = hist(temp$f_mae_full, plot = F, breaks = 100)$counts,
                     var3 = hist(temp$s_mae_full, plot = F, breaks = 100)$counts,
                     var4 = hist(temp$s_crps_full, plot = F, breaks = 100)$counts)

fwrite(rh_dat,"Synthetic_rankhist.csv")
rh_dat[,`:=`(cdf_v1 = cumsum(var1),
             cdf_v2 = cumsum(var2),
             cdf_v3 = cumsum(var3),
             cdf_v4 = cumsum(var4))]
rh_dat[, cdf_unif := cumsum(rep(sum(var1)/length(var1),length(var1)))]
pdat <- rh_dat[,.(cdf_v1,cdf_v2,cdf_v3,cdf_v4,cdf_unif)]
pdat[,rank := seq_along(cdf_unif)]
temp <- as.data.table(matrix(rep(0,ncol(pdat)),nrow = 1))
setnames(temp,names(pdat))
pdat <- rbind(temp,pdat)
pdat <- melt(pdat, id.vars = "rank", value.name = "CDF", variable.name = "Model")
pdat[,CDF := CDF/max(CDF)]
pdat[,rank := rank/max(rank)]

ggplot(pdat[!Model %in%  c("cdf_unif"),], aes(x = rank, y = CDF, col = Model)) +
  geom_line(linewidth = 1) +
  #geom_line(data = pdat[Model == "cdf_noisecov",], aes(x = rank, y = CDF), linetype = "dotted", col = "black", linewidth = 1) +
  geom_line(data = pdat[Model == "cdf_unif",], aes(x = rank, y = CDF, col = "Uniform"), linetype = "dashed", col = "black", linewidth = 0.5)+
  theme_bw() +
  scale_color_manual(values = c("black",'#dfe0af', '#4696c6', '#00429d') ,label = lapply(c("$F_{NC}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  xlab("Normalised Rank")

ggsave("Synthetic_RankHist_all.png", height = 5, width = 7, dpi = 400)

##Synthetic RALSD
library(ggplot2)
library(latex2exp)
dat <- py$res
dat2 <- as.data.table(rbind(dat$f_mae_nc,dat$f_mae_full,dat$s_mae_full,dat$s_crps_full))
dat2[,Model := rep(c("Baseline","+ Noise Inject","+ Stoch Train","+ CRPS"), each = 64)]
setnames(dat2, c("Mean","StDev","Model"))
dat2[,FreqBand := rep(1:64, length(dat))]
dat2[,Model := factor(Model, levels = c("Baseline","+ Noise Inject","+ Stoch Train","+ CRPS"))]

ggplot(dat2, aes(x = FreqBand, y = Mean, col = Model)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = Mean - StDev, ymax = Mean + StDev, fill = Model), alpha = 0.2, linetype = 0)+
  geom_hline(yintercept = 1) +
  theme_bw()+
  scale_color_manual(values = c("black",'#dfe0af', '#4696c6', '#00429d'),label = lapply(c("$F_{NC}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  scale_fill_manual(values = c("black",'#dfe0af', '#4696c6', '#00429d'),label = lapply(c("$F_{NC}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  #scale_color_jama(labels = lapply(c("$F_{low}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  #scale_fill_jama(labels = lapply(c("$F_{low}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  ylab("Standardised Amplitude") +
  xlab("Frequency Band")

ggsave("Synthetic_RALSD_all.png", height = 5, width = 7, dpi = 400)

##Real data RALSD
library(ggplot2)
library(latex2exp)
dat <- py$res
dat2 <- as.data.table(rbind(dat$Basic,dat$Freqsep,dat$Stochastic,dat$Stochastic_CRPS))
dat2[,Model := rep(c("Baseline","+ Noise Inject","+ Stoch Train","+ CRPS"), each = 64)]
setnames(dat2, c("Mean","StDev","Model"))
dat2[,FreqBand := rep(1:64, length(dat))]
dat2[,Model := factor(Model, levels = c("Baseline","+ Noise Inject","+ Stoch Train","+ CRPS"))]

ggplot(dat2, aes(x = FreqBand, y = Mean, col = Model)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = Mean - StDev, ymax = Mean + StDev, fill = Model), alpha = 0.2, linetype = 0)+
  geom_hline(yintercept = 1) +
  theme_bw()+
  scale_color_manual(values = c("black",'#dfe0af', '#4696c6', '#00429d'),label = lapply(c("$F_{NC}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  scale_fill_manual(values = c("black",'#dfe0af', '#4696c6', '#00429d'),label = lapply(c("$F_{NC}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  #scale_color_jama(labels = lapply(c("$F_{low}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  #scale_fill_jama(labels = lapply(c("$F_{low}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  ylab("Standardised Amplitude") +
  xlab("Frequency Band")

ggsave("Real_RALSD_Merid.png", height = 5, width = 7, dpi = 400)


###synthetic KS violin plot###########
dat <- data.table(NC = py$ks_ncov, Low = py$ks_l1, Med = py$ks_l2, Full = py$ks_l3,
                 S_MAE = py$ks_s_mae, S_CRPS = py$ks_s_crps)

dat <- melt(dat)

ggplot(dat, aes(x = variable, y = value, fill = variable)) +
  geom_violin(draw_quantiles = c(0.25,0.5,0.75), show.legend = FALSE) +
  xlab("Model") +
  ylab("KS Statistic") +
  scale_fill_manual(values = c("black",'#dfe0af','#dfe0af','#dfe0af', '#4696c6', '#00429d'))+
  scale_x_discrete(labels = unname(TeX(c("$F_{NC}^{MAE}$","$F_{low}^{MAE}$","$F_{med}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"))))+
  theme_bw()

ggsave("NoiseIncrease_KS_final.png", height = 5, width = 6, dpi = 400)

dat <- data.table(Real = py$real, LowNoise = py$gen_ninj1, MedNoise = py$gen_ninj2, FullNoise = py$gen_ninj3)
dat2 <- melt(dat)

ggplot(dat2[variable != "Real",], aes(x = value, col = variable)) +
  geom_density(linewidth = 0.8) +
  geom_density(data = dat2[variable == "Real",], aes(x = value), col = "black", linewidth = 1)
ggsave("IncreasedNoise_Density.png", height = 5, width = 6, dpi = 400)

##bimodal density
dat <- data.table(Real = py$samp2, ninj = py$gen_ninj1[1:500], ncov = py$gen_nc[1:500])
dat2 <- melt(dat)
dat2[,variable := factor(variable,levels = c("Real","ncov","ninj"))]

ggplot(dat2[variable != "Real",], aes(x = value, col = variable)) +
  geom_density(linewidth = 1) +
  geom_density(data = dat2[variable == "Real",], aes(x = value), col = "black", linetype = "dashed", linewidth = 1) +
  scale_color_manual(name = "Model",values = c("black",'#dfe0af'), labels = lapply(c("$F_{NC}^{MAE}$","$F_{full}^{MAE}$"), TeX)) +
  #scale_color_manual(name = "Model", values = c("#DF8F44FF","grey40"), labels = c("Baseline","+ Noise Inject")) +
  theme_bw() + 
  xlab("Pixel Value") +
  ylab("Marginal Density")

ggsave("Bimodal_Marginal.png", height = 5, width = 6, dpi = 400)

##Synthetic density
dat <- py$gens_res
dat <- data.table(Real = py$samp2, ninj = py$gen_ni[1:500], ncov = py$gen_nc[1:500])
dat2 <- melt(dat)
dat2[,variable := factor(variable,levels = c("Real","ncov","ninj"))]

ggplot(dat2[variable != "Real",], aes(x = value, col = variable)) +
  geom_density(linewidth = 1) +
  geom_density(data = dat2[variable == "Real",], aes(x = value), col = "black",linetype = "dashed", linewidth = 1) +
  #scale_color_jama(name = "Model", labels = lapply(c("$F_{Full}^{MAE}$","$F_{NC}^{MAE}$"), TeX)) +
  scale_color_manual(name = "Model", values = c("black",'#dfe0af'), label = lapply(c("$F_{NC}^{MAE}$","$F_{full}^{MAE}$"), TeX)) +
  theme_bw() + 
  xlab("Pixel Value") +
  ylab("Marginal Density") +
  theme(legend.position=c(0.8,0.5),
        legend.box.background = element_rect(colour = "black"))

ggsave("Synthetic_Marginal.png", height = 5, width = 6, dpi = 400)


# library(ggplot2)
# hist(py$res)
# hist(py$res_nc)
dat <- data.table(NoiseInject = py$res, NoiseCovar = py$res_nc)
dat2 <- melt(dat)

rh_dat <- data.table(noisecov = hist(py$res_nc, plot = F, breaks = 50)$counts,
                     noiseinj = hist(py$res, plot = F, breaks = 50)$counts)

rh_dat[,`:=`(cdf_noisecov = cumsum(noisecov),
             cdf_noiseinj = cumsum(noiseinj))]
rh_dat[, cdf_unif := cumsum(rep(sum(noisecov)/length(noisecov),length(noisecov)))]

pdat <- rh_dat[,.(cdf_noiseinj,cdf_noisecov,cdf_unif)]
pdat[,rank := seq_along(cdf_unif)]
temp <- as.data.table(matrix(rep(0,4),nrow = 1))
setnames(temp, names(pdat))
pdat <- rbind(temp,pdat)

pdat <- melt(pdat, id.vars = "rank", value.name = "CDF", variable.name = "Model")
pdat[,CDF := CDF/max(CDF)]
pdat[,rank := rank/max(rank)]


ggplot(pdat[Model != "cdf_unif",], aes(x = rank, y = CDF, col = Model)) +
  geom_line(linewidth = 1) +
  geom_line(data = pdat[Model == "cdf_unif",], aes(x = rank, y = CDF), linetype = "dashed", col = "black", linewidth = 1)+
  theme_bw() +
  scale_color_jama(labels = c("NoiseInject","NoiseCovar")) +
  xlab("Normalised Rank")

ggsave("SyntheticRankCDF.png", height = 4, width = 6, dpi = 400)

dat2[,variable := factor(variable,levels = c("NoiseCovar","NoiseInject"))]
ggplot(dat2, aes(x = value, fill = variable)) +
  geom_histogram(bins = 18, show.legend = FALSE) +
  facet_wrap(~variable, labeller = as_labeller(c(NoiseCovar = "Baseline",NoiseInject = "+ Noise Inject"))) +
  scale_fill_manual(name = "Model", values = c("#DF8F44FF","grey40"), labels = c("Baseline","+ Noise Inject")) +
  xlab("Rank") +
  ylab("Frequency") +
  theme_bw()
ggsave("SyntheticRankHist.png", height = 4, width = 6, dpi = 400)


exposures <- seq(from = -4,
                 to = 4,
                 by = 0.01)
probabilities <- dnorm(exposures)

# calculating measures
measures <- quantile(x = exposures,
                     probs = c(0.48, 0.52))

png("exampledistrib.png", width = 5, height = 5, units = "in", res = 300)
# plotting distribution
plot(x = exposures,
     y = probabilities,
     type = "l")


polygon(x = c(measures[1], exposures[(exposures >= measures[1]) & (exposures <= measures[2])], measures[2]),
        y = c(-1, probabilities[(exposures >= measures[1]) & (exposures <= measures[2])], 0),
        border = NA,
        col = adjustcolor(col = "darkgreen",
                          alpha.f = 1))

dev.off()

dat <- data.table(Covariate = py$ks_ncov, Injection = py$ks_ninj)
dat2 <- melt(dat)

ggplot(dat2, aes(x = variable, y = value, fill = variable)) +
  geom_violin(draw_quantiles = c(0.025,0.5,0.75), show.legend = FALSE, col = "black") +
  ylab("KS Statistic") +
  xlab("Noise Type")+
  theme_bw() +
  scale_fill_jama()
ggsave("figs_for_paper/NoiseType_Violin.png", height = 5, width = 5, dpi = 400)

##example marginal distribution
dat <- data.table(Real = py$real_marg, Injection = py$gen_ninj, Covariate = py$gen_ncov)
dat2 <- melt(dat, variable.name = "Model", value.name = "Value")

ggplot(dat2, aes(x = Value,  col = Model)) +
  geom_density(linewidth = 0.8) +
  scale_y_log10() +
  xlab("Pixel Value") +
  theme_bw() +
  scale_colour_jama()

ggsave("Example_Marginal.png", height = 5, width = 6, dpi = 400)


###value difference
dat <- data.table(ks_stat = py$ks_ninj, mean_value = py$fine_mn_val)

ggplot(dat, aes(x = mean_value,  y = ks_stat)) +
  geom_point() +
  geom_smooth() +
  xlab("Pixel Value") +
  ylab("Wasserstein Distance") +
  theme_bw() +
  scale_colour_jama() +
  scale_fill_jama()

ggsave("WassersteinvsValue_Synthetic_not_standardised.png", height = 5, width = 6, dpi = 400)

####################################################################

dat <- data.table(Real = py$real_marg, NoiseCov = py$gen_ncov, NoiseInj = py$gen_ninj)
dat2 <- melt(dat, id.vars = NULL)

ggplot(dat2, aes(x = value, col = variable, group = variable)) +
  geom_density() +
  ylab("Density") +
  xlab("Pixel Value")

ggsave("figs_for_paper/f1a.png", dpi = 600, width = 5, height = 5)

library(SpecsVerification)
load("justcrps_rh.rdata")
ens = t(py$ens)
obs = py$real
ens2 <- ens[,1:39]

rh <- Rankhist(ens,obs)
PlotRankhist(rh, mode = "raw")

rh_basic <- rh
rh_mae <- rh
rh_crps <- rh

basic <- cumsum(rh_basic)
mae <- cumsum(rh_mae)
crps <- cumsum(rh_crps)
plot(crps, type = "l")
lines(mae, type = "l", col = "red")
lines(basic, type = "l", col = "blue")

rh_dat <- data.table(basic = rh_basic, mae = rh_mae, crps = rh_crps)
rh_dat[,`:=`(cdf_basic = cumsum(basic),cdf_mae = cumsum(mae), cdf_crps = cumsum(crps))]
rh_dat[, cdf_unif := cumsum(rep(16221.78,101))]
fwrite(rh_dat, "RankHistData.csv")
pdat <- rh_dat[,.(cdf_basic,cdf_mae,cdf_crps,cdf_unif)]
pdat[,rank := 1:101]
pdat <- melt(pdat, id.vars = "rank", value.name = "CDF", variable.name = "Model")
pdat[,CDF := CDF/max(CDF)]
pdat[,rank := rank/max(rank)]

library(ggsci)

ggplot(pdat[Model != "cdf_unif",], aes(x = rank, y = CDF, col = Model)) +
  geom_line(linewidth = 1) +
  geom_line(data = pdat[Model == "cdf_unif",], aes(x = rank, y = CDF), linetype = "dashed", col = "black", linewidth = 1)+
  theme_bw() +
  scale_color_jama() +
  xlab("Normalised Rank")

ggsave("RankHist_CDF_RealData.png", height = 3, width = 6, dpi = 400)

########################
temp <- py$results
rh_dat <- data.table(noisecov = hist(temp$Basic, plot = F, breaks = 100)$counts,
                     freqsep = hist(temp$FreqSep, plot = F, breaks = 100)$counts,
                     stochastic = hist(temp$Stochastic, plot = F, breaks = 100)$counts,
                     crps = hist(temp$Stochastic_CRPS, plot = F, breaks = 100)$counts)
rh_dat[,`:=`(cdf_noisecov = cumsum(noisecov),
             cdf_freqsep = cumsum(freqsep),
             cdf_stochastic = cumsum(stochastic),
             cdf_crps = cumsum(crps))]
rh_dat[, cdf_unif := cumsum(rep(sum(crps)/length(crps),length(crps)))]
fwrite(rh_dat, "RankHistData.csv")
rh_dat <- fread("RankHistData.csv")
pdat <- rh_dat[,.(cdf_noisecov,cdf_freqsep,cdf_stochastic,cdf_crps,cdf_unif)]
pdat[,rank := seq_along(cdf_unif)]
pdat <- rbind(matrix(rep(0,6),nrow = 1),pdat, use.names = FALSE)
setnames(pdat, c(names(rh_dat[,.(cdf_noisecov,cdf_freqsep,cdf_stochastic,cdf_crps, cdf_unif)]),"rank"))
pdat <- melt(pdat, id.vars = "rank", value.name = "CDF", variable.name = "Model")
pdat[,CDF := CDF/max(CDF)]
pdat[,rank := rank/max(rank)]

pdat1 <- pdat[Model %in% c("cdf_noisecov","cdf_unif","cdf_freqsep"),]
library(ggsci)
ggplot(pdat1[Model == "cdf_noisecov",], aes(x = rank, y = CDF, col = Model)) +
  geom_line(linewidth = 1) +
  geom_line(data = pdat1[Model == "cdf_unif",], aes(x = rank, y = CDF), linetype = "dashed", col = "black", linewidth = 0.5)+
  theme_bw() +
  scale_color_manual(values = c("#DF8F44FF","grey40") ,labels = c("Baseline", "+ Noise Inject")) +
  xlab("Normalised Rank")

ggsave("ExampleRH.png", height = 5, width = 8, dpi = 400)


ggplot(pdat[!Model %in%  c("cdf_unif"),], aes(x = rank, y = CDF, col = Model)) +
  geom_line(linewidth = 1) +
  #geom_line(data = pdat[Model == "cdf_noisecov",], aes(x = rank, y = CDF), linetype = "dotted", col = "black", linewidth = 1) +
  geom_line(data = pdat[Model == "cdf_unif",], aes(x = rank, y = CDF, col = "Uniform"), linetype = "dashed", col = "black", linewidth = 0.5)+
  theme_bw() +
  scale_color_manual(values = c("black",'#dfe0af', '#4696c6', '#00429d') ,labels = lapply(c("$F_{NC}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  xlab("Normalised Rank")

ggsave("RankHist_CDF_RealData_FourModels.png", height = 5, width = 8, dpi = 400)

###spat comp ks stats
dat <- data.table(low = py$ks_w01, med = py$ks_w1, high = py$ks_w10)
fwrite(dat,"spatcomplex_ks_stat.csv")
dat2 <- melt(dat)
ggplot(dat2, aes(x = variable, y = value)) +
  geom_violin()

ggplot(dat2, aes(x = variable, y = value, fill = variable)) +
  geom_violin(draw_quantiles = c(0.25,0.5,0.75), show.legend = FALSE) +
  xlab("Spatial Heterogeneity") +
  ylab("KS Statistic") +
  scale_fill_manual(values = c('#dfe0af', '#4696c6', '#00429d'))+
  scale_x_discrete(labels = c("Low", "Medium", "High"))+
  theme_bw()
ggsave("KS_Violin_SpatComplex.png", height = 5, width = 5, dpi = 400)


###Rankhist of Max Values
temp <- py$results1
rh_dat <- data.table(noisecov = hist(temp$Basic, plot = F, breaks = 36)$counts,
                     freqsep = hist(temp$FreqSep, plot = F, breaks = 36)$counts,
                     stochastic = hist(temp$Stochastic, plot = F, breaks = 36)$counts,
                     crps = hist(temp$Stochastic_CRPS, plot = F, breaks = 36)$counts)
rh_dat[,`:=`(cdf_noisecov = cumsum(noisecov),
             cdf_freqsep = cumsum(freqsep),
             cdf_stochastic = cumsum(stochastic),
             cdf_crps = cumsum(crps))]
rh_dat[, cdf_unif := cumsum(rep(sum(crps)/length(crps),length(crps)))]
pdat <- rh_dat[,.(cdf_noisecov,cdf_freqsep,cdf_stochastic,cdf_crps,cdf_unif)]
pdat[,rank := seq_along(cdf_unif)]
pdat <- rbind(matrix(rep(0,6),nrow = 1),pdat, use.names = FALSE)
setnames(pdat, c(names(rh_dat[,.(cdf_noisecov,cdf_freqsep,cdf_stochastic,cdf_crps, cdf_unif)]),"rank"))
pdat <- melt(pdat, id.vars = "rank", value.name = "CDF", variable.name = "Model")
pdat[,CDF := CDF/max(CDF)]
pdat[,rank := rank/max(rank)]
pdat[,Quantile := "0.01%"]

temp <- py$results9
rh_dat <- data.table(noisecov = hist(temp$Basic, plot = F, breaks = 36)$counts,
                     freqsep = hist(temp$FreqSep, plot = F, breaks = 36)$counts,
                     stochastic = hist(temp$Stochastic, plot = F, breaks = 36)$counts,
                     crps = hist(temp$Stochastic_CRPS, plot = F, breaks = 36)$counts)
rh_dat[,`:=`(cdf_noisecov = cumsum(noisecov),
             cdf_freqsep = cumsum(freqsep),
             cdf_stochastic = cumsum(stochastic),
             cdf_crps = cumsum(crps))]
rh_dat[, cdf_unif := cumsum(rep(sum(crps)/length(crps),length(crps)))]
pdat2 <- rh_dat[,.(cdf_noisecov,cdf_freqsep,cdf_stochastic,cdf_crps,cdf_unif)]
pdat2[,rank := seq_along(cdf_unif)]
pdat2 <- rbind(matrix(rep(0,6),nrow = 1),pdat2, use.names = FALSE)
setnames(pdat2, c(names(rh_dat[,.(cdf_noisecov,cdf_freqsep,cdf_stochastic,cdf_crps, cdf_unif)]),"rank"))
pdat2 <- melt(pdat2, id.vars = "rank", value.name = "CDF", variable.name = "Model")
pdat2[,CDF := CDF/max(CDF)]
pdat2[,rank := rank/max(rank)]
pdat2[,Quantile := "99.99%"]
pdat <- rbind(pdat, pdat2)


library(ggsci)
library(latex2exp)

ggplot(pdat[Model != "cdf_unif",], aes(x = rank, y = CDF, col = Model)) +
  geom_line(linewidth = 1) +
  geom_line(data = pdat[Model == "cdf_unif",], aes(x = rank, y = CDF), linetype = "dashed", col = "black", linewidth = 1)+
  theme_bw() +
  facet_wrap(~ Quantile) +
  scale_color_manual(values = c("black",'#dfe0af', '#4696c6', '#00429d') ,labels = lapply(c("$F_{NC}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  xlab("Normalised Rank")
ggsave("RankHist_Quantiles_Merid.png", height = 4, width = 8, dpi = 500)

hist(temp$Stochastic_CRPS)

##Synth DEM
res_list <- py$results
rh_dat <- data.table(Low = hist(res_list$w1, plot = F, breaks = 50)$counts,
                     Medium = hist(res_list$w2, plot = F,breaks = 50)$counts, 
                     High = hist(res_list$w4, plot = F,breaks = 50)$counts)
rh_dat[,`:=`(cdf_low = cumsum(Low),cdf_medium = cumsum(Medium), cdf_high = cumsum(High))]
rh_dat[, cdf_unif := cumsum(rep(sum(Low)/length(Low),length(Low)))]
fwrite(rh_dat, "RankHistData_SynthDEM.csv")
pdat <- rh_dat[,.(cdf_low,cdf_medium,cdf_high,cdf_unif)]
pdat[,rank := seq_along(cdf_unif)]
pdat <- rbind(matrix(rep(0,5),nrow = 1),pdat, use.names = FALSE)
setnames(pdat, c(names(rh_dat[,.(cdf_low,cdf_medium,cdf_high,cdf_unif)]),"rank"))
pdat <- melt(pdat, id.vars = "rank", value.name = "CDF", variable.name = "Model")
pdat[,CDF := CDF/max(CDF)]
pdat[,rank := rank/max(rank)]

pdat[Model == "cdf_low",Spatial_Complexity := "0.1"]
pdat[Model == "cdf_medium",Spatial_Complexity := "1"]
pdat[Model == "cdf_high",Spatial_Complexity := "10"]

ggplot(pdat[Model != "cdf_unif",], aes(x = rank, y = CDF, col = Spatial_Complexity)) +
  geom_line(linewidth = 1) +
  geom_line(data = pdat[Model == "cdf_unif",], aes(x = rank, y = CDF), linetype = "dashed", col = "black", linewidth = 1)+
  theme_bw() +
  scale_color_manual(values = c('#dfe0af', '#4696c6', '#00429d'), labels = c("Low","Med","High"), name = "Heterogeneity") +
  xlab("Normalised Rank")
ggsave("RankHist_SynthDEM_all.png", height = 5, width = 6, dpi = 400)

##Max pool
rh_dat <- data.table(basic = hist(res$Basic, plot = F)$counts, mae = hist(res$Stochastic, plot = F)$counts, 
                     crps = hist(res$Stochastic_CRPS, plot = F)$counts)
rh_dat[,`:=`(cdf_basic = cumsum(basic),cdf_mae = cumsum(mae), cdf_crps = cumsum(crps))]
rh_dat[, cdf_unif := cumsum(rep(sum(basic)/length(basic),length(basic)))]
fwrite(rh_dat, "RankHistData_MaxPool.csv")
pdat <- rh_dat[,.(cdf_basic,cdf_mae,cdf_crps,cdf_unif)]
pdat[,rank := seq_along(cdf_unif)]
pdat <- melt(pdat, id.vars = "rank", value.name = "CDF", variable.name = "Model")
pdat[,CDF := CDF/max(CDF)]
pdat[,rank := rank/max(rank)]

library(ggsci)

ggplot(pdat[Model != "cdf_unif",], aes(x = rank, y = CDF, col = Model)) +
  geom_line(linewidth = 1) +
  geom_line(data = pdat[Model == "cdf_unif",], aes(x = rank, y = CDF), linetype = "dashed", col = "black", linewidth = 1)+
  theme_bw() +
  scale_color_jama() +
  xlab("Normalised Rank")
ggsave("RankHistMP_CDF_RealData.png", height = 3, width = 6, dpi = 400)

res <- py$results
test <- hist(res$Stochastic_CRPS, breaks = 50, plot = F)

save(ens,obs, file = "nocrps_rh.rdata")

###extremes
cdif = py$cdif
nocdif = py$nocdif
basic = py$regdif

t.test(cdif,basic)

dat <- data.table(CRPS = cdif, MAE = nocdif, Basic = basic)
dat2 <- melt(dat, value.name = "ExtremeDiff", variable.name = "Model")
ggplot(dat2, aes(x = Model, y = ExtremeDiff, fill = Model)) +
  geom_boxplot(outlier.shape = NA, notch = T, show.legend = F) +
  ylim(NA,5)+
  ylab("Difference in 99.99% quantile") +
  theme_bw() +
  scale_fill_jama()

ggsave("Extreme_compare.png", height = 5, width = 6, dpi = 400)

##extremes
dat <- data.table(Basic = py$qbasic, FreqSep = py$qfs, MAE = py$qmae, CRPS = py$qcrps)
dat1 <- melt(dat, variable.name = "Model")
dat1[,Quantile := 0.0001]
dat <- data.table(Basic = py$qbasic9, FreqSep = py$qfs9, MAE = py$qmae9, CRPS = py$qcrps9)
dat9 <- melt(dat, variable.name = "Model")
dat9[,Quantile := 0.9999]
dat_all <- rbind(dat1,dat9)

fwrite(dat_all,"Quantile_Data_Merid.csv")
dat_all <- fread("Quantile_Data_Merid.csv")
dat_all[,Quantile := as.factor(Quantile)]
dat_all[,Model := factor(Model, levels = c("Basic","FreqSep","MAE","CRPS"))]

ggplot(dat_all, aes(x = Model, y = value, fill = Quantile, color = Model)) +
  geom_boxplot(notch = TRUE, outlier.shape = NA, show.legend = T, linewidth = 0.8) +
  geom_hline(yintercept = 0)+
  scale_x_discrete(labels = unname(TeX(c("$F_{NC}^{MAE}$","$F_{low}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$")))) +
  #geom_violin() +
  theme_bw()+
  scale_fill_manual(values = c("white","grey"),labels = c("0.01%","99.99%")) +
  scale_color_manual(values = c("black",'#dfe0af', '#4696c6', '#00429d')) +
  ylim(c(-2,3.5)) +
  ylab("Difference in Ground-Truth and Modeled Quantiles")

ggsave("Extremes_Final_Merid.png", height = 5, width = 6, dpi = 400)
unname(TeX(c("$F_{low}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$")))
#########################
mat <- py$qcrps_test
image(mat)
matdt <- data.table(mat)
matdt[,row := 1:128]
matdt <- melt(matdt, id.vars = "row")

ggplot(matdt, aes(x = variable, y = row, fill = value)) +
  geom_raster()
###RALSD
library(ggplot2)
library(latex2exp)
dat <- py$res_zonal
dat2 <- as.data.table(rbind(dat$Basic,dat$Freqsep,dat$Stochastic,dat$Stochastic_CRPS))
dat2[,Model := rep(c("Basic","Freqsep","Stochastic","Stochastic_CRPS"), each = 64)]
setnames(dat2, c("Mean","StDev","Model"))
dat2[,FreqBand := rep(1:64, length(dat))]
dat2[,Component := "Zonal"]

dat <- py$res_merid
datmerid <- as.data.table(rbind(dat$Basic,dat$Freqsep,dat$Stochastic,dat$Stochastic_CRPS))
datmerid[,Model := rep(c("Basic","Freqsep","Stochastic","Stochastic_CRPS"), each = 64)]
setnames(datmerid, c("Mean","StDev","Model"))
datmerid[,FreqBand := rep(1:64, length(dat))]
datmerid[,Component := "Meridional"]

dat2 <- rbind(dat2,datmerid)

#dat2[,Model := factor(Model, levels = c("NoFS","CRPS50","CRPS20"))]

ggplot(dat2, aes(x = FreqBand, y = Mean, col = Model)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = Mean - StDev, ymax = Mean + StDev, fill = Model), alpha = 0.2, linetype = 0)+
  geom_hline(yintercept = 1) +
  theme_bw()+
  scale_color_manual(values = c("black",'#dfe0af', '#4696c6', '#00429d') ,labels = lapply(c("$F_{NC}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  scale_fill_manual(values = c("black",'#dfe0af', '#4696c6', '#00429d') ,labels = lapply(c("$F_{NC}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  facet_wrap(~ Component) +
  ylab("Standardised Amplitude") +
  xlab("Frequency Band")

ggsave("RALSD_Realdata_Final.png", height = 5, width = 8, dpi = 400)

ggplot(dat2, aes(x = FreqBand, y = Mean, col = Model)) +
  geom_line(linewidth = 1) +
  #geom_ribbon(aes(ymin = Mean - StDev, ymax = Mean + StDev, fill = Model), alpha = 0.2, linetype = 0)+
  geom_hline(yintercept = 1) +
  theme_bw()+
  scale_y_log10() +
  scale_color_jama(labels = lapply(c("$F_{low}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  scale_fill_jama(labels = lapply(c("$F_{low}^{MAE}$","$F_{full}^{MAE}$","$S_{full}^{MAE}$","$S_{full}^{CRPS}$"), TeX)) +
  ylab("Standardised Amplitude") +
  xlab("Frequency Band")

ggsave("RALSD_Temperature.png", height = 5, width = 7, dpi = 400)


##temperature
temp <- py$results
rh_dat <- data.table(var1 = hist(temp$Temp, plot = F, breaks = 100)$counts,
                     var2 = hist(temp$`Temp + Humid`, plot = F, breaks = 100)$counts)
rh_dat[,`:=`(cdf_v1 = cumsum(var1),
             cdf_v2 = cumsum(var2))]
rh_dat[, cdf_unif := cumsum(rep(sum(var1)/length(var1),length(var1)))]
pdat <- rh_dat[,.(cdf_v1,cdf_v2,cdf_unif)]
pdat[,rank := seq_along(cdf_unif)]
temp <- as.data.table(matrix(rep(0,ncol(pdat)),nrow = 1))
setnames(temp,names(pdat))
pdat <- rbind(temp,pdat)
pdat <- melt(pdat, id.vars = "rank", value.name = "CDF", variable.name = "Model")
pdat[,CDF := CDF/max(CDF)]
pdat[,rank := rank/max(rank)]

ggplot(pdat[!Model %in%  c("cdf_unif"),], aes(x = rank, y = CDF, col = Model)) +
  geom_line(linewidth = 1) +
  #geom_line(data = pdat[Model == "cdf_noisecov",], aes(x = rank, y = CDF), linetype = "dotted", col = "black", linewidth = 1) +
  geom_line(data = pdat[Model == "cdf_unif",], aes(x = rank, y = CDF, col = "Uniform"), linetype = "dashed", col = "black", linewidth = 0.5)+
  theme_bw() +
  scale_color_manual(values = c("#DF8F44FF","black"), labels = c("Temperature","Temp+Humid")) +
  xlab("Normalised Rank")

ggsave("Temperature_TempHumid_RankHist.png", height = 5, width = 7, dpi = 400)


##temp humid correlation
dat <- data.table(Real = py$real_corr, SameMod = py$same_corr, DiffMod = py$diff_corr)

dat2 <- melt(dat)
ggplot(dat2, aes(x = variable, y = value)) +
  geom_violin(draw_quantiles = c(0.25,0.5,0.75)) +
  ylab("Mutual information(2d)") +
  xlab("Model") +
  ggtitle("Temp and Humid Correlation")
ggsave("TempHumidMI_Normalised.png", width = 5, height = 4, dpi = 400)  

library(terra)
fine <- rast("../Data/ds_humid_v2/fine_train.nc")
plot(fine[[4235]])
dem <- rast("../../FFEC/Common_Files/WNA_DEM_SRT_30m_cropped.tif")
d2 <- crop(dem, fine)
d3 <- resample(d2, fine)
plot(d3)
plot(fine[[5]])
d3 <- d3/sd(values(d3))
writeRaster(d3, "../Data/hr_topography.tif")
writeRaster(d3, "../Data/hr_topography.nc")

##relative humid
vars <- rast("../Data/ds_humid_v2/RH_test.nc")
plot(vars$T2_1)
temp <- vars$T2_1042
q2 <- vars$Q2_1042
ps <- vars$PSFC_1042
plot(ps)
plot(q2)
plot(temp)
rh <- 0.263*ps*q2*(exp((17.67*(temp - 273.16))/(temp - 29.65)))^(-1)
rh[rh > 100] <- 100
plot(rh)

##precip
library(terra)
dat <- rast("../Data/ds_precip_v2/fine_test.nc")
temp <- dat[[1:8]]
hist(values(temp))
library(e1071)
kurtosis(values(temp))
x <- rnorm(100)
kurtosis(x)

dat <- py$res
datmerid <- as.data.table(rbind(dat$All_Data,dat$No_Zero))
datmerid[,Model := rep(c("AllData","NoZero"), each = 64)]
setnames(datmerid, c("Mean","StDev","Model"))
datmerid[,FreqBand := rep(1:64, length(dat))]

ggplot(datmerid, aes(x = FreqBand, y = Mean, col = Model)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = Mean - StDev, ymax = Mean + StDev, fill = Model), alpha = 0.2, linetype = 0)+
  geom_hline(yintercept = 1) +
  theme_bw()+
  ylab("Standardised Amplitude") +
  xlab("Frequency Band")

ggsave("RALSD_Precip.png", width = 5, height = 4, dpi = 400)
