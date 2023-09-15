library(ggplot2)
library(data.table)

dat <- data.table(LowNoise = py$wass_l1, MediumNoise = py$wass_l2, FullNoise = py$wass_l3)
dat2 <- melt(dat)

ggplot(dat2, aes(x = variable, y = value, fill = variable)) +
  geom_violin(draw_quantiles = c(0.25,0.5,0.75), show.legend = FALSE) +
  xlab("Noise Level") +
  ylab("Wasserstein Distance")
ggsave("IncreasedNoise.png", height = 5, width = 6, dpi = 400)

dat <- data.table(Real = py$real, LowNoise = py$gen_ninj1, MedNoise = py$gen_ninj2, FullNoise = py$gen_ninj3)
dat2 <- melt(dat)

ggplot(dat2[variable != "Real",], aes(x = value, col = variable)) +
  geom_density(linewidth = 0.8) +
  geom_density(data = dat2[variable == "Real",], aes(x = value), col = "black", linewidth = 1)
ggsave("IncreasedNoise_Density.png", height = 5, width = 6, dpi = 400)

dat <- data.table(NoiseCov = py$ks_ncov, NoiseInj = py$ks_ninj)
dat2 <- melt(dat)

ggplot(dat2, aes(x = variable, y = value)) +
  geom_boxplot(col = "black", fill = "purple") +
  ylab("KS Statistic") +
  xlab("Model")
ggsave("figs_for_paper/f1b.png")

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

save(ens,obs, file = "nocrps_rh.rdata")

cdif = py$cdif
nocdif = py$nocdif

t.test(cdif,nocdif)

dat <- data.table(CRPS = cdif, NoCRPS = nocdif)
dat2 <- melt(dat)
ggplot(dat2, aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(outlier.shape = NA, notch = T, show.legend = F) +
  ylim(-1,1.5) +
  xlab("Model") +
  ylab("Difference in 99.9% quantile")

ggsave("Extreme_compare.png", height = 5, width = 6, dpi = 400)
