import numpy as np
import torch
import mlflow
from matplotlib import pyplot as plt
from matplotlib import colorbar, colors, gridspec

device = torch.device("cuda:0")
gen_location = "../Generators/real_data/freqsep_noisecov/"
G = mlflow.pytorch.load_model(gen_location)

import xarray as xr
batchsize = 16
data_folder = "../Data/processed_data/ds_wind_full/"
cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1)
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1)
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy())

##check extremes
random = torch.randint(0, 17500, (350, ))
inv_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()

all_gens = []
for j in random:
  print(j)
  coarse_in = coarse[j,...]
  coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()
  gen_out = G(coarse_in, inv_in).cpu().detach()
  for i in range(2):
      fine_gen = G(coarse_in,inv_in)
      gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
      del fine_gen
  all_gens.append(gen_out)

noisecov_fs = torch.cat(all_gens, 0)
torch.save(noisecov_fs, "Generated_noisecov_all.pt")
no_crps_allgen = torch.cat(all_gens, 0)
torch.save(no_crps_allgen, "Generated_nocrps_all.pt")
crps_allgen = torch.cat(all_gens, 0)
torch.save(crps_allgen, "Generated_crps_all.pt")

noisecov_fs = noisecov_fs[0:39200,...]
c_dif = []
noc_dif = []
nc_dif = []
for i in range(128):
  for j in range(128):
    zonal_gen_nocrps = no_crps_allgen[:,1,i,j].numpy()
    zonal_gen = crps_allgen[:,1,i,j].numpy()
    zonal_nc = noisecov_fs[:,1,i,j].numpy()   
    zonal_real = fine[:,1,i,j].numpy()
    crps = np.quantile(zonal_gen, 0.999)
    nocrps = np.quantile(zonal_gen_nocrps, 0.999)
    noisecov = np.quantile(zonal_nc, 0.999)
    real = np.quantile(zonal_real, 0.999)
    c_dif.append(real - crps)
    noc_dif.append(real - nocrps)
    nc_dif.append(real - noisecov)

cdif = np.array(c_dif)
nocdif = np.array(noc_dif)
regdif = np.array(nc_dif)
import seaborn as sns
sns.boxplot([cdif,nocdif,regdif])
plt.show()
##make nice grid figure - stolen from Leinonen
def plot_img(img, value_range=(np.log10(0.1), np.log10(100)), extent=None):
    plt.imshow(img, interpolation='nearest',
        norm=colors.Normalize(*value_range), extent=extent)
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)

num_samples=4; num_instances=4; out_fn=None; plot_stride=1; num_frames = 1

num_rows = num_samples*num_frames
num_cols = 2+num_instances

plt.close()
figsize = (num_cols*4, num_rows*4)
plt.figure(figsize=figsize)

gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.05, hspace=0.05)

#value_range = batch_gen.decoder.value_range

batchsize = num_instances
inv_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()
import random
random.seed(3.14159)

for s in range(num_samples):
    sample = random.randint(0,8000)
    for t in range(num_frames):
        i = s*num_frames+t
        plt.subplot(gs[i,0])
        plot_img(fine[sample,0,...])
        if(s == 3):
            plt.xlabel("WRF", fontsize=10)
        plt.subplot(gs[i,1])
        plot_img(coarse[sample,0,...])
        if(s == 3):
            plt.xlabel("ERA5", fontsize=10)
        for k in range(num_instances):
            coarse_in = coarse[sample,...]
            coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()

            gen_out = G(coarse_in, inv_in).cpu().detach()
            j = 2+k
            plt.subplot(gs[i,j])
            plot_img(gen_out[k,0,...]) 
            if(s == 3):
                plt.xlabel("Gen {}".format(k+1), fontsize=10)

plt.savefig("Example_Realisations_crps.png", bbox_inches='tight')

##rank histogram
def rankhist_preds(G, coarse, fine, invariant):
  batchsize = 4
  random = torch.randint(0, 1000, (100, ))
  inv_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()
  ens = []
  truth = []
  for sample in range(100):
      print("Processing",sample)
      coarse_in = coarse[sample,...]
      coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()

      gen_out = G(coarse_in, inv_in).cpu().detach()
      for i in range(9):
          fine_gen = G(coarse_in,inv_in)
          gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
          del fine_gen
      
      # real = torch.squeeze((fine[sample,0,120,120])).numpy()
      # fake = torch.squeeze((gen_out[:,0,120,120]))
      # ens.append(fake.unsqueeze(1))
      # truth.append(real)
      
      real = torch.squeeze((fine[sample,0,0:16,0:16]))
      fake = torch.squeeze((gen_out[:,0,0:16,0:16]))
      ens.append(torch.reshape(fake, [40,16*16]))
      truth.append(torch.reshape(real, [16*16]))
  return(torch.cat(ens, 1).numpy(), np.array(truth))

ens, real = rankhist_preds(G, coarse, fine, invariant)
real = real.flatten()
ens2 = ens.numpy()
real2 = real.numpy()

###specify generator
gen_location = "../Generators/Noise_Increase/level1"
G = mlflow.pytorch.load_model(gen_location)

###specify data
data_folder = "../Data/synthetic/no_small/"
coarse_train = np.load(data_folder+"coarse_stochastic.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_stochastic.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]

coarse_mn = torch.mean(coarse, dim = 0)
plt.imshow(coarse_mn[0,...])
plt.show()
batchsize = 8
coarse_in = coarse_mn.repeat(batchsize, 1,1,1).to(device).float()

gen_out = G(coarse_in).cpu().detach()
for i in range(64):
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
import scipy.stats
ks_res_ninj = np.empty([128,128])
for i in range(128):
  for j in range(128):
    gen = gen_out[0:500,0,i,j]
    target = fine[:,0,i,j]
    test = scipy.stats.wasserstein_distance(gen,target)
    ks_res_ninj[i,j] = test
    
wass_l1 = ks_res_ninj.flatten()

plt.close()
plt.imshow(ks_res_ninj)
plt.show()

t1 = ks_ninj
t2 = t1[t1 > 0.05]
print(t2.size/t1.size)

import seaborn as sns
xp = 5
yp = 120
gen_ninj1 = gen_out[:,0,xp,yp].flatten()
gen_ninj1 = gen_ninj1.numpy()
samp2 = fine[:,0,xp,yp].flatten()
plt.close()
sns.kdeplot(gen_ninj,label = "Gen")
sns.kdeplot(samp2,label = "Real")
plt.legend()
plt.show()

###specify generator
gen_location = "../Generators/Noise_Increase/level2"
G = mlflow.pytorch.load_model(gen_location)

gen_out = G(coarse_in).cpu().detach()
for i in range(64):
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
    
ks_res_ninj = np.empty([128,128])
for i in range(128):
  for j in range(128):
    gen = gen_out[0:500,0,i,j]
    target = fine[:,0,i,j]
    test = scipy.stats.wasserstein_distance(gen,target)
    ks_res_ninj[i,j] = test
    
wass_l2 = ks_res_ninj.flatten()

xp = 5
yp = 120
gen_ninj2 = gen_out[:,0,xp,yp].flatten()
gen_ninj2 = gen_ninj2.numpy()
samp2 = fine[:,0,xp,yp].flatten()
plt.close()
sns.kdeplot(gen_ninj,label = "Gen")
sns.kdeplot(samp2,label = "Real")
plt.legend()
plt.show()

##level 3
gen_location = "../Generators/Noise_Increase/level3"
G = mlflow.pytorch.load_model(gen_location)

gen_out = G(coarse_in).cpu().detach()
for i in range(64):
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
ks_res_ninj = np.empty([128,128])
for i in range(128):
  for j in range(128):
    gen = gen_out[0:500,0,i,j]
    target = fine[:,0,i,j]
    test = scipy.stats.wasserstein_distance(gen,target)
    ks_res_ninj[i,j] = test
    
wass_l3 = ks_res_ninj.flatten()
plt.close()
sns.violinplot(y = [wass_l1,wass_l2,wass_l3],x = ["Low Noise","Med Noise", "Full Noise"], cut = 0)
plt.show()
xp = 5
yp = 120
gen_ninj3 = gen_out[:,0,xp,yp].flatten()
gen_ninj3 = gen_ninj3.numpy()
real = fine[:,0,xp,yp].flatten().numpy()
plt.close()
sns.kdeplot(gen_ninj,label = "Gen")
sns.kdeplot(samp2,label = "Real")
plt.legend()
plt.show()


##noise covariate
gen_location = "../Generators/synthetic_noisecov_standardised/"
G = mlflow.pytorch.load_model(gen_location)
noise = torch.normal(0,1,size = [batchsize, 1, coarse.shape[2], coarse.shape[3]])
coarse_orig = coarse_mn.repeat(batchsize, 1,1,1)
coarse_in = torch.cat([coarse_orig,noise], dim = 1).to(device).float()

gen_out = G(coarse_in).cpu().detach()
for i in range(64):
    noise = torch.normal(0,1,size = [batchsize, 1, coarse.shape[2], coarse.shape[3]])
    coarse_in = torch.cat([coarse_orig,noise], dim = 1).to(device).float()
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen

ks_res_ncov = np.empty([128,128])
for i in range(128):
  for j in range(128):
    gen = gen_out[0:500,0,i,j]
    target = fine[:,0,i,j]
    test = scipy.stats.kstest(gen,target)
    ks_res_ncov[i,j] = test.statistic

ks_ncov = ks_res_ncov.flatten()
plt.close()
plt.imshow(ks_res_ncov)
plt.show()

t1 = ks_ncov
t2 = t1[t1 > 0.05]
print(t2.size/t1.size)

xp = 5
yp = 120
gen_ncov = gen_out[:,0,xp,yp].flatten().numpy()
real_marg = fine[:,0,xp,yp].flatten().numpy()
plt.close()
sns.kdeplot(gen_ncov,label = "Gen")
sns.kdeplot(real_marg,label = "Real")
plt.legend()
plt.show()



quant = 0.5
gen_quantile = torch.quantile(gen_out, quant, dim = 0)
gen_quantile = torch.squeeze(gen_quantile)
fine_quantile = torch.quantile(fine[0:gen_out.shape[0],...],quant, dim = 0)
fine_quantile = torch.squeeze(fine_quantile)
minval = min(torch.min(gen_quantile), torch.min(fine_quantile))
maxval = max(torch.max(gen_quantile), torch.max(fine_quantile))

plt.imshow(gen_quantile, vmin=minval, vmax=maxval)
plt.imshow(fine_quantile, vmin=minval, vmax=maxval)


################noise covariates
###specify generator
gen_location = "Generators/Synthetic/NoiseCov_LR/Generator_210/"
G = mlflow.pytorch.load_model(gen_location)

###specify data
data_folder = "../Data/SynthReg/"
coarse_train = np.load(data_folder+"coarse_stochastic.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_stochastic.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]


coarse_mn = torch.mean(coarse, dim = 0)
plt.imshow(coarse_mn[0,...])
plt.show()
batchsize = 8
noise = torch.normal(0,1,size = [batchsize, 1, coarse.shape[2], coarse.shape[3]])
coarse_orig = coarse_mn.repeat(batchsize, 1,1,1)
coarse_in = torch.cat([coarse_orig,noise], dim = 1).to(device).float()

gen_out = G(coarse_in).cpu().detach()
for i in range(32):
    noise = torch.normal(0,1,size = [batchsize, 1, coarse.shape[2], coarse.shape[3]])
    coarse_in = torch.cat([coarse_orig,noise], dim = 1).to(device).float()
    fine_gen = G(coarse_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
quant = 0.5
gen_quantile = torch.quantile(gen_out, quant, dim = 0)
gen_quantile = torch.squeeze(gen_quantile)
fine_quantile = torch.quantile(fine[0:gen_out.shape[0],...],quant, dim = 0)
fine_quantile = torch.squeeze(fine_quantile)
minval = min(torch.min(gen_quantile), torch.min(fine_quantile))
maxval = max(torch.max(gen_quantile), torch.max(fine_quantile))

plt.imshow(gen_quantile, vmin=minval, vmax=maxval)
plt.imshow(fine_quantile, vmin=minval, vmax=maxval)


plt.close()

plt.show()

torch.max(fine_quantile)

plt.close()
plt.imshow(fine_quantile - gen_quantile)
plt.show()
##example KDEs

import seaborn as sns
xp = 5
yp = 5
samp1 = gen_out[:,0,xp,yp].flatten()
samp2 = fine[0:264,0,xp,yp].flatten()
plt.close()
sns.kdeplot(samp1,label = "Gen")
sns.kdeplot(samp2,label = "Real")
plt.legend()
plt.show()

##examples images
import xarray as xr
batchsize = 4
data_folder = "./ds_wind/"
cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1)
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1)
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy())

invariant_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()

sample = 1000
coarse_in = coarse[sample,...]
coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()

G = mlflow.pytorch.load_model("./Generators/wind/just_crps/")
# gen_out = G(coarse_in, invariant_in).cpu().detach()
# #gen_out = G(coarse_in).cpu().detach()

# plt.imshow(fine[sample,0,...])
# plt.show()

# figure, axis = plt.subplots(2, 2)
  
# # For Sine Function
# axis[0, 0].imshow(gen_out[0,0,...])
# axis[0, 1].imshow(gen_out[1,0,...])
# axis[1, 0].imshow(gen_out[2,0,...])
# axis[1, 1].imshow(gen_out[3,0,...])
  
# Combine all the operations and display

sample = 1042
coarse_in = coarse[sample,...]
coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()
invariant_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()

gen_out = G(coarse_in, invariant_in).cpu().detach()
for i in range(24):
    fine_gen = G(coarse_in, invariant_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen
    
gen_var = torch.std(gen_out, dim = 0)
plt.close()
plt.imshow(gen_var[0,...])
plt.savefig("Example_var_nocrps.png", dpi=300)
for i in range(30):
    plt.imshow(gen_out[i,0,...])
    plt.savefig("Realisation_"+str(i)+".png",dpi = 300)
    plt.close()
    
plt.imshow(fine[1042,0,...])
plt.show()


zonal = zonal.to(device)
torch.quantile(zonal, torch.tensor([0.001,0.5,0.99]).to(device))
##############
###RALSD
import scipy
def ralsd(img,real):
    # Input data
    ynew = img # Generated data
    npix = ynew.shape[-1] # Shape of image in one dimension

    # Define the wavenumbers basically
    kfreq = np.fft.fftfreq(npix) * npix 
    kfreq2D = np.meshgrid(kfreq, kfreq) 
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2) # Magnitude of wavenumber/vector
    knrm = knrm.flatten() 

    # Computes the fourier transform and returns the amplitudes
    def calculate_2dft(image):
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image)**2
        return fourier_amplitudes.flatten()

    powers = []
    for i in range(ynew.shape[0]):
        wind_2d = calculate_2dft(ynew[i, ...])
        wind_real = calculate_2dft(real[i,...])
        kbins = np.arange(0.5, npix//2+1, 1.) # Bins to average the spectra
        # kvals = 0.5 * (kbins[1:] + kbins[:-1]) # "Interpolate" at the bin center
        # This ends up computing the radial average (kinda weirdly because knrm and wind_2d are flat, but
        # unique knrm bins correspond to different radii (calculated above)
        Abins, _, _ = scipy.stats.binned_statistic(knrm, wind_2d, statistic = "mean", bins = kbins) 
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
        
        # now for ground truth
        Abins_R, _, _ = scipy.stats.binned_statistic(knrm, wind_real, statistic = "mean", bins = kbins) 
        Abins_R *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
        Abins_stand = Abins/Abins_R
        # Add to a list -- each element is a RASPD
        powers.append(Abins_stand)
    return(powers)

class NetCDFSR:
    """Data loader from torch.Tensors"""
    def __init__(
        self,
        coarse: torch.Tensor,
        fine: torch.Tensor,
        invarient: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Initializes the dataset.
        Returns:
            torch.Tensor: The dataset batches.
        """
        self.fine = fine
        self.coarse = coarse
        self.invarient = invarient

    def __len__(self):
        return self.fine.size(0)

    def __getitem__(self, idx):
        noise_inv = False
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #fine_ = self.fine[idx, ...]
        #fine_ = torch.cat([self.fine[idx,...],self.invarient],0)
        #coarse_ = torch.cat([self.coarse[idx, ...],self.invarient],0)
        fine_ = self.fine[idx,...]
        coarse_ = self.coarse[idx,...]
        if(self.invarient is None):
            return coarse_, fine_, -1
        else:
            if(noise_inv):
                noise_inv = torch.normal(0,1, size = [1,self.invarient.shape[1],self.invarient.shape[2]],device=self.device)
                invarient_ = torch.cat([self.invarient,noise_inv],0)
            else:
                invarient_ = self.invarient
            return coarse_, fine_, invarient_



def calc_ralsd(G,dataloader):
    torch.cuda.empty_cache()
    RALSD = []
    gen_img = []
    real_img = []
    for i, data in enumerate(dataloader):
        if(i > 100):
            break
        ##print("running batch ", i)
        #torch.cuda.empty_cache()
        out = G(data[0].to("cuda:0").float())
        #out = G(data[0].to("cuda:0").float())
        #print(data[1][:,0,...].size())
        real = data[1][:,0,...].cpu().detach()
        zonal = out[:,0,...].cpu().detach()
        gen_img.append(zonal)
        real_img.append(real)
        
        distMetric = ralsd(zonal.numpy(),real.numpy())
        t1 = np.mean(distMetric,axis = 0)
        RALSD.append(t1)
        #print("RALSD: ",log_dist)
        del data
        del out
        del real
    #gen = torch.stack(gen_img,0)
    #real = torch.stack(real_img,0)
    return(RALSD)

models = ["Generators/Synthetic/New/Generator_250/","Generators/Synthetic/NoiseCov_LR/Generator_210/"]
modNm = ['Inject', 'Covariate']

data_folder = "../Data/SynthReg/"
coarse_train = np.load(data_folder+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]

noise = torch.normal(0,1,size = [coarse.shape[0], 1, coarse.shape[2], coarse.shape[3]])
coarse_noise = torch.cat([coarse,noise], dim = 1)
ds_inject = NetCDFSR(coarse, fine, None, device=device)
ds_covariate = NetCDFSR(coarse_noise, fine, None, device=device)

datasets = [ds_inject,ds_covariate] ##datasets for each model
pred_num = [0,0]

res = dict()
for i in range(len(models)):
    print("Analysing model",modNm[i])
    G = mlflow.pytorch.load_model(models[i])
    dataloader = torch.utils.data.DataLoader(
        dataset=datasets[i], batch_size=4, shuffle=True
    )
    
    RALSD = calc_ralsd(G, dataloader)
    ral = np.mean(RALSD,axis = 0)
    sdral = np.std(RALSD,axis = 0)
    res[modNm[i]] = np.column_stack((ral,sdral))


cols = ['orange','blue']
for i,nm in enumerate(modNm):
    plt.plot(res[nm][:,0], label = nm, color = cols[i])
    plt.fill_between(range(64),res[nm][:,0]+res[nm][:,1],res[nm][:,0]-res[nm][:,1], alpha = 0.1, color = cols[i])
plt.hlines(y = 1, xmin=0, xmax=64, color = "black")
plt.xlabel("Frequency Band")
plt.ylabel("Standardised Amplitude")
plt.legend()
plt.savefig('RALSD_Synthetic.png',dpi = 600)

##############################################
###########DEM weights
def calc_rankhist(coarse, fine, invariant, G):
    batchsize = 4
    random = torch.randint(0, 1000, (30, ))
    invariant_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()

    allrank = []
    #mp = torch.nn.MaxPool2d(8)
    for sample in random:
        print("Processing",sample)
        coarse_in = coarse[sample,...]
        coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()

        gen_out = G(coarse_in, invariant_in).cpu().detach()
        for i in range(24):
            fine_gen = G(coarse_in, invariant_in)
            gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
            del fine_gen
        
        real = torch.squeeze((fine[sample,0,...]))
        fake = torch.squeeze((gen_out[:,0,...]))

        rankvals = []
        for i in range(128):
            for j in range(128):
                obs = real[i,j].numpy()
                ensemble = fake[:,i,j].flatten().numpy()
                allvals = np.append(ensemble,obs)
                rankvals.append(sorted(allvals).index(obs))

        allrank.append(rankvals)
    l2 = np.array([item for sub in allrank for item in sub])
    return(l2)

### CRPS temperature
import xarray as xr
batchsize = 4
data_folder = "../Data/ds_temp/"
cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1)
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1)
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy())

G = mlflow.pytorch.load_model("Generators\crps\increaseweight\Generator_500")

res_crps = calc_rankhist(coarse, fine, invariant, G)
plt.hist(res_crps, bins=10, density=True)
plt.ylim((0,0.055))
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistCRPS.png',dpi = 600)

##W1
invar = np.load("../Data/SynthDEM/dem_crop.npy")
invariant = torch.from_numpy(invar)[None,...]
data_folder = "../Data/SynthDEM/W1/"
coarse_train = np.load(data_folder+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]
G = mlflow.pytorch.load_model("Generators\Synthetic\SynthDEM\W1")

plt.imshow(fine[42,0,...])
plt.savefig('ExampleW1.png',dpi = 600)

res_w1 = calc_rankhist(coarse, fine, invariant, G)

plt.hist(res_w1, bins=10, density=True)
plt.ylim((0,0.025))
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistW1.png',dpi = 600)

data_folder = "../Data/SynthDEM/W5/"
coarse_train = np.load(data_folder+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]
G = mlflow.pytorch.load_model("Generators\Synthetic\SynthDEM\W5")

plt.imshow(fine[42,0,...])
plt.savefig('ExampleW5.png',dpi = 600)

res_w5 = calc_rankhist(coarse, fine, invariant, G)
plt.hist(res_w5, bins=10, density=True)
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistW5.png',dpi = 600)


data_folder = "../Data/SynthDEM/W10/"
coarse_train = np.load(data_folder+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]
G = mlflow.pytorch.load_model("Generators\Synthetic\SynthDEM\W10")

batchsize = 4
coarse_mn = torch.mean(coarse, dim = 0)
coarse_in = coarse_mn.repeat(batchsize, 1,1,1).to(device).float()
invariant_in = invariant.repeat(batchsize, 1,1,1).to(device).float()

gen_out = G(coarse_in, invariant_in).cpu().detach()
for i in range(32):
    fine_gen = G(coarse_in, invariant_in)
    gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
    del fine_gen

def crps_empirical2(pred, truth):
    """
    Computes negative Continuous Ranked Probability Score CRPS* [1] between a
    set of samples ``pred`` and true data ``truth``. This uses an ``n log(n)``
    time algorithm to compute a quantity equal that would naively have
    complexity quadratic in the number of samples ``n``::

        CRPS* = E|pred - truth| - 1/2 E|pred - pred'|
              = (pred - truth).abs().mean(0)
              - (pred - pred.unsqueeze(1)).abs().mean([0, 1]) / 2

    Note that for a single sample this reduces to absolute error.

    **References**

    [1] Tilmann Gneiting, Adrian E. Raftery (2007)
        `Strictly Proper Scoring Rules, Prediction, and Estimation`
        https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    :param torch.Tensor pred: A set of sample predictions batched on rightmost dim.
        This should have shape ``(num_samples,) + truth.shape``.
    :param torch.Tensor truth: A tensor of true observations.
    :return: A tensor of shape ``truth.shape``.
    :rtype: torch.Tensor
    """
    if pred.shape[1:] != (1,) * (pred.dim() - truth.dim() - 1) + truth.shape:
        raise ValueError(
            "Expected pred to have one extra sample dim on left. "
            "Actual shapes: {} versus {}".format(pred.shape, truth.shape)
        )
    opts = dict(device=pred.device, dtype=pred.dtype)
    num_samples = pred.size(0)
    if num_samples == 1:
        return (pred[0] - truth).abs()

    pred = pred.sort(dim=0).values
    diff = pred[1:] - pred[:-1]
    weight = torch.arange(1, num_samples, **opts) * torch.arange(
        num_samples - 1, 0, -1, **opts
    )
    weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))

    return (pred - truth).abs().mean(0) - (diff * weight).sum(0) / num_samples**2

crps_res = crps_empirical2(gen_out[:,0,...], fine[42,0,...])
plt.imshow(crps_res)
import seaborn as sns
xp = 5
yp = 125
samp1 = gen_out[:,0,xp,yp].flatten()
samp2 = fine[0:132,0,xp,yp].flatten()
plt.close()
sns.kdeplot(samp1,label = "Gen")
sns.kdeplot(samp2,label = "Real")
plt.legend()

fine_out = fine[0:132,0,...]
fine_mn = torch.mean(fine_out, dim = 0)
fine_mn.shape

gen_mn = torch.mean(gen_out[:,0,...], dim = 0)
gen_mn.shape

diff = torch.abs(fine_mn - gen_mn)
plt.imshow(diff)
diff2 = diff/fine_mn
plt.imshow(diff2)
plt.hist(diff2.flatten())

plt.imshow(fine[42,0,...])
plt.savefig('ExampleW10.png',dpi = 600)



res_w10 = calc_rankhist(coarse, fine, invariant, G)
plt.hist(res_w10, bins=10, density=True)
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistW10.png',dpi = 600)

###rankhist synthetic
def calc_rankhist_synth(coarse, fine, G):
    batchsize = 4
    random = torch.randint(0, 1000, (30, ))

    allrank = []
    #mp = torch.nn.MaxPool2d(8)
    for sample in random:
        print("Processing",sample)
        coarse_in = coarse[sample,...]
        coarse_in = coarse_in.unsqueeze(0).repeat(int(batchsize),1,1,1).to(device).float()

        gen_out = G(coarse_in).cpu().detach()
        for i in range(24):
            fine_gen = G(coarse_in)
            gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
            del fine_gen
        
        real = torch.squeeze((fine[sample,...]))
        fake = torch.squeeze((gen_out[:,...]))

        rankvals = []
        for i in range(128):
            for j in range(128):
                obs = real[i,j].numpy()
                ensemble = fake[:,i,j].flatten().numpy()
                allvals = np.append(ensemble,obs)
                rankvals.append(sorted(allvals).index(obs))

        allrank.append(rankvals)
    l2 = np.array([item for sub in allrank for item in sub])
    return(l2)

gen_location = "Generators/Synthetic/New/Generator_250/"
G = mlflow.pytorch.load_model(gen_location)

###specify data
data_folder = "../Data/SynthReg/"
coarse_train = np.load(data_folder+"coarse_val_reg.npy")
coarse_train = np.swapaxes(coarse_train, 0, 2)
fine_train = np.load(data_folder+"fine_val_reg.npy")
fine_train = np.swapaxes(fine_train, 0, 2)
coarse = torch.from_numpy(coarse_train)[:,None,...]
fine = torch.from_numpy(fine_train)[:,None,...]

res = calc_rankhist_synth(coarse, fine, G)

###rankhist for wind data
import xarray as xr
batchsize = 4
data_folder = "../Data/ds_wind/"
cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1)
noise = torch.normal(0,1,size = [coarse.shape[0], 1, coarse.shape[2], coarse.shape[3]])
coarse_noise = torch.cat([coarse,noise], dim = 1)
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1)
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy())

invariant_in = invariant.repeat(int(batchsize),1,1,1).to(device).float()

G = mlflow.pytorch.load_model("Generators\\wind\\noise_covar_LR")

test = G(coarse_noise[0:4,...].to(device), invariant_in.to(device))
plt.imshow(test[0,0,...].cpu().detach())

res = calc_rankhist(coarse_noise, fine, invariant, G)
plt.hist(res, bins=10, density=True)
plt.ylim((0,0.055))
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistWind_Covar.png',dpi = 600)

G = mlflow.pytorch.load_model("Generators\\wind\\noise_inject")

test = G(coarse[1000:1004,...].to(device), invariant_in.to(device))
plt.imshow(test[0,1,...].cpu().detach())
plt.savefig('ExampleWind2.png',dpi = 600)

res_inject = calc_rankhist(coarse, fine, invariant, G)
plt.hist(res_inject, bins=10, density=True)
plt.ylim((0,0.055))
plt.xlabel("Rank")
plt.ylabel("Density")
plt.savefig('RankHistWind_Inject.png',dpi = 600)
