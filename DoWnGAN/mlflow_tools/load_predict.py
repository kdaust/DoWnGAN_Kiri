#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:28:05 2022

@author: kiridaust
"""

import mlflow
import mlflow.pytorch

from xarray.core.dataset import Dataset
import xarray as xr
import netCDF4
import numpy as np

import torch
import matplotlib.pyplot as plt
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# HR = torch.load("C:/Users/kirid/Desktop/Masters/GAN_Results/Validation/HR_topo_GEN.pt")
# LR = torch.load("C:/Users/kirid/Desktop/Masters/GAN_Results/Validation/LR_topo_GEN.pt")

# num = 7
# plt.imshow(HR[num,1,...])
# plt.imshow(LR[num,1,...])
# np.savetxt("HR_Generate.csv", HR[7,0,...],delimiter=',')
# np.savetxt("LR_Generate.csv", LR[7,0,...],delimiter=',')


device = torch.device("cuda:0")

import scipy.stats

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

class NetCDFSR(Dataset):
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

        if torch.is_tensor(idx):
            idx = idx.tolist()
        fine_ = self.fine[idx,...]
        coarse_ = self.coarse[idx,...]
        invarient_ = self.invarient
        return coarse_, fine_, invarient_


def calc_ralsd(G,dataloader):
    torch.cuda.empty_cache()
    RALSD = []
    for i, data in enumerate(dataloader):
        if(i > 100):
            break
        #coarse = data[0].to(device)
        #print(data[0])
        #inv = data[2].to(device)
        print("running batch ", i)
        #torch.cuda.empty_cache()
        out = G(data[0],data[2])
        #print(data[1][:,0,...].size())
        real = data[1][:,0,...].cpu().detach().numpy()
        zonal = out[:,0,...].cpu().detach().numpy()
        merid = out[:,1,...].cpu().detach().numpy()
        #real = data[1].cpu().detach().numpy()
        #fake = out.cpu().detach().numpy()
        # zquant = np.quantile(zonal, qval, axis = (1,2))
        # mquant = np.quantile(merid, qval, axis = (1,2))
        # u99 = np.append(u99,zquant)
        # v99 = np.append(v99, mquant)
        
        distMetric = ralsd(zonal,real)
        t1 = np.mean(distMetric,axis = 0)
        RALSD.append(t1)
        #print("RALSD: ",log_dist)
        del data
        del out
        del real
    return(RALSD)

mod_noise = "/media/data/mlflow_exp/4/b56771fd635d414d9586dc72019237be/artifacts/Generator/Generator_420"
G = mlflow.pytorch.load_model(mod_noise)

cond_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/coarse_validation.nc", engine="netcdf4")
fine_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/fine_validation.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
invariant = xr.open_dataset("~/Masters/Data/temperature/DEM_Use.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy().squeeze(0)).to(device).float()

time_slice = 3600
noise_gen = G(coarse[time_slice,...],invariant)
for i in range(100):
    print("Generating",i)
    temp = G(coarse[time_slice,...],invariant)
    noise_gen = torch.cat((noise_gen,temp),1)
    
torch.save(noise_gen.cpu().detach(),"ExampleNoiseGen.pt")
print("Done")


mod_smallres = "/media/data/mlflow_exp/4/17d56bf78c714b18a44ae2f5116d0d15/artifacts/Generator/Generator_370"
mod_bigres = "/media/data/mlflow_exp/4/e286bcc85c8540be938305892ae3ab4c/artifacts/Generator/Generator_370"
cond_fields = xr.open_dataset("~/Masters/Data/PredictTest/coarse_val_sht.nc", engine="netcdf4")
fine_fields = xr.open_dataset("~/Masters/Data/PredictTest/fine_val_sht.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()

G = mlflow.pytorch.load_model(mod_bigres)
invariant = xr.open_dataset("~/Masters/Data/PredictTest/DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy().squeeze(0)).to(device).float()

ds = NetCDFSR(coarse, fine, invariant, device=device)
dataloader = torch.utils.data.DataLoader(
    dataset=ds, batch_size=16, shuffle=False
)
HR_RALSD = calc_ralsd(G, dataloader)
test = next(iter(dataloader))
HR_gen = G(test[0],test[2])
plt.imshow(HR_gen[1,0,...].cpu().detach())

G = mlflow.pytorch.load_model(mod_smallres)
invariant = xr.open_dataset("~/Masters/Data/PredictTest/DEM_Coarse.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy().squeeze(0)).to(device).float()

ds = NetCDFSR(coarse, fine, invariant, device=device)
dataloader = torch.utils.data.DataLoader(
    dataset=ds, batch_size=16, shuffle=False
)
LR_RALSD = calc_ralsd(G, dataloader)
test = next(iter(dataloader))
LR_gen = G(test[0],test[2])

plt.imshow(HR_gen[7,0,...].cpu().detach())
plt.imshow(LR_gen[7,0,...].cpu().detach())
torch.save(HR_gen.cpu().detach(),"HR_topo_GEN.pt")
torch.save(LR_gen.cpu().detach(),"LR_topo_GEN.pt")


LRral = np.mean(LR_RALSD,axis = 0)
LRsd = np.std(LR_RALSD,axis = 0)
HRral = np.mean(HR_RALSD,axis = 0)
HRsd = np.std(HR_RALSD,axis = 0)

plt.plot(HRral, label = "HighRes")
plt.plot(LRral, label = "LowRes")
plt.fill_between(range(64), HRral+HRsd,HRral-HRsd, alpha = .1)
plt.fill_between(range(64), LRral+LRsd,LRral-LRsd, alpha = .1)
plt.xlabel("Frequency Group")
plt.ylabel("Amplitude proportion")
plt.legend()





LR = np.array(LR_RALSD)
np.savetxt("/home/kiridaust/Masters/DoWnGAN_Kiri/Results/LR_RALSD.csv",LR,delimiter = ',')

HR = np.array(HR_RALSD)
np.savetxt("/home/kiridaust/Masters/DoWnGAN_Kiri/Results/HR_RALSD.csv",HR,delimiter = ',')
torch.save(zonal, "/home/kiridaust/Masters/DoWnGAN_Kiri/Results/LRTopo_WindGen.pt")


# plt.boxplot((HRU,LRU),notch=True,labels=("HighRes","LowRes"))
# plt.title("1% quantile of zonal wind speed")

# plt.hist(u99,25)
# plt.title("99th Qunatile of Jan Zonal Wind Speed (LRT)")

# plt.hist(v99,25)
# plt.title("99th Qunatile of Jan Meridional Wind Speed (LRT)")

# print("Zonal = ",np.mean(u99))
# print("Merid = ",np.mean(v99))




