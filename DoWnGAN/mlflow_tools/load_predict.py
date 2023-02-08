#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:28:05 2022

@author: kiridaust
"""

import mlflow
import mlflow.pytorch
import torch
from xarray.core.dataset import Dataset
import xarray as xr
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

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


mod_smallres = "/media/data/mlflow_exp/4/69a19619c17647ae8665266d393008be/artifacts/Generator/Generator_370"
mod_bigres = "/media/data/mlflow_exp/4/1e436bfdac40440690d7ae17b6879598/artifacts/Generator/Generator_340"
G = mlflow.pytorch.load_model(mod_bigres)
#G = mlflow.pytorch.load_model("/media/data/mlflow_exp/4/17d56bf78c714b18a44ae2f5116d0d15/artifacts/Generator/Generator_410")


cond_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/coarse_validation.nc", engine="netcdf4")
fine_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/fine_validation.nc", engine="netcdf4")
invariant = xr.open_dataset("~/Masters/Data/temperature/DEM_Use.nc", engine = "netcdf4")
#invariant = xr.open_dataset("~/Masters/Data/PredictTest/DEM_Coarse.nc", engine = "netcdf4")

coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
invariant = torch.from_numpy(invariant.to_array().to_numpy().squeeze(0)).to(device).float()

ds = NetCDFSR(coarse, fine, invariant, device=device)
dataloader = torch.utils.data.DataLoader(
    dataset=ds, batch_size=16, shuffle=True
)
# coarse_curr = coarse[32,:,0:6,0:6]
# fine2 = fine[:,0:48,0:48]
torch.cuda.empty_cache()
RALSD = []
for i, data in enumerate(dataloader):
    if(i > 60):
        break
    #coarse = data[0].to(device)
    #print(data[0])
    #inv = data[2].to(device)
    print("running batch ", i)
    #torch.cuda.empty_cache()
    out = G(data[0],data[2])
    #print(data[1][:,0,...].size())
    # real = data[1][:,0,...].cpu().detach().numpy()
    # zonal = out[:,0,...].cpu().detach().numpy()
    # merid = out[:,1,...].cpu().detach().numpy()
    real = data[1].cpu().detach().numpy()
    fake = out.cpu().detach().numpy()
    # zquant = np.quantile(zonal, qval, axis = (1,2))
    # mquant = np.quantile(merid, qval, axis = (1,2))
    # u99 = np.append(u99,zquant)
    # v99 = np.append(v99, mquant)
    
    distMetric = ralsd(fake,real)
    t1 = np.mean(distMetric,axis = 0)
    RALSD.append(t1)
    #print("RALSD: ",log_dist)
    del data
    del out
    del real
    #i = i+1
    

LR_RALSD = RALSD.copy()
HR_RALSD = RALSD.copy()

LRral = np.mean(LR_RALSD,axis = 0)
LRsd = np.std(LR_RALSD,axis = 0)
HRral = np.mean(HR_RALSD,axis = 0)
HRsd = np.std(HR_RALSD,axis = 0)


plt.plot(HRral, label = "Noise")
plt.plot(LRral, label = "NoNoise")
plt.fill_between(range(64), HRral+HRsd,HRral-HRsd, alpha = .1)
plt.fill_between(range(64), LRral+LRsd,LRral-LRsd, alpha = .1)
plt.xlabel("Frequency Group")
plt.ylabel("Amplitude proportion")
plt.legend()

# plt.boxplot((HRU,LRU),notch=True,labels=("HighRes","LowRes"))
# plt.title("1% quantile of zonal wind speed")

# plt.hist(u99,25)
# plt.title("99th Qunatile of Jan Zonal Wind Speed (LRT)")

# plt.hist(v99,25)
# plt.title("99th Qunatile of Jan Meridional Wind Speed (LRT)")

# print("Zonal = ",np.mean(u99))
# print("Merid = ",np.mean(v99))




