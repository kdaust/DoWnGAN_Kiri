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
import pickle
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
        invarient_ = self.invarient[idx,...]
        #invarient_ = self.invarient
        return coarse_, fine_, invarient_


def calc_ralsd(G,dataloader):
    torch.cuda.empty_cache()
    RALSD = []
    for i, data in enumerate(dataloader):
        if(i > 100):
            break
        print("running batch ", i)
        #torch.cuda.empty_cache()
        out = G(data[0],data[2])
        #print(data[1][:,0,...].size())
        real = data[1][:,0,...].cpu().detach().numpy()
        zonal = out[:,0,...].cpu().detach().numpy()
        #merid = out[:,1,...].cpu().detach().numpy()
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


models = ['d4c12d8ef6b84871bc0cb5fd18d638ef','4b906c3c6fe54f09832fcb9f22011f98','d3211ab32ecc4b41a5181c6ebdb3f83f','65e9cd4ba68045bdb79526d0196b654e']
modNm = ['Cov_LR','Cov_Both','Inject_LowCL','Inject_PFS']

data_folder = "/home/kiridaust/Masters/Data/processed_data/ds_wind/"
cond_fields = xr.open_dataset(data_folder + "coarse_validation.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_validation.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy()).to(device).float()
invariant = invariant.repeat(coarse.shape[0],1,1,1)
noise_f = torch.normal(0,1,size = [invariant.shape[0],1,128,128], device=device)
noise_c = torch.normal(0,1,size = [coarse.shape[0], 1, coarse.shape[2],coarse.shape[3]], device=device)
coarse_noise = torch.cat([coarse,noise_c],1)
invariant_noise = torch.cat([invariant,noise_f],1)

ds_nc = NetCDFSR(coarse_noise, fine, invariant, device=device)
ds_nb = NetCDFSR(coarse_noise, fine, invariant_noise, device=device)
ds = NetCDFSR(coarse, fine, invariant, device=device)
datasets = [ds_nc,ds_nb,ds,ds] ##datasets for each model

res = dict()
for i in range(len(models)):
    print("Analysing model",modNm[i])
    mod_noise = "/media/data/mlflow_exp/4/" + models[i] +"/artifacts/Generator/Generator_500"
    G = mlflow.pytorch.load_model(mod_noise)
    dataloader = torch.utils.data.DataLoader(
        dataset=datasets[i], batch_size=16, shuffle=False
    )
    
    RALSD = calc_ralsd(G, dataloader)
    ral = np.mean(RALSD,axis = 0)
    sdral = np.std(RALSD,axis = 0)
    res[modNm[i]] = np.column_stack((ral,sdral))


for nm in modNm:
    plt.plot(res[nm][:,0], label = nm)
    plt.fill_between(range(64),res[nm][:,0]+res[nm][:,1],res[nm][:,0]-res[nm][:,1], alpha = 0.1)
plt.legend()

with open('ralsd_data.pkl','wb') as fp:
    pickle.dump(res,fp)
    print("Done!!!")


#cond_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/coarse_validation.nc", engine="netcdf4")
#fine_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/fine_validation.nc", engine="netcdf4")
#coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
#fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
#invariant = xr.open_dataset("~/Masters/Data/temperature/DEM_Use.nc", engine = "netcdf4")
#invariant = torch.from_numpy(invariant.to_array().to_numpy().squeeze(0)).to(device).float()





