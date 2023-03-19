# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 17:51:22 2023

@author: kirid
"""
import mlflow
import mlflow.pytorch
from xarray.core.dataset import Dataset
import xarray as xr
import netCDF4
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pysteps as ps
device = torch.device("cuda:0")

data_folder = "/home/kiridaust/Masters/Data/processed_data/ds_wind/"
fine_fields = xr.open_dataset(data_folder + "fine_validation.nc", engine="netcdf4")
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1)
real = fine[508,0,...]

gen = torch.load("/home/kiridaust/Masters/DoWnGAN_Kiri/Generated_Stochastic/Wind_Generated_958a0ed957424681843fe9ebc0a279e7.pt")
fake = gen[:,0,...]
rhist = ps.verification.ensscores.rankhist(fake.numpy(),real.numpy(),None)
plt.plot(rhist)

rankvals = []
for i in range(128):
    for j in range(128):
        obs = real[i,j].numpy()
        ensemble = fake[:,i,j].flatten().numpy()
        allvals = np.append(ensemble,obs)
        rankvals.append(sorted(allvals).index(obs))
        
plt.hist(rankvals)


plt.imshow(real)
plt.imshow(gen[1,0,...])
plt.imshow(gen[2,0,...])
plt.imshow(gen[3,0,...])
plt.imshow(gen[4,0,...])

mod_noise = "/media/data/mlflow_exp/4/ccfb1a914fcd43c58aae2ef2c27a54ef/artifacts/Generator/Generator_500"
G = mlflow.pytorch.load_model(mod_noise)

data_folder = "/home/kiridaust/Masters/Data/ToyDataSet/"
coarse_val = np.load(data_folder+"coarse_val_toydat.npy")
coarse_val = np.swapaxes(coarse_val, 0, 2)
fine_val = np.load(data_folder+"fine_val_toydat.npy")
fine_val = np.swapaxes(fine_val, 0, 2)

fine_in = torch.from_numpy(fine_val)[:,None,...]
coarse_in = torch.from_numpy(coarse_val)[:,None,...].to(device).float()

plt.imshow(coarse_in[32,0,...].cpu())

coarse_sht = coarse_in[0:50,...]
fine_gen = G(coarse_sht)
fine_gent1 = fine_gen.cpu().detach()
del fine_gen
fine_gen = G(coarse_sht)
fine_gent2 = fine_gen.cpu().detach()
del fine_gen
fine_gen = G(coarse_sht)
fine_gent3 = fine_gen.cpu().detach()
del fine_gen
#plt.imshow(fine_gent3[5,0,...])
fine_gen = torch.cat([fine_gent1,fine_gent2,fine_gent3],0)


sns.set_style('white')
xp = 100
yp = 32
sample = fine_gen[:,0,xp,yp].flatten()
sns.kdeplot(sample,label = "Generated")
real = fine_in[0:150,0,xp,yp].flatten()
sns.kdeplot(real,label = "real")

plt.imshow(fine_gen[1,0,...])
plt.imshow(fine_gen[2,0,...])
plt.imshow(fine_gen[130,0,...])

noise_inside = torch.load("C:/Users/kirid/Desktop/Masters/DoWnGAN_Kiri/Wind_Noise_Inside.pt")
plt.imshow(noise_inside[132,0,...])

noise_cov = torch.load("C:/Users/kirid/Desktop/Masters/DoWnGAN_Kiri/Wind_Noise_Input.pt")
plt.imshow(noise_cov[132,0,...])
plt.imshow(noise_cov[1,0,...])

fine_in = np.load("C:/Users/kirid/Desktop/Masters/DoWnGAN_Kiri/fine_val_toydat.npy")
fine_in = torch.from_numpy(np.swapaxes(fine_in,0,2))[:,None,...]

import scipy as sp
wass_dist = np.zeros([128,128])
for x in range(128):
    for y in range(128):
        fake_dist = fine_gen[:,0,x,y].flatten()
        true_dist = fine_in[0:150,0,x,y].flatten()
        wass_dist[x,y] = sp.stats.wasserstein_distance(true_dist,fake_dist)
        
plt.imshow(wass_dist)

for i in range(8):
    plt.figure()
    plt.imshow(noise_inside[i,0,...])
    plt.show()

img_std = torch.std(noise_cov,dim=0)
plt.imshow(img_std[0,...])
plt.imshow(img_std[1,...])

sns.set_style('white')
xp = 64
yp = 64
sample = noise_cov[:,0,xp,yp].flatten()
sns.kdeplot(sample,label = "Generated")
real = fine_in[:,:,xp,yp].flatten()
sns.kdeplot(real,label = "real")

xp = 5
yp = 5
sample = fine_gen[:,0,xp,yp].flatten()
sns.kdeplot(sample,label = "Generated")
real = fine_in[:,:,xp,yp].flatten()
sns.kdeplot(real,label = "real")
print("Done")




# class NetCDFSR(Dataset):
#     """Data loader from torch.Tensors"""
#     def __init__(
#         self,
#         coarse: torch.Tensor,
#         fine: torch.Tensor,
#         invarient: torch.Tensor,
#         device: torch.device,
#     ) -> torch.Tensor:
#         """
#         Initializes the dataset.
#         Returns:
#             torch.Tensor: The dataset batches.
#         """
#         self.fine = fine
#         self.coarse = coarse
#         self.invarient = invarient

#     def __len__(self):
#         return self.fine.size(0)

#     def __getitem__(self, idx):

#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         fine_ = self.fine[idx,...]
#         coarse_ = self.coarse[idx,...]
#         invarient_ = self.invarient
#         return coarse_, fine_, invarient_

# mod_noise = "/media/data/mlflow_exp/4/b56771fd635d414d9586dc72019237be/artifacts/Generator/Generator_420"
# G = mlflow.pytorch.load_model(mod_noise)

# cond_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/coarse_validation.nc", engine="netcdf4")
# fine_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/fine_validation.nc", engine="netcdf4")
# coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
# fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
# invariant = xr.open_dataset("~/Masters/Data/temperature/DEM_Use.nc", engine = "netcdf4")
# invariant = torch.from_numpy(invariant.to_array().to_numpy().squeeze(0)).to(device).float()

# ds = NetCDFSR(coarse, fine, invariant, device=device)
# dataloader = torch.utils.data.DataLoader(
#     dataset=ds, batch_size=16, shuffle=False
# )

# data_in = None
# for i in range(225):
#     data_in = next(iter(dataloader))

# temp = G(data_in[0],data_in[2])
# noise_gen = temp[0,...].cpu().detach()
# for i in range(100):
#     torch.cuda.empty_cache()
#     print("Generating",i)
#     temp = G(data_in[0],data_in[2])
#     noise_gen = torch.cat((noise_gen,temp[0,...].cpu().detach()),0)
#     del temp

# print(noise_gen.size())
# torch.save(noise_gen,"ExampleNoiseGen.pt")
# print("Done")


# ############
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
gen = torch.load("NoiseGens.pt")
plt.imshow(gen[0])
plt.imshow(gen[1])
plt.imshow(gen[50])

# np.save("GAN_Noise_Res.npy",gen)

# tfine = np.load("C:/Users/kirid/Desktop/Masters/ToyDataSet/fine_train.npy")
# t1 = np.swapaxes(tfine, 0, 2)
# plt.imshow(t1[0,...])

