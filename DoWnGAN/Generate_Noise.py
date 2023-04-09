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
device = torch.device("cuda:0")


mod_noise = "/media/data/mlflow_exp/4/44216d42cbd8429e99310844d53633e7/artifacts/Generator/Generator_70"
G = mlflow.pytorch.load_model(mod_noise)
data_folder = "/home/kiridaust/Masters/Data/processed_data/ds_wind/"
#data_folder = "/home/kiridaust/Masters/Data/ToyDataSet/VerticalSep/"
# coarse = np.load(data_folder+"coarse_val.npy")
# coarse = np.swapaxes(coarse, 0, 2)
# coarse = torch.from_numpy(coarse)[:,None,...].to(device).float()
cond_fields = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
fine_fields = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
invariant = xr.open_dataset(data_folder + "DEM_Crop.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy()).to(device).float()



batchsize = 32
invariant = invariant.repeat(batchsize,1,1,1)
# noise_f = torch.normal(0,1,size = [batchsize,1,128,128], device=device)
# invariant = torch.cat([invariant, noise_f], 1)
print(invariant.size())
#coarse_val = np.load(data_folder+"coarse_val_toydat.npy")
#coarse_val = np.swapaxes(coarse_val, 0, 2)
#fine_val = np.load(data_folder+"fine_val_toydat.npy")
#fine_val = np.swapaxes(fine_val, 0, 2)

#fine_in = torch.from_numpy(fine_val)[:,None,...]
#plt.imshow(coarse[508,0,...].cpu())
random = torch.randint(0, 5000, (30, ))

allrank = []
for sample in random:
    print("Processing",sample)
    coarse_in = coarse[sample,...]
    coarse_in = coarse_in.unsqueeze(0).repeat(batchsize,1,1,1)

    gen_out = G(coarse_in,invariant).cpu().detach()
    for i in range(5):
        fine_gen = G(coarse_in,invariant)
        gen_out = torch.cat([gen_out,fine_gen.cpu().detach()],0)
        del fine_gen
    
    real = fine[sample,0,...].cpu().detach()
    fake = gen_out[:,0,...].cpu().detach()
    rankvals = []
    for i in range(128):
        for j in range(128):
            obs = real[i,j].numpy()
            ensemble = fake[:,i,j].flatten().numpy()
            allvals = np.append(ensemble,obs)
            rankvals.append(sorted(allvals).index(obs))
            
    allrank.append(rankvals)
        
l2 = np.array([item for sub in allrank for item in sub])
np.save("Rank_Hist_Data.npy", l2)
# plt.hist(l2)

# #torch.save(gen_out,"Wind_NoiseInject_6884.pt")

# for i in range(5):
#     plt.imshow(gen_out[i,0,...])
#     plt.show()
    

# gen_mean = torch.mean(gen_out,0)
# plt.imshow(gen_mean[1,...])
# plt.imshow(coarse_in[0,...].cpu())
# plt.imshow(fine[sample,0,...].cpu())
# print(coarse_in.size())

# torch.manual_seed(0)
# random = torch.randint(0, 32, (20, ))
# plt.imshow(coarse[9,0,...].cpu())
# fake = G(coarse[0:32,...],invariant)
# sampnum = 31
# plt.imshow(coarse[sampnum,0,...].cpu())
# plt.imshow(fake[sampnum,0,...].cpu().detach())
# plt.imshow(fine[sampnum,0,...].cpu().detach())
# gen_cov = fine_gen
# gen_inject = fine_gen

# plt.rcParams["figure.figsize"] = (7,6)
# plt.imshow(fine_gen[1,0,...])

# plt.savefig('Example_PFS_NoiseInject.svg',bbox_inches = 'tight')
# #plt.imshow(fine_gen[2,0,...])
# #plt.imshow(fine_gen[3,0,...])
# #plt.imshow(fine_gen[42,0,...])


# plt.savefig('RankHist_PFS_NoiseInject.svg', bbox_inches = 'tight', dpi = 600)
# sdres = torch.std(fine_gen,dim = 0)
# plt.imshow(sdres[0,...])

# #torch.save(fine_gen,"Wind_Generated_d4c12d8ef6b84871bc0cb5fd18d638ef.pt")
# #plt.imshow(fine_gen[1,0,...])
# #plt.imshow(fine_gen[56,0,...])
# # plt.imshow(fine_gen[233,0,...])

# xp = 5
# yp = 5
# samp1 = gen_inject[:,0,xp,yp].flatten()
# samp2 = gen_cov[:,0,xp,yp].flatten()
# sns.set_style('whitegrid')
# sns.kdeplot(samp1,label = "Inject")
# sns.kdeplot(samp2,label = "Covar")
# plt.legend()

# sample = fine_gen[42,0,...].flatten()
# sns.set_style('whitegrid')
# sns.kdeplot(sample,bw = 0.5)
#real = fine_in[:,:,xp,yp].flatten()
#myplt = sns.kdeplot(real,bw = 0.5)
#myplt.figure.savefig("ToyDat2_Figure.png")
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
# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# gen = torch.load("Wind_NoiseInject_6884.pt")
# plt.imshow(gen[0,0,...])
# plt.imshow(gen[42,0,...])
# plt.imshow(gen[175,0,...])
# plt.imshow(gen[130,0,...])

# avg = torch.mean(gen,0)
# plt.imshow(avg[0,...])
# hf = gen[3,0,...] - avg[0,]
# plt.imshow(hf)
# test = torch.var(avg,(1,2))

# fig, ax = plt.subplots(1, 3)

# for i,num in enumerate([0,42,175]):
#     ax[i].imshow(gen[num,0,...])
#     ax[i].axis('off')
# fig.show()
# np.save("GAN_Noise_Res.npy",gen)

# tfine = np.load("C:/Users/kirid/Desktop/Masters/ToyDataSet/fine_train.npy")
# t1 = np.swapaxes(tfine, 0, 2)
# plt.imshow(t1[0,...])

