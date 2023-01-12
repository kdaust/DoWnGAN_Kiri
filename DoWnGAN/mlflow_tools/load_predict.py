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

device = torch.device("cuda:0")

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

        #fine_ = self.fine[idx, ...]
        #fine_ = torch.cat([self.fine[idx,...],self.invarient],0)
        #coarse_ = torch.cat([self.coarse[idx, ...],self.invarient],0)
        fine_ = self.fine[idx,...]
        coarse_ = self.coarse[idx,...]
        invarient_ = self.invarient
        # print(self.fine[idx,...])
        # print(self.invarient)
        # print(fine_)
        return coarse_, fine_, invarient_

G = mlflow.pytorch.load_model("/media/data/mlflow_exp/4/e286bcc85c8540be938305892ae3ab4c/artifacts/Generator/Generator_410")

cond_fields = xr.open_dataset("~/Masters/Data/PredictTest/coarse_val_sht.nc", engine="netcdf4")
fine_fields = xr.open_dataset("~/Masters/Data/PredictTest/fine_val_sht.nc", engine="netcdf4")
invariant = xr.open_dataset("~/Masters/Data/PredictTest/DEM_Crop.nc", engine = "netcdf4")

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
i = 0
u99 = np.array(0)
v99 = np.array(0)
qval = 0.99
for data in dataloader:
    #coarse = data[0].to(device)
    #print(data[0])
    #inv = data[2].to(device)
    print("running batch ", i)
    #torch.cuda.empty_cache()
    out = G(data[0],data[2])
    zonal = out[:,0,...].cpu().detach().numpy()
    merid = out[:,1,...].cpu().detach().numpy()
    zquant = np.quantile(zonal, qval, axis = (1,2))
    mquant = np.quantile(merid, qval, axis = (1,2))
    u99 = np.append(u99,zquant)
    v99 = np.append(v99, mquant)
    del data
    del out
    i = i+1
    
print("Zonal = ",u99)
print("Merid = ",v99)


