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
        fine_ = self.fine[idx,...]
        coarse_ = self.coarse[idx,...]
        invarient_ = self.invarient
        return coarse_, fine_, invarient_

mod_noise = "/media/data/mlflow_exp/4/b56771fd635d414d9586dc72019237be/artifacts/Generator/Generator_420"
G = mlflow.pytorch.load_model(mod_noise)

cond_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/coarse_validation.nc", engine="netcdf4")
fine_fields = xr.open_dataset("~/Masters/Data/processed_data/ds_humid/fine_validation.nc", engine="netcdf4")
coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine = torch.from_numpy(fine_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
invariant = xr.open_dataset("~/Masters/Data/temperature/DEM_Use.nc", engine = "netcdf4")
invariant = torch.from_numpy(invariant.to_array().to_numpy().squeeze(0)).to(device).float()

ds = NetCDFSR(coarse, fine, invariant, device=device)
dataloader = torch.utils.data.DataLoader(
    dataset=ds, batch_size=16, shuffle=False
)

data_in = None
for i in range(225):
    data_in = next(iter(dataloader))

temp = G(data_in[0],data_in[2])
noise_gen = temp[0,...]
for i in range(100):
    print("Generating",i)
    temp = G(data_in[0],data_in[2])
    noise_gen = torch.cat((noise_gen,temp[0,...]),1)
    
torch.save(noise_gen.cpu().detach(),"ExampleNoiseGen.pt")
print("Done")