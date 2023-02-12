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