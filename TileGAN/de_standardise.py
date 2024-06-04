import xarray as xr
import netCDF4
import numpy as np
import torch

data_folder = "/home/kdaust/Masters/ds_wind_full"

coarse_train = xr.open_dataset(data_folder + "coarse_train.nc", engine="netcdf4")
fine_train = xr.open_dataset(data_folder + "fine_train.nc", engine="netcdf4")
coarse_test = xr.open_dataset(data_folder + "coarse_test.nc", engine="netcdf4")
fine_test = xr.open_dataset(data_folder + "fine_test.nc", engine="netcdf4")
invarient = xr.open_dataset(data_folder + "DEM_Crop.nc", engine="netcdf4")

coarse_train = torch.from_numpy(coarse_train.to_array().to_numpy()).transpose(0, 1).to(config.device).float()
fine_train = torch.from_numpy(fine_train.to_array().to_numpy()).transpose(0, 1).to(config.device).float()
coarse_test = torch.from_numpy(coarse_test.to_array().to_numpy()).transpose(0, 1).to(config.device).float()
fine_test = torch.from_numpy(fine_test.to_array().to_numpy()).transpose(0, 1).to(config.device).float()
invarient = torch.from_numpy(invarient.to_array().to_numpy().squeeze(0)).to(config.device).float()