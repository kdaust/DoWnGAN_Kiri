#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:28:05 2022

@author: kiridaust
"""

import csv
import mlflow
import mlflow.pytorch
import torch
from xarray.core import dataset
from xarray.core.dataset import Dataset
import xarray as xr
import netCDF4
device = torch.device("cuda:0")

G = mlflow.pytorch.load_model("/media/data/mlflow_exp/4/995fb1e0ae364207b9872d20a5fab639/artifacts/Generator/Generator_200")

cond_fields = xr.open_dataset("~/Masters/Data/PredictTest/predict_crop.nc", engine="netcdf4")
invariant = xr.open_dataset("~/Masters/Data/PredictTest/DEM_Crop.nc", engine = "netcdf4")

coarse = torch.from_numpy(cond_fields.to_array().to_numpy()).transpose(0, 1).to(device).float()
fine = torch.from_numpy(invariant.to_array().to_numpy().squeeze(0)).to(device).float()
coarse_curr = coarse[32,:,0:6,0:6]
fine2 = fine[:,0:48,0:48]

out = G(coarse_curr,fine2)
