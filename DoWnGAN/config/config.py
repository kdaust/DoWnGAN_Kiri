from DoWnGAN.helpers.wrf_times import datetime_wrf_period

import os
from datetime import datetime

import torch

# Path to HR data. Files are organized by variable and are loaded by xarray.open_mfdataset()
FINE_DATA_PATH_U10 = '~/Masters/Data/WRF/U10_WRF2.nc'
FINE_DATA_PATH_V10 = '~/Masters/Data/WRF/V10_WRF2.nc'
# Root dir for the covariates. Individual files defined below
COVARIATE_DATA_PATH = '~/Masters/Data/Era5/'
# Where you want the processed data
PROC_DATA = '~/Masters/Processed'

# Where to store the mlflow tracking information. Make sure there is plenty of storage. 
# This repo is NOT conservative with IO.
EXPERIMENT_PATH = '/media/data/mlflow_exp'
EXPERIMENT_TAG="Kiri's Tests"

# Whether to load preprocessed data
already_preprocessed = True

# Which CUDA device to see
device = torch.device("cuda:0")

# One of florida, west, or central
# One of florida, west, or central
# region = "florida"
region = "kiri_test"
invariant_fields = []

# Choose a reference field
ref_coarse = "u10"

# Masking years
mask_years = [2003]

# Scale factor for the covariates
scale_factor = 8

# WRF Time slice
# Add extra  6 hour step early due to peculiarities in WRF (extra field)
# Actual starting time of WRF is 2000-10-01 00:00:00
start_time = datetime(2001, 1, 1, 0, 0)
end_time = datetime(2007, 12 ,31, 0, 0)
range_datetimes = datetime_wrf_period(start_time, end_time)

# Compute constants, machine dependent.
cpu_count = os.cpu_count()
chunk_size = 150


###########################################################
#### These are all of the options that can be configured###
###########################################################

"""
This file contains all of the configurable options that can be accessed for this project abd us exclusively dictionaries.
It uses paths defined in config.py
"""

# Variables in HR fields, and paths to those netcdfs
# This assumes they are separate files
fine_paths_dict = {
    "u10": FINE_DATA_PATH_U10,
    "v10": FINE_DATA_PATH_V10
}


# Rename attributes to a coherent naming convention
non_standard_attributes = {
    "latitude": "lat",
    "longitude": "lon",
    "Times": "time",
    "Time": "time",
    "times": "time",
    "U10": "u10",
    "V10": "v10",
    "uas": "u10",
    "vas": "v10",
}

# Covariate paths list
cov_paths_dict = {
    "u10": COVARIATE_DATA_PATH+"/U10_ERA2.nc",
    "v10": COVARIATE_DATA_PATH+"/V10_ERA2.nc",
}

# Common names ordered, Just add variables into this dictionary when extending.
covariate_names_ordered = {
    # Standard name: variable name in netcdf
    "u10": 'u10',
    "v10": "v10",
}

fine_names_ordered = {"u10": "u10", "v10": "v10"}


# These define the region indices in the coarse resolution
# They are multiplied by the scale factor to get them in the HR field.
# This assumes that the HR grids fit perfectly into the LR grids
regions = {
    "florida": {"lat_min": 4, "lat_max": 20, "lon_min": 70, "lon_max": 86},
    "central": {"lat_min": 30, "lat_max": 46, "lon_min": 50, "lon_max": 66},
    "central_larger": {"lat_min": 9, "lat_max": 47, "lon_min": 29, "lon_max": 67},
    "west": {"lat_min": 30, "lat_max": 46, "lon_min": 15, "lon_max": 31},
    "kiri_test": {"lat_min": 2, "lat_max": 40, "lon_min": 2, "lon_max": 19}
}

