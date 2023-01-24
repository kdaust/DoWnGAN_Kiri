# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:22:36 2023

@author: kirid
"""

import nctoolkit as nc
import numpy as np

##era5
fine = False

coarse_covars = {
    "temp": "/home/kiridaust/Masters/Data/temperature/temp_era5.nc",
    #"humid": "/home/kiridaust/Masters/Data/temperature/humid_era5.nc",
    "pressure": "/home/kiridaust/Masters/Data/temperature/PSLR.nc"
    }

coarse_out = {
    "train": "/home/kiridaust/Masters/Data/temperature/just_temp/coarse_train.nc",
    "test": "/home/kiridaust/Masters/Data/temperature/just_temp/coarse_test.nc",
    }

fine_covars = {
    "temp": "/home/kiridaust/Masters/Data/temperature/wrf_temp.nc",
    #"humid": "/home/kiridaust/Masters/Data/temperature/wrf_humid.nc"
    }

fine_out = {
    "train": "/home/kiridaust/Masters/Data/temperature/just_temp/fine_train.nc",
    "test": "/home/kiridaust/Masters/Data/temperature/just_temp/fine_test.nc",
    }

train_region = [-126,-122,49,53]
if(fine):
    spatial_res = 0.03125
    file_dict = fine_covars
    out_dict = fine_out
else:
    spatial_res = 0.25
    file_dict = coarse_covars
    out_dict = coarse_out
    
bounds = [train_region[0]+spatial_res/2,train_region[1]-spatial_res/2,train_region[2]+spatial_res/2,train_region[3]-spatial_res/2]


for i,covar in enumerate(file_dict.keys()):
    print("Processing ",covar)
    temp = nc.open_data(file_dict[covar])
    oldnm = temp.contents.variable[0]
    temp.rename({oldnm: covar})
    
    temp.subset(years = range(2001,2007)) #what we're working with
    temp.to_latlon(lon = [bounds[0],bounds[1]], lat = [bounds[2],bounds[3]], res = [spatial_res,spatial_res])
    num_slices = int(temp.contents.ntimes)
    num_points = int(temp.contents.npoints)
    temp_mean = temp.copy()
    temp_mean.spatial_mean()
    temp_mean.tmean()
    temp_mean.run()


    temp_var = temp.copy()
    t3 = temp.copy()
    temp_var.spatial_var()
    temp_var.tsum()
    t3.spatial_mean()
    t3.tvar()
    t3.multiply((num_points*(num_slices-1))/(num_points-1))
    temp_var.add(t3)
    temp_var.multiply((num_points-1)/(num_points*num_slices-1)) ##this is now the variance
    temp_var.run()

    var_mean = float(temp_mean.to_xarray()[covar])
    var_var = np.sqrt(float(temp_var.to_xarray()[covar]))
    temp.subtract(var_mean)
    temp.divide(var_var)
    temp.run()
    if(i == 0):
        ds_out = temp.copy()
    else:
        ds_out.append(temp)

ds_out.merge()
train = ds_out.copy()
train.subset(years = range(2001,2004))
train.to_nc(out_dict["train"])
test = ds_out.copy()
test.subset(years = 2005)
test.to_nc(out_dict["test"])
# ds_out.subset(years = 2007)
# ds_out.to_nc("/home/kiridaust/Masters/Data/temperature/fine_validation.nc")

#t2.assign(avg = lambda x: spatial_mean(x.tas), drop = True)


# cdo sub $IN -enlarge,$IN -timmean -fldmean $IN centred.nc
# cdo div centred.nc -enlarge,centred.nc -timmean -fldstd centred.nc $OUT
# gridtype  = lonlat
# gridsize  = 256
# xsize     = 16
# ysize     = 16
# xname     = longitude
# xlongname = "longitude"
# xunits    = "degrees_east"
# yname     = latitude
# ylongname = "latitude"
# yunits    = "degrees_north"
# xfirst    = 236.125
# xinc      = 0.25
# yfirst    = 52.875
# yinc      = -0.25