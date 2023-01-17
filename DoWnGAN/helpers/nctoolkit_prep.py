# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:22:36 2023

@author: kirid
"""

import nctoolkit as nc
import numpy as np

##era5
coarse_covars = {
    "temp": "temp_era5.nc",
    "humid": "humid_era5.nc",
    "pressure": "pressure.nc"
    }

train_region = [-126.125,-121.875,49.125,52.875]
x_res = 0.25
y_res = 0.25

for covar in coarse_covars.keys():
    temp = nc.open_data(coarse_covars[covar])
    oldnm = temp.contents.variable[0]
    temp.rename({oldnm: covar})
    
    temp.subset(years = range(2001,2010)) #what we're working with
    temp.to_latlon(lon = [train_region[0],train_region[1]], lat = [train_region[2],train_region[3]], res = [x_res,y_res])
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
    var_var = float(temp_var.to_xarray()[covar])
    temp.subtract(var_mean)
    temp.divide(var_var)
    temp.run()


temp.to_nc("Test_Process.nc")

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