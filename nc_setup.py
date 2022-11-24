# -*- coding: utf-8 -*-
##Process raw data with nctoolkit

import nctoolkit as nc
import numpy as np
import matplotlib.pyplot as plt

infile = 'U10_remap.nc'
dat = nc.open_data(infile)
dat.start

dat.tmean("time")
dat.run()
dat.tmean("time")
dat.spatial_mean()
dat.assign(mean = lambda x: x.u10 - nc.spatial_mean(x.u10))

nc.cdo_version()

### merge WRF
### crop time
### crop area
### fix var names
### remap to fit
### standardise to 0,1
