# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import ranky
import cfgrib
#%%
gribname = '/home/tzech/data/weather/numerical_weather/ecmwf/enfo_germany_pf/01_grib/20170701-20170731.grib'

ds = cfgrib.open_dataset(gribname)
#%%
netcdfname = '/home/tzech/data/weather/numerical_weather/ecmwf/enfo_germany_pf/02_netcdf/20170701-20170731.nc'
ds.to_netcdf(netcdfname)