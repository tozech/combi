#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:13:22 2019

@author: tzech
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

#%%
netcdfname = '/home/tzech/data/weather/numerical_weather/ecmwf/enfo_germany_pf/02_netcdf/20170701-20170731.nc'

ds = xr.open_dataset(netcdfname)
#%%
def low_func(x):
    print(x.name)
    return 2 * x
#%%
def func(data):
    grps_ssrd = data.groupby_bins('ssrd', bins=3)
    applied_f = grps_ssrd.apply(low_func)
    return applied_f

ds_bin_applied = func(ds)
print(ds_bin_applied)
#%%
assert False
#%%
grps_step = ds.groupby('step')
applied = grps_step.apply(func)
print(applied)
#%%
df = ds.to_dataframe()