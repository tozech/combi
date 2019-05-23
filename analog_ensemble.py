#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:20:38 2019

@author: tzech
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

import seaborn as sns   
#%%
fname = '/home/tzech/ownCloud/Data/combi/cmv_nwp/Q1/20160401-20160430.nc'
cmv_nwp = xr.open_dataset(fname)
#%%
loc_id = 100910
cmv_nwp = cmv_nwp.sel(location_id=loc_id)
#%%
df_cmv_nwp = cmv_nwp.to_dataframe()
df_cmv_nwp = df_cmv_nwp.reset_index()
#%%
fname = '/home/tzech/ownCloud/Data/ground_measurements/dwd/10min/historical/20160401-20160430.nc'
meas = xr.open_dataset(fname)
#%%
loc_id = 183
meas = meas.sel(station_id=loc_id)
meas = meas.rename({'datetime': 'valid_time', 'global': 'measurements'})
df_meas = meas.to_dataframe()
df_meas = df_meas.loc[:, 'measurements']
df_meas = df_meas.reset_index()
#%%
df = df_cmv_nwp.merge(df_meas, on='valid_time')
#%%
fil = df['step'] == pd.Timedelta('15min')
df_fil = df.loc[fil, ['forecast', 'measurements']]
#%%
df_fil.plot()