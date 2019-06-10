#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:01:10 2019

@author: tzech
"""

from collections import OrderedDict
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

import seaborn as sns

import ranky
import properscoring

fname = '/home/tzech/ownCloud/Data/ground_measurements/pvk_monitoring/20160401-20160430.nc'
ds = xr.open_dataset(fname)
#%%
ds['P_per_peak'] = ds['E_wr__0'] * 12 / ds['pv_P_dc']
ds['G_kW'] = ds['G_m__0'] / 1000.
ds['P_sim'] = ds['G_kW'] #Simplest model
ds['P_frac_P_sim'] = ds['P_per_peak'] / ds['P_sim']
#%%
train_locIds = ['a000270', 'a000255', 'a000267', 'a000279', 'a000257', 'a000268',
                'a000331', 'a000281']
test_locIds = ['a000132', 'a000272', 'a000202', 'a000160', 'a000302', 'a000067',
               'a000142', 'a000122']
#%%
train = ds.sel(location_id=train_locIds)
train = train.to_dataframe()
train = train[['T_m__0', 'G_m__0', 'E_wr__0', 'P_per_peak', 'pv_P_dc', 'G_kW', 'P_sim', 'P_frac_P_sim']]
train = train.dropna()
#%%

#%%
for locId in train_locIds:
    ts = train.loc[locId]
    fig, ax = plt.subplots()
    ts[['P_per_peak', 'P_sim']].plot(ax=ax, marker='.', secondary_y='G_kW')
    ax.set_title(locId)
    
    fig, ax = plt.subplots()
    ax.plot(ts['P_sim'], ts['P_per_peak'], '.')
    ax.plot([0, ts['P_sim'].max()], [0, ts['P_per_peak'].max()], color='grey')
    ax.set_title(locId)
#%%
sns.pairplot(train[['P_per_peak', 'P_sim', 'T_m__0', 'P_frac_P_sim']].reset_index(), 'location_id')
#%%
assert False
#%%
plt.close('all')