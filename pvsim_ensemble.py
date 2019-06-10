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

#%%
def split_locations(ds, locIds):
    ret = ds.sel(location_id=locIds)
    ret = ret.to_dataframe()
    columns = ['T_m__0', 'G_m__0', 'E_wr__0',
               'P_per_peak', 'pv_P_dc', 'G_kW', 'P_sim', 'P_frac_P_sim']
    ret = ret[columns]
    ret = ret.dropna()
    fil = ret['G_kW'] > 0.05 #Filtering night
    ret = ret.loc[fil, :]
    return ret

def compute_derived(x):
    x['P_per_peak'] = x['E_wr__0'] * 12 / x['pv_P_dc']
    x['G_kW'] = x['G_m__0'] / 1000.
    x['P_sim'] = x['G_kW'] #Simplest model
    x['P_frac_P_sim'] = x['P_per_peak'] / x['P_sim']
    return x
#%%
fname = '/home/tzech/ownCloud/Data/ground_measurements/pvk_monitoring/20160401-20160430.nc'
ds = xr.open_dataset(fname)
#%%
ds = compute_derived(ds)
#%%
train_locIds = ['a000270', 'a000255', 'a000267', 'a000279', 'a000257', 'a000268',
                'a000331', 'a000281']
#%%
#a000268: No power on 2016-04-05 13h
#a000255: No power on 2016-04-12

#%%
test_locIds = ['a000132', 'a000272', 'a000202', 'a000160', 'a000302', 'a000067',
               'a000142', 'a000122']
#%%
train = split_locations(ds, train_locIds)
test = split_locations(ds, test_locIds)
#%%
train['is_zero_meas'] = train['P_per_peak'].apply(lambda x: np.allclose(x, 0.))
train['is_zero_sim'] = train['P_sim'].apply(lambda x: np.allclose(x, 0.))
train['is_zero_despite'] = train['is_zero_meas'] & (~train['is_zero_sim'])

#%%
for locId in train_locIds:
    ts = train.loc[locId]
    fig, ax = plt.subplots()
    ts[['P_per_peak', 'P_sim', 'is_zero_despite']].plot(ax=ax, marker='.', secondary_y='is_zero_despite')
    ax.set_title(locId)

    fig, ax = plt.subplots()
    ax.plot(ts['P_sim'], ts['P_per_peak'], '.')
    ax.plot([0, ts['P_sim'].max()], [0, ts['P_per_peak'].max()], color='grey')
    ax.set_title(locId)
#%%
sns.pairplot(train[['P_per_peak', 'P_sim', 'T_m__0', 'P_frac_P_sim']].reset_index(), diag_kws={'bins': 30})
#%%
fig, ax = plt.subplots()
train['P_frac_P_sim'].hist(ax=ax, bins=100, log=False)
#%%
sns.pairplot(train[['P_per_peak', 'P_sim', 'T_m__0', 'P_frac_P_sim']].reset_index(), hue='location_id')
#%% aggregated
total_train = train[['E_wr__0', 'G_m__0', 'pv_P_dc', 'is_zero_despite']].groupby(level=1).sum(skipna=True)
total_train = compute_derived(total_train)
#%%
total_train = total_train.dropna()
#%%
cols = ['P_per_peak', 'P_sim', 'P_frac_P_sim', 'is_zero_despite']
sns.pairplot(total_train[cols].reset_index(), hue='is_zero_despite')
#%%
assert False
#%%
plt.close('all')