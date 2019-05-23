#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:20:38 2019

@author: tzech
"""
from collections import OrderedDict

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
#%% Add time of day
df['valid_time_of_day'] = df.valid_time.dt.time
#%%
fil_step = df['step'] == pd.Timedelta('15min')
df_fil = df.loc[fil_step, ['forecast', 'measurements']]
#%%
plt.ioff()
grps_step = df.groupby('step')
for k, grp in grps_step:
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = grp[['forecast', 'measurements']].plot(ax=ax)
    k_str = int(k.total_seconds()/60.)
    plt.title(k)
    plt.ylim([0, 1000])
    plt.savefig('/home/tzech/results/plots/ts_{0}.png'.format(k_str))

plt.close('all')
plt.ion()
#%% Leave one day out cross validation
# Start with leave 2016-04-30 out and use 15min step only
test_date = pd.Timestamp('2016-04-30')
fil_test = df['base_time'] >= test_date
fil_train = df['base_time'] < test_date
df_test = df.loc[fil_test]
df_train = df.loc[fil_train]
#%% Build persistence ensemble for same datetime
def latest_samples(data, n=10):
    data = data.sort_values('base_time')
    return data.iloc[-n:, :]

def latest_measurements(data, n=10):
    latest = latest_samples(data)
    latest = latest.reset_index()
    meas = latest['measurements']
    return meas
    

fil_now = df['step'] == pd.Timedelta('0min')
grps_valid_time_of_day = df_train.loc[fil_now, :].groupby('valid_time_of_day')
grps_valid_time_of_day[['base_time', 'forecast', 'measurements']].last()
pers_list = [(k, latest_measurements(grp)) for k, grp in grps_valid_time_of_day]
pers = pd.DataFrame(OrderedDict(pers_list))
pers = pers.transpose()
pers = pers.reset_index()
pers['time_of_day'] = pers['index'].apply(lambda t: pd.Timedelta(t.strftime('%H:%M:%S')))
pers['valid_time'] = pers['time_of_day'] + test_date
pers = pers[[col for col in pers.columns if col not in ['index', 'time_of_day']]]
pers = pers.set_index('valid_time')
#%%
fig, ax = plt.subplots()
df_test.loc[fil_now, :].set_index('valid_time')['measurements'].plot(ax=ax, lw=3)
pers.plot(ax=ax)

