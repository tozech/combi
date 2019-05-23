#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:20:38 2019

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
cmv_nwp.close()
del(cmv_nwp)
meas.close()
del(meas)
del(df_meas)
del(df_cmv_nwp)
gc.collect()
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
fil_test = df['base_time'].dt.date == test_date.date()
fil_train = df['base_time'].dt.date != test_date.date()
df_test = df.loc[fil_test]
df_train = df.loc[fil_train]
#%% Build persistence ensemble for same datetime
def get_samples(data, key, n=10, ascending=True):
    data = data.sort_values(key, ascending=ascending)
    return data.iloc[-n:, :]

def latest_samples(data, n=10):
    return get_samples(data, 'base_time')


def latest_measurements(data, n=10):
    latest = latest_samples(data, n)
    latest = latest.reset_index()
    meas = latest['measurements']
    return meas

def sort_ensemble(data):
    values = data.values
    values.sort(axis=1)
    return pd.DataFrame(values, index=data.index, columns=data.columns)

def time_of_day(time_col):
    return time_col.apply(lambda t: pd.Timedelta(t.strftime('%H:%M:%S')))

fil_now = df['step'] == pd.Timedelta('0min')
grps_valid_time_of_day = df_train.loc[fil_now, :].groupby('valid_time_of_day')
grps_valid_time_of_day[['base_time', 'forecast', 'measurements']].last()
pers_list = [(k, latest_measurements(grp)) for k, grp in grps_valid_time_of_day]
pers = pd.DataFrame(OrderedDict(pers_list))
pers = pers.transpose()
pers = pers.reset_index()
pers['time_of_day'] = time_of_day(pers['index'])
pers['valid_time'] = pers['time_of_day'] + test_date
pers = pers[[col for col in pers.columns if col not in ['index', 'time_of_day']]]
pers = pers.set_index('valid_time')
pers = sort_ensemble(pers)
#%%
meas_ts = df_test.loc[fil_now, :].set_index('valid_time')['measurements']
meas_time_of_day = time_of_day(meas_ts.reset_index()['valid_time'])
current_fc = df_test.loc[fil_step, :].set_index('valid_time')['forecast']
fig, ax = plt.subplots()
pers.plot(ax=ax, color='k', alpha=0.5)
meas_ts.plot(ax=ax, lw=2)
current_fc.plot(ax=ax, lw=2)
#%%
fig, ax = plt.subplots()
mask = [True for x in meas_ts.values]
rh, bins = ranky.rankz(meas_ts.values, pers.values.T, mask)
ax.bar(bins[:-1], rh)
#%% Analog ensemble only on current values
df_train_test = df_train.merge(df_test, on='valid_time_of_day', suffixes=('', '_test'))
#%%
df_train_test['abs_train_test'] = (df_train_test['forecast'] - df_train_test['forecast_test']).abs()
#%%
def analogs_one_val(data, key='abs_train_test', n=10):
    analogs = get_samples(data, key, n, ascending=False)
    analogs = analogs.reset_index()
    meas = analogs['measurements']
    return meas

grps_valid_time_of_day = df_train_test.groupby('valid_time_of_day')
analogs_list = [(k, analogs_one_val(grp)) for k, grp in grps_valid_time_of_day]
analog = pd.DataFrame(OrderedDict(analogs_list))
analog = analog.transpose()
analog = analog.reset_index()
analog['time_of_day'] = analog['index'].apply(lambda t: pd.Timedelta(t.strftime('%H:%M:%S')))
analog['valid_time'] = analog['time_of_day'] + test_date
analog = analog[[col for col in analog.columns if col not in ['index', 'time_of_day']]]
analog = analog.set_index('valid_time')
analog = sort_ensemble(analog)
#%%
fil_times = [t + test_date for t in time_of_day(analog.reset_index()['valid_time']) if t in list(meas_time_of_day)]
analog = analog.loc[fil_times, :]
#%%
fig, ax = plt.subplots()
analog.plot(ax=ax, color='k', alpha=0.5)
df_test.loc[fil_now, :].set_index('valid_time')['measurements'].plot(ax=ax, lw=2)
df_test.loc[fil_step, :].set_index('valid_time')['forecast'].plot(ax=ax, lw=2)
#%%
fig, ax = plt.subplots()
mask = [True for x in meas_ts.values]
rh, bins = ranky.rankz(meas_ts.values, analog.values.T, mask)
ax.bar(bins[:-1], rh)