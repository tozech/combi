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
netcdfname = '/home/tzech/data/weather/numerical_weather/ecmwf/enfo_germany_pf/02_netcdf/20170701-20170731.nc'

ds = xr.open_dataset(netcdfname)
#%%
csvname = '/home/tzech/data/weather/ground_measurements/dwd/recent/01_csv/produkt_zehn_min_sd_20170419_20181020_00183.txt'
dwd = pd.read_csv(csvname, sep=';', parse_dates=['MESS_DATUM'])
#%%
dwd = dwd.rename(columns={'MESS_DATUM': 'valid_time'})
dwd = dwd.set_index('valid_time')
#%%
metadata = '/home/tzech/data/weather/ground_measurements/dwd/recent/01_csv/Metadaten_Geographie_00183.txt'
dwd_meta = pd.read_csv(metadata, sep=';')
#%%
altitude, latitude, longitude = tuple(dwd_meta[['Stationshoehe', 'Geogr.Breite', 'Geogr.Laenge']].iloc[-1].values)
#%%
ds['GHI_fc'] = ds['ssrd'].diff('step')
ds['GHI_fc'] = ds['GHI_fc'] / 3600 / 12 #Convert to Wh/m^2
#%%
ds_dayahead = ds.sel(step=slice(pd.Timedelta('23:59:59'), pd.Timedelta('1d 23:59:59')))
#%%
ds_station = ds_dayahead.sel(latitude=latitude, longitude=longitude, method='nearest')
#%%
df = ds_station.to_dataframe()
df = df.reset_index()
df['valid_time'] = df['time'] + df['step']
df = df.set_index('valid_time')
#%%
dwd = dwd.rename(columns={'GS_10': 'GHI_meas'})
#%%
dwd['GHI_meas'] = dwd['GHI_meas'] / 3600 * 100**2 * 6
dwd_12h = dwd.resample('12h', closed='right', label='right').mean()
#%%
dwd_12h_centered = dwd_12h.copy(deep=True)
dwd_12h_centered.index = dwd_12h_centered.index - pd.Timedelta('6h')
df_centered = df.copy(deep=True)
df_centered.index = df_centered.index - pd.Timedelta('6h')
#%%
start, end = '2017-07-02', '2017-07-31'
fig, ax = plt.subplots()
dwd.loc[start:end, 'GHI_meas'].plot(ax=ax)
dwd_12h_centered.loc[start:end, 'GHI_meas'].plot(ax=ax)
df_centered.loc[start:end, 'GHI_fc'].plot(ax=ax, lw=0, marker='.')

meas_ts = dwd_12h.loc[start:end, 'GHI_meas']
fc_ts = df.loc[start:end, ['GHI_fc', 'number']]
#%%
fc = fc_ts.reset_index()
fc = fc.pivot(index='valid_time', columns='number', values='GHI_fc')
#%%
fig, ax = plt.subplots()
mask = [True for x in meas_ts.values]
rh, bins = ranky.rankz(meas_ts.values, fc.values.T, mask)
ax.bar(bins[:-1], rh)
