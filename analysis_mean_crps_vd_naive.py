#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 14:31:34 2019

@author: tzech
"""
import os.path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


datadir = '/home/tzech/ownCloud/Results/time_corr/'
methods = ['vd_naive']
calib_freqs = ['15min', '3h']
issorteds = [False]
months = ['2018-01']

def read_single_file(datadir, method, calib_freq, issorted, month):
    if calib_freq == '15min':
        method_calib_freq = method
    else:
        method_calib_freq = '_'.join([method, calib_freq])
    if issorted:
        fname = '_'.join(['ens', method_calib_freq, 'sort', f'{month}-01.csv'])
    else:
        fname = '_'.join(['ens', method_calib_freq, f'{month}-01.csv'])
    
    df = pd.read_csv(os.path.join(datadir, fname), sep=';')
    df = df.set_index('winsize')
    df = df.stack()
    df = df.reset_index()
    df.columns = ['winsize', 'metric', 'value']
    df['method'] = method
    df['calib_freq'] = calib_freq
    df['issorted'] = issorted
    df['month'] = month
    return df

df_list = []
for method in methods:
    for calib_freq in calib_freqs:
        for issorted in issorteds:
            for month in months:
                tdf = read_single_file(datadir, method, calib_freq, issorted, month)
                df_list.append(tdf)
                
df = pd.concat(df_list)
#%%
#df['winsize'] = df['winsize'] * pd.Timedelta('15min')
#%%
cols = ['mean_crps', 'reliability', 'resolution', 'uncertainty']
#%%
sns.relplot('winsize', 'value', 'calib_freq', row='metric', row_order=cols,
            kind='line', facet_kws=dict(sharex=True, sharey=False),
            data=df)
#%%
#fig, axs = plt.subplots(2, 2, sharex=True)
#for col, ax in zip(cols, np.reshape(axs, (4))):
#    sns.relplot('winsize', col, 'calib_freq', 
#                     kind='line', data=df, ax=ax)
##%%
#fig, ax = plt.subplots()
#
#grps = df.groupby('calib_freq')
#for k, grp in grps:
#    grp.set_index('winsize')[cols].plot(ax=ax, subplots=True, sharex=True)