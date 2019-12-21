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
    for c in ['mean_crps', 'reliability', 'resolution']:
        df['{0}_by_unc'.format(c)] = df[c] / df['uncertainty']
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
def plot_by_unc(data):
    cols_by_unc = ['{0}_by_unc'.format(c)  for c in ['mean_crps', 'reliability', 'resolution']]
    g = sns.relplot('winsize', 'value', 'calib_freq', row='metric', row_order=cols_by_unc,
                    kind='line', facet_kws=dict(sharex=True, sharey=False),
                    data=data)
    return g

g = plot_by_unc(df)
#%%
g = plot_by_unc(df[df['winsize'] <= 24])
for ax, ylim in zip(np.squeeze(g.axes), [(0.32, 0.42), (0, 0.1), (0.6, 0.7)]):
    ax.set_ylim(ylim)
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