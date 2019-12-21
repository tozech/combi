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
months = ['2018-01', '2018-07']

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

df_list.append(read_single_file(datadir, 'uncalib', '15min', True, '2018-01'))
                
df = pd.concat(df_list)
#%%
#df['winsize'] = df['winsize'] * pd.Timedelta('15min')
#%%
cols = ['mean_crps', 'reliability', 'resolution', 'uncertainty']
#%%
sns.relplot('winsize', 'value', 'calib_freq', 
            row='metric', row_order=cols, col='month',
            kind='line', facet_kws=dict(sharex=True, sharey=False),
            data=df)

#%%
def plot_by_unc(data):
    cols_by_unc = ['{0}_by_unc'.format(c)  for c in ['mean_crps', 'reliability', 'resolution']]
    g = sns.relplot('winsize', 'value', 'calib_freq', 
                    row='metric', row_order=cols_by_unc, col='month',
                    kind='line', facet_kws=dict(sharex=True, sharey=False),
                    data=data)
    return g
#%% unsorted VD naive comparison
g = plot_by_unc(df[~df['issorted']])
#%%
g = plot_by_unc(df[(df['winsize'] <= 24) & (~df['issorted'])])
ax_squeezed = np.reshape(g.axes, 6)
ylims = [(0.32, 0.42), (0.16, 0.26), (0, 0.1), (0, 0.1), (0.6, 0.7), (0.79, 0.89)]
for ax, ylim in zip(ax_squeezed, ylims):
    ax.set_ylim(ylim)
#%% sorted vs. unsorted
# reliability is improved, but resolution is lost due to calibration
fil = df['month'] == '2018-01'
cols_by_unc = ['{0}_by_unc'.format(c)  for c in ['mean_crps', 'reliability', 'resolution']]
g = sns.relplot('winsize', 'value', 'calib_freq', 
            row='metric', row_order=cols_by_unc, col='issorted',
            kind='line', facet_kws=dict(sharex=True, sharey=False),
            data=df[fil])
ax_squeezed = np.reshape(g.axes, 6)
ylims = [(0.3, 0.8), (0.3, 0.8), (0., 0.4), (0., 0.4), (0.3, 0.8), (0.3, 0.8)]
for ax, ylim in zip(ax_squeezed, ylims):
    ax.set_ylim(ylim)
    ax.grid(True)
#%%
fil = df['month'] == '2018-01'
fil &= df['calib_freq'] == '15min'
cols_by_unc = ['{0}_by_unc'.format(c)  for c in ['mean_crps', 'reliability', 'resolution']]
g = sns.relplot('winsize', 'value', 'issorted', 
            row='metric', row_order=cols_by_unc,
            kind='line', facet_kws=dict(sharex=True, sharey=False),
            data=df[fil])
ax_squeezed = np.reshape(g.axes, 3)
ylims = [(0.3, 0.8), (0, 0.5), (0.3, 0.8)]
for ax, ylim in zip(ax_squeezed, ylims):
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