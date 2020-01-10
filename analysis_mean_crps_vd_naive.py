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


datadir = '/home/tzech/ownCloud/Results/time_corr/run_2019-12-22'
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
    df['crps_pot'] = df['uncertainty'] - df['resolution']
    for c in ['mean_crps', 'reliability', 'resolution', 'crps_pot']:
        df['{0}_by_unc'.format(c)] = df[c] / df['uncertainty']
    for c in ['mean_crps', 'reliability', 'resolution', 'uncertainty', 'crps_pot']:
        df['{0}_by_mae'.format(c)] = df[c] / df['mae']
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

df_list.append(read_single_file(datadir, 'uncalib', '15min', False, '2018-01'))
df_list.append(read_single_file(datadir, 'uncalib', '15min', True, '2018-01'))
df_list.append(read_single_file(datadir, 'uncalib', '15min', False, '2018-07'))
df_list.append(read_single_file(datadir, 'uncalib', '15min', True, '2018-07'))

df_list.append(read_single_file(datadir, 'vd_naive', '15min', True, '2018-01'))
df_list.append(read_single_file(datadir, 'vd_naive', '15min', True, '2018-07'))

df = pd.concat(df_list)
#assert False
#%% Add combined method and calib_freq
df['method_freq'] = df.apply(lambda row: "{0}_{1}".format(row['method'], row['calib_freq']), axis=1)
#%%
def join_sorted(row):
    if row['issorted']:
        return "{0}_sorted".format(row['method_freq'])
    else:
        return row['method_freq']
df['method_freq_issorted'] = df.apply(join_sorted, axis=1)
#%%
#df['winsize'] = df['winsize'] * pd.Timedelta('15min')


#%%
g = sns.relplot('winsize', 'value', 'method_freq_issorted',
                row='metric', row_order=['mean_crps', 'mae'], col='month',
                kind='line', facet_kws=dict(sharex=True, sharey=False),
                data=df)
ax_squeezed = np.reshape(g.axes, 4)
ylims = [(0.3, 0.8), (0.3, 0.8), (0., 0.4), (0., 0.4)]
for ax, ylim in zip(ax_squeezed, ylims):
#    ax.set_ylim(ylim)
    ax.grid(True)
g.set_ylabels('Irradiance [$W/m^2$]')
g.set_xlabels('Winsize n [15min]')
#%%
g = sns.relplot('winsize', 'value', 'method_freq_issorted',
                row='metric', row_order=['bias', 'rmse', 'std_mean'], col='month',
                kind='line', facet_kws=dict(sharex=True, sharey=False),
                data=df)
ax_squeezed = np.reshape(g.axes, 6)
ylims = [(0.3, 0.8), (0.3, 0.8), (0., 0.4), (0., 0.4), (0.3, 0.8), (0.3, 0.8)]
for ax, ylim in zip(ax_squeezed, ylims):
#    ax.set_ylim(ylim)
    ax.grid(True)
g.set_ylabels('Irradiance [$W/m^2$]')
g.set_xlabels('Winsize n [15min]')
#%%
cols = ['mean_crps', 'reliability', 'crps_pot', 'resolution', 'uncertainty']

g = sns.relplot('winsize', 'value', 'method_freq_issorted',
                row='metric', row_order=cols, col='month',
                kind='line', facet_kws=dict(sharex=True, sharey=False),
                data=df)

ax_squeezed = np.reshape(g.axes, 10)
ylims = [(0.3, 0.8), (0.3, 0.8), (0., 0.4), (0., 0.4), (0., 0.4), (0., 0.4),
         (0.3, 0.8), (0.3, 0.8), (0.3, 0.8), (0.3, 0.8)]
for ax, ylim in zip(ax_squeezed, ylims):
#    ax.set_ylim(ylim)
    ax.grid(True)
g.set_ylabels('Irradiance [$W/m^2$]')
g.set_xlabels('Winsize n [15min]')
#%% by UNC
cols_by_unc = ['{0}_by_unc'.format(c)  for c in ['mean_crps', 'reliability', 'crps_pot', 'resolution']]
g = sns.relplot('winsize', 'value', 'method_freq_issorted',
            row='metric', row_order=cols_by_unc, col='month',
            kind='line', facet_kws=dict(sharex=True, sharey=False),
            data=df)
ax_squeezed = np.reshape(g.axes, 8)
ylims = [(0.1, 0.8), (0.1, 0.8), (0., 0.4), (0., 0.4), (0.3, 0.9), (0.3, 0.9), (None, None), (None, None)]
for ax, ylim in zip(ax_squeezed, ylims):
#    ax.set_ylim(ylim)
    ax.grid(True)
g.set_ylabels('Irradiance / UNC [-]')
g.set_xlabels('Winsize n [15min]')
#%% by_MAE
cols_by_unc = ['{0}_by_mae'.format(c)  for c in ['mean_crps', 'reliability', 'crps_pot', 'resolution', 'uncertainty']]
g = sns.relplot('winsize', 'value', 'method_freq_issorted',
            row='metric', row_order=cols_by_unc, col='month',
            kind='line', facet_kws=dict(sharex=True, sharey=False),
            data=df)
ylims = [(0.7, 0.9), (0.7, 0.9), (0., 0.5), (0., 0.5), (0.3, 0.8), (0.3, 0.8),
         (0, 5), (0, 5), (0, 5), (0, 5)]
ax_squeezed = np.reshape(g.axes, len(ylims))
for ax, ylim in zip(ax_squeezed, ylims):
    ax.set_ylim(ylim)
    ax.grid(True)
g.set_ylabels('Irradiance / MAE [-]')
g.set_xlabels('Winsize n [15min]')
#%%============================================================================
#%% Old plots

#%%
def plot_by_var(data, var='unc'):
    cols_by_unc = ['{0}_by_{1}'.format(c, var)  for c in ['mean_crps', 'reliability', 'resolution']]
    g = sns.relplot('winsize', 'value', 'method_freq',
                    row='metric', row_order=cols_by_unc, col='month',
                    kind='line', facet_kws=dict(sharex=True, sharey=False),
                    data=data)
    return g
#%% unsorted VD naive comparison
if False:
    g = plot_by_var(df[~df['issorted']], 'unc')
    #%%
    g = plot_by_var(df[~df['issorted']], 'mae')
    #%%
    g = plot_by_var(df[(df['winsize'] <= 24) & (~df['issorted'])], 'unc')
    ax_squeezed = np.reshape(g.axes, 6)
    ylims = [(0.32, 0.42), (0.16, 0.26), (0, 0.1), (0, 0.1), (0.6, 0.7), (0.79, 0.89)]
    for ax, ylim in zip(ax_squeezed, ylims):
        ax.set_ylim(ylim)
    #%% sorted uncalib vs. unsorted calib!
# reliability is improved, but resolution is lost due to calibration
if False:
    fil = df['month'] == '2018-01'
    cols_by_unc = ['{0}_by_unc'.format(c)  for c in ['mean_crps', 'reliability', 'resolution']]
    g = sns.relplot('winsize', 'value', 'method_freq',
                row='metric', row_order=cols_by_unc, col='issorted',
                kind='line', facet_kws=dict(sharex=True, sharey=False),
                data=df[fil])
    ax_squeezed = np.reshape(g.axes, 6)
    ylims = [(0.3, 0.8), (0.3, 0.8), (0., 0.4), (0., 0.4), (0.3, 0.8), (0.3, 0.8)]
    for ax, ylim in zip(ax_squeezed, ylims):
        ax.set_ylim(ylim)
        ax.grid(True)
#%%
if False:
    fil = df['month'] == '2018-01'
    #fil &= df['calib_freq'] == '15min'
    cols_by_unc = ['{0}_by_unc'.format(c)  for c in ['mean_crps', 'reliability', 'resolution']]
    g = sns.relplot('winsize', 'value', 'method_freq_issorted',
                row='metric', row_order=cols_by_unc,
                kind='line', facet_kws=dict(sharex=True, sharey=False),
                data=df[fil])
    ax_squeezed = np.reshape(g.axes, 3)
    ylims = [(0.3, 0.8), (0., 0.4), (0.3, 0.8)]
    for ax, ylim in zip(ax_squeezed, ylims):
        ax.set_ylim(ylim)
#%% Same with by_MAE
if False:
    fil = df['month'] == '2018-01'
    #fil &= df['calib_freq'] == '15min'
    cols_by_unc = ['{0}_by_mae'.format(c)  for c in ['mean_crps', 'reliability', 'resolution']]
    g = sns.relplot('winsize', 'value', 'method_freq_issorted',
                row='metric', row_order=cols_by_unc,
                kind='line', facet_kws=dict(sharex=True, sharey=False),
                data=df[fil])
    ax_squeezed = np.reshape(g.axes, 3)
    ylims = [(0.3, 0.8), (0., 0.4), (0.3, 0.8)]
    for ax, ylim in zip(ax_squeezed, ylims):
        pass#ax.set_ylim(ylim)


#%%
if False:
    mae_vd_naive_15min = df.loc[(df.metric=='mae') & (df.method_freq_issorted=='vd_naive_15min'), 'value'].values
    for c in ['mean_crps', 'reliability', 'resolution']:
        metric_3h = df.loc[(df.metric==c) & (df.method_freq_issorted=='vd_naive_3h'), 'value'].values
        col_by_mae = '{0}_by_mae'.format(c)
        print(f'WARNING: Setting MAE for {col_by_mae} for vd_naive_3h to MAE of vd_naive_15min!!!')
        df.loc[(df.metric==col_by_mae) & (df.method_freq_issorted=='vd_naive_3h'), 'value'] = metric_3h / mae_vd_naive_15min

    cols_by_unc = ['{0}_by_mae'.format(c)  for c in ['mean_crps', 'reliability', 'resolution']]
    g = sns.relplot('winsize', 'value', 'method_freq_issorted',
                row='metric', row_order=cols_by_unc, col='month',
                kind='line', facet_kws=dict(sharex=True, sharey=False),
                data=df)
    ax_squeezed = np.reshape(g.axes, 6)
    ylims = [(0.7, 0.9), (0.7, 0.9), (0., 0.5), (0., 0.5), (0., 5.), (0., 5.)]
    for ax, ylim in zip(ax_squeezed, ylims):
        ax.set_ylim(ylim)
        ax.grid(True)
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