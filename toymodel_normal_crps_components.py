#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:06:40 2019

@author: tzech
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from properscoring import mean_crps, crps_ensemble, crps_gaussian

#%%
df = pd.DataFrame(columns=['n_ens', 'm_times', 'mean_crps_dist', 'mean_crps_ens', 'rel', 'res', 'unc'])
#%%
n_ens = 100
max_exp_m_times = 5
mu_pred = 0
sig_pred = 0
for exp_m_times in range(2, max_exp_m_times):
    m_times = 10**exp_m_times
    print('exp_m_times', exp_m_times, 'm_times', m_times)
    ens = sig_pred * np.random.randn(m_times, n_ens) + mu_pred
    obs = np.random.randn(m_times)

    crps_dist = crps_gaussian(obs, mu_pred, sig_pred)
    mean_crps_dist = np.mean(crps_dist)
    crps_values = crps_ensemble(obs, ens)
    mean_crps_ens, rel, res, unc = mean_crps(obs, ens)
    assert np.allclose(mean_crps_ens, np.mean(crps_values))
    df = df.append({'n_ens': n_ens,
                    'm_times': m_times,
                    'mean_crps_dist': mean_crps_dist,
                    'mean_crps_ens': mean_crps_ens,
                    'rel': rel,
                    'res': res,
                    'unc': unc},
                    ignore_index=True)

#%% unc component converges towards CRPS dist
print(df['mean_crps_dist'] - df['unc'])
#%% rel component of ensemble converges for a fixed n_ens with larger m_times
# to a finite number and therefore the
# difference between mean_crps_dist and mean_crps_ens
fname = 'crps_components_m_times_100-1M.csv'
df_store = pd.read_csv(fname, index_col=0)
save = False
if save:
    df_store = pd.concat([df_store, df])
    df_store = df_store.sort_values(['n_ens', 'm_times'])
    df_store.to_csv(fname)
#%% rel component converges to zero with larger n_ens for large m_times
# Hence, finite reliability here is only a finite size effect of the ensemble
df_store_max = df_store[df_store.m_times == 10**6]
#plt.plot(df_store_max.n_ens, df_store_max.rel)

#%% sig_pred = sig_obs = 1-> res tends to zero for larger n_ens
#   sig_pred = 1/2 while sig_obs = 1 -> rel and res increases
#   sig_pred = 2 while sig_obs = 1 -> rel and res increases, res becomes negative
#   sig_pred = 0 while sig_obs = 1 -> mean_crps == rel, res == unc
df_sig = pd.DataFrame()
for sig_pred in np.linspace(0, 2, 100):
    m_times = 10**4
    print('sig_pred', sig_pred, 'm_times', m_times)
    ens = sig_pred * np.random.randn(m_times, n_ens) + mu_pred
    obs = np.random.randn(m_times)

    crps_dist = crps_gaussian(obs, mu_pred, sig_pred)
    mean_crps_dist = np.mean(crps_dist)
    crps_values = crps_ensemble(obs, ens)
    mean_crps_ens, rel, res, unc = mean_crps(obs, ens)
    assert np.allclose(mean_crps_ens, np.mean(crps_values))
    df_sig = df_sig.append({'sig_pred': sig_pred,
                            'n_ens': n_ens,
                            'm_times': m_times,
                            'mean_crps_dist': mean_crps_dist,
                            'mean_crps_ens': mean_crps_ens,
                            'rel': rel,
                            'res': res,
                            'unc': unc},
                            ignore_index=True)
#%%
fname = f'crps_components_vs_sig_pred_m_times_{m_times}.csv'
df_sig.to_csv(fname)
#%%
fig, ax = plt.subplots()
cols = ['sig_pred', 'mean_crps_ens', 'rel', 'res', 'unc']
df_sig[cols].set_index('sig_pred').plot(ax=ax)
ax.set_title(f'Toy model normal distribution, n_ens: {n_ens}, m_times: {m_times}')
#%% For a perfect forecast
# mean_crps = rel = 0
# res = unc > 0
mean_crps(obs, np.tile(obs, (n_ens, 1)).T)

