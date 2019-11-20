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
n_ens = 50
for exp_m_times in range(2, 7):
    m_times = 10**exp_m_times
    print('exp_m_times', exp_m_times, 'm_times', m_times)
    ens = np.random.randn(m_times, n_ens)
    obs = np.random.randn(m_times)
    
    crps_dist = crps_gaussian(obs, 0, 1)
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
    

