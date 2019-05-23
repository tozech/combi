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
import properscoring
#%%
def get_samples(data, key, n=10, ascending=True):
    data = data.sort_values(key, ascending=ascending)
    return data.iloc[-n:, :]

def latest_samples(data, n=10):
    return get_samples(data, 'delta_t', n)


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

def analogs_one_val(data, key='abs_train_test', n=10):
    analogs = get_samples(data, key, n, ascending=False)
    analogs = analogs.reset_index()
    meas = analogs['measurements']
    return meas

def create_ensemble(df, func, date):
    grps_valid_time_of_day = df.groupby('valid_time_of_day')
    ens_list = [(k, func(grp)) for k, grp in grps_valid_time_of_day]
    ens = pd.DataFrame(OrderedDict(ens_list))
    ens = ens.transpose()
    ens = ens.reset_index()
    ens['time_of_day'] = time_of_day(ens['index'])
    ens['valid_time'] = ens['time_of_day'] + date
    ens = ens[[col for col in ens.columns if col not in ['index', 'time_of_day']]]
    ens = ens.set_index('valid_time')
    ens = sort_ensemble(ens)
    return ens

def align(meas_ts, date, df_ens):
    meas_time_of_day = time_of_day(meas_ts.reset_index()['valid_time'])
    ens_time_of_day = time_of_day(df_ens.reset_index()['valid_time'])
    fil_times = [pd.Timedelta(t) + pd.Timestamp(date) for t in ens_time_of_day
                 if t in list(meas_time_of_day)]
    return fil_times

def drop_date(df):
    return df[[col for col in df.columns if col != 'date']]

def plot_ts(meas_ts, current_fc, df_ens):
    fig, ax = plt.subplots()
    df_ens.plot(ax=ax, color='k', alpha=0.5)
    meas_ts.plot(ax=ax, lw=2)
    current_fc.plot(ax=ax, lw=2)
    return fig, ax

def rankhist(meas_ts, df_ens):
    mask = [True for x in meas_ts.values]
    rh, bins = ranky.rankz(meas_ts.values, df_ens.values.T, mask)
    return rh, bins

def plot_rankhist(rh, bins, normalize=False):
    fig, ax = plt.subplots()
    if normalize:
        rh = rh / np.sum(rh)
    ax.bar(bins[:-1], rh)
    return fig, ax

# Ensemble definition functions
def persistence_ens(df, target_date):
    fil_now = df['step'] == pd.Timedelta('0min')
    func = lambda x: latest_measurements(x, n=ens_size)
    ens = create_ensemble(df.loc[fil_now, :], func, target_date)
    return ens

def analog_ens_one_val(df, target_date):
    """Analog ensemble only on current values
    """
    func = lambda x: analogs_one_val(x, n=ens_size)
    ens = create_ensemble(df, func, target_date)
    return ens

def delta_t_from_date(df, date):
    delta_t = date - df['base_time']
    delta_t[delta_t < pd.Timedelta(0)] += pd.Timedelta('32d')
    return delta_t

def gen_ens(df, data, ens_func, step=pd.Timedelta('15min')):
    """Leave one day out cross validation
    """
    df_test = df
    test_date = df['date'].unique()
    assert len(test_date) == 1
    test_date = test_date[0]
    print(test_date)
    fil_train = data['date'] != test_date
    df_train = data.loc[fil_train]
    fil_step = df_test['step'] == step

    df_test_meas = df_test.loc[fil_step, ['measurements', 'forecast', 'valid_time_of_day']]
    df_train_test = df_train.merge(df_test_meas, how='left', on='valid_time_of_day', suffixes=('', '_test'))

    df_train_test['delta_t'] = delta_t_from_date(df_train_test, test_date)

    pd.Timestamp('2016-04-10') - df['base_time']

    fil_now = df_test['step'] == pd.Timedelta('0min')
    fil_step = df_test['step'] == pd.Timedelta('15min')
    meas_ts = df_test.loc[fil_now, :].set_index('valid_time')['measurements']

    df_train_test['abs_train_test'] = (df_train_test['forecast'] - df_train_test['forecast_test']).abs()
    # Build ensemble
    ens = ens_func(df_train_test, test_date)

    fil_times = align(meas_ts, test_date, ens)
    ens = ens.loc[fil_times, :]
    return ens

#%% Ensemble size
ens_size = 30
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
df['date'] = df.valid_time.dt.date
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



#%% Loop over step
fil_now = df['step'] == pd.Timedelta('0min')
fil_step = df['step'] == pd.Timedelta('15min')
meas_ts = df.loc[fil_now, :].set_index('valid_time')['measurements']
current_fc = df.loc[fil_step, :].set_index('valid_time')['forecast']

grps_date = df.groupby('date')

models = {'PeEn': persistence_ens, 'AnEn': analog_ens_one_val}
step_range = pd.timedelta_range('15min', '6h', freq='15min')

crps_step = OrderedDict()
for key, ens_func in models.items():
    crps_step[key] = OrderedDict()
    for step in step_range:
        step_min = int(step.total_seconds() / 60)
        ens = grps_date.apply(gen_ens, data=df, ens_func=ens_func, step=step)
        ens['crps'] = properscoring.crps_ensemble(meas_ts, ens)
        crps_step[key][step] = ens['crps'].mean()

        ens = ens.reset_index().set_index('valid_time')
        ens = drop_date(ens)

        plot_ts(meas_ts, current_fc, ens)
        plt.legend([])
        plt.title('{2} CMV+NWP step={0}min n={1}: Arkona, Arpil 2016'.format(step_min, ens_size, key))
        plt.savefig('/home/tzech/ownCloud/Data/plots/ts_{2}_Arkona_2016-04_step_{0}min_n{1}.png'.format(step_min, ens_size, key))

        rh, bins = rankhist(meas_ts, ens)
        plot_rankhist(rh, bins, normalize=True)
        plt.legend([])
        plt.ylabel('$rel.\;freq\;of\;rank$')
        plt.xlabel('$ranks$')
        plt.title('Ranks: {2} CMV+NWP step={0}min n={1}: Arkona, April 2016'.format(step_min, ens_size, key))
        plt.savefig('/home/tzech/ownCloud/Data/plots/RankHist_{2}_Arkona_2016-04_step_{0}min_n{1}.png'.format(step_min, ens_size, key))

#%%
fig, ax = plt.subplots()
x = step_range.total_seconds()/60
y = np.array(list(crps_step['PeEn'].values())) / meas_ts.mean()
ax.plot(x, y)
y = np.array(list(crps_step['AnEn'].values())) / meas_ts.mean()
ax.plot(x, y)
ax.set_xlabel('$step \; [min]$')
ax.set_ylabel('$rel. CRPS_{mean} \; [-]$')
ax.legend(['PeEn', 'AnEn'])
ax.set_title('Analog Ensemble CMV+NWP n={0}: Arkona, April 2016'.format(ens_size))
plt.savefig('/home/tzech/ownCloud/Data/plots/CRPS_AnEn_Arkona_2016-04_n{0}.png'.format(ens_size))
#%%
plt.close('all')
plt.ion()
assert False, 'Done. The remainer is only test code.'
#%% Test code
#   ===========================================================================
test_date = pd.Timestamp('2016-04-30')
fil_test = df['base_time'].dt.date == test_date.date()
df_test = df.loc[fil_test]
#%%
pers = gen_ens(df_test, df, persistence_ens)
#%%
analog = gen_ens(df_test, df, analog_ens_one_val)
#%%
step_str = '15min'
step = pd.Timedelta(step_str)
#%%
grps_date = df.groupby('date')
pers = grps_date.apply(gen_ens, data=df, ens_func=persistence_ens, step=step)
#%%
pers = pers.reset_index().set_index('valid_time')
pers = drop_date(pers)
#%%
fil_now = df['step'] == pd.Timedelta('0min')
fil_step = df['step'] == pd.Timedelta('15min')
meas_ts = df.loc[fil_now, :].set_index('valid_time')['measurements']
current_fc = df.loc[fil_step, :].set_index('valid_time')['forecast']
plot_ts(meas_ts, current_fc, pers)
plt.legend([])
#%%
rh_pers, bins = rankhist(meas_ts, pers)
plot_rankhist(rh_pers, bins, normalize=True)
plt.legend([])
plt.ylabel('$rel.\;freq\;of\;rank$')
plt.xlabel('$ranks$')
plt.title('Ranks: Analog Ensemble CMV+NWP step={1} n={0}: Arkona, June 2016'.format(step_str, ens_size))
#%%
pers['crps'] = properscoring.crps_ensemble(meas_ts, pers)
print("CRPS perEns: {0} for step=15min".format(pers['crps'].mean()))
#%%
analog = grps_date.apply(gen_ens, data=df, ens_func=analog_ens_one_val, step=step)
#%%
analog = analog.reset_index().set_index('valid_time')
analog = drop_date(analog)
#%%
fil_now = df['step'] == pd.Timedelta('0min')
fil_step = df['step'] == pd.Timedelta('15min')
meas_ts = df.loc[fil_now, :].set_index('valid_time')['measurements']
current_fc = df.loc[fil_step, :].set_index('valid_time')['forecast']
plot_ts(meas_ts, current_fc, analog)
plt.legend([])
#%%
rh_analog, bins = rankhist(meas_ts, analog)
plot_rankhist(rh_analog, bins, normalize=True)
plt.legend([])
#%%
analog['crps'] = properscoring.crps_ensemble(meas_ts, analog)
print("CRPS AnEns: {0} for step=15min".format(analog['crps'].mean()))
#%%
msg = 'Nan values for different times between persistence and analog'
assert (pers['crps'].isnull() == analog['crps'].isnull()).all(), msg