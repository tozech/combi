# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import scipy.stats as st
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import properscoring

# %matplotlib inline

# %%
plt.plot(np.linspace(0, 1, 10))

# %%
np.random.seed(42)

# %% [markdown]
# # Actual distribution

# %%
mu_G = np.array([0, 0])
sigma_G = 1
rho_G = 0.8
cov_G = np.array([[1, rho_G], [rho_G, 1]])

def G(m):
    return np.random.multivariate_normal(mu_G, sigma_G**2 * cov_G, m)



# %%
#Alternative: st.multivariate_normal?

# %%
m = 10000
samples_G = G(m)

# %%
df_G = pd.DataFrame.from_records(samples_G, columns=['x', 'y'])

# %%
sns.jointplot(x="x", y="y", data=df_G, marker=".", alpha=0.3).plot_joint(sns.kdeplot, zorder=0, n_levels=6);

# %% [markdown]
# ### Error in mean

# %%
mu_Fs = np.linspace(-15, 5, 10)
sigma_F = sigma_G
rho_F = 0.5

def F(m, mu, sigma, rho):
    mu_F = np.tile(mu, 2)
    sigma_F = sigma
    cov_F = np.array([[1, rho], [rho, 1]])
    df = pd.DataFrame.from_records(np.random.multivariate_normal(mu_F, sigma_F**2 * cov_F, m), 
                                   columns=['F_x', 'F_y'])
    df = df.join(pd.DataFrame.from_records(np.random.multivariate_normal(mu_F, sigma_F**2 * cov_F, m), 
                                           columns=['F_prime_x', 'F_prime_y']))
    df['mu'] = mu
    df['sigma'] = sigma
    df['sigma_err'] = 1 - sigma**2
    df['rho'] = rho
    df['G_x'] = df_G['x']
    df['G_y'] = df_G['y']
    return df



# %%
df_all = pd.concat([F(m, x, sigma_F, rho_F) for x in mu_Fs])

# %%
sns.scatterplot(x='F_x', y='F_y', hue='mu', data=df_all[df_all.mu < -10], marker=".", alpha=0.3)


# %%
def compute(df_all):
    df_all['err_x'] = df_all['F_x'] - df_all['G_x']
    df_all['err_y'] = df_all['F_y'] - df_all['G_y']
    df_all['abs_err_x'] = df_all['err_x'].abs()
    df_all['abs_err_y'] = df_all['err_y'].abs()
    df_all['sum_F'] = df_all['F_x'] + df_all['F_y']
    df_all['sum_F_prime'] = df_all['F_prime_x'] + df_all['F_prime_y']
    df_all['sum_G'] = df_all['G_x'] + df_all['G_y']
    df_all['err_sum'] = df_all['sum_F'] - df_all['sum_G']
    df_all['abs_err_sum'] = df_all['err_sum'].abs()
    df_all['rm_err'] = np.sqrt(df_all['err_x']**2 + df_all['err_y']**2)
    df_all['delta_prime_x'] = df_all['F_x'] - df_all['F_prime_x']
    df_all['delta_prime_y'] = df_all['F_y'] - df_all['F_prime_y']
    df_all['delta_prime_sum'] = df_all['sum_F'] - df_all['sum_F_prime']
    df_all['abs_delta_prime_x'] = df_all['delta_prime_x'].abs()
    df_all['abs_delta_prime_y'] = df_all['delta_prime_y'].abs()
    df_all['abs_delta_prime_sum'] = df_all['delta_prime_sum'].abs()
    df_all['rm_delta_prime'] = np.sqrt(df_all['delta_prime_x']**2 + df_all['delta_prime_y']**2)
    return df_all


# %%
df_all = compute(df_all)

# %%
df_all.head()


# %%
def plot_energy_score(df_all, x='mu'):
    means = df_all.groupby(x).mean()
    means['ES'] = means['rm_err'] - 1/2* means['rm_delta_prime']
    means['ES'].plot()
    return means


# %%
means = plot_energy_score(df_all, x='mu')


# %%
def plot_crps(df_all, x='mu'):
    means = df_all.groupby(x).mean()
    means['crps_x'] = means['abs_err_x'] - 1/2 * means['abs_delta_prime_x']
    means['crps_y'] = means['abs_err_y'] - 1/2 * means['abs_delta_prime_y']
    means['crps_sum'] = means['abs_err_sum'] - 1/2 * means['abs_delta_prime_sum']
    fig, ax = plt.subplots()
    means[['crps_x', 'crps_y', 'crps_sum']].plot(ax=ax)
    return means


# %%
plot_crps(df_all, x='mu')

# %%
assert False

# %% [markdown]
# ### Error in variance

# %%
mu_F = 0.
sigma_Fs = np.concatenate([np.linspace(0.1, 0.9, 9), np.linspace(1, 10, 10)])
rho_F = 0.5

# %%
df_all = pd.concat([F(m, mu_F, x, rho_F) for x in sigma_Fs])

# %%
df_all = compute(df_all)

# %%
#sns.scatterplot(x='F_x', y='F_y', hue='sigma', data=df_all, marker=".", alpha=0.3)

# %%
means = plot_energy_score(df_all, x='sigma_err')

# %%
plot_crps(df_all, x='sigma_err')

# %% [markdown]
# ### Error in correlation

# %%
mu_F = 0.
sigma_F = 1
rho_Fs = np.linspace(0, 1., 10)

# %%
df_all = pd.concat([F(m, mu_F, sigma_F, x) for x in rho_Fs])

# %%
df_all = compute(df_all)

# %%
#sns.scatterplot(x='F_x', y='F_y', hue='rho', data=df_all, marker=".", alpha=0.3)

# %%
means = plot_energy_score(df_all, x='rho')

# %%
plot_crps(df_all, x='rho')
