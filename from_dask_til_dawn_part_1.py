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
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline

# %%
@np.vectorize
def function(x: float):
    assert x >= 0.
    assert x <= 100.
    if x <= 99:
        ret = x / 100. * 9.81
    else:
        ret = 0
    return ret


# %%
n_train = 101
x = np.linspace(0, 100, n_train)
x

# %%
y = function(x)

# %%
plt.plot(x, y, '+')


# %%
def add_noise(arr, scale=1):
    white_noise = np.random.randn((len(arr)))
    return arr + scale * white_noise


# %%
y_noise = add_noise(y)
plt.plot(x, y_noise, '+')
plt.savefig('few.png', dpi=300)

# %%
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(x.reshape(n_train, 1), y_noise)
reg.coef_

# %%
n_test = 100
test_x = np.linspace(0, 100, n_test)

# %%
y_linreg = reg.predict(test_x.reshape((n_test, 1)))

# %%
plt.plot(x, y_noise, '+')
plt.plot(test_x, y_linreg, '-')
plt.savefig('few_linreg.png', dpi=300)

# %%
n_train_many = 100000
x = np.linspace(0, 100, n_train_many)

# %%
y = function(x)
y_noise = add_noise(y)
reg.fit(x.reshape(n_train_many, 1), y_noise)
y_linreg = reg.predict(test_x.reshape((n_test, 1)))
reg.coef_

# %%
#fig, ax = plt.subplots(figsize=(20, 12))
plt.hexbin(x, y_noise, mincnt=1)
plt.plot(test_x, y_linreg, '-', color='tab:red')
plt.savefig('many_linreg.png', dpi=300)

# %%
test_actual = function(test_x)
resi = y_linreg - test_actual
rmse = np.sqrt(np.mean(resi**2))
plt.plot(test_x, resi)
rmse

# %%
from sklearn import neighbors
n_neighbors = 100
weights = 'uniform' # 'distance' #
knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
knn.fit(x.reshape(n_train_many, 1), y_noise)
y_knn = knn.predict(test_x.reshape((n_test, 1)))

# %%
y_knn

# %%
resi = y_knn - test_actual
rmse = np.sqrt(np.mean(resi**2))
plt.plot(test_x, resi)
rmse

# %%
plt.hexbin(x, y_noise, mincnt=1)
plt.plot(x, knn.predict(x.reshape((n_train_many, 1))), '-', color='tab:red')
plt.savefig('many_knn.png', dpi=300)

# %%
#TODO Repeat with Dask ML
