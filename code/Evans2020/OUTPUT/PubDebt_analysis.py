'''
This script analyzes the results from the two-period overlapping
generations model
'''

# Import packages
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

results = pickle.load(open('results_sims.pkl', 'rb'))
dict_params = results['dict_params']
dict_endog = results['dict_endog']

# Unpack dictionaries
print('dict_params keys:', dict_params.keys())
for k, v in dict_params.items():
    exec(k + ' = v')
print('dict_endog keys:', dict_endog.keys())
for k, v in dict_endog.items():
    exec(k + ' = v')

# Calculate average annual interest rates
rt_an_arr = (1 + rt_arr) ** (1 / yrs_in_per) - 1
rt_an_arr_avg = np.zeros((Hbar_size, k20_size, x1_size))
rbart_an_arr_avg = np.zeros((Hbar_size, k20_size, x1_size))
for H_ind in range(Hbar_size):
    for k_ind in range(k20_size):
        for x_ind in range(x1_size):
            rt_an_mat = rt_an_arr[H_ind, k_ind, x_ind, :, :]
            rbart_an_mat = rbart_an_arr[H_ind, k_ind, x_ind, :, :]
            default_mat = default_arr[H_ind, k_ind, x_ind, :, :]
            nodefault_mat = default_mat == 0
            rt_an_arr_avg[H_ind, k_ind, x_ind] = \
                rt_an_mat[nodefault_mat].mean()
            rbart_an_arr_avg[H_ind, k_ind, x_ind] = \
                rbart_an_mat[nodefault_mat].mean()

print('rt_an_arr_avg x1=0.0:')
print(rt_an_arr_avg[:, :, 0])
print('')

print('rbart_an_arr_avg x1=0.0:')
print(rbart_an_arr_avg[:, :, 0])
print('')

print('An int rate spread, x1=0.0:')
print(rt_an_arr_avg[:, :, 0] - rbart_an_arr_avg[:, :, 0])
print('')

print('rt_an_arr_avg x1=0.1:')
print(rt_an_arr_avg[:, :, 1])
print('')

print('rbart_an_arr_avg x1=0.1:')
print(rbart_an_arr_avg[:, :, 1])
print('')

print('An int rate spread, x1=0.1:')
print(rt_an_arr_avg[:, :, 1] - rbart_an_arr_avg[:, :, 1])
print('')

# Plot interest rate differentials
