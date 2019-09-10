'''
------------------------------------------------------------------------
Generate tables and figures from simulations
------------------------------------------------------------------------
'''

# Import packages
import numpy as np
import PubDebt_latex_tabs as textabs
# import dask
# import multiprocessing
# import scipy.stats as sts
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# from mpl_toolkits.mplot3d import Axes3D
import os

# Create directory if images directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
images_fldr = 'images'
images_dir = os.path.join(cur_path, images_fldr)
if not os.access(images_dir, os.F_OK):
    os.makedirs(images_dir)

# Create OUTPUT directory if does not already exist
output_fldr = 'OUTPUT'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

'''
------------------------------------------------------------------------
Printout table output
------------------------------------------------------------------------
GameOver_p1b      =
med_wt_Hk         =
med_k2t_Hk        =
wt_Hk             =
k2t_Hk            =
GameOver_p1Hk     =
med_Hbar_wtmed_Hk =
med_k20_k2tmed_Hk =
per2GO            =
min_per2GO_Hk     =
med_per2GO_Hk     =
avg_per2GO_Hk     =
max_per2GO_Hk     =
min_cdf2GO_Hk     =
med_cdf2GO_Hk     =
avg_cdf2GO_Hk     =
max_cdf2GO_Hk     =
dict_stats        =
dict_results      =
------------------------------------------------------------------------
'''
outputfile = os.path.join(output_dir, 'results_sims_tA0.pkl')
results_sims = pickle.load(open(outputfile, 'rb'))

dict_params = results_sims['dict_params']
dict_endog = results_sims['dict_endog']

for k, v in dict_params.items():
    exec(k + ' = v')
for k, v in dict_endog.items():
    exec(k + ' = v')

# Table 5.2
med_wt_tk = np.zeros((tau_size, k20_size))
med_Ht_tk = np.zeros_like(med_wt_tk)
med_k2t_tk = np.zeros_like(med_wt_tk)
t0_rbart_an_tk = np.zeros_like(med_wt_tk)
min_rbart_an_tk = np.zeros_like(med_wt_tk)
med_rbart_an_tk = np.zeros_like(med_wt_tk)
avg_rbart_tk = np.zeros_like(med_wt_tk)
avg_rbart_an_tk = np.zeros_like(med_wt_tk)
avg_Rbart_tk = np.zeros_like(med_wt_tk)
avg_Rbart_an_tk = np.zeros_like(med_wt_tk)
max_rbart_an_tk = np.zeros_like(med_wt_tk)
avg_rt_tk = np.zeros_like(med_wt_tk)
avg_rt_an_tk = np.zeros_like(med_wt_tk)
avg_Rt_tk = np.zeros_like(med_wt_tk)
avg_Rt_an_tk = np.zeros_like(med_wt_tk)
std_Rt_tk = np.zeros_like(med_wt_tk)
std_Rt_an_tk = np.zeros_like(med_wt_tk)
avg_eqprem_tk = np.zeros_like(med_wt_tk)
avg_eqprem_an_tk = np.zeros_like(med_wt_tk)
avg_shrp_tk = np.zeros_like(med_wt_tk)
avg_shrp_an_tk = np.zeros_like(med_wt_tk)
t0_rbartcdf_an_tk = np.zeros_like(med_wt_tk)
min_rbartcdf_an_tk = np.zeros_like(med_wt_tk)
med_rbartcdf_an_tk = np.zeros_like(med_wt_tk)
avg_rbartcdf_an_tk = np.zeros_like(med_wt_tk)
max_rbartcdf_an_tk = np.zeros_like(med_wt_tk)
for tau_ind in range(tau_size):
    for k_ind in range(k20_size):
        wt_tk = wt_arr[tau_ind, k_ind, :, 1:]
        Ht_tk = Ht_arr[tau_ind, k_ind, :, 1:]
        k2t_tk = k2t_arr[tau_ind, k_ind, :, 1:]
        rt_tk = rt_arr[tau_ind, k_ind, :, 1:]
        rt_an_tk = (1 + rt_tk) ** (1 / yrs_in_per) - 1
        avg_rt_tk[tau_ind, k_ind] = rt_tk.mean()
        avg_Rt_tk[tau_ind, k_ind] = 1 + avg_rt_tk[tau_ind, k_ind]
        std_Rt_tk[tau_ind, k_ind] = (1 + rt_tk).std()
        avg_rt_an_tk[tau_ind, k_ind] = rt_an_tk.mean()
        avg_Rt_an_tk[tau_ind, k_ind] = 1 + avg_rt_an_tk[tau_ind, k_ind]
        std_Rt_an_tk[tau_ind, k_ind] = (1 + rt_an_tk).std()
        rbart_tk = rbart_arr[tau_ind, k_ind, :, 1:]
        rbart_an_tk = rbart_an_arr[tau_ind, k_ind, :, 1:]
        med_wt_tk[tau_ind, k_ind] = np.median(wt_tk)
        med_Ht_tk[tau_ind, k_ind] = np.median(Ht_tk)
        med_k2t_tk[tau_ind, k_ind] = np.median(k2t_tk)
        t0_rbart_an_tk[tau_ind, k_ind] = rbart_an_arr[tau_ind, k_ind, 0,
                                                      0]
        t0_rbartcdf_an_tk[tau_ind, k_ind] = \
            ((rbart_an_tk <= t0_rbart_an_tk[tau_ind, k_ind]).sum() /
             len(rbart_an_tk.flatten()))
        min_rbart_an_tk[tau_ind, k_ind] = rbart_an_tk.min()
        min_rbartcdf_an_tk[tau_ind, k_ind] = \
            ((rbart_an_tk <= min_rbart_an_tk[tau_ind, k_ind]).sum() /
             len(rbart_an_tk.flatten()))
        med_rbart_an_tk[tau_ind, k_ind] = np.median(rbart_an_tk)
        med_rbartcdf_an_tk[tau_ind, k_ind] = \
            ((rbart_an_tk <= med_rbart_an_tk[tau_ind, k_ind]).sum() /
             len(rbart_an_tk.flatten()))
        avg_rbart_tk[tau_ind, k_ind] = rbart_tk.mean()
        avg_rbart_an_tk[tau_ind, k_ind] = rbart_an_tk.mean()
        avg_Rbart_tk[tau_ind, k_ind] = 1 + avg_rbart_tk[tau_ind, k_ind]
        avg_Rbart_an_tk[tau_ind, k_ind] = \
            1 + avg_rbart_an_tk[tau_ind, k_ind]
        avg_rbartcdf_an_tk[tau_ind, k_ind] = \
            ((rbart_an_tk <= avg_rbart_an_tk[tau_ind, k_ind]).sum() /
             len(rbart_an_tk.flatten()))
        max_rbart_an_tk[tau_ind, k_ind] = rbart_an_tk.max()
        max_rbartcdf_an_tk[tau_ind, k_ind] = \
            ((rbart_an_tk <= max_rbart_an_tk[tau_ind, k_ind]).sum() /
             len(rbart_an_tk.flatten()))
        avg_eqprem_tk[tau_ind, k_ind] = (avg_Rt_tk[tau_ind, k_ind] -
                                         avg_Rbart_tk[tau_ind, k_ind])
        avg_eqprem_an_tk[tau_ind, k_ind] = \
            (avg_Rt_an_tk[tau_ind, k_ind] -
             avg_Rbart_an_tk[tau_ind, k_ind])
        avg_shrp_tk[tau_ind, k_ind] = \
            ((avg_Rt_tk[tau_ind, k_ind] - avg_Rbart_tk[tau_ind, k_ind]) /
             std_Rt_tk[tau_ind, k_ind])
        avg_shrp_an_tk[tau_ind, k_ind] = \
            ((avg_Rt_an_tk[tau_ind, k_ind] -
              avg_Rbart_an_tk[tau_ind, k_ind]) /
             std_Rt_an_tk[tau_ind, k_ind])

med_Htmed_wtmed_tk = med_Ht_tk / med_wt_tk
med_k20_k2tmed_tk = np.tile(k20_vec.reshape((1, k20_size)),
                            (tau_size, 1)) / med_k2t_tk

print('med_wt_tk')
print(med_wt_tk)
print('med_k2t_tk')
print(med_k2t_tk)
print('med_Htmed_wtmed_tk')
print(med_Htmed_wtmed_tk)
print('med_k20_k2tmed_tk')
print(med_k20_k2tmed_tk)
four_matrices = (med_wt_tk, med_k2t_tk, med_Htmed_wtmed_tk,
                 med_k20_k2tmed_tk)
textabs.print_latex_tab_tau_med(tau_vec, k20_vec, four_matrices)


# Print statistics on rbar
print('t0_rbart_an_tk')
print(t0_rbart_an_tk)
print('min_rbart_an_tk')
print(min_rbart_an_tk)
print('med_rbart_an_tk')
print(med_rbart_an_tk)
print('avg_rbart_an_tk')
print(avg_rbart_an_tk)
print('max_rbart_an_tk')
print(max_rbart_an_tk)
print('t0_rbartcdf_an_tk')
print(t0_rbartcdf_an_tk)
print('min_rbartcdf_an_tk')
print(min_rbartcdf_an_tk)
print('med_rbartcdf_an_tk')
print(med_rbartcdf_an_tk)
print('avg_rbartcdf_an_tk')
print(avg_rbartcdf_an_tk)
print('max_rbartcdf_an_tk')
print(max_rbartcdf_an_tk)
rate_matrices = (t0_rbart_an_tk * 100, min_rbart_an_tk * 100,
                 med_rbart_an_tk * 100, avg_rbart_an_tk * 100,
                 max_rbart_an_tk * 100)
cdf_matrices = (t0_rbartcdf_an_tk, min_rbartcdf_an_tk,
                med_rbartcdf_an_tk, avg_rbartcdf_an_tk,
                max_rbartcdf_an_tk)
textabs.print_latex_tab_tau_riskl(tau_vec, k20_vec, rate_matrices,
                                  cdf_matrices)

# Create equity premium table

print('avg_Rt_tk')
print(avg_Rt_tk)
print('std_Rt_tk')
print(std_Rt_tk)
print('avg_Rbart_tk')
print(avg_Rbart_tk)
print('avg_eqprem_tk')
print(avg_eqprem_tk)
print('avg_shrp_tk')
print(avg_shrp_tk)

print('avg_Rt_an_tk')
print(avg_Rt_an_tk)
print('std_Rt_an_tk')
print(std_Rt_an_tk)
print('avg_Rbart_an_tk')
print(avg_Rbart_an_tk)
print('avg_eqprem_an_tk')
print(avg_eqprem_an_tk)
print('avg_shrp_an_tk')
print(avg_shrp_an_tk)

per_matrices = (avg_Rt_tk * 100, std_Rt_tk * 100, avg_Rbart_tk * 100,
                avg_eqprem_tk * 100,
                avg_shrp_tk)
an_matrices = (avg_Rt_an_tk * 100, std_Rt_an_tk * 100,
               avg_Rbart_an_tk * 100, avg_eqprem_an_tk * 100,
               avg_shrp_an_tk)
textabs.print_latex_tab_tau_eqprem(tau_vec, k20_vec, per_matrices,
                                   an_matrices)


dict_tabs = \
    {'med_wt_tk': med_wt_tk, 'med_k2t_tk': med_k2t_tk,
     'med_Htmed_wtmed_tk': med_Htmed_wtmed_tk,
     'med_k20_k2tmed_tk': med_k20_k2tmed_tk,
     't0_rbart_an_tk': t0_rbart_an_tk,
     'min_rbart_an_tk': min_rbart_an_tk,
     'med_rbart_an_tk': med_rbart_an_tk,
     'avg_rbart_tk': avg_rbart_tk,
     'avg_rbart_an_tk': avg_rbart_an_tk,
     'avg_Rbart_tk': avg_Rbart_tk,
     'avg_Rbart_an_tk': avg_Rbart_an_tk,
     'max_rbart_an_tk': max_rbart_an_tk,
     'avg_rt_tk': avg_rt_tk,
     'avg_rt_an_tk': avg_rt_an_tk,
     'avg_Rt_tk': avg_Rt_tk,
     'avg_Rt_an_tk': avg_Rt_an_tk,
     'std_Rt_tk': std_Rt_tk, 'std_Rt_an_tk': std_Rt_an_tk,
     't0_rbartcdf_an_tk': t0_rbartcdf_an_tk,
     'min_rbartcdf_an_tk': min_rbartcdf_an_tk,
     'med_rbartcdf_an_tk': med_rbartcdf_an_tk,
     'avg_rbartcdf_an_tk': avg_rbartcdf_an_tk,
     'max_rbartcdf_an_tk': max_rbartcdf_an_tk,
     'avg_eqprem_tk': avg_eqprem_tk,
     'avg_eqprem_an_tk': avg_eqprem_an_tk,
     'avg_shrp_tk': avg_shrp_tk, 'avg_shrp_an_tk': avg_shrp_an_tk}
results_tabs = {'dict_params': dict_params, 'dict_endog': dict_endog,
                'dict_stats': dict_tabs}

outputfile = os.path.join(output_dir, 'results_tabs_tA0.pkl')
pickle.dump(results_tabs, open(outputfile, 'wb'))
