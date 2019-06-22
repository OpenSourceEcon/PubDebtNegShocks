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
outputfile = os.path.join(output_dir, 'results_sims_A0.pkl')
results_sims = pickle.load(open(outputfile, 'rb'))

dict_params = results_sims['dict_params']
dict_endog = results_sims['dict_endog']

for k, v in dict_params.items():
    exec(k + ' = v')
for k, v in dict_endog.items():
    exec(k + ' = v')

GameOver_p1 = \
    np.append(np.zeros((Hbar_size, k20_size, S, 1), dtype=bool),
              GameOver_arr[:, :, :, 1:], axis=3)
GameOver_p1b = GameOver_p1 == 1

# Table 5.2
med_wt_Hk = np.zeros((Hbar_size, k20_size))
med_k2t_Hk = np.zeros((Hbar_size, k20_size))
t0_rbart_an_Hk = np.zeros((Hbar_size, k20_size))
min_rbart_an_Hk = np.zeros((Hbar_size, k20_size))
med_rbart_an_Hk = np.zeros((Hbar_size, k20_size))
avg_rbart_an_Hk = np.zeros((Hbar_size, k20_size))
max_rbart_an_Hk = np.zeros((Hbar_size, k20_size))
t0_rbartcdf_an_Hk = np.zeros((Hbar_size, k20_size))
min_rbartcdf_an_Hk = np.zeros((Hbar_size, k20_size))
med_rbartcdf_an_Hk = np.zeros((Hbar_size, k20_size))
avg_rbartcdf_an_Hk = np.zeros((Hbar_size, k20_size))
max_rbartcdf_an_Hk = np.zeros((Hbar_size, k20_size))
for H_ind in range(Hbar_size):
    for k_ind in range(k20_size):
        wt_Hk = wt_arr[H_ind, k_ind, :, 1:]
        k2t_Hk = k2t_arr[H_ind, k_ind, :, 1:]
        rbart_an_Hk = rbart_an_arr[H_ind, k_ind, :, 1:]
        GameOver_p1Hk = GameOver_p1b[H_ind, k_ind, :, 1:]
        med_wt_Hk[H_ind, k_ind] = np.median(wt_Hk[~GameOver_p1Hk])
        med_k2t_Hk[H_ind, k_ind] = np.median(k2t_Hk[~GameOver_p1Hk])
        t0_rbart_an_Hk[H_ind, k_ind] = rbart_an_arr[H_ind, k_ind, 0, 0]
        t0_rbartcdf_an_Hk[H_ind, k_ind] = \
            ((rbart_an_Hk[~GameOver_p1Hk] <=
              t0_rbart_an_Hk[H_ind, k_ind]).sum() /
             len(rbart_an_Hk[~GameOver_p1Hk]))
        min_rbart_an_Hk[H_ind, k_ind] = \
            rbart_an_Hk[~GameOver_p1Hk].min()
        min_rbartcdf_an_Hk[H_ind, k_ind] = \
            ((rbart_an_Hk[~GameOver_p1Hk] <=
              min_rbart_an_Hk[H_ind, k_ind]).sum() /
             len(rbart_an_Hk[~GameOver_p1Hk]))
        med_rbart_an_Hk[H_ind, k_ind] = \
            np.median(rbart_an_Hk[~GameOver_p1Hk])
        med_rbartcdf_an_Hk[H_ind, k_ind] = \
            ((rbart_an_Hk[~GameOver_p1Hk] <=
              med_rbart_an_Hk[H_ind, k_ind]).sum() /
             len(rbart_an_Hk[~GameOver_p1Hk]))
        avg_rbart_an_Hk[H_ind, k_ind] = \
            rbart_an_Hk[~GameOver_p1Hk].mean()
        avg_rbartcdf_an_Hk[H_ind, k_ind] = \
            ((rbart_an_Hk[~GameOver_p1Hk] <=
              avg_rbart_an_Hk[H_ind, k_ind]).sum() /
             len(rbart_an_Hk[~GameOver_p1Hk]))
        max_rbart_an_Hk[H_ind, k_ind] = \
            rbart_an_Hk[~GameOver_p1Hk].max()
        max_rbartcdf_an_Hk[H_ind, k_ind] = \
            ((rbart_an_Hk[~GameOver_p1Hk] <=
              max_rbart_an_Hk[H_ind, k_ind]).sum() /
             len(rbart_an_Hk[~GameOver_p1Hk]))

med_Hbar_wtmed_Hk = np.tile(Hbar_vec.reshape((Hbar_size, 1)),
                            (1, k20_size)) / med_wt_Hk
med_k20_k2tmed_Hk = np.tile(k20_vec.reshape((1, k20_size)),
                            (Hbar_size, 1)) / med_k2t_Hk

print('med_wt_Hk')
print(med_wt_Hk)
print('med_k2t_Hk')
print(med_k2t_Hk)
print('med_Hbar_wtmed_Hk')
print(med_Hbar_wtmed_Hk)
print('med_k20_k2tmed_Hk')
print(med_k20_k2tmed_Hk)
four_matrices = (med_wt_Hk, med_k2t_Hk, med_Hbar_wtmed_Hk,
                 med_k20_k2tmed_Hk)
textabs.print_latex_tab_med(Hbar_vec, k20_vec, four_matrices)


# Create histogram of periods to game over for each specification
num_bins = 30
for H_ind in range(Hbar_size):
    for k_ind in range(k20_size):
        per2GO_Hk = GameOver_arr[H_ind, k_ind, :, :].argmax(axis=1)

        # Plot steady-state consumption and savings distributions
        fig, ax = plt.subplots()
        n, bin_cuts, patches = plt.hist(per2GO_Hk, num_bins,
                                        edgecolor='k', density=True)
        plt.title(r'Histogram of periods to game over across ' +
                  'simulations \n for $H$=' + str(Hbar_vec[H_ind]) +
                  ', and $k_{2,0}$=' + str(k20_vec[k_ind]))
        plt.xlabel(r'Periods to game over')
        plt.xticks((1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
                   ('1', '5', '10', '15', '20', '25', '30', '35', '40',
                    '45', '50'))
        plt.ylabel(r'Percent of simulations ending in $p$ periods')
        plt.yticks((0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                    0.40, 0.45, 0.50),
                   ('0%', '5%', '10%', '15%', '20%', '25%', '30%',
                    '35%', '40%', '45%', '50%'))
        plt.grid(b=True, which='major', color='0.65', linestyle=':')
        image_name = 'per2GOhist_' + str(H_ind) + str(k_ind) + '.png'
        fig_path = os.path.join(images_dir, image_name)
        plt.savefig(fig_path)
        # plt.show()
        plt.close()


# Table 5.3
per2GO = GameOver_arr.argmax(axis=3)
per2GO[per2GO == 0] = T
min_per2GO_Hk = per2GO.min(axis=2)
med_per2GO_Hk = np.median(per2GO, axis=2)
avg_per2GO_Hk = per2GO.mean(axis=2)
max_per2GO_Hk = per2GO.max(axis=2)

print('min_per2GO_Hk')
print(min_per2GO_Hk)
print('med_per2GO_Hk')
print(med_per2GO_Hk)
print('avg_per2GO_Hk')
print(avg_per2GO_Hk)
print('max_per2GO_Hk')
print(max_per2GO_Hk)

min_cdf2GO_Hk = np.zeros((Hbar_size, k20_size))
med_cdf2GO_Hk = np.zeros((Hbar_size, k20_size))
avg_cdf2GO_Hk = np.zeros((Hbar_size, k20_size))
max_cdf2GO_Hk = np.zeros((Hbar_size, k20_size))
for H_ind in range(Hbar_size):
    for k_ind in range(k20_size):
        min_cdf2GO_Hk[H_ind, k_ind] = \
            (per2GO[H_ind, k_ind, :] <=
             min_per2GO_Hk[H_ind, k_ind]).sum() / S
        med_cdf2GO_Hk[H_ind, k_ind] = \
            (per2GO[H_ind, k_ind, :] <=
             med_per2GO_Hk[H_ind, k_ind]).sum() / S
        # Linearly interpolate CDF of average between value below and
        # value above
        pct = 1 - (((per2GO[H_ind, k_ind, :] <=
                     avg_per2GO_Hk[H_ind, k_ind]).sum() / S) % 1)
        avg_cdf2GO_Hk[H_ind, k_ind] = \
            (pct * ((per2GO[H_ind, k_ind, :] <=
                     avg_per2GO_Hk[H_ind, k_ind]).sum() / S) +
             (1 - pct) * ((per2GO[H_ind, k_ind, :] <=
                           avg_per2GO_Hk[H_ind, k_ind] + 1).sum() / S))
        max_cdf2GO_Hk[H_ind, k_ind] = \
            (per2GO[H_ind, k_ind, :] <=
             max_per2GO_Hk[H_ind, k_ind]).sum() / S

print('min_cdf2GO_Hk')
print(min_cdf2GO_Hk)
print('med_cdf2GO_Hk')
print(med_cdf2GO_Hk)
print('avg_cdf2GO_Hk')
print(avg_cdf2GO_Hk)
print('max_cdf2GO_Hk')
print(max_cdf2GO_Hk)
per_matrices = (min_per2GO_Hk, med_per2GO_Hk, avg_per2GO_Hk,
                max_per2GO_Hk)
cdf_matrices = (min_cdf2GO_Hk, med_cdf2GO_Hk, avg_cdf2GO_Hk,
                max_cdf2GO_Hk)
textabs.print_latex_tab_shut(Hbar_vec, k20_vec, per_matrices,
                             cdf_matrices)

# Print statistics on rbar
print('t0_rbart_an_Hk')
print(t0_rbart_an_Hk)
print('min_rbart_an_Hk')
print(min_rbart_an_Hk)
print('med_rbart_an_Hk')
print(med_rbart_an_Hk)
print('avg_rbart_an_Hk')
print(avg_rbart_an_Hk)
print('max_rbart_an_Hk')
print(max_rbart_an_Hk)
print('t0_rbartcdf_an_Hk')
print(t0_rbartcdf_an_Hk)
print('min_rbartcdf_an_Hk')
print(min_rbartcdf_an_Hk)
print('med_rbartcdf_an_Hk')
print(med_rbartcdf_an_Hk)
print('avg_rbartcdf_an_Hk')
print(avg_rbartcdf_an_Hk)
print('max_rbartcdf_an_Hk')
print(max_rbartcdf_an_Hk)
rate_matrices = (t0_rbart_an_Hk * 100, min_rbart_an_Hk * 100,
                 med_rbart_an_Hk * 100, avg_rbart_an_Hk * 100,
                 max_rbart_an_Hk * 100)
cdf_matrices = (t0_rbartcdf_an_Hk, min_rbartcdf_an_Hk,
                med_rbartcdf_an_Hk, avg_rbartcdf_an_Hk,
                max_rbartcdf_an_Hk)
textabs.print_latex_tab_riskl(Hbar_vec, k20_vec, rate_matrices,
                              cdf_matrices)

dict_tabs = \
    {'med_wt_Hk': med_wt_Hk, 'med_k2t_Hk': med_k2t_Hk,
     'med_Hbar_wtmed_Hk': med_Hbar_wtmed_Hk,
     'med_k20_k2tmed_Hk': med_k20_k2tmed_Hk,
     'min_per2GO_Hk': min_per2GO_Hk, 'med_per2GO_Hk': med_per2GO_Hk,
     'avg_per2GO_Hk': avg_per2GO_Hk, 'max_per2GO_Hk': max_per2GO_Hk,
     'min_cdf2GO_Hk': min_cdf2GO_Hk, 'med_cdf2GO_Hk': med_cdf2GO_Hk,
     'avg_cdf2GO_Hk': avg_cdf2GO_Hk, 'max_cdf2GO_Hk': max_cdf2GO_Hk,
     't0_rbart_an_Hk': t0_rbart_an_Hk,
     'min_rbart_an_Hk': min_rbart_an_Hk,
     'med_rbart_an_Hk': med_rbart_an_Hk,
     'avg_rbart_an_Hk': avg_rbart_an_Hk,
     'max_rbart_an_Hk': max_rbart_an_Hk,
     't0_rbartcdf_an_Hk': t0_rbartcdf_an_Hk,
     'min_rbartcdf_an_Hk': min_rbartcdf_an_Hk,
     'med_rbartcdf_an_Hk': med_rbartcdf_an_Hk,
     'avg_rbartcdf_an_Hk': avg_rbartcdf_an_Hk,
     'max_rbartcdf_an_Hk': max_rbartcdf_an_Hk}
results_tabs = {'dict_params': dict_params, 'dict_endog': dict_endog,
                'dict_stats': dict_tabs}

outputfile = os.path.join(output_dir, 'results_tabs_A0.pkl')
pickle.dump(results_tabs, open(outputfile, 'wb'))
