'''
------------------------------------------------------------------------
Simulate many runs of the economy under different parameterizations for
Hbar and k20, assuming z0=mu
------------------------------------------------------------------------
'''

# Import packages
import time
import numpy as np
import scipy.stats as sts
import multiprocessing
from dask import delayed
from dask.distributed import Client
import pickle
import PubDebt_funcs as funcs
import PubDebt_parameters as params

import os


# Create OUTPUT directory if does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'OUTPUT'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

'''
------------------------------------------------------------------------
Set exogenous parameters
------------------------------------------------------------------------
yrs_in_per = integer >= 1, number of years in a model period
beta_an    = scalar in (0, 1), annual discount factor
beta       = scalar in (0, 1), model period discount factor
gamma      = scalar >= 1, coefficient of relative risk aversion
c_min      = scalar > 0, minimum individual consumption
K_min      = scalar > 0, minimum aggregate capital stock
n_1        = scalar >= 0, exogenous labor supply when young
n_2        = scalar >= 0, exogenous labor supply when old
n_vec      = (2,) vector, lifetime exogenous labor supply
alpha      = scalar in (0, 1), capital share of income
delta_an   = scalar in (0, 1], annual depreciation rate
delta      = scalar in (0, 1], model period depreciation rate
rho_an     = scalar in (-1, 1), annual persistence of normally
             distributed TFP process
rho        = scalar in (-1, 1), model period persistence of normally
             distributed TFP process
mu         = scalar, unconditional mean of normally distributed TFP
             process
sigma_an   = scalar > 0, annual standard deviation of normally
             distributed TFP process
rho_sum    = scalar, recursive geometric sum of rho ** 2
y_ind      = integer >= 0, index of yrs_in_per
sigma      = scalar > 0, model period standard deviation of normally
             distributed TFP process
A_min      = scalar >= 0, minimum value in support of lognormally
             distributed TFP process
z_min      = scalar, minimum value in support of normally distributed
             TFP process
Hbar_vec   = (Hbar_size,) vector, values of Hbar
Hbar_size  = integer >= 1, number of values of Hbar
k20_vec    = (Hbar_size,) vector, values of k20
k20_size   = integer >= 1, number of values of k20_vec
z0         = scalar, initial normally distributed TFP value
T          = integer > 1, maximum number of periods to simulate
S          = integer >= 1, number of simulations
rand_seed  = integer > 0, random seed for simulation
------------------------------------------------------------------------
'''
p = params.parameters()

# Set up parallel processing
max_cores = multiprocessing.cpu_count()
print('Cores available on this machine =', max_cores)
num_workers = min(max_cores, p.S)
print('Number of workers =', num_workers)
client = Client(processes=False)


'''
------------------------------------------------------------------------
Simulate series
------------------------------------------------------------------------
start_time   =
GameOver_arr =
unif_mat     =
zt_mat       =
T_ind        =
cut_lb_vec   =
eps_t_vec    =
c1t_arr      =
c2t_arr      =
Ht_arr       =
rt_arr       =
k2t_arr      =
k_ind        =
H_ind        =
k2tp1_args   =
S_ind        =
GameOver     =
k2t          =
zt           =
k2tp1        =
c1t          =
Ht           =
c2t          =
wt           =
rt           =
elapsed_time =
GameOver_p1  =
zt_arr       =
Kt_arr       =
Yt_arr       =
Ct_arr       =
dict_params  =
dict_endog   =
------------------------------------------------------------------------
'''
start_time = time.process_time()
default_arr = np.zeros((p.Hbar_size, p.k20_size, p.x1_size, p.S, p.T),
                       dtype=bool)
unif_mat = \
    sts.uniform.rvs(loc=0, scale=1, size=((p.S, p.T - 1)),
                    random_state=p.rand_seed)
zt_mat = np.zeros((p.S, p.T))
zt_mat[:, 0] = p.z0
for t_ind in range(1, p.T):
    cut_lb_vec = (p.z_min - p.rho * zt_mat[:, t_ind - 1] -
                  (1 - p.rho) * p.mu)
    eps_t_vec = funcs.trunc_norm_draws(unif_mat[:, t_ind - 1], 0,
                                       p.sigma, cut_lb_vec)
    zt_mat[:, t_ind] = (p.rho * zt_mat[:, t_ind - 1] +
                        (1 - p.rho) * p.mu + eps_t_vec)

c1t_arr = np.zeros_like(default_arr)
c2t_arr = np.zeros_like(default_arr)
Ht_arr = np.zeros_like(default_arr)
wt_arr = np.zeros_like(default_arr)
rt_arr = np.zeros_like(default_arr)
k2t_arr = np.zeros_like(default_arr)
rbart_arr = np.zeros_like(default_arr)
rbart_an_arr = np.zeros_like(default_arr)
EulErr_arr = np.zeros_like(default_arr)
PathTime_arr = np.zeros((p.Hbar_size, p.k20_size, p.x1_size, p.S))
S_ind_arr = np.zeros((p.Hbar_size, p.k20_size, p.x1_size, p.S))
for k_ind in range(p.k20_size):
    k2t_arr[:, k_ind, :, :, 0] = p.k20_vec[k_ind]

for H_ind in range(p.Hbar_size):
    p.Hbar = p.Hbar_vec[H_ind]
    for k_ind in range(p.k20_size):
        p.k20 = p.k20_vec[k_ind]
        for x1_ind in range(p.x1_size):
            p.x1 = p.x1_vec[x1_ind]
            simulations = []
            for S_ind in range(p.S):
                # (H_ind, k_ind, x1_ind, S_ind, zt_vec, default_vec,
                #     c1t_vec, c2t_vec, Ht_vec, wt_vec, rt_vec, k2t_vec,
                #     rbart_vec, rbart_an_vec, EulErr_vec, path_time) = \
                timepaths_s = delayed(funcs.sim_timepath)(
                    p, H_ind=H_ind, k_ind=k_ind, x1_ind=x1_ind,
                    S_ind=S_ind, zt_vec=zt_mat[S_ind, :],
                    rand_seed=p.rand_seed)
                simulations.append(timepaths_s)

            simulations = delayed(simulations).compute()

            for S_ind in range(p.S):
                S_ind_arr[H_ind, k_ind, x1_ind, S_ind] = \
                    simulations[S_ind][3]  # original S_ind
                default_arr[H_ind, k_ind, x1_ind, S_ind, :] = \
                    simulations[S_ind][5]  # default_vec
                c1t_arr[H_ind, k_ind, x1_ind, S_ind, :] = \
                    simulations[S_ind][6]  # c1t_vec
                c2t_arr[H_ind, k_ind, x1_ind, S_ind, :] = \
                    simulations[S_ind][7]  # c2t_vec
                Ht_arr[H_ind, k_ind, x1_ind, S_ind, :] = \
                    simulations[S_ind][8]  # Ht_vec
                wt_arr[H_ind, k_ind, x1_ind, S_ind, :] = \
                    simulations[S_ind][9]  # wt_vec
                rt_arr[H_ind, k_ind, x1_ind, S_ind, :] = \
                    simulations[S_ind][10]  # rt_vec
                k2t_arr[H_ind, k_ind, x1_ind, S_ind, :] = \
                    simulations[S_ind][11][:-1]  # k2t_vec[:-1]
                rbart_arr[H_ind, k_ind, x1_ind, S_ind, :] = \
                    simulations[S_ind][12]  # rbart_vec
                rbart_an_arr[H_ind, k_ind, x1_ind, S_ind, :] = \
                    simulations[S_ind][13]  # rbart_an_vec
                EulErr_arr[H_ind, k_ind, x1_ind, S_ind, :] = \
                    simulations[S_ind][14]  # EulErr_vec
                PathTime_arr[H_ind, k_ind, x1_ind, S_ind] = \
                    simulations[S_ind][15]  # path_time

# Print computation time
total_time = time.process_time() - start_time
funcs.print_time(total_time, 'All cases-simulations')

default_p1 = \
    np.append(np.zeros((p.Hbar_size, p.k20_size, p.x1_size, p.S, 1),
                       dtype=bool), default_arr[:, :, :, :, 1:], axis=4)
zt_arr = np.tile(zt_mat.reshape((1, 1, 1, p.S, p.T)),
                 (p.Hbar_size, p.k20_size, p.x1_size, 1, 1))
Kt_arr = (1 - default_p1) * k2t_arr
Yt_arr = (1 - default_p1) * funcs.get_Y(Kt_arr, zt_arr, p)
Ct_arr = (1 - default_p1) * funcs.get_C(c1t_arr, c2t_arr)
dict_params = \
    {
        'yrs_in_per': p.yrs_in_per,
        'beta_an': p.beta_an,
        'beta': p.beta,
        'gamma': p.gamma,
        'c_min': p.c_min,
        'K_min': p.K_min,
        'n1': p.n1,
        'n2': p.n2,
        'nvec': p.nvec,
        'alpha': p.alpha,
        'epsilon': p.epsilon,
        'delta_an': p.delta_an,
        'delta': p.delta,
        'rho_an': p.rho_an,
        'rho': p.rho,
        'mu_an': p.mu_an,
        'sigma_an': p.sigma_an,
        'sigma': p.sigma,
        'mu': p.mu,
        'A_min': p.A_min,
        'z_min': p.z_min,
        'Hbar_vec': p.Hbar_vec,
        'Hbar_size': p.Hbar_size,
        'Hbar': p.Hbar,
        'k20_vec': p.k20_vec,
        'k20_size': p.k20_size,
        'k20': p.k20,
        'x1_size': p.x1_size,
        'w1n1_avg': p.w1n1_avg,
        'x1_vec': p.x1_vec,
        'x1': p.x1,
        'x2': p.x2,
        'z0': p.z0,
        'tau': p.tau,
        'T': p.T,
        'S': p.S,
        'rand_seed': p.rand_seed
    }
dict_endog = \
    {
        'unif_mat': unif_mat,
        'zt_mat': zt_mat,
        'c1t_arr': c1t_arr,
        'c2t_arr': c2t_arr,
        'Ht_arr': Ht_arr,
        'wt_arr': wt_arr,
        'rt_arr': rt_arr,
        'rbart_arr': rbart_arr,
        'rbart_an_arr': rbart_an_arr,
        'k2t_arr': k2t_arr,
        'EulErr_arr': EulErr_arr,
        'PathTime_arr': PathTime_arr,
        'Kt_arr': Kt_arr,
        'Yt_arr': Yt_arr,
        'Ct_arr': Ct_arr,
        'default_arr': default_arr,
        'S_ind_arr': S_ind_arr,
        'total_time': total_time
    }

results_sims = {'dict_params': dict_params, 'dict_endog': dict_endog}
outputfile = os.path.join(output_dir, 'results_sims.pkl')
pickle.dump(results_sims, open(outputfile, 'wb'))
