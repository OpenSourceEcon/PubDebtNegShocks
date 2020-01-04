'''
------------------------------------------------------------------------
Replicate Blanchard (2019)
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
import B19_funcs as funcs
# import B19_parameters as params

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
yrs_in_per = 25
beta_an = 0.96
beta = beta_an ** yrs_in_per
gamma = 2.2
c_min = 1e-5
K_min = 1e-5

# Firm parameters
alpha = 1 / 3
epsilon = 1.0  # np.inf
delta_an = 0.0
delta = 1 - ((1 - delta_an) ** yrs_in_per)
nvec = np.array([1.0, 0.0])

# Aggregate shock z parameters
rho_an = 0.95
rho = rho_an ** yrs_in_per
mu_an = 0.0
sigma_an = 0.2  # 0.4946
rho_sum = 0.0
rho2_sum = 0.0
for y_ind in range(yrs_in_per):
    rho_sum += rho_an ** y_ind
    rho2_sum += rho_an ** (2 * y_ind)
sigma = np.sqrt(rho2_sum * (sigma_an ** 2))
mu = mu_an * rho_sum
A_min = 0.0
if A_min == 0.0:
    z_min = -np.inf
elif (A_min > 0.0) and (A_min < np.exp(mu)):
    z_min = np.log(A_min)
elif A_min >= np.exp(mu):
    err_msg = 'Parameter Error: A_min >= e ** (mu)'
    raise ValueError(err_msg)

# Set government parameters, transfer parameters, and initial values
Hbar_vec = np.array([0.0, 0.05])
# self.Hbar_vec = np.array([0.0, 0.05, 0.11, 0.17])
Hbar_size = Hbar_vec.shape[0]
Hbar = Hbar_vec[0]
tau = None
z0 = mu

# Set simulation parameters
T = 25
S = 15
rand_seed = 25

# Set up parallel processing
max_cores = multiprocessing.cpu_count()
print('Cores available on this machine =', max_cores)
num_workers = min(max_cores, S)
print('Number of workers =', num_workers)
client = Client(processes=False)


'''
------------------------------------------------------------------------
Re-set some parameter values
------------------------------------------------------------------------
'''
# print('sigma')
# print(sigma)
# print('')
# print('beta')
# print(beta)
# print('')

'''
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
'''

# Put functions hers

'''
------------------------------------------------------------------------
Calibrate beta, mu, gamma
------------------------------------------------------------------------
'''
avg_rtp1_size = 3
avg_rtp1_an_vec = np.linspace(0.00, 0.04, avg_rtp1_size)
avg_Rtp1_vec = (1 + avg_rtp1_an_vec) ** yrs_in_per
avg_rbart_size = 3
avg_rbart_an_vec = np.linspace(-0.02, 0.01, avg_rbart_size)
avg_Rbart_vec = (1 + avg_rbart_an_vec) ** yrs_in_per

# print('avg_Rtp1_vec')
# print(avg_Rtp1_vec)
# print('')
# print('avg_Rbart_vec')
# print(avg_Rbart_vec)
# print('')

avgRtp1_mat = np.tile(avg_Rtp1_vec.reshape((avg_rtp1_size, 1)),
                      (1, avg_rbart_size))
avgRbart_mat = np.tile(avg_Rbart_vec.reshape((1, avg_rbart_size)),
                       (avg_rtp1_size, 1))
avgRtp1_gt_avgRbart = avgRtp1_mat - avgRbart_mat > 0
# print(avgRtp1_gt_avgRbart)

# Calibrate mu using linear production expected MPK
mu_vec = 1.0786 * np.ones_like(avg_Rtp1_vec)
# mu_vec = np.log(avg_Rtp1_vec) - np.log(alpha) - ((sigma ** 2) / 2)
mu_mat = np.tile(mu_vec.reshape((avg_rtp1_size, 1)),
                 (1, avg_rbart_size))
mu_mat[~avgRtp1_gt_avgRbart] = np.nan

# Calibrate beta using Cobb-Douglas expected values expression for beta
beta_vec = (alpha / (1 - alpha)) * (1 / (2 * avg_Rtp1_vec))
beta_an_vec = beta_vec ** (1 / yrs_in_per)
beta_mat = np.tile(beta_vec.reshape((avg_rtp1_size, 1)),
                   (1, avg_rbart_size))
beta_mat[~avgRtp1_gt_avgRbart] = np.nan

# Calibrate gamma
gamma_mat = ((np.log(avgRtp1_mat) - np.log(avgRbart_mat)) /
             (sigma ** 2))
gamma_mat[~avgRtp1_gt_avgRbart] = np.nan

# Calibrate x_1
x1_mat = (((1 - alpha) * np.exp(mu_mat + ((sigma ** 2) / 2)) *
           ((2 * beta_mat) ** alpha)) ** (1 / (1 - alpha)))

# Calibrate kbar_2
kbar2_mat = 2 * beta_mat * x1_mat
Hbar_vec[1] = 0.05 * kbar2_mat[0, 0]  # 1.0786
# print('Hbar_vec=', Hbar_vec)

# Calibrate x_1 array for different values of x1, given calibration
x1_arr = np.zeros((avg_rtp1_size, avg_rbart_size, 3))
x1_arr[:, :, 0] = x1_mat
x1_arr[:, :, 1] = 0.5 * x1_mat
x1_arr[:, :, 2] = 0.0 * x1_mat

# Calibrate sigma vector of 5% and 10% increases
sigma_vec = np.zeros(3)
sigma_vec[0] = sigma
sigma_vec[1] = 1.05 * sigma
sigma_vec[2] = 1.10 * sigma

# Calibrate corresponding mu_arr that holds the expected value of the
# TFP shock while expanding the variance. If ExpA is the expected value of
# the TFP shock, then ExpA = exp(mu + (sig ** 2) / 2), then
# log(ExpA) = mu + (sig ** 2) / 2
ExpA = mu_mat + (sigma ** 2) / 2
mu_arr = np.zeros((avg_rtp1_size, avg_rbart_size, 3))
mu_arr[:, :, 0] = mu_mat
mu_arr[:, :, 1] = mu_mat
mu_arr[:, :, 2] = mu_mat
# mu_arr[:, :, 1] = ExpA - (sigma_vec[1] ** 2) / 2
# mu_arr[:, :, 2] = ExpA - (sigma_vec[2] ** 2) / 2

# print('mu_mat')
# print(mu_mat)
# print('')
# print('beta_mat')
# print(beta_mat)
# print('')
# print('gamma_mat')
# print(gamma_mat)
# print('')
# print('x1_mat')
# print(x1_mat)
# print('')
# print('kbar2_mat')
# print(kbar2_mat)
# print('')
# print('x1_arr 0')
# print(x1_arr[:, :, 0])
# print('')
# print('x1_arr 1')
# print(x1_arr[:, :, 1])
# print('')
# print('x1_arr 2')
# print(x1_arr[:, :, 2])
# print('')
# print('sigma_vec')
# print(sigma_vec)
# print('')
# print('mu_arr 0')
# print(mu_arr[:, :, 0])
# print('')
# print('mu_arr 1')
# print(mu_arr[:, :, 1])
# print('')
# print('mu_arr 2')
# print(mu_arr[:, :, 2])
# print('')


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
default_arr = np.zeros((Hbar_size, 2, 3, avg_rtp1_size,
                        avg_rbart_size, S, T), dtype=bool)
unif_mat = \
    sts.uniform.rvs(loc=0, scale=1, size=((S, T)),
                    random_state=rand_seed)
# First three dimensions of zt_arr correspond to mu_arr in different
# order
zt_arr = np.zeros((3, avg_rtp1_size, avg_rbart_size, S, T))
for sig_ind in range(3):
    sigma = sigma_vec[sig_ind]
    for avgrtp1_ind in range(avg_rtp1_size):
        for avgrbart_ind in range(avg_rbart_size):
            mu = mu_arr[avgrtp1_ind, avgrbart_ind, sig_ind]
            for s_ind in range(S):
                for t_ind in range(T):
                    unif = unif_mat[s_ind, t_ind]
                    if t_ind == 0 and avgRtp1_gt_avgRbart[avgrtp1_ind,
                                                          avgrbart_ind]:
                        cut_lb = z_min - mu
                        eps_t = funcs.trunc_norm_draws(unif, 0, sigma,
                                                       cut_lb)
                        z_t = mu + eps_t
                    elif ((t_ind > 0) and
                          avgRtp1_gt_avgRbart[avgrtp1_ind,
                                              avgrbart_ind]):
                        z_tm1 = zt_arr[sig_ind, avgrtp1_ind,
                                       avgrbart_ind, s_ind, t_ind - 1]
                        cut_lb = z_min - rho * z_tm1 - (1 - rho) * mu
                        eps_t = funcs.trunc_norm_draws(unif, 0, sigma,
                                                       cut_lb)
                        z_t = rho * z_tm1 + (1 - rho) * mu + eps_t
                    else:
                        z_t = np.nan
                    zt_arr[sig_ind, avgrtp1_ind, avgrbart_ind, s_ind,
                           t_ind] = z_t

c1t_arr = np.zeros_like(default_arr)
c2t_arr = np.zeros_like(default_arr)
ut_arr = np.zeros((Hbar_size, 2, 3, avg_rtp1_size, avg_rbart_size, S,
                   T - 1))
Ht_arr = np.zeros_like(default_arr)
wt_arr = np.zeros_like(default_arr)
rt_arr = np.zeros_like(default_arr)
k2t_arr = np.zeros_like(default_arr)
rbart_arr = np.zeros_like(default_arr)
rbart_an_arr = np.zeros_like(default_arr)
EulErr_arr = np.zeros_like(default_arr)
PathTime_arr = np.zeros((Hbar_size, 2, 3, avg_rtp1_size,
                         avg_rbart_size, S))
s_ind_arr = np.zeros((Hbar_size, 2, 3, avg_rtp1_size, avg_rbart_size,
                      S))
for rtp1_ind in range(avg_rtp1_size):
    for rbart_ind in range(avg_rbart_size):
        k2t_arr[:, :, :, rtp1_ind, rbart_ind, :, 0] = \
            kbar2_mat[rtp1_ind, rbart_ind]

for H_ind in range(Hbar_size):
    Hbar_in = Hbar_vec[H_ind]
    for risk_type_ind in range(2):  # 0=xval, 1=sigval
        for risk_val_ind in range(3):
            for avgrtp1_ind in range(avg_rtp1_size):
                for avgrbart_ind in range(avg_rbart_size):
                    if avgRtp1_gt_avgRbart[avgrtp1_ind, avgrbart_ind]:
                        simulations = []
                        beta_in = beta_mat[avgrtp1_ind, avgrbart_ind]
                        gamma_in = gamma_mat[avgrtp1_ind, avgrbart_ind]
                        k20_in = kbar2_mat[avgrtp1_ind, avgrbart_ind]
                        if risk_type_ind == 0:
                            mu_in = mu_mat[avgrtp1_ind, avgrbart_ind]
                            sigma_in = sigma_vec[0]
                            x1_in = x1_arr[avgrtp1_ind, avgrbart_ind,
                                           risk_val_ind]
                            z0_vec_in = zt_arr[0, avgrtp1_ind,
                                               avgrbart_ind, :, 0]
                        elif risk_type_ind == 1:
                            mu_in = mu_arr[avgrtp1_ind, avgrbart_ind,
                                           risk_val_ind]
                            sigma_in = sigma_vec[risk_val_ind]
                            x1_in = x1_mat[avgrtp1_ind, avgrbart_ind]
                            z0_vec_in = zt_arr[risk_val_ind,
                                               avgrtp1_ind,
                                               avgrbart_ind, :, 0]
                        for s_ind in range(S):
                            z0_in = z0_vec_in[s_ind]
                            if risk_type_ind == 0:
                                zt_vec_in = \
                                    zt_arr[0, avgrtp1_ind, avgrbart_ind,
                                           s_ind, :]
                            elif risk_type_ind == 1:
                                zt_vec_in = \
                                    zt_arr[risk_type_ind, avgrtp1_ind,
                                           avgrbart_ind, s_ind, :]
                            timepaths_s = delayed(funcs.sim_timepath)(
                                Hbar_in, beta_in, gamma_in, k20_in,
                                sigma_in, x1_in, T, z0_in, z_min, rho,
                                mu_in, nvec, epsilon, alpha, delta, tau,
                                c_min, K_min, A_min, yrs_in_per,
                                H_ind=H_ind,
                                risk_type_ind=risk_type_ind,
                                risk_val_ind=risk_val_ind,
                                avgrtp1_ind=avgrtp1_ind,
                                avgrbart_ind=avgrbart_ind, S_ind=s_ind,
                                zt_vec=zt_vec_in,
                                rand_seed=rand_seed)
                            simulations.append(timepaths_s)

                        simulations = delayed(simulations).compute()

                        for s_ind in range(S):
                            s_ind_arr[H_ind, risk_type_ind,
                                      risk_val_ind, avgrtp1_ind,
                                      avgrbart_ind, s_ind] = \
                                simulations[s_ind][5]  # original S_ind
                            default_arr[H_ind, risk_type_ind,
                                        risk_val_ind, avgrtp1_ind,
                                        avgrbart_ind, s_ind, :] = \
                                simulations[s_ind][7]  # default_vec
                            c1t_arr[H_ind, risk_type_ind, risk_val_ind,
                                    avgrtp1_ind, avgrbart_ind, s_ind,
                                    :] = \
                                simulations[s_ind][8]  # c1t_vec
                            c2t_arr[H_ind, risk_type_ind, risk_val_ind,
                                    avgrtp1_ind, avgrbart_ind, s_ind,
                                    :] = \
                                simulations[s_ind][9]  # c2t_vec
                            ut_arr[H_ind, risk_type_ind, risk_val_ind,
                                   avgrtp1_ind, avgrbart_ind, s_ind,
                                   :] = \
                                simulations[s_ind][10]  # ut_vec
                            Ht_arr[H_ind, risk_type_ind, risk_val_ind,
                                   avgrtp1_ind, avgrbart_ind, s_ind,
                                   :] = \
                                simulations[s_ind][11]  # Ht_vec
                            wt_arr[H_ind, risk_type_ind, risk_val_ind,
                                   avgrtp1_ind, avgrbart_ind, s_ind,
                                   :] = \
                                simulations[s_ind][12]  # wt_vec
                            rt_arr[H_ind, risk_type_ind, risk_val_ind,
                                   avgrtp1_ind, avgrbart_ind, s_ind,
                                   :] = \
                                simulations[s_ind][13]  # rt_vec
                            k2t_arr[H_ind, risk_type_ind, risk_val_ind,
                                    avgrtp1_ind, avgrbart_ind, s_ind,
                                    :] = \
                                simulations[s_ind][14][:-1]  # k2t_vec[:-1]
                            rbart_arr[H_ind, risk_type_ind,
                                      risk_val_ind, avgrtp1_ind,
                                      avgrbart_ind, s_ind, :] = \
                                simulations[s_ind][15]  # rbart_vec
                            rbart_an_arr[H_ind, risk_type_ind,
                                         risk_val_ind, avgrtp1_ind,
                                         avgrbart_ind, s_ind, :] = \
                                simulations[s_ind][16]  # rbart_an_vec
                            EulErr_arr[H_ind, risk_type_ind,
                                       risk_val_ind, avgrtp1_ind,
                                       avgrbart_ind, s_ind, :] = \
                                simulations[s_ind][17]  # EulErr_vec
                            PathTime_arr[H_ind, risk_type_ind,
                                         risk_val_ind, avgrtp1_ind,
                                         avgrbart_ind, s_ind] = \
                                simulations[s_ind][18]  # path_time

                    else:  # avg_Rtp1 <= avg_rbart
                        s_ind_arr[H_ind, risk_type_ind, risk_val_ind,
                                  avgrtp1_ind, avgrbart_ind, :] = np.nan
                        default_arr[H_ind, risk_type_ind, risk_val_ind,
                                    avgrtp1_ind, avgrbart_ind, :, :] = \
                            np.nan  # default_vec
                        c1t_arr[H_ind, risk_type_ind, risk_val_ind,
                                avgrtp1_ind, avgrbart_ind, :, :] = \
                            np.nan  # c1t_vec
                        c2t_arr[H_ind, risk_type_ind, risk_val_ind,
                                avgrtp1_ind, avgrbart_ind, :, :] = \
                            np.nan  # c2t_vec
                        ut_arr[H_ind, risk_type_ind, risk_val_ind,
                               avgrtp1_ind, avgrbart_ind, :, :] = \
                            np.nan  # ut_vec
                        Ht_arr[H_ind, risk_type_ind, risk_val_ind,
                               avgrtp1_ind, avgrbart_ind, :, :] = \
                            np.nan  # Ht_vec
                        wt_arr[H_ind, risk_type_ind, risk_val_ind,
                               avgrtp1_ind, avgrbart_ind, :, :] = \
                            np.nan  # wt_vec
                        rt_arr[H_ind, risk_type_ind, risk_val_ind,
                               avgrtp1_ind, avgrbart_ind, :, :] = \
                            np.nan  # rt_vec
                        k2t_arr[H_ind, risk_type_ind, risk_val_ind,
                                avgrtp1_ind, avgrbart_ind, :, :] = \
                            np.nan  # k2t_vec[:-1]
                        rbart_arr[H_ind, risk_type_ind, risk_val_ind,
                                  avgrtp1_ind, avgrbart_ind, :, :] = \
                            np.nan  # rbart_vec
                        rbart_an_arr[H_ind, risk_type_ind, risk_val_ind,
                                     avgrtp1_ind, avgrbart_ind, :,
                                     :] = np.nan  # rbart_an_vec
                        EulErr_arr[H_ind, risk_type_ind, risk_val_ind,
                                   avgrtp1_ind, avgrbart_ind, :, :] = \
                            np.nan  # EulErr_vec
                        PathTime_arr[H_ind, risk_type_ind, risk_val_ind,
                                     avgrtp1_ind, avgrbart_ind, :] = \
                            np.nan  # path_time

            dict_endog_new = \
                {
                    'unif_mat': unif_mat,
                    'zt_arr': zt_arr,
                    'c1t_arr': c1t_arr,
                    'c2t_arr': c2t_arr,
                    'ut_arr': ut_arr,
                    'Ht_arr': Ht_arr,
                    'wt_arr': wt_arr,
                    'rt_arr': rt_arr,
                    'rbart_arr': rbart_arr,
                    'rbart_an_arr': rbart_an_arr,
                    'k2t_arr': k2t_arr,
                    'EulErr_arr': EulErr_arr,
                    'PathTime_arr': PathTime_arr,
                    'default_arr': default_arr,
                    's_ind_arr': s_ind_arr,
                }
            exec('outputfile = os.path.join(output_dir, \'dict_endog_' +
                 str(H_ind) + str(risk_type_ind) + str(risk_val_ind) +
                 '.pkl\')')
            exec('pickle.dump(dict_endog_new, open(outputfile, ' +
                 '\'wb\'))')

# Print computation time
total_time = time.process_time() - start_time
funcs.print_time(total_time, 'All cases-simulations')

default_p1 = \
    np.append(np.zeros((H_ind, risk_type_ind, risk_val_ind, avgrtp1_ind,
                        avgrbart_ind, S, 1), dtype=bool),
              default_arr[:, :, :, :, :, :, 1:], axis=6)
zt_arr_macro = np.tile(zt_arr.reshape((1, 1, 3, avg_rtp1_size,
                                       avg_rbart_size, S, T)),
                       (Hbar_size, 2, 1, 1, 1, 1, 1))
Kt_arr = (1 - default_p1) * k2t_arr
Y_args = (nvec, epsilon, alpha)
Yt_arr = (1 - default_p1) * funcs.get_Y(Kt_arr, zt_arr_macro, Y_args)
Ct_arr = (1 - default_p1) * funcs.get_C(c1t_arr, c2t_arr)
dict_params = \
    {
        'yrs_in_per': yrs_in_per,
        'beta_an': beta_an,
        'beta': beta,
        'gamma': gamma,
        'c_min': c_min,
        'K_min': K_min,
        'nvec': nvec,
        'n1': nvec[0],
        'n2': nvec[1],
        'alpha': alpha,
        'epsilon': epsilon,
        'delta_an': delta_an,
        'delta': delta,
        'rho_an': rho_an,
        'rho': rho,
        'mu_an': mu_an,
        'sigma_an': sigma_an,
        'sigma': sigma,
        'mu': mu,
        'A_min': A_min,
        'z_min': z_min,
        'Hbar_vec': Hbar_vec,
        'Hbar_size': Hbar_size,
        'Hbar': Hbar,
        'tau': tau,
        'T': T,
        'S': S,
        'rand_seed': rand_seed,
        'max_cores': max_cores,
        'num_workers': num_workers,
        'avg_rtp1_size': avg_rtp1_size,
        'avg_rtp1_an_vec': avg_rtp1_an_vec,
        'avg_Rtp1_vec': avg_Rtp1_vec,
        'avg_rbart_size': avg_rbart_size,
        'avg_rbart_an_vec': avg_rbart_an_vec,
        'avg_Rbart_vec': avg_Rbart_vec,
        'avgRtp1_mat': avgRtp1_mat,
        'avgRbart_mat': avgRbart_mat,
        'avgRtp1_gt_avgRbart': avgRtp1_gt_avgRbart,
        'mu_vec': mu_vec,
        'mu_mat': mu_mat,
        'mu_arr': mu_arr,
        'beta_vec': beta_vec,
        'beta_mat': beta_mat,
        'gamma_mat': gamma_mat,
        'x1_mat': x1_mat,
        'x1_arr': x1_arr,
        'kbar2_mat': kbar2_mat,
        'sigma_vec': sigma_vec,
        'ExpA': ExpA,
    }
dict_endog = \
    {
        'unif_mat': unif_mat,
        'zt_arr': zt_arr,
        'c1t_arr': c1t_arr,
        'c2t_arr': c2t_arr,
        'ut_arr': ut_arr,
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
        's_ind_arr': s_ind_arr,
        'total_time': total_time
    }

results_sims = {'dict_params': dict_params, 'dict_endog': dict_endog}
outputfile = os.path.join(output_dir, 'results_sims.pkl')
pickle.dump(results_sims, open(outputfile, 'wb'))
