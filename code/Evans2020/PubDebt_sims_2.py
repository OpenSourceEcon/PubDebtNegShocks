'''
------------------------------------------------------------------------
Simulate many runs of the economy under different parameterizations for
Hbar and k20, assuming z0=mu
------------------------------------------------------------------------
'''

# Import packages
import numpy as np
import pickle
import PubDebt_funcs_2 as funcs
# import PubDebt_parameters as params

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
# Household parameters
yrs_in_per = 25
beta_an = 0.96
beta = beta_an ** yrs_in_per
gamma = 2.2
c_min = 1e-5
K_min = 1e-5
n1 = 1.0
n2 = 0.0
nvec = np.array([n1, n2])

# Firm parameters
alpha = 0.33
epsilon = 1.0
delta_an = 0.05
delta = 1 - ((1 - delta_an) ** yrs_in_per)

# Aggregate shock z parameters
rho_an = 0.95
rho = rho_an ** yrs_in_per
mu_an = 0.0
sigma_an = 0.4946
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
Hbar_vec = np.array([0.0, 0.05, 0.11, 0.17])
Hbar_size = Hbar_vec.shape[0]
Hbar = Hbar_vec[0]
k20_vec = np.array([0.11, 0.14, 0.17])
k20_size = k20_vec.shape[0]
k20 = k20_vec[0]
x1_size = 3
w1n1_avg = 0.1
x1_vec = np.linspace(0.0, w1n1_avg, x1_size)
x1 = x1_vec[0]
x2 = 0.0
z0 = mu
tau = None

# Set simulation parameters
T = 20
S = 3000
rand_seed = 25

# p = params.parameters()
mod_args = \
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
     alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
     sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
     k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T, S)

(H_ind, k_ind, x1_ind, S_ind, zt_vec, default_vec, c1t_vec, c2t_vec,
    Ht_vec, wt_vec, rt_vec, k2t_vec, rbart_vec, rbart_an_vec,
    EulErr_vec, elapsed_time) = funcs.sim_timepath(mod_args,
                                                   rand_seed=rand_seed)

# Print computation time
funcs.print_time(elapsed_time, 'Single time path')

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
# start_time = timeit.default_timer()
# GameOver_arr = np.zeros((Hbar_size, k20_size, x1_size, S, T))
# unif_mat = \
#     sts.uniform.rvs(loc=0, scale=1, size=((S, T - 1)),
#                     random_state=rand_seed)
# zt_mat = np.zeros((S, T))
# zt_mat[:, 0] = z0
# for t_ind in range(1, T):
#     cut_lb_vec = z_min - rho * zt_mat[:, t_ind - 1] - (1 - rho) * mu
#     eps_t_vec = funcs.trunc_norm_draws(unif_mat[:, t_ind - 1], 0, sigma,
#                                        cut_lb_vec)
#     zt_mat[:, t_ind] = (rho * zt_mat[:, t_ind - 1] + (1 - rho) * mu +
#                         eps_t_vec)

# c1t_arr = np.zeros_like(GameOver_arr)
# c2t_arr = np.zeros_like(GameOver_arr)
# Ht_arr = np.zeros_like(GameOver_arr)
# wt_arr = np.zeros_like(GameOver_arr)
# rt_arr = np.zeros_like(GameOver_arr)
# k2t_arr = np.zeros_like(GameOver_arr)
# rbart_arr = np.zeros_like(GameOver_arr)
# rbart_an_arr = np.zeros_like(GameOver_arr)
# for k_ind in range(k20_size):
#     k2t_arr[:, k_ind, :, 0] = k20_vec[k_ind]

# for H_ind in range(Hbar_size):
#     k2tp1_args = (n_vec, c_min, K_min, Hbar_vec[H_ind], beta, gamma,
#                   alpha, delta, mu, rho, sigma, A_min, yrs_in_per)
#     for k_ind in range(k20_size):
#         for S_ind in range(S):
#             GameOver = False
#             t_ind = 0
#             while (t_ind < T - 1) and not GameOver:
#                 print('H_ind=', H_ind, ',k_ind=', k_ind,
#                       ',S_ind=', S_ind, ',t_ind=', t_ind)
#                 k2t = k2t_arr[H_ind, k_ind, S_ind, t_ind]
#                 zt = zt_mat[S_ind, t_ind]
#                 (k2tp1, c1t, Ht, c2t, wt, rt, rbart, rbart_an,
#                     GameOver) = funcs.get_k2tp1(k2t, zt, k2tp1_args)
#                 k2t_arr[H_ind, k_ind, S_ind, t_ind + 1] = k2tp1
#                 c1t_arr[H_ind, k_ind, S_ind, t_ind] = c1t
#                 Ht_arr[H_ind, k_ind, S_ind, t_ind] = Ht
#                 c2t_arr[H_ind, k_ind, S_ind, t_ind] = c2t
#                 wt_arr[H_ind, k_ind, S_ind, t_ind] = wt
#                 rt_arr[H_ind, k_ind, S_ind, t_ind] = rt
#                 rbart_arr[H_ind, k_ind, S_ind, t_ind] = rbart
#                 rbart_an_arr[H_ind, k_ind, S_ind, t_ind] = rbart_an
#                 if GameOver:
#                     GameOver_arr[H_ind, k_ind, S_ind, t_ind:] = GameOver
#                 t_ind += 1

# elapsed_time = timeit.default_timer() - start_time
# print('Elapsed time=', elapsed_time)
# GameOver_p1 = \
#     np.append(np.zeros((Hbar_size, k20_size, S, 1), dtype=bool),
#               GameOver_arr[:, :, :, 1:], axis=3)
# zt_arr = np.tile(zt_mat.reshape((1, 1, S, T)),
#                  (Hbar_size, k20_size, 1, 1))
# Kt_arr = (1 - GameOver_p1) * k2t_arr
# Yt_arr = (1 - GameOver_p1) * funcs.get_Y(Kt_arr, n_vec, zt_arr, alpha)
# Ct_arr = (1 - GameOver_p1) * funcs.get_C(c1t_arr, c2t_arr)
dict_params = \
    {'yrs_in_per': yrs_in_per, 'beta_an': beta_an, 'beta': beta,
     'gamma': gamma, 'c_min': c_min, 'K_min': K_min, 'nvec': nvec,
     'alpha': alpha, 'delta_an': delta_an, 'delta': delta,
     'rho_an': rho_an, 'rho': rho, 'mu': mu, 'sigma_an': sigma_an,
     'sigma': sigma, 'Hbar_vec': Hbar_vec, 'k20_vec': k20_vec,
     'Hbar_size': Hbar_size, 'k20_size': k20_size, 'z0': z0, 'T': T,
     'S': S, 'A_min': A_min, 'z_min': z_min, 'rand_seed': rand_seed}
dict_endog = \
    {'H_ind': H_ind, 'k_ind': k_ind, 'x1_ind': x1_ind, 'S_ind': S_ind,
     'zt_vec': zt_vec, 'default_vec': default_vec, 'c1t_vec': c1t_vec,
     'c2t_vec': c2t_vec, 'Ht_vec': Ht_vec, 'wt_vec': wt_vec,
     'rt_vec': rt_vec, 'k2t_vec': k2t_vec, 'rbart_vec': rbart_vec,
     'rbart_an_vec': rbart_an_vec, 'EulErr_vec': EulErr_vec,
     'elapsed_time': elapsed_time}
# dict_endog = \
#     {'unif_mat': unif_mat, 'zt_mat': zt_mat, 'c1t_arr': c1t_arr,
#      'c2t_arr': c2t_arr, 'Ht_arr': Ht_arr, 'wt_arr': wt_arr,
#      'rt_arr': rt_arr, 'rbart_arr': rbart_arr,
#      'rbart_an_arr': rbart_an_arr, 'k2t_arr': k2t_arr, 'Kt_arr': Kt_arr,
#      'Yt_arr': Yt_arr, 'Ct_arr': Ct_arr, 'GameOver_arr': GameOver_arr,
#      'elapsed_time': elapsed_time}

results_sims = {'dict_params': dict_params, 'dict_endog': dict_endog}
outputfile = os.path.join(output_dir, 'results_sims_2.pkl')
pickle.dump(results_sims, open(outputfile, 'wb'))

print(EulErr_vec)
