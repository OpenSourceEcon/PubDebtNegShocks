'''
------------------------------------------------------------------------
This module contains the functions used in the Public Debt and Negative
Shocks paper
------------------------------------------------------------------------
'''

# Import packages
import numpy as np
from numba import jit
import scipy.optimize as opt
import scipy.stats as sts
import scipy.integrate as intgr

'''
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
'''


@jit
def trunc_norm_draws(unif_vals, mu, sigma, cut_lb=None, cut_ub=None):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a truncated normal
    distribution based on a normal distribution with mean mu and
    standard deviation sigma and cutoffs (cut_lb, cut_ub). These draws
    correspond to an (N x S) matrix of randomly generated draws from a
    uniform distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()

    OBJECTS CREATED WITHIN FUNCTION:
    cut_ub_cdf  = scalar in [0, 1], cdf of N(mu, sigma) at upper bound
                  cutoff of truncated normal distribution
    cut_lb_cdf  = scalar in [0, 1], cdf of N(mu, sigma) at lower bound
                  cutoff of truncated normal distribution
    unif2_vals  = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  rescaled uniform derived from original.
    tnorm_draws = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  values drawn from truncated normal PDF with base
                  normal distribution N(mu, sigma) and cutoffs
                  (cut_lb, cut_ub)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: tnorm_draws
    --------------------------------------------------------------------
    '''
    # No cutoffs: truncated normal = normal
    if (cut_lb is None) & (cut_ub is None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = 0.0
    # Lower bound truncation, no upper bound truncation
    elif (cut_lb is not None) & (cut_ub is None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)
    # Upper bound truncation, no lower bound truncation
    elif (cut_lb is None) & (cut_ub is not None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = 0.0
    # Lower bound and upper bound truncation
    elif (cut_lb is not None) & (cut_ub is not None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)

    unif2_vals = unif_vals * (cut_ub_cdf - cut_lb_cdf) + cut_lb_cdf
    tnorm_draws = sts.norm.ppf(unif2_vals, loc=mu, scale=sigma)

    return tnorm_draws


@jit
def get_Y(k2t, n_vec, zt, alpha):
    Y = np.exp(zt) * (k2t ** alpha) * (n_vec.sum() ** (1 - alpha))

    return Y


@jit
def get_C(c1t, c2t):
    C = c1t + c2t

    return C


@jit
def get_w(k2t, n_vec, zt, alpha):
    wt = (1 - alpha) * np.exp(zt) * ((k2t / n_vec.sum()) ** alpha)

    return wt


@jit
def get_r(k2t, n_vec, zt, alpha, delta):
    rt = (alpha * np.exp(zt) * ((n_vec.sum() / k2t) ** (1 - alpha)) -
          delta)

    return rt


@jit
def get_Ht(Hbar, w, n1, c_min, K_min):
    Ht = np.minimum(Hbar, w * n1 - c_min - K_min)

    return Ht


@jit
def get_Htaut(w, n1, tau):
    Ht = tau * w * n1

    return Ht


@jit
def get_Hbar_err(zt, *args):
    k2t, n_vec, c_min, K_min, Hbar, alpha = args
    n_1 = n_vec[0]
    wt = get_w(k2t, n_vec, zt, alpha)
    Hbar_err = Hbar - wt * n_1 + c_min + K_min

    return Hbar_err


@jit
def get_zstar(k2t, ztm1, args):
    n_vec, c_min, K_min, Hbar, alpha, mu, rho, sigma = args
    z_init = 1.5 * mu
    z_mu = rho * ztm1 + (1 - rho) * mu
    zst_args = (k2t, n_vec, c_min, K_min, Hbar, alpha)
    results = opt.root(get_Hbar_err, z_init, args=zst_args)
    z_star = results.x.item(0)
    eps_star = z_star - rho * ztm1 - (1 - rho) * mu
    A_star = np.exp(z_star)
    prob_shut = sts.norm.cdf(z_star, z_mu, sigma)
    if not results.success:
        err_msg = 'Root finder did not solve in get_zstar().'
        print('zstar=', z_star)
        print('Hbar_err=', results.fun.item(0))
        raise ValueError(err_msg)

    return z_star, eps_star, A_star, prob_shut


@jit
def get_c2t(k2t, zt, args):
    n_vec, c_min, K_min, tau, alpha, delta = args
    n_1, n_2 = n_vec
    wt = get_w(k2t, n_vec, zt, alpha)
    rt = get_r(k2t, n_vec, zt, alpha, delta)
    c2t = wt * n_2 + (1 + rt) * k2t + get_Htaut(wt, n_1, tau)

    return c2t


@jit
def get_MUc(c, gamma):
    '''
    --------------------------------------------------------------------
    Generate marginal utility(ies) of consumption with CRRA consumption
    utility and stitched function at lower bound such that the new
    hybrid function is defined over all consumption on the real
    line but the function has similar properties to the Inada condition.

    u'(c) = c ** (-sigma) if c >= epsilon
          = g'(c) = 2 * b2 * c + b1 if c < epsilon

        such that g'(epsilon) = u'(epsilon)
        and g''(epsilon) = u''(epsilon)

        u(c) = (c ** (1 - sigma) - 1) / (1 - sigma)
        g(c) = b2 * (c ** 2) + b1 * c + b0
    --------------------------------------------------------------------
    INPUTS:
    c  = scalar, individual consumption in a particular period
    gamma = scalar >= 1, coefficient of relative risk aversion for CRRA
            utility function: (c**(1-gamma) - 1) / (1 - gamma)
    graph = boolean, =True if want plot of stitched marginal utility of
            consumption function

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    epsilon    = scalar > 0, positive value close to zero
    c_s        = scalar, individual consumption
    c_s_cnstr  = boolean, =True if c_s < epsilon
    b1         = scalar, intercept value in linear marginal utility
    b2         = scalar, slope coefficient in linear marginal utility
    MU_c       = scalar or (p,) vector, marginal utility of consumption
                 or vector of marginal utilities of consumption
    p          = integer >= 1, number of periods remaining in lifetime
    cvec_cnstr = (p,) boolean vector, =True for values of cvec < epsilon

    FILES CREATED BY THIS FUNCTION:
        MU_c_stitched.png

    RETURNS: MU_c
    --------------------------------------------------------------------
    '''
    epsilon = 1e-5
    if c > epsilon:
        MUc = c ** (-gamma)
    elif c <= epsilon:
        b2 = (-gamma * (epsilon ** (-gamma - 1))) / 2
        b1 = (epsilon ** (-gamma)) - 2 * b2 * epsilon
        MUc = 2 * b2 * c + b1
    # print('c=', c, ', MUc=', MUc)

    return MUc


@jit
def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    This function gives the PDF of the lognormal distribution for xvals
    given mu and sigma

    (LN): f(x; mu, sigma) = (1 / (x * sigma * sqrt(2 * pi))) *
            exp((-1 / 2) * (((log(x) - mu) / sigma) ** 2))
            x in [0, infty), mu in (-infty, infty), sigma > 0
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, data
    mu    = scalar, mean of the ln(x)
    sigma = scalar > 0, standard deviation of ln(x)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals        = (N,) vector, probability of each observation given
                      the parameter values

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    pdf_vals = \
        np.float64(((1 / (np.sqrt(2 * np.pi) * sigma * xvals)) *
                    np.exp((-1.0 / 2.0) *
                           (((np.log(xvals) - mu) / sigma) ** 2))))

    return pdf_vals


@jit
def get_1pr_MU_c2_pdf(Atp1, *args):
    (k2tp1, zt, n_vec, c_min, K_min, tau, gamma, alpha, delta, mu, rho,
        sigma, A_min_cdf) = args
    ztp1 = np.log(Atp1)
    z_mu = rho * zt + (1 - rho) * mu
    c2tp1_args = (n_vec, c_min, K_min, tau, alpha, delta)
    c2tp1 = get_c2t(k2tp1, ztp1, c2tp1_args)
    rtp1 = get_r(k2tp1, n_vec, ztp1, alpha, delta)
    MU_c2tp1 = get_MUc(c2tp1, gamma)
    MU_c2tp1_pdf = (1 + rtp1) * MU_c2tp1 * (LN_pdf(Atp1, z_mu, sigma) /
                                            (1 - A_min_cdf))

    return MU_c2tp1_pdf


@jit
def get_MU_c2_pdf(Atp1, *args):
    (k2tp1, zt, n_vec, c_min, K_min, tau, gamma, alpha, delta, mu, rho,
        sigma, A_min_cdf) = args
    ztp1 = np.log(Atp1)
    z_mu = rho * zt + (1 - rho) * mu
    # ztp1 = rho * zt + (1 - rho) * mu + eps
    c2tp1_args = (n_vec, c_min, K_min, tau, alpha, delta)
    c2tp1 = get_c2t(k2tp1, ztp1, c2tp1_args)
    MU_c2tp1 = get_MUc(c2tp1, gamma)
    MU_c2tp1_pdf = MU_c2tp1 * (LN_pdf(Atp1, z_mu, sigma) /
                               (1 - A_min_cdf))

    return MU_c2tp1_pdf


@jit
def get_U_c2_pdf(Atp1, *args):
    (k2tp1, zt, n_vec, c_min, K_min, Hbar, gamma, alpha, delta, mu, rho,
        sigma, A_min_cdf) = args
    ztp1 = np.log(Atp1)
    z_mu = rho * zt + (1 - rho) * mu
    # ztp1 = rho * zt + (1 - rho) * mu + eps
    c2tp1_args = (n_vec, c_min, K_min, Hbar, alpha, delta)
    c2tp1 = get_c2t(k2tp1, ztp1, c2tp1_args)
    # print(c2tp1)
    U_c2tp1 = get_MUc(c2tp1, gamma)
    U_c2tp1_pdf = U_c2tp1 * LN_pdf(Atp1, z_mu, sigma) / (1 - A_min_cdf)

    return U_c2tp1_pdf


@jit
def get_Eul_err(k2tp1, *args):
    (k2t, zt, n_vec, c_min, K_min, Ht, tau, beta, gamma, alpha, delta,
        mu, rho, sigma, A_min) = args
    n1, n2 = n_vec
    wt = get_w(k2t, n_vec, zt, alpha)
    c1 = wt * n1 - k2tp1 - Ht
    # print('c1', c1)
    MU_c1 = get_MUc(c1, gamma)
    mu_ztp1 = rho * zt + (1 - rho) * mu
    if A_min == 0.0:
        A_min_cdf = 0.0
    elif A_min > 0.0:
        A_min_cdf = sts.norm.cdf(np.log(A_min), loc=mu_ztp1,
                                 scale=sigma)
    muc2_args = (k2tp1, zt, n_vec, c_min, K_min, tau, gamma, alpha,
                 delta, mu, rho, sigma, A_min_cdf)
    (Exp_1pr_MU_c2, _) = intgr.quad(get_1pr_MU_c2_pdf, A_min, np.inf,
                                    args=muc2_args)
    # print('Vals:', MU_c1, beta, Exp_MU_c2)
    # print('Types:', type(MU_c1), type(beta), type(Exp_MU_c2))
    Eul_err = MU_c1 - beta * Exp_1pr_MU_c2

    return Eul_err


@jit
def get_neg_lf_util(k2tp1, *args):
    (k2t, zt, n_vec, c_min, K_min, Hbar, beta, gamma, alpha, delta, mu,
        rho, sigma, A_min) = args
    n1, n2 = n_vec
    wt = get_w(k2t, n_vec, zt, alpha)
    c1 = wt * n1 - k2tp1 - Hbar
    U_c1 = ((c1 ** (1 - gamma)) - 1) / (1 - gamma)

    mu_ztp1 = rho * zt + (1 - rho) * mu
    if A_min == 0.0:
        A_min_cdf = 0.0
    elif A_min > 0.0:
        A_min_cdf = sts.norm.cdf(np.log(A_min), loc=mu_ztp1,
                                 scale=sigma)
    Uc2_args = (k2tp1, zt, n_vec, c_min, K_min, Hbar, gamma, alpha,
                delta, mu, rho, sigma, A_min_cdf)
    (Exp_U_c2, _) = intgr.quad(get_U_c2_pdf, A_min, np.inf,
                               args=Uc2_args)
    # print('Vals:', MU_c1, beta, Exp_MU_c2)
    # print('Types:', type(MU_c1), type(beta), type(Exp_MU_c2))
    neg_lf_util = U_c1 - beta * Exp_U_c2

    return neg_lf_util


@jit
def get_k2tp1(k2t, zt, args):
    '''
    --------------------------------------------------------------------
    Solve for k2tp1
    c1t + k2tp1 = wt * n1 - tau * w1 * n1
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    '''
    (n_vec, c_min, K_min, tau, beta, gamma, alpha, delta, mu, rho,
        sigma, A_min, yrs_in_per) = args
    n_1, n_2 = n_vec
    wt = get_w(k2t, n_vec, zt, alpha)
    Ht = get_Htaut(wt, n_1, tau)
    rt = get_r(k2t, n_vec, zt, alpha, delta)
    c2t = wt * n_2 + (1 + rt) * k2t + Ht
    # k2tp1_init = 0.5 * (wt * n1 - cmin - Hbar) + 0.5 * K_min
    k2tp1_max = wt * n_1 - c_min - Ht
    if (k2tp1_max - K_min < 0.01) and (k2tp1_max - K_min > 0):
        print('Too small maximization range: ' +
              'k2tp1_max - K_min too small.')
        k2tp1 = 0.5 * K_min + 0.5 * k2tp1_max
        c1t = wt * n_1 - k2tp1 - Ht
    else:
        k_args = (k2t, zt, n_vec, c_min, K_min, Ht, tau, beta, gamma, alpha,
                  delta, mu, rho, sigma, A_min)
        # print('K_min=', K_min, ', k2tp1_max=', k2tp1_max)
        results = opt.root_scalar(get_Eul_err, args=k_args, method='brenth',
                                  bracket=[K_min, k2tp1_max])
        # k2tp1 = results.root
        # results = opt.minimize_scalar(get_neg_lf_util,
        #                               bounds=(K_min, k2tp1_max),
        #                               method='bounded', args=k_args)
        k2tp1 = results.root
        # k2tp1 = results.x
        c1t = wt * n_1 - k2tp1 - Ht
        # if not results.success:
        if not results.converged:
            # err_msg = 'Root finder did not solve in get_Eul_err().'
            err_msg = 'Minimization did not solve in get_neg_lf_util().'
            raise ValueError(err_msg)

    # Compute price of riskless one-period bond
    MU_c1 = get_MUc(c1t, gamma)
    mu_ztp1 = rho * zt + (1 - rho) * mu
    if A_min == 0.0:
        A_min_cdf = 0.0
    elif A_min > 0.0:
        A_min_cdf = sts.norm.cdf(np.log(A_min), loc=mu_ztp1,
                                 scale=sigma)
    muc2_args = (k2tp1, zt, n_vec, c_min, K_min, tau, gamma, alpha,
                 delta, mu, rho, sigma, A_min_cdf)
    (Exp_MU_c2, _) = intgr.quad(get_MU_c2_pdf, A_min, np.inf,
                                args=muc2_args)
    pbar_t = beta * Exp_MU_c2 / MU_c1
    rbar_t = (1 / pbar_t) - 1
    rbar_t_an = ((1 / pbar_t) ** (1 / yrs_in_per)) - 1

    return k2tp1, c1t, Ht, c2t, wt, rt, rbar_t, rbar_t_an
