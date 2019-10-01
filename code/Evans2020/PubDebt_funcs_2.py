'''
------------------------------------------------------------------------
This module contains the functions used in the Public Debt and Negative
Shocks paper
------------------------------------------------------------------------
'''

# Import packages
import time
import numpy as np
import random
import numba
import scipy.optimize as opt
import scipy.stats as sts
import scipy.integrate as intgr

'''
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
'''


@numba.jit(forceobj=True)
def print_time(seconds, type):
    '''
    --------------------------------------------------------------------
    Takes a total amount of time in seconds and prints it in terms of
    more readable units (days, hours, minutes, seconds)
    --------------------------------------------------------------------
    INPUTS:
    seconds = scalar > 0, total amount of seconds
    type    = string, either "SS" or "TPI"

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    secs = scalar > 0, remainder number of seconds
    mins = integer >= 1, remainder number of minutes
    hrs  = integer >= 1, remainder number of hours
    days = integer >= 1, number of days

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Nothing
    --------------------------------------------------------------------
    '''
    if seconds < 60:  # seconds
        secs = round(seconds, 4)
        print(type + ' computation time: ' + str(secs) + ' sec')
    elif seconds >= 60 and seconds < 3600:  # minutes
        mins = int(seconds / 60)
        secs = round(((seconds / 60) - mins) * 60, 1)
        print(type + ' computation time: ' + str(mins) + ' min, ' +
              str(secs) + ' sec')
    elif seconds >= 3600 and seconds < 86400:  # hours
        hrs = int(seconds / 3600)
        mins = int(((seconds / 3600) - hrs) * 60)
        secs = round(((seconds / 60) - hrs * 60 - mins) * 60, 1)
        print(type + ' computation time: ' + str(hrs) + ' hrs, ' +
              str(mins) + ' min, ' + str(secs) + ' sec')
    elif seconds >= 86400:  # days
        days = int(seconds / 86400)
        hrs = int(((seconds / 86400) - days) * 24)
        mins = int(((seconds / 3600) - days * 24 - hrs) * 60)
        secs = round(
            ((seconds / 60) - days * 24 * 60 - hrs * 60 - mins) * 60, 1)
        print(type + ' computation time: ' + str(days) + ' days, ' +
              str(hrs) + ' hrs, ' + str(mins) + ' min, ' +
              str(secs) + ' sec')


@numba.jit(forceobj=True)
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


@numba.jit(forceobj=True)
def get_Y(k2t, zt, args):
    '''
    --------------------------------------------------------------------
    Calculate aggregate output
    --------------------------------------------------------------------
    INPUTS:
    Kt   =
    Lt   =
    zt   =
    args =

    RETURNS: Yt
    '''
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = args
    close_tol = 1e-6
    Kt = k2t
    Lt = nvec.sum()
    At = np.exp(zt)
    if np.isclose(epsilon, 1.0, atol=close_tol):
        Yt = At * ((Kt) ** alpha) * ((Lt) ** (1 - alpha))
    elif np.isinf(epsilon):
        Yt = At * (alpha * Kt + (1 - alpha) * Lt)
    elif (
        (epsilon > 0) and
        (not np.isclose(epsilon, 1.0, atol=close_tol)) and
        (not np.isinf(epsilon))
    ):
        Yt = At * ((alpha * (Kt ** ((epsilon - 1) / epsilon)) +
                    (1 - alpha) * (Lt ** ((epsilon - 1) / epsilon))) **
                   (epsilon / (epsilon - 1)))
    elif epsilon <= 0:
        err_msg = 'ERROR get_Y(): epsilon <= 0'
        raise ValueError(err_msg)

    return Yt


@numba.jit(forceobj=True)
def get_C(c1t, c2t):
    C = c1t + c2t

    return C


@numba.jit(forceobj=True)
def get_w(k2t, zt, args):
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = args
    Lt = nvec.sum()
    At = np.exp(zt)
    if np.isinf(epsilon):
        wt = (1 - alpha) * At
    elif epsilon > 0 and not np.isinf(epsilon):
        Yt = get_Y(k2t, zt, args)
        wt = ((1 - alpha) * (At ** ((epsilon - 1) / epsilon)) *
              ((Yt / Lt) ** (1 / epsilon)))
    elif epsilon <= 0:
        err_msg = 'ERROR get_w(): epsilon <= 0'
        raise ValueError(err_msg)

    return wt


@numba.jit(forceobj=True)
def get_r(k2t, zt, args):
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = args
    Kt = k2t
    At = np.exp(zt)
    if np.isinf(epsilon):
        rt = alpha * At - delta
    elif epsilon > 0 and not np.isinf(epsilon):
        Yt = get_Y(k2t, zt, args)
        rt = (alpha * (At ** ((epsilon - 1) / epsilon)) *
              ((Yt / Kt) ** (1 / epsilon)) - delta)
    elif epsilon <= 0:
        err_msg = 'ERROR get_r(): epsilon <= 0'
        raise ValueError(err_msg)

    return rt


@numba.jit(forceobj=True)
def get_Ht(wt, args):
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = args
    default = False
    if tau is None:
        Ht = np.minimum(Hbar, wt * n1 + x1 - c_min - K_min)
        if Ht < Hbar:
            default = True
    elif tau > 0.0 and tau < 1.0:
        Ht = tau * (wt * n1 + x1)
    elif tau <= 0.0 or tau >= 1.0:
        err_msg = ('ERROR get_Ht(): tau=' + str(tau) + ' is not ' +
                   'valid value.')
        raise ValueError(err_msg)

    return Ht, default


@numba.jit(forceobj=True)
def get_Hbar_err(zt, *args):
    '''
    This function is the error function that solves for the current
    period shock that sets w * n1 + x1 - c_min - K_min = Hbar. This is
    the minimum shock that does not create default.
    '''
    k2t, p_args = args
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = p_args
    wt = get_w(k2t, zt, p_args)
    Hbar_err = Hbar - wt * n1 - x1 + c_min + K_min

    return Hbar_err


@numba.jit(forceobj=True)
def get_zstar(k2t, ztm1, args):
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = args
    z_init = 1.5 * mu
    z_mu = rho * ztm1 + (1 - rho) * mu
    zst_args = (k2t, args)
    results = opt.root(get_Hbar_err, z_init, args=zst_args)
    z_star = results.x.item(0)
    eps_star = z_star - rho * ztm1 - (1 - rho) * mu
    A_star = np.exp(z_star)
    prob_shut = sts.norm.cdf(z_star, z_mu, sigma)
    if not results.success:
        err_msg = ('zstar ERROR: Root finder did not solve in ' +
                   'get_zstar().')
        print('zstar=', z_star)
        print('Hbar_err=', results.fun.item(0))
        raise ValueError(err_msg)

    return z_star, eps_star, A_star, prob_shut


@numba.jit(forceobj=True)
def get_c2t(k2t, zt, args):
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = args
    wt = get_w(k2t, zt, args)
    rt = get_r(k2t, zt, args)
    Ht, default = get_Ht(wt, args)
    c2t = (1 + rt) * k2t + wt * n2 + x2 + Ht

    return c2t


@numba.jit(forceobj=True)
def get_MUc_CRRA(c, gamma):
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
    c_epsilon = 1e-5
    if c > c_epsilon:
        MUc = c ** (-gamma)
    elif c <= c_epsilon:
        b2 = (-gamma * (c_epsilon ** (-gamma - 1))) / 2
        b1 = (c_epsilon ** (-gamma)) - 2 * b2 * c_epsilon
        MUc = 2 * b2 * c + b1
    # print('c=', c, ', MUc=', MUc)

    return MUc


@numba.jit(forceobj=True)
def get_c1mgam(c, gamma):
    '''
    --------------------------------------------------------------------
    Generate marginal utility(ies) of consumption with CRRA consumption
    utility and stitched function at lower bound such that the new
    hybrid function is defined over all consumption on the real
    line but the function has similar properties to the Inada condition.

    f(c) = c ** (1-sigma) if c >= epsilon
    g(c) = b2 * c + b1    if c < epsilon

        such that g(epsilon) = f(epsilon)
        and g'(epsilon) = f'(epsilon)

        f(c) = c ** (1 - sigma)
        g(c) = b2 * c + b1

        s.t. b2 = (1 - gamma) * (epsilon ** (-gamma))
             b1 = epsilon**(-gamma) - (1-gamma) * (epsilon ** (1-gamma))
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
    b1         = scalar, intercept value in linear marginal utility
    b2         = scalar, slope coefficient in linear marginal utility
    MU_c       = scalar or (p,) vector, marginal utility of consumption
                 or vector of marginal utilities of consumption
    p          = integer >= 1, number of periods remaining in lifetime
    cvec_cnstr = (p,) boolean vector, =True for values of cvec < epsilon

    FILES CREATED BY THIS FUNCTION:
        MU_c_stitched.png

    RETURNS: f_c
    --------------------------------------------------------------------
    '''
    c_epsilon = 1e-5
    if c > c_epsilon:
        f_c = c ** (1 - gamma)
    elif c <= c_epsilon:
        b2 = (1 - gamma) * (c_epsilon ** (-gamma))
        b1 = (c_epsilon ** (-gamma)) - b2 * c_epsilon
        f_c = b2 * c + b1
    # print('c=', c, ', MUc=', MUc)

    return f_c


@numba.jit(forceobj=True)
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


@numba.jit(forceobj=True)
def get_1pr_MU_c2_pdf(Atp1, *args):
    '''
    This function is the target for calculating the integral
    (expectation): E[(1+r_{tp1})*(c_{2,t+1})**(-gamma)]. This function
    returns the value of
    (1 + r_{tp1})*((c_{2,t+1})**(-gamma)) * pdf(A|mu,sigma)
    for a given value of A and k2tp1
    '''
    (k2tp1, zt, A_min_cdf, p_args) = args
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = p_args
    ztp1 = np.log(Atp1)
    z_mu = rho * zt + (1 - rho) * mu
    c2tp1 = get_c2t(k2tp1, ztp1, p_args)
    rtp1 = get_r(k2tp1, ztp1, p_args)
    MU_CRRA_c2tp1 = get_MUc_CRRA(c2tp1, gamma)
    MU_c2tp1_pdf = ((1 + rtp1) * MU_CRRA_c2tp1 *
                    (LN_pdf(Atp1, z_mu, sigma) / (1 - A_min_cdf)))

    return MU_c2tp1_pdf


@numba.jit(forceobj=True)
def get_MU_c2_pdf(Atp1, *args):
    '''
    This function is the target for calculating the integral
    (expectation): E[(c_{2,t+1})**(-gamma)]. This function returns the
    value of ((c_{2,t+1})**(-gamma)) * pdf(A|mu,sigma)
    for a given value of A and k2tp1
    '''
    (k2tp1, zt, A_min_cdf, p_args) = args
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = p_args
    ztp1 = np.log(Atp1)
    z_mu = rho * zt + (1 - rho) * mu
    c2tp1 = get_c2t(k2tp1, ztp1, p_args)
    MU_CRRA_c2tp1 = get_MUc_CRRA(c2tp1, gamma)
    MU_c2tp1_pdf = (MU_CRRA_c2tp1 *
                    (LN_pdf(Atp1, z_mu, sigma) / (1 - A_min_cdf)))

    return MU_c2tp1_pdf


@numba.jit(forceobj=True)
def get_c2tp1_1mgam_pdf(Atp1, *args):
    '''
    This function is the target for calculating the integral
    (expectation): E[(c_{2,t+1})**(1-gamma)]. This function returns the
    value of ((c_{2,t+1})**(1-gamma)) * pdf(A|mu,sigma)
    for a given value of A and k2tp1
    '''
    (k2tp1, zt, A_min_cdf, p_args) = args
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = p_args
    ztp1 = np.log(Atp1)
    z_mu = rho * zt + (1 - rho) * mu
    c2tp1 = get_c2t(k2tp1, ztp1, p_args)
    c2tp1_1mgam = get_c1mgam(c2tp1, gamma)
    c2tp1_1mgam_pdf = (c2tp1_1mgam *
                       (LN_pdf(Atp1, z_mu, sigma) / (1 - A_min_cdf)))

    return c2tp1_1mgam_pdf


@numba.jit(forceobj=True)
def get_ExpMU_c2tp1_k(k2tp1, zt, args):
    (A_min_cdf, p_args) = args
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = p_args
    Ex_args = (k2tp1, zt, A_min_cdf, p_args)
    (Exp_1pr_MU_CRRA_c2, _) = intgr.quad(get_1pr_MU_c2_pdf, A_min,
                                         np.inf, args=Ex_args)
    (Exp_c2_1mgam, _) = intgr.quad(get_c2tp1_1mgam_pdf, A_min, np.inf,
                                   args=Ex_args)
    MU = Exp_1pr_MU_CRRA_c2 / Exp_c2_1mgam

    return MU


@numba.jit(forceobj=True)
def get_ExpMU_c2_b(k2tp1, zt, args):
    (A_min_cdf, p_args) = args
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = p_args
    Ex_args = (k2tp1, zt, A_min_cdf, p_args)
    (Exp_MU_CRRA_c2, _) = intgr.quad(get_MU_c2_pdf, A_min, np.inf,
                                     args=Ex_args)
    (Exp_c2_1mgam, _) = intgr.quad(get_c2tp1_1mgam_pdf, A_min, np.inf,
                                   args=Ex_args)
    MU = Exp_MU_CRRA_c2 / Exp_c2_1mgam

    return MU


@numba.jit(forceobj=True)
def get_Eul_err(k2tp1, *args):
    (k2t, zt, Ht, p_args) = args
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = p_args
    wt = get_w(k2t, zt, p_args)
    c1 = wt * n1 + x1 - k2tp1 - Ht
    MU_c1 = get_MUc_CRRA(c1, 1.0)
    mu_ztp1 = rho * zt + (1 - rho) * mu
    if A_min == 0.0:
        A_min_cdf = 0.0
    elif A_min > 0.0:
        A_min_cdf = sts.norm.cdf(np.log(A_min), loc=mu_ztp1,
                                 scale=sigma)
    MU_args = (A_min_cdf, p_args)
    Exp_MU_ctp1 = get_ExpMU_c2tp1_k(k2tp1, zt, MU_args)
    Eul_err = MU_c1 - (beta / (1 - beta)) * Exp_MU_ctp1

    return Eul_err


@numba.jit(forceobj=True)
def get_k2tp1(k2t, zt, args):
    '''
    --------------------------------------------------------------------
    Solve for k2tp1
    c1t + k2tp1 = wt * n1 - tau * w1 * n1
    --------------------------------------------------------------------
    --------------------------------------------------------------------
    '''
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = args
    krange_tol = 0.01
    wt = get_w(k2t, zt, args)
    Ht, default = get_Ht(wt, args)
    rt = get_r(k2t, zt, args)
    c2t = get_c2t(k2t, zt, args)
    if default:
        print('Default: Ht < Hbar ==> ' +
              'wt * n1 + x1 - c_min - K_min < Hbar')
        k2tp1 = K_min
        c1t = c_min
        Eul_err = 0.0
    elif not default:  # wt * n1 + x1 - c_min - K_min >= Hbar
        k2tp1_max = wt * n1 + x1 - c_min - Ht
        if (
            (k2tp1_max - K_min < krange_tol) and
            (k2tp1_max - K_min >= 0.0)
        ):
            print('Too small maximization range: ' +
                  'k2tp1_max - K_min too small.')
            k2tp1 = 0.5 * K_min + 0.5 * k2tp1_max
            c1t = wt * n1 + x1 - k2tp1 - Ht
            eul_args = (k2t, zt, Ht, args)
            Eul_err = get_Eul_err(k2tp1, *eul_args)
        elif k2tp1_max - K_min < 0.0:
            err_msg = 'Problem in get_k2tp1(): k2tp1_max - K_min <= 0.'
            raise ValueError(err_msg)
        elif k2tp1_max - K_min >= krange_tol:
            k2_init = 0.5 * k2tp1_max - 0.5 * K_min
            k_args = (k2t, zt, Ht, args)
            # print('K_min=', K_min, ', k2tp1_max=', k2tp1_max)
            results = opt.root(get_Eul_err, k2_init, args=k_args)
            # k2tp1 = results.root
            # results = opt.minimize_scalar(get_neg_lf_util,
            #                               bounds=(K_min, k2tp1_max),
            #                               method='bounded', args=k_args)
            # k2tp1 = results.root
            k2tp1 = results.x
            c1t = wt * n1 + x1 - k2tp1 - Ht
            Eul_err = results.fun
            # if not results.converged:
            if not results.success:
                err_msg = 'Root finder did not solve in get_Eul_err().'
                # err_msg = ('Minimization did not solve in ' +
                #            'get_neg_lf_util().')
                raise ValueError(err_msg)

        # Compute price of riskless one-period bond
        MU_c1 = get_MUc_CRRA(c1t, 1.0)
        mu_ztp1 = rho * zt + (1 - rho) * mu
        if A_min == 0.0:
            A_min_cdf = 0.0
        elif A_min > 0.0:
            A_min_cdf = sts.norm.cdf(np.log(A_min), loc=mu_ztp1,
                                     scale=sigma)
        Ex_args = (A_min_cdf, args)
        Exp_MU_c2tp1 = get_ExpMU_c2_b(k2tp1, zt, Ex_args)
        pbar_t = (beta / (1 - beta)) * (Exp_MU_c2tp1 / MU_c1)
        rbar_t = (1 / pbar_t) - 1
        rbar_t_an = ((1 / pbar_t) ** (1 / yrs_in_per)) - 1

    return (k2tp1, c1t, Ht, c2t, wt, rt, rbar_t, rbar_t_an, default,
            Eul_err)


@numba.jit(forceobj=True)
def sim_timepath(
    args, H_ind=None, k_ind=None, x1_ind=None, S_ind=None, zt_vec=None,
    rand_seed=None
):
    start_time = time.process_time()
    (yrs_in_per, beta_an, beta, gamma, c_min, K_min, n1, n2, nvec,
        alpha, epsilon, delta_an, delta, rho_an, rho, mu_an, sigma_an,
        sigma, mu, A_min, z_min, Hbar_vec, Hbar_size, Hbar, k20_vec,
        k20_size, k20, x1_size, w1n1_avg, x1_vec, x1, x2, z0, tau, T,
        S) = args
    if H_ind is None:
        H_ind = 0
    if k_ind is None:
        k_ind = 0
    if S_ind is None:
        S_ind = 0
    if zt_vec is None:
        if rand_seed is None:
            rand_seed = random.randint(1, 1000)
        zt_vec = np.zeros(T)
        unif_vec = sts.uniform.rvs(loc=0, scale=1, size=(T - 1),
                                   random_state=rand_seed)
        zt_vec[0] = z0
        for t_ind in range(1, T):
            cut_lb = (z_min - rho * zt_vec[t_ind - 1] - (1 - rho) * mu)
            eps_t = trunc_norm_draws(unif_vec[t_ind - 1], 0, sigma,
                                     cut_lb)
            zt_vec[t_ind] = (rho * zt_vec[t_ind - 1] + (1 - rho) * mu +
                             eps_t)

    default_vec = np.zeros(T)
    c1t_vec = np.zeros(T)
    c2t_vec = np.zeros(T)
    Ht_vec = np.zeros(T)
    wt_vec = np.zeros(T)
    rt_vec = np.zeros(T)
    k2t_vec = np.zeros(T + 1)
    EulErr_vec = np.zeros(T)
    k2t_vec[0] = k20
    rbart_vec = np.zeros(T)
    rbart_an_vec = np.zeros(T)

    default = False
    t_ind = 0
    while (t_ind < T) and not default:
        print('H_ind=', H_ind, ',k_ind=', k_ind,
              ',S_ind=', S_ind, ',t_ind=', t_ind)
        k2t = k2t_vec[t_ind]
        zt = zt_vec[t_ind]
        (k2tp1, c1t, Ht, c2t, wt, rt, rbart, rbart_an, default,
            eul_err) = get_k2tp1(k2t, zt, args)
        k2t_vec[t_ind + 1] = k2tp1
        EulErr_vec[t_ind] = eul_err
        c1t_vec[t_ind] = c1t
        Ht_vec[t_ind] = Ht
        c2t_vec[t_ind] = c2t
        wt_vec[t_ind] = wt
        rt_vec[t_ind] = rt
        rbart_vec[t_ind] = rbart
        rbart_an_vec[t_ind] = rbart_an
        if default:
            default_vec[t_ind:] = default
        t_ind += 1

    elapsed_time = time.process_time() - start_time

    return (H_ind, k_ind, x1_ind, S_ind, zt_vec, default_vec, c1t_vec,
            c2t_vec, Ht_vec, wt_vec, rt_vec, k2t_vec, rbart_vec,
            rbart_an_vec, EulErr_vec, elapsed_time)
