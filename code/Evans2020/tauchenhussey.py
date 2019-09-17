import numpy as np
import scipy.stats as st


def tauchenhussey(N, mu, rho, sigma, baseSigVal='combo'):
    '''
    --------------------------------------------------------------------
    This function solves for a Markov chain whose sample paths
    approximate those of the continuous AR(1) process
    z(t+1) = (1-rho)*mu + rho * z(t) + eps(t+1) where eps are normally
    distributed with stddev sigma. This procedure is an implementation
    of the Tauchen and Hussey (1992, Econometrica, 59:2, pp. 371-396)
    algorithm.

    [This code is an adaptation of Ben Tengelsen's adaptation of Marin
    Floden's implementation of Tauchen and Hussey (1991).]
    --------------------------------------------------------------------
    INPUTS:
    N         = integer >= 2, number of nodes for discretized support of
                z
    mu        = scalar, unconditional mean of AR(1) process
    rho       = scalar in (-1, 1), persistence of AR(1) process
    sigma     = scalar > 0, standard deviation of shocks (eps) to AR(1)
    baseSigma = scalar, std. dev. used to calculate Gaussian quadrature
                weights and nodes, i.e. to build the grid. I recommend
                that you use baseSigma = w * sigma + (1 - w) * sigmaZ
                where sigmaZ = sigma / sqrt(1 - rho **2), and
                w = 0.5 + rho / 4. Tauchen and Hussey (1991) recommend
                baseSigma = sigma, and also mention baseSigma = sigmaZ.
    Format:     {Z, Zprob} = TauchenHussey(N,mu,rho,sigma,m)
    Input:      N         scalar, number of nodes for Z

    Output:     Z       N*1 vector, nodes for Z
                Zprob   N*N matrix, transition probabilities
    Author:		Benjamin J. Tengelsen, Brigham Young University (python)
                Martin Floden, Stockholm School of Economics (original)
                January 2007 (updated August 2007)
    --------------------------------------------------------------------
    '''
    if baseSigVal == 'combo':
        sigmaZ = sigma / np.sqrt(1 - rho ** 2)
        w = 0.5 + rho / 4
        baseSigma = w * sigma + (1 - w) * sigmaZ
    if baseSigVal == 'sigma':
        baseSigma = sigma
    Z = np.zeros((N, 1))
    Zprob = np.zeros((N, N))
    [Z, w] = gaussnorm(N, mu, baseSigma ** 2)
    for i in range(N):
        for j in range(N):
            EZprime = (1 - rho) * mu + rho * Z[i]
            Zprob[i, j] = (w[j] * st.norm.pdf(Z[j], EZprime, sigma) /
                           st.norm.pdf(Z[j], mu, baseSigma))

    for i in range(N):
        Zprob[i, :] = Zprob[i, :] / sum(Zprob[i, :])

    return Z.flatten(), Zprob


def gaussnorm(n, mu, s2):
    '''
    --------------------------------------------------------------------
    Find Gaussian nodes and weights for the normal distribution
    --------------------------------------------------------------------
    n  = # nodes
    mu = mean
    s2 = variance
    --------------------------------------------------------------------
    '''
    [x0, w0] = gausshermite(n)
    x = x0 * np.sqrt(2 * s2) + mu
    # print(s2, mu)
    # print(x)
    w = w0 / np.sqrt(np.pi)
    return [x, w]


def gausshermite(n):
    '''
    Gauss Hermite nodes and weights following 'Numerical Recipes for C'
    '''
    MAXIT = 10
    EPS = 3e-14
    PIM4 = 0.7511255444649425

    x = np.zeros((n, 1))
    w = np.zeros((n, 1))

    m = int((n + 1) / 2)
    for i in range(m):
        if i == 0:
            z = np.sqrt((2 * n + 1) - 1.85575 * (2 * n + 1) **
                        (-0.16667))
        elif i == 1:
            z = z - 1.14 * (n ** 0.426) / z
        elif i == 2:
            z = 1.86 * z - 0.86 * x[0]
        elif i == 3:
            z = 1.91 * z - 0.91 * x[1]
        else:
            z = 2 * z - x[i - 1]

        for iter in range(MAXIT):
            p1 = PIM4
            p2 = 0
            for j in range(n):
                p3 = p2
                p2 = p1
                p1 = (z * np.sqrt(2 / (j + 1)) * p2 -
                      np.sqrt(float(j) / (j + 1)) * p3)
            pp = np.sqrt(2 * n) * p2
            z1 = z
            z = z1 - p1 / pp
            if np.absolute(z - z1) <= EPS:
                break

        if iter > MAXIT:
            err_msg = 'Too many iterations'
            raise ValueError(err_msg)
        x[i, 0] = z
        x[n - i - 1, 0] = -z
        w[i, 0] = 2 / pp / pp
        w[n - i - 1, 0] = w[i]

    x = x[::-1]
    return [x, w]
