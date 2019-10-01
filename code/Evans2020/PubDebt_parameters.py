'''
Create a parameters class that instantiates initial values of all
exogenous parameters
'''

# Import packages
import numpy as np


class parameters:
    '''
    Parameters class for exogenous objects
    '''

    def __init__(self):
        '''
        Instantiate the parameters class with the input name
        paramclass_name
        '''
        # Household parameters
        self.yrs_in_per = 25
        self.beta_an = 0.96
        self.beta = self.beta_an ** self.yrs_in_per
        self.gamma = 2.2
        self.c_min = 1e-5
        self.K_min = 1e-5
        self.n1 = 1.0
        self.n2 = 0.0
        self.nvec = np.array([self.n1, self.n2])

        # Firm parameters
        self.alpha = 0.33
        self.epsilon = 1.0
        self.delta_an = 0.05
        self.delta = 1 - ((1 - self.delta_an) ** self.yrs_in_per)

        # Aggregate shock z parameters
        self.rho_an = 0.95
        self.rho = self.rho_an ** self.yrs_in_per
        self.mu_an = 0.0
        self.sigma_an = 0.4946
        rho_sum = 0.0
        rho2_sum = 0.0
        for y_ind in range(self.yrs_in_per):
            rho_sum += self.rho_an ** y_ind
            rho2_sum += self.rho_an ** (2 * y_ind)
        self.sigma = np.sqrt(rho2_sum * (self.sigma_an ** 2))
        self.mu = self.mu_an * rho_sum
        self.A_min = 0.0
        if self.A_min == 0.0:
            self.z_min = -np.inf
        elif (self.A_min > 0.0) and (self.A_min < np.exp(self.mu)):
            self.z_min = np.log(self.A_min)
        elif self.A_min >= np.exp(self.mu):
            err_msg = 'Parameter Error: A_min >= e ** (mu)'
            raise ValueError(err_msg)

        # Set government parameters, transfer parameters, and initial values
        self.Hbar_vec = np.array([0.0, 0.05, 0.11, 0.17])
        self.Hbar_size = self.Hbar_vec.shape[0]
        self.Hbar = self.Hbar_vec[0]
        self.k20_vec = np.array([0.11, 0.14, 0.17])
        self.k20_size = self.k20_vec.shape[0]
        self.k20 = self.k20_vec[0]
        self.x1_size = 3
        self.w1n1_avg = 0.1
        self.x1_vec = np.linspace(0.0, self.w1n1_avg, self.x1_size)
        self.x1 = self.x1_vec[0]
        self.x2 = 0.0
        self.z0 = self.mu
        self.tau = None

        # Set simulation parameters
        self.T = 20
        self.S = 3000
        self.rand_seed = 25
