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

        # Firm parameters
        self.alpha = 1 / 3
        self.epsilon = 1.0

        # Aggregate shock z parameters
        self.rho_an = 0.95
        self.rho = self.rho_an ** self.yrs_in_per
        self.mu_an = 0.0
        self.sigma_an =  0.2  # 0.4946
        rho_sum = 0.0
        rho2_sum = 0.0
        for y_ind in range(self.yrs_in_per):
            rho_sum += self.rho_an ** y_ind
            rho2_sum += self.rho_an ** (2 * y_ind)
        self.sigma = np.sqrt(rho2_sum * (self.sigma_an ** 2))
        self.mu = self.mu_an * rho_sum

        # Set government parameters, transfer parameters, and initial values
        self.Hbar_vec = np.array([0.0, 0.05])
        # self.Hbar_vec = np.array([0.0, 0.05, 0.11, 0.17])
        self.Hbar_size = self.Hbar_vec.shape[0]
        self.Hbar = self.Hbar_vec[0]
        self.k20_vec = np.array([0.11, 0.14])
        # self.k20_vec = np.array([0.11, 0.14, 0.17])
        self.k20_size = self.k20_vec.shape[0]
        self.k20 = self.k20_vec[0]
        self.x1_size = 2
        self.w1n1_avg = 0.1
        self.x1_vec = np.linspace(0.0, self.w1n1_avg, self.x1_size)
        self.x1 = self.x1_vec[0]
        self.z0 = self.mu

        # Set simulation parameters
        self.T = 25
        self.S = 30
        self.rand_seed = 25
