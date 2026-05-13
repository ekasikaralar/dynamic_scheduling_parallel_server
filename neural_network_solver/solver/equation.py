import numpy as np
import pandas as pd
import torch

class Equation(object):
    """Base class for defining PDE-related functions."""

    def __init__(self, eqn_config):
        
        # PDE dimensions and time parameters
        self.dim = eqn_config.dim # Dimensionality of the PDE
        self.total_time = eqn_config.total_time # Total time horizon
        self.num_time_interval = eqn_config.num_time_interval  # Number of time intervals partitioning the time horizon
        self.delta_t = self.total_time / self.num_time_interval # Length of each time partition
        self.sqrt_delta_t = np.sqrt(self.delta_t) # Square root of time partition length
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load system data (rates, costs)
        self.theta = torch.tensor(pd.read_csv(eqn_config.theta_file, header=None).values, dtype=torch.float32).T.to(self.device)  # Hourly abandonment rates
        self.cost = torch.tensor(pd.read_csv(eqn_config.cost_file, header=None).values, dtype=torch.float32).T.to(self.device)  # Hourly cost rates
        self.lambd = torch.tensor(pd.read_csv(eqn_config.lambd_file, header=None).values, dtype=torch.float32).T.to(self.device)  # Hourly arrival rates
        self.zeta = torch.tensor(pd.read_csv(eqn_config.zeta_file, header=None).values, dtype=torch.float32).T.to(self.device)  # Second-order drift terms
        self.sigma = torch.sqrt(2 * self.lambd)  # Diffusion coefficient for the state process
           
    
class HJB(Equation):

    def __init__(self, eqn_config, drift_net = None, h_net = None):
        super(HJB, self).__init__(eqn_config)

        self.drift_net = drift_net if drift_net is not None else None
        self.h_net = h_net if h_net is not None else None

        if h_net is not None:
            self.h_net = h_net.to(self.device)
            self.h_net.eval()
            for param in self.h_net.parameters():
                param.requires_grad = False

        if drift_net is not None:
            self.drift_net = drift_net.to(self.device)
            self.drift_net.eval()
            for param in self.drift_net.parameters():  
                param.requires_grad = False
        
    def sample(self, num_sample):
        """ Generate sample paths for the forward SDE."""
        
        # Generate Brownian increments
        dw_sample = torch.randn(num_sample, self.dim, self.num_time_interval).to(self.device) * self.sqrt_delta_t
            
        # Initialize state trajectory array
        x_sample = torch.zeros(num_sample, self.dim, self.num_time_interval + 1).to(self.device) 
        x_sample[:, :, 0] = torch.rand(num_sample, self.dim) * 20 - 10 # Initial state (x0): Uniform distribution between -10 and 10

        # Generate the sample paths
        for i in range(self.num_time_interval):
            drift = self.zeta - self.theta * x_sample[:, :, i] + self.drift_net(x_sample[:, :, i])
            x_sample[:, :, i + 1] = x_sample[:, :, i] + drift * self.delta_t + self.sigma * dw_sample[:, :, i]
        return dw_sample, x_sample


    def f_tf(self, x, z):
        """Generator function in the PDE."""   
        first_term = torch.sum(self.drift_net(x) * z, dim = 1, keepdim = True) ## This has a dimension of num_sample x 1        
        second_term = torch.sum(self.cost * x, dim = 1, keepdim = True) ## This has a dimension of num_sample x 1
        h_term = self.h_net(torch.cat([x, z], dim=1)) if self.h_net else 0.0
        return first_term - second_term + h_term