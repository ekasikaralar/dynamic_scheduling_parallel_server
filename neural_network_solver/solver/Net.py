import numpy as np
import torch
from torch import nn

class Y_Net(nn.Module):
    # neural network for the value function approximation
    def __init__(self, config):
        super(Y_Net, self).__init__()
        self.config = config
        self.net_config = config.net_config  # Neural Network Hyperparameters
        self.eqn_config = config.eqn_config  # PDE-related system parameters
        self.dim = self.eqn_config.dim  # Dimensionality of the PDE
        
        units = self.net_config.num_neurons # Neurons in each hidden layer
        num_layers = self.net_config.num_layers   # Number of hidden layers
        activation = self.net_config.activation   # Activation function for hidden layers
        slope = self.net_config.slope # Alpha for Leaky ReLU or CELU activations
        self.softplus_output = getattr(self.net_config, 'softplus_output', False)

        self.linearlayers = nn.ModuleList([nn.Linear(self.dim, units)])

        for _ in range(num_layers - 1): # num_layers is the number of hidden layers
            self.linearlayers.append(nn.Linear(units, units))

        self.linearlayers.append(nn.Linear(units, 1, bias = False)) # output dimension is 1

        if activation == 'ReLU':
            self.activation = nn.ReLU()
            init_method = nn.init.kaiming_uniform_

        elif activation == 'eLU':
            self.activation = nn.ELU()
            init_method = nn.init.kaiming_uniform_
            
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope = slope)
            init_method = nn.init.kaiming_uniform_

        elif activation == "GELU":
            self.activation = nn.GELU()
            init_method = nn.init.xavier_uniform_

        elif activation == "SELU":
            self.activation = nn.SELU()
            init_method = nn.init.xavier_uniform_
        
        elif activation == "SiLU":
            self.activation = nn.SiLU()
            init_method = nn.init.xavier_uniform_
        
        elif activation == "CELU":
            self.activation = nn.CELU(alpha=slope)
            init_method = nn.init.xavier_uniform_

        else:
            raise Exception(f"{activation} not an available activation. Available options: ReLU, LeakyReLU, GELU, SELU, SiLU, CELU")
        
        for layer in self.linearlayers:
            if isinstance(layer, nn.Linear):
                init_method(layer.weight)

    
    def forward(self, x, training=False):

        # hidden layers
        for layer in self.linearlayers[:-1]:
            x = self.activation(layer(x))

        # pass through the output layer
        x = self.linearlayers[-1](x)

        if self.softplus_output:
            x = torch.nn.functional.softplus(x)

        return x



class Z_Net(nn.Module):
    # neural network for the gradient approximation
    def __init__(self, config):
        super(Z_Net, self).__init__()
        self.config = config
        self.net_config = config.net_config  # Neural Network Hyperparameters
        self.eqn_config = config.eqn_config  # PDE-related system parameters
        self.dim = self.eqn_config.dim  # Dimensionality of the PDE
        
        units = self.net_config.num_neurons # Neurons in each hidden layer
        num_layers = self.net_config.num_layers   # Number of hidden layers
        activation = self.net_config.activation   # Activation function for hidden layers
        slope = self.net_config.slope # Alpha for Leaky ReLU or CELU activations
        self.softplus_output = getattr(self.net_config, 'softplus_output', False)

        self.linearlayers = nn.ModuleList([nn.Linear(self.dim, units)])

        for _ in range(num_layers - 1):
            self.linearlayers.append(nn.Linear(units, units))

        self.linearlayers.append(nn.Linear(units, self.dim, bias=False)) # output dimension is equal to the dimension of the problem

        if activation == 'ReLU':
            self.activation = nn.ReLU()
            init_method = nn.init.kaiming_uniform_
            
        elif activation == 'eLU':
            self.activation = nn.ELU()
            init_method = nn.init.kaiming_uniform_
            
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope = slope)
            init_method = nn.init.kaiming_uniform_

        elif activation == "GELU":
            self.activation = nn.GELU()
            init_method = nn.init.xavier_uniform_

        elif activation == "SELU":
            self.activation = nn.SELU()
            init_method = nn.init.xavier_uniform_
        
        elif activation == "SiLU":
            self.activation = nn.SiLU()
            init_method = nn.init.xavier_uniform_
        
        elif activation == "CELU":
            self.activation = nn.CELU(alpha=slope)
            init_method = nn.init.xavier_uniform_
        else:
            raise Exception(f"{activation} not an available activation. Available options: ReLU, LeakyReLU, GELU, SELU, SiLU, CELU")
        
        for layer in self.linearlayers:
            if isinstance(layer, nn.Linear):
                init_method(layer.weight)

    
    def forward(self, x, training=False):

        # hidden layers
        for layer in self.linearlayers[:-1]:
            x = self.activation(layer(x))

        # pass through the output layer
        x = self.linearlayers[-1](x)

        if self.softplus_output:
            x = torch.nn.functional.softplus(x)

        return x