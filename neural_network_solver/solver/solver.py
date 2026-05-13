import numpy as np
import torch
import time
from torch import nn
import os
import json
import pandas as pd


from Net import Y_Net
from Net import Z_Net

torch.manual_seed(73)

class BSDE_solver(object):
    """
    BSDE Solver Class for solving PDE-related systems using neural networks.

    Attributes:
        config (object): General configuration object containing network and equation parameters.
        bsde (object): Object implementing the PDE method (e.g., HJB, FBSDE).
    """
        
    def __init__(self, config, bsde):
        # Configuration objects
        self.config = config
        self.net_config = config.net_config  # Neural Network Hyperparameters
        self.eqn_config = config.eqn_config  # PDE-related system parameters
        self.bsde = bsde  # PDE method instance

        self.model = NonsharedModel(config, bsde) # Neural network model class 

        # Device setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

        # Learning rate parameters
        self.gamma = self.net_config.gamma  # Decay rate for learning rate
        self.init_learning_rate = self.net_config.init_learning_rate  # Initial learning rate
        self.milestones = self.net_config.milestones  # Milestones for adjusting learning rate

        self.print_interval = self.net_config.print_interval
        self.lambd_const = self.net_config.lambd_const  # Negative gradient approximation constant

        # Load system data to calculate the diffusion coefficient 
        self.lambd = torch.tensor(pd.read_csv(self.eqn_config.lambd_file, header=None).values, dtype=torch.float32).T.to(self.device)  # Hourly arrival rates
        self.sigma = torch.sqrt(2 * self.lambd)  # Diffusion coefficient for the state process
           
        
    def loss_fn(self, dw, x, training):
        """
        Calculate the loss function for the given model and data.

        Args:
            dw (torch.Tensor): Brownian increments.
            x (torch.Tensor): State process.
            training: Flag indicating whether the model is in training mode.

        Returns:
            torch.Tensor: The calculated negative loss.
        """
        # Compute the model output
        delta, negative_loss = self.model((dw, x), training)

        # Mean squared error with a regularization term
        return torch.mean(delta ** 2) + self.lambd_const * negative_loss


    def save_weights_as_txt(self, network, network_name, save_path_cppweights):
        # Iterate through each layer in the network
        for j, layer in enumerate(network.linearlayers):
            # Save weights
            if hasattr(layer, 'weight'):
                weight_filename = os.path.join(save_path_cppweights, f"{network_name}_w{j}.txt")
                np.savetxt(weight_filename, layer.weight.detach().cpu().numpy(), delimiter = ",")
            # Save biases if they are not None
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_filename = os.path.join(save_path_cppweights, f"{network_name}_b{j}.txt")
                np.savetxt(bias_filename, layer.bias.detach().cpu().numpy(), delimiter = ",")
    
    def train(self, save_path_logs, save_path_cppweights):
        
        training_history = []
        start_time = time.time()
        self.valid_size = self.net_config.valid_size
        self.batch_size = self.net_config.batch_size

        # Initialize optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.init_learning_rate) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.milestones, gamma = self.gamma)
        
        # Generate validation data for the interval
        dw_valid, x_valid = self.bsde.sample(self.valid_size)

        for step in range(self.net_config.total_iterations):

            # Generate training data
            dw_train, x_train = self.bsde.sample(self.batch_size)

            # Calculate the training loss 
            loss = self.loss_fn(dw_train, x_train, training = True)
            loss_train = loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            # Validation step
            with torch.no_grad():

                loss_valid = self.loss_fn(dw_valid, x_valid, training = False)
                loss_valid = loss_valid.item()
            
            # Log progress
            if step % self.print_interval == 0:
                elapsed_time = time.time() - start_time
                training_history.append([step, loss_train, loss_valid, elapsed_time])
                print(" iter: ", step, " train_loss: ", loss_train, " validation_loss: ", loss_valid, " elapsed_time: ", elapsed_time)
        
        # Save the trained models for V and gradient of V
        torch.save(self.model.y_net, save_path_logs + f"y_network.pth")
        torch.save(self.model.z_net, save_path_logs + f"z_network.pth")

        # Save the final weights for use in C++ simulation
        self.save_weights_as_txt(self.model.z_net, f"z_network", save_path_cppweights)

        return training_history


class NonsharedModel(nn.Module):
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()

        # Configuration objects
        self.config = config
        self.net_config = config.net_config # Neural Network Hyperparameters
        self.eqn_config = config.eqn_config # PDE-related system parameters
        self.bsde = bsde # PDE method instance 
        self.alpha = self.eqn_config.discount_rate # Discount rate

        # Device setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

        # Get sigma from bsde or set to vector of ones if not found
        self.sigma = getattr(self.bsde, 'sigma', torch.ones(self.eqn_config.dim, device=self.device))
        self.sigma = self.sigma.squeeze(0)
        self.sigma = torch.diag(self.sigma) # Turn the sigma into a diagonal matrix of dimensions dim * dim
        
        self.y_net = Y_Net(config).to(self.device) # Initialize the neural network to approximate the value function
        self.z_net = Z_Net(config).to(self.device) # Initialize the neural network to approximate the gradient of the value function

    # Helper function
    def calculate_negative_loss(self, func):
        """
        Calculate the negative loss as the sum of squared negative values in the function.

        Args:
            func (torch.Tensor): Input tensor to calculate the negative loss.

        Returns:
            torch.Tensor: The calculated negative loss.
        """

        zero_func = torch.clamp(func.min(dim=1, keepdim=True)[0], max=0.0)  # Keep only negative values
        negative_loss = torch.sum(zero_func ** 2)  # Sum of squares of negative values
        return negative_loss
    

    def forward(self, inputs, training):
        """Forward pass of the model.
        
        Args:
            inputs: Tuple of (dw, x) where:
                dw: Brownian increments tensor
                x: State process tensor
            training: Boolean indicating training vs inference mode
        """
        discount = 1
        dw, x = inputs  

        y = self.y_net(x[:,:,0], training)
        negative_loss = self.calculate_negative_loss(y)
        
        # Forward propagation through time steps
        for t in range(0, self.bsde.num_time_interval):

            z = self.z_net(x[:,:,t], training)
            negative_loss += self.calculate_negative_loss(z)
            
            y = y + self.bsde.f_tf(x[:,:,t], z) * discount * self.bsde.delta_t + torch.sum(torch.matmul(z, self.sigma) * dw[:, :, t], dim = 1, keepdim=True) * discount 
            
            discount *= np.exp(-self.alpha * self.bsde.delta_t)

        y_terminal = self.y_net(x[:,:,-1], training) * discount
        negative_loss += self.calculate_negative_loss(y_terminal)
        
        delta = y_terminal - y 
        return delta, negative_loss