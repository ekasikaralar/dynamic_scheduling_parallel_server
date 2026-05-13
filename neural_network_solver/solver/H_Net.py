import numpy as np
import torch
import time
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import json
import pandas as pd

torch.manual_seed(73)

class H_Net(object):
    def __init__(self, config):
        self.config = config
        self.net_config = config.net_config
        self.data_config = config.data_config
        
        # Device setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

        # Load data
        self.state = torch.tensor(pd.read_csv(self.data_config.state_file, header=None).values, dtype=torch.float32).to(self.device) 
        self.gradient = torch.tensor(pd.read_csv(self.data_config.gradient_file, header=None).values, dtype=torch.float32).to(self.device) 
        self.obj_value = torch.tensor(pd.read_csv(self.data_config.obj_file, header=None).values, dtype=torch.float32).to(self.device) 

        # Dimensions of the data
        self.input_dim = self.state.size(1) + self.gradient.size(1)  # input dimension
        self.output_dim = self.obj_value.size(1) if self.obj_value.ndim > 1 else 1

        print(self.state.size(0))
        print(self.gradient.size(0))
        
        # Model 
        self.model = FeedForwardNet(config, self.input_dim, self.output_dim).to(self.device)

        # Learning rate parameters 
        self.gamma = self.net_config.gamma # Decay rate for learning rate
        self.init_learning_rate = self.net_config.init_learning # Initial learning rate
        self.milestones = self.net_config.milestones # Milestones for adjusting learning rate
        self.epochs = self.net_config.epochs # Number of epochs

        self.print_interval = self.net_config.print_interval

    def train(self, save_path_logs):

        training_history = []
        start_time = time.time()
        self.batch_size = self.net_config.batch_size

        # Initialize optimizer and learning rate scheduler, define the loss function 
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.init_learning_rate) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.milestones, gamma = self.gamma)
        loss_fn = nn.MSELoss()

        # Data
        x = torch.cat([self.state, self.gradient], dim=1)
        y = self.obj_value # output data
        
        # Create dataset
        dataset = TensorDataset(x.float(), y.float())

        # Split into train and validation sets (80/20 split)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  

        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = self.batch_size)

        # Training loop
        for epoch in range(self.epochs):
            
            self.model.train()
            train_loss = 0.0

            for xb, yb in train_loader: 
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)

            scheduler.step()

            # Validation step
            val_loss = 0.0
            with torch.no_grad():
                self.model.eval()
                for xb, yb in val_loader: 
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    loss = loss_fn(self.model(xb), yb)
                    val_loss += loss.item() * xb.size(0)
            
            train_loss /= train_size
            val_loss /= val_size

            # Log progress
            if epoch % self.print_interval == 0:
                elapsed_time = time.time() - start_time
                training_history.append([epoch, train_loss, val_loss, elapsed_time])
                print(" epoch: ", epoch, " train_loss: ", train_loss, " validation_loss: ", val_loss, " elapsed_time: ", elapsed_time)
        
        # Save the trained models for H function 
        torch.save(self.model, save_path_logs + f"h_network.pth")
        return training_history

class FeedForwardNet(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super(FeedForwardNet, self).__init__()
        self.net_config = config.net_config  # Neural Network Hyperparameters

        units = self.net_config.num_neurons # Neurons in each hidden layers
        num_layers = self.net_config.num_layers # Number of hidden layers
        
        activation = self.net_config.activation # Activation function for hidden layers

        # Input batch normalization layer
        self.input_batchnorm = nn.BatchNorm1d(input_dim)
        
        # Layers
        self.linearlayers = nn.ModuleList([nn.Linear(input_dim, units)])
        self.batchnormlayers = nn.ModuleList([nn.BatchNorm1d(units)])

        for _ in range(num_layers - 1):
            self.linearlayers.append(nn.Linear(units, units))
            self.batchnormlayers.append(nn.BatchNorm1d(units))

        self.output_layer = nn.Linear(units, output_dim, bias=False)
    
        if activation == 'ReLU':
            self.activation = nn.ReLU()
            init_method = nn.init.kaiming_uniform_

        elif activation == 'eLU':
            self.activation = nn.ELU()
            init_method = nn.init.kaiming_uniform_

        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
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
            self.activation = nn.CELU()
            init_method = nn.init.xavier_uniform_

        else:
            raise Exception(f"{activation} not an available activation. Available options: ReLU, eLU, LeakyReLU, GELU, SELU, SiLU, CELU")
        
        # Initialize weights
        for layer in self.linearlayers:
            if isinstance(layer, nn.Linear):
                init_method(layer.weight)
        init_method(self.output_layer.weight)

    def forward(self, x):
        x = self.input_batchnorm(x)  
        for linear, bn in zip(self.linearlayers, self.batchnormlayers):
            x = self.activation(bn(linear(x)))
        x = self.output_layer(x)
        return x 